"""Receipt-driven collection of scientific LACE rollout cells.

Only a receipt published last by :mod:`instrument_runtime` makes a cell
collectable.  The collector re-hashes every bound file, reconstructs the
instrumented JSONL from immutable raw rows, validates exact frozen schedule
coordinates, and emits cells in schedule order.  It performs no simulator or
GPU work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
    SCIENTIFIC_SIGNATURE_CONFIG,
    validate_analysis_protocol,
    validate_instrument_against_analysis_protocol,
)
from gear_sonic.research.lace.analysis_protocol_lock import (
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    execution_cell_binding,
    load_analysis_protocol_lock,
)
from gear_sonic.research.lace.instrument import (
    INSTRUMENT_DIGEST_FIELD,
    MEASUREMENT_FAMILY_DIGEST_FIELD,
    RUNTIME_RNG_SEED_SEMANTICS,
    episode_instrument_sha256,
    episode_measurement_family_sha256,
    validate_measurement_family,
    validate_scientific_instrument,
)
from gear_sonic.research.lace.instrument_runtime import (
    INSTRUMENT_BINDING_DIGEST_FIELD,
    ROLLOUT_BINDING_DIGEST_FIELD,
    RUNTIME_HANDSHAKE_DIGEST_FIELD,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    _strict_json_loads,
    _validate_dataset_binding,
    _validate_plan_and_handshake,
    build_plan_binding,
    file_sha256,
    validate_instrument_binding,
    validate_rollout_binding,
    validate_runtime_handshake,
    validate_scientific_cache_environment,
)
from gear_sonic.research.lace.normalizer import fit_d_atlas_normalizer
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import (
    DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schedule_batch_loader import (
    load_locked_rollout_schedule,
    validate_scientific_materialized_dataset_binding,
)
from gear_sonic.research.lace.schema import (
    canonical_sha256,
    validate_robot_contract_readback,
    validate_scientific_probe_episode,
    validate_split_manifest,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

ROLLOUT_COLLECTION_KIND = "lace_scientific_rollout_collection"
ROLLOUT_COLLECTION_SCHEMA_VERSION = 2
ROLLOUT_COLLECTION_DIGEST_FIELD = "rollout_collection_sha256"

_CELL_FIELDS = (
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "repeat_index",
)
_RESERVED_INSTRUMENT_FIELDS = (
    "instrument_sha256",
    "instrument_manifest_sha256",
    "instrument_sidecar_file_sha256",
    "instrument_binding_sha256",
    "measurement_family_sha256",
    "measurement_family_manifest_sha256",
)
_COLLECTED_RECEIPT_FIELDS = {
    "cell",
    "receipt_path",
    "receipt_file_sha256",
    "rollout_binding_sha256",
    "receipt",
}
_COLLECTION_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_use",
    "schedule_sha256",
    "split_sha256",
    "split_selection_sha256",
    "reference_length_inventory_sha256",
    "analysis_protocol",
    "analysis_protocol_path",
    "analysis_protocol_file_sha256",
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    "analysis_protocol_lock",
    "analysis_protocol_lock_path",
    "analysis_protocol_lock_file_sha256",
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    "measurement_family",
    "measurement_family_sha256",
    "cell_instruments",
    "receipts",
    "episodes",
    "cell_count",
    "rollout_count",
    ROLLOUT_COLLECTION_DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_absolute(), f"{name} path must be absolute")
    _require(not path.is_symlink(), f"{name} path may not be a symlink")
    _require(path.is_file(), f"{name} is missing: {path}")
    raw = path.read_text(encoding="utf-8")
    _require(bool(raw.strip()), f"{name} must not be blank")
    value = _strict_json_loads(raw, name)
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    _require(path.is_absolute(), f"{name} path must be absolute")
    _require(not path.is_symlink(), f"{name} path may not be a symlink")
    _require(path.is_file(), f"{name} is missing: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(bool(line.strip()), f"{name} line {line_number} is blank")
            value = _strict_json_loads(line, f"{name} line {line_number}")
            _require(isinstance(value, dict), f"{name} line {line_number} is not an object")
            records.append(value)
    _require(bool(records), f"{name} must not be empty")
    return records


def _cell_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in _CELL_FIELDS}


def _cell_token(cell: Mapping[str, Any]) -> str:
    return canonical_sha256({"cell": dict(cell)})


def _expected_schedule_cells(
    schedule: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[Mapping[str, Any]]]]:
    ordered_cells: list[dict[str, Any]] = []
    rows_by_token: dict[str, list[Mapping[str, Any]]] = {}
    for row in schedule["rollouts"]:
        cell = _cell_from_row(row)
        token = _cell_token(cell)
        if token not in rows_by_token:
            ordered_cells.append(cell)
            rows_by_token[token] = []
        rows_by_token[token].append(row)
    return ordered_cells, rows_by_token


def _canonical_instrumented_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _validate_raw_episode(
    record: Mapping[str, Any],
    *,
    scheduled: Mapping[str, Any],
    instrument: Mapping[str, Any],
    robot_contract_sha256: str,
    index: int,
    allow_instrument_fields: bool = False,
    instrument_file_sha256: str | None = None,
    instrument_binding_sha256: str | None = None,
    family: Mapping[str, Any] | None = None,
) -> None:
    from gear_sonic.research.lace.atlas import (
        _validate_episode_against_schedule,
        _validate_runtime_realization,
        _validate_termination_multi_hot,
    )

    _validate_episode_against_schedule(
        record,
        scheduled,
        schedule_sha256=instrument["schedule_sha256"],
        index=index,
    )
    _require(record.get("policy_id") == scheduled["probe_policy_id"], "policy alias drifted")
    _require(record.get("scientific_runtime_ready") is True, "episode is not runtime-ready")
    _require(record.get("scientific_runtime_blockers") == [], "episode has runtime blockers")
    _require(record.get("completion_reason") == "episode_end", "episode is incomplete")
    _require(
        record.get("domain_randomization_seed_semantics") == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "episode DR seed semantics drifted",
    )
    _require(
        record.get("runtime_rng_seed_semantics") == RUNTIME_RNG_SEED_SEMANTICS,
        "episode runtime RNG semantics drifted",
    )
    _require(
        record.get("runtime_rng_seed_readback") == scheduled["runtime_rng_seed"]
        and record.get("runtime_rng_seed_readback_source") == "env.cfg.seed",
        "episode runtime RNG readback drifted",
    )
    _require(
        record.get("probe_thresholds") == instrument["probe_thresholds"]
        and record.get("probe_thresholds_sha256") == instrument["probe_thresholds_sha256"],
        "episode probe threshold contract drifted",
    )
    _validate_runtime_realization(
        record.get("domain_randomization_realization"),
        record.get("domain_randomization_realization_sha256"),
        episode=record,
        index=index,
    )
    realization = record["domain_randomization_realization"]
    _require(
        realization.get("resolved_event_configuration")
        == instrument["domain_randomization_config"],
        "episode resolved event configuration drifted",
    )
    _validate_termination_multi_hot(record, index=index)
    trace = record["termination_multi_hot"]
    _require(
        trace.get("contract") == instrument["termination_predicates"]
        and trace.get("contract_sha256") == instrument["termination_predicates_sha256"],
        "episode termination contract drifted",
    )
    validate_scientific_probe_episode(
        record,
        score_window_config=instrument["score_window_config"],
        probe_thresholds=instrument["probe_thresholds"],
    )
    robot = record.get("robot_contract_readback")
    _require(isinstance(robot, Mapping), "episode robot contract readback is missing")
    validate_robot_contract_readback(robot)
    _require(
        record.get("robot_contract_readback_sha256") == robot_contract_sha256
        and canonical_sha256(robot) == robot_contract_sha256,
        "episode robot contract readback digest drifted",
    )
    if not allow_instrument_fields:
        for field in _RESERVED_INSTRUMENT_FIELDS:
            _require(field not in record, f"raw episode already contains reserved field {field}")
        return
    _require(instrument_file_sha256 is not None, "instrument file digest is missing")
    _require(instrument_binding_sha256 is not None, "instrument binding digest is missing")
    _require(family is not None, "measurement family is missing")
    _require(
        record.get("instrument_sha256") == episode_instrument_sha256(instrument)
        and record.get("instrument_manifest_sha256") == instrument[INSTRUMENT_DIGEST_FIELD]
        and record.get("instrument_sidecar_file_sha256") == instrument_file_sha256
        and record.get("instrument_binding_sha256") == instrument_binding_sha256
        and record.get("measurement_family_sha256") == episode_measurement_family_sha256(family)
        and record.get("measurement_family_manifest_sha256")
        == family[MEASUREMENT_FAMILY_DIGEST_FIELD],
        "instrumented episode identity drifted",
    )


def _validate_cell_receipt(
    receipt_path: Path,
    *,
    expected_cell: Mapping[str, Any],
    scheduled_rows: Sequence[Mapping[str, Any]],
    schedule_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    analysis_protocol_path: Path,
    analysis_protocol_file_sha256: str,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    receipt = _load_json_object(receipt_path, "rollout receipt")
    validate_rollout_binding(receipt, verify_artifacts=True)
    _require(receipt["cell"] == dict(expected_cell), "receipt cell drifted from frozen schedule")
    expected_ids = [str(row["rollout_id"]) for row in scheduled_rows]
    _require(receipt["rollout_ids"] == expected_ids, "receipt rollout order drifted")
    _require(
        receipt["schedule_sha256"] == schedule_manifest["schedule_sha256"],
        "receipt schedule digest drifted",
    )
    _require(receipt["split_sha256"] == split_manifest["split_sha256"], "receipt split drifted")
    _require(
        receipt["split_selection_sha256"] == split_manifest["selection_sha256"],
        "receipt split selection drifted",
    )
    _require(
        receipt[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "receipt analysis protocol digest drifted",
    )
    _require(
        receipt["analysis_protocol_path"] == str(analysis_protocol_path)
        and receipt["analysis_protocol_file_sha256"] == analysis_protocol_file_sha256,
        "receipt does not bind the exact external analysis protocol file",
    )

    artifacts = receipt["artifacts"]
    locked = load_locked_rollout_schedule(
        artifacts["schedule_lock"]["path"],
        repo_root=repo_root,
    )
    _require(locked.manifest == dict(schedule_manifest), "locked schedule artifact drifted")
    _require(locked.split_manifest == dict(split_manifest), "locked split artifact drifted")
    _require(
        locked.reference_length_inventory == dict(reference_length_inventory),
        "locked reference-length inventory drifted",
    )
    _require(
        str(locked.schedule_path) == artifacts["schedule"]["path"]
        and str(locked.split_path) == artifacts["split_manifest"]["path"]
        and str(locked.reference_length_inventory_path)
        == artifacts["reference_length_inventory"]["path"]
        and str(locked.schedule_spec_path) == artifacts["schedule_spec"]["path"],
        "receipt artifact paths drifted from the validated schedule lock",
    )
    _require(
        artifacts["analysis_protocol"]["path"] == receipt["analysis_protocol_path"]
        and artifacts["analysis_protocol"]["file_sha256"]
        == receipt["analysis_protocol_file_sha256"]
        and _load_json_object(
            Path(artifacts["analysis_protocol"]["path"]),
            "analysis protocol",
        )
        == dict(analysis_protocol),
        "receipt analysis protocol artifact drifted",
    )
    handshake_path = Path(artifacts["runtime_handshake"]["path"])
    handshake = _load_json_object(handshake_path, "handshake")
    validate_runtime_handshake(handshake, repo_root=repo_root)
    _require(
        handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD] == receipt["runtime_handshake_sha256"],
        "receipt handshake digest drifted",
    )
    _require(
        handshake["rollout_binding_output_path"] == str(receipt_path),
        "handshake receipt path drifted",
    )
    _require(
        handshake["launch_plan_path"] == artifacts["launch_plan"]["path"]
        and handshake["instrument_output_path"] == artifacts["instrument"]["path"]
        and handshake["instrument_binding_output_path"] == artifacts["instrument_binding"]["path"]
        and handshake["rollout_output_path"] == artifacts["raw_rollouts"]["path"]
        and handshake["instrumented_rollout_output_path"]
        == artifacts["instrumented_rollouts"]["path"],
        "handshake output paths drifted from receipt artifacts",
    )
    _require(
        handshake["analysis_protocol_path"] == artifacts["analysis_protocol"]["path"]
        and handshake["analysis_protocol_file_sha256"]
        == artifacts["analysis_protocol"]["file_sha256"]
        and handshake["analysis_protocol_sha256"]
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "handshake analysis protocol binding drifted",
    )
    _require(
        artifacts["runtime_handshake"]["file_sha256"]
        == file_sha256(Path(artifacts["runtime_handshake"]["path"])),
        "handshake artifact digest drifted",
    )

    plan = _load_json_object(Path(artifacts["launch_plan"]["path"]), "launch plan")
    _require(
        plan.get("launch_plan_sha256")
        == canonical_sha256(plan, digest_field="launch_plan_sha256")
        == receipt["launch_plan_sha256"],
        "receipt launch plan digest drifted",
    )
    validated_handshake, validated_plan, validated_plan_digest = _validate_plan_and_handshake(
        handshake_path=handshake_path,
        expected_handshake_file_sha256=artifacts["runtime_handshake"]["file_sha256"],
        launch_plan_path=artifacts["launch_plan"]["path"],
        repo_root=repo_root,
        verify_git_head=False,
    )
    _require(validated_handshake == handshake, "collector handshake validation drifted")
    _require(validated_plan == plan, "collector launch-plan validation drifted")
    _require(
        validated_plan_digest == receipt["launch_plan_sha256"],
        "collector launch-plan digest drifted",
    )
    _require(
        handshake["schedule_sha256"] == schedule_manifest["schedule_sha256"]
        and handshake["checkpoint_sha256"] == expected_cell["checkpoint_sha256"]
        and handshake["probe_policy_id"] == expected_cell["probe_policy_id"]
        and handshake["rollout_ids"] == expected_ids
        and handshake["motion_count"] == len(scheduled_rows),
        "runtime handshake drifted from the frozen execution cell",
    )
    _require(plan.get("cell") == dict(expected_cell), "launch plan cell drifted")
    _require(
        plan.get("analysis_protocol_path") == artifacts["analysis_protocol"]["path"]
        and plan.get("analysis_protocol_file_sha256")
        == artifacts["analysis_protocol"]["file_sha256"]
        and plan.get("analysis_protocol_sha256")
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "launch plan analysis protocol binding drifted",
    )
    _require(
        plan.get("schedule_sha256") == schedule_manifest["schedule_sha256"]
        and plan.get("schedule_lock_path") == artifacts["schedule_lock"]["path"]
        and plan.get("schedule_path") == artifacts["schedule"]["path"]
        and plan.get("schedule_file_sha256") == artifacts["schedule"]["file_sha256"]
        and plan.get("schedule_spec_path") == artifacts["schedule_spec"]["path"],
        "launch plan schedule provenance drifted from the validated lock",
    )
    _require(plan.get("atlas_probe_assignments") == list(scheduled_rows), "plan rows drifted")
    _require(plan.get("rollout_ids") == expected_ids, "plan rollout ids drifted")
    expected_motion_keys = [str(row["motion_key"]) for row in scheduled_rows]
    _require(
        plan.get("motion_keys") == expected_motion_keys,
        "launch plan motion order drifted from the frozen cell",
    )
    _require(plan.get("num_envs") == len(scheduled_rows), "launch plan num_envs drifted")
    _require(
        plan.get("rollout_output_path") == artifacts["raw_rollouts"]["path"],
        "launch plan raw rollout path drifted",
    )
    launch_environment = plan.get("launch_environment")
    _require(isinstance(launch_environment, Mapping), "launch plan environment is missing")
    cache_environment = launch_environment.get("scientific_cache_environment")
    _require(isinstance(cache_environment, Mapping), "launch plan cache environment is missing")
    expected_cache_token = canonical_sha256(
        {
            "schedule_sha256": schedule_manifest["schedule_sha256"],
            "cell": dict(expected_cell),
            "plan_output": artifacts["launch_plan"]["path"],
            "rollout_output": artifacts["raw_rollouts"]["path"],
        }
    )
    _require(
        launch_environment.get("launch_cache_token") == expected_cache_token,
        "launch plan cache token drifted from the frozen execution cell",
    )
    validate_scientific_cache_environment(
        cache_environment,
        launch_cache_token=expected_cache_token,
    )
    _require(
        launch_environment.get("scientific_cache_environment_semantics")
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "launch plan cache environment semantics drifted",
    )
    _require(
        plan.get("instrument_runtime")
        == build_plan_binding(Path(artifacts["runtime_handshake"]["path"]), handshake),
        "plan runtime binding drifted",
    )
    _validate_dataset_binding(plan)
    dataset_binding = plan["dataset_binding"]
    inventory_by_motion = {
        str(row["motion_key"]): row for row in reference_length_inventory["motions"]
    }
    expected_robot_files = [
        {
            "motion_key": motion_key,
            "path": str(Path(inventory_by_motion[motion_key]["source_path"]).resolve()),
            "sha256": inventory_by_motion[motion_key]["source_file_sha256"],
        }
        for motion_key in expected_motion_keys
    ]
    _require(
        dataset_binding["robot"]["files"] == expected_robot_files,
        "launch plan robot dataset differs from the locked reference-length inventory",
    )
    validate_scientific_materialized_dataset_binding(
        reference_length_inventory,
        motion_keys=expected_motion_keys,
        dataset_binding=dataset_binding,
    )
    _require(
        canonical_sha256(plan["dataset_binding"]) == receipt["dataset_binding_sha256"],
        "receipt dataset binding drifted",
    )
    checkpoint_bundle = plan.get("checkpoint_bundle")
    _require(isinstance(checkpoint_bundle, Mapping), "plan checkpoint bundle is missing")
    _require(
        checkpoint_bundle.get("checkpoint_path") == artifacts["checkpoint"]["path"]
        and checkpoint_bundle.get("checkpoint_sha256")
        == artifacts["checkpoint"]["file_sha256"]
        == receipt["cell"]["checkpoint_sha256"]
        and checkpoint_bundle.get("config_path") == artifacts["checkpoint_config"]["path"]
        and checkpoint_bundle.get("config_sha256") == artifacts["checkpoint_config"]["file_sha256"],
        "receipt checkpoint bundle drifted",
    )

    _require(
        _load_json_object(Path(artifacts["schedule"]["path"]), "schedule")
        == dict(schedule_manifest),
        "receipt schedule artifact differs from the supplied frozen schedule",
    )
    _require(
        _load_json_object(Path(artifacts["split_manifest"]["path"]), "split manifest")
        == dict(split_manifest),
        "receipt split artifact differs from the supplied frozen split",
    )
    _require(
        _load_json_object(
            Path(artifacts["reference_length_inventory"]["path"]),
            "reference-length inventory",
        )
        == dict(reference_length_inventory),
        "receipt inventory artifact differs from the supplied frozen inventory",
    )

    instrument = _load_json_object(Path(artifacts["instrument"]["path"]), "instrument")
    validate_scientific_instrument(instrument, repo_root=repo_root)
    resolved_config = instrument["resolved_hydra_config"]
    _require(
        resolved_config.get("checkpoint") == artifacts["checkpoint"]["path"]
        and resolved_config.get("lace_instrument_handshake_path")
        == artifacts["runtime_handshake"]["path"]
        and resolved_config.get("lace_instrument_handshake_file_sha256")
        == artifacts["runtime_handshake"]["file_sha256"]
        and resolved_config.get("lace_launch_plan_path") == artifacts["launch_plan"]["path"],
        "instrument resolved runtime paths drifted from receipt artifacts",
    )
    _require(
        resolved_config.get("lace_analysis_protocol_path") == artifacts["analysis_protocol"]["path"]
        and resolved_config.get("lace_analysis_protocol_file_sha256")
        == artifacts["analysis_protocol"]["file_sha256"]
        and resolved_config.get("lace_analysis_protocol_sha256")
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "instrument resolved analysis protocol binding drifted",
    )
    validate_instrument_against_analysis_protocol(instrument, analysis_protocol)
    _require(
        instrument["git_commit"] == handshake["git_commit"],
        "instrument git commit drifted from runtime handshake",
    )
    environment_fingerprint = instrument["environment_fingerprint"]
    _require(
        environment_fingerprint.get("scientific_cache_environment") == cache_environment
        and environment_fingerprint.get("scientific_cache_environment_semantics")
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "instrument cache environment drifted from the launch plan",
    )
    _require(
        environment_fingerprint.get("process_executable") == plan["command"][0]
        and environment_fingerprint.get("process_argv") == plan["command"][1:]
        and environment_fingerprint.get("pythonpath") == launch_environment["PYTHONPATH"],
        "instrument process invocation drifted from the launch plan",
    )
    motion_config = resolved_config["manager_env"]["commands"]["motion"]
    motion_library_config = motion_config["motion_lib_cfg"]
    for config_field, dataset_group in (
        ("motion_file", "robot"),
        ("smpl_motion_file", "smpl"),
    ):
        configured_root = motion_library_config.get(config_field)
        expected_root = dataset_binding[dataset_group]["motion_root"]
        _require(
            isinstance(configured_root, str)
            and configured_root == expected_root
            and Path(configured_root).is_absolute()
            and not Path(configured_root).is_symlink()
            and Path(configured_root).resolve() == Path(configured_root)
            and Path(configured_root).is_dir(),
            f"instrument {config_field} drifted from the launch-plan dataset binding",
        )
    recorder_config = instrument["recorder_config"]
    recorder_runtime = recorder_config.get("runtime")
    _require(isinstance(recorder_runtime, Mapping), "instrument recorder runtime is missing")
    _require(
        recorder_config.get("resolved_hydra_term")
        == resolved_config["manager_env"]["recorders"]["failure_atlas"],
        "instrument recorder term drifted from resolved Hydra",
    )
    _require(
        recorder_config["resolved_hydra_term"].get("output_path")
        == artifacts["raw_rollouts"]["path"],
        "instrument recorder output path drifted",
    )
    _require(
        recorder_runtime.get("effective_probe_thresholds_sha256")
        == instrument["probe_thresholds_sha256"],
        "instrument recorder threshold readback drifted",
    )
    score_window = instrument["score_window_config"]
    _require(
        score_window.get("rule") == "fixed_window_ending_at_first_failure_or_censored_end_v1"
        and score_window.get("score_window_seconds")
        == instrument["probe_thresholds"].get("score_window_seconds")
        and score_window.get("timestep_seconds") == recorder_runtime.get("step_dt_seconds")
        and score_window.get("sample_count_rule")
        == "max(1,floor(score_window_seconds/timestep_seconds+1e-12))",
        "instrument score-window contract drifted from live recorder inputs",
    )
    instrument_digest = episode_instrument_sha256(instrument)
    _require(
        instrument[INSTRUMENT_DIGEST_FIELD] == receipt["instrument_manifest_sha256"]
        and instrument_digest == receipt["episode_instrument_sha256"],
        "receipt instrument binding drifted",
    )
    family = receipt["measurement_family"]
    _require(
        instrument["source_bundle"] == handshake["source_bundle"]
        and instrument["source_bundle_sha256"]
        == handshake["source_bundle_sha256"]
        == family["source_bundle_sha256"]
        == receipt["source_bundle_sha256"],
        "receipt source-bundle provenance chain drifted",
    )
    validate_measurement_family(family, instruments=[instrument])
    _require(
        episode_measurement_family_sha256(family) == receipt["episode_measurement_family_sha256"],
        "receipt measurement family drifted",
    )
    binding = _load_json_object(
        Path(artifacts["instrument_binding"]["path"]),
        "instrument binding",
    )
    validate_instrument_binding(binding)
    _require(
        binding[INSTRUMENT_BINDING_DIGEST_FIELD] == receipt["instrument_binding_sha256"],
        "receipt instrument binding digest drifted",
    )
    _require(binding["measurement_family"] == family, "binding measurement family drifted")
    _require(
        binding["instrument_file_sha256"] == artifacts["instrument"]["file_sha256"],
        "binding instrument file digest drifted",
    )
    _require(
        binding["launch_plan_sha256"] == receipt["launch_plan_sha256"]
        and binding["launch_plan_file_sha256"] == artifacts["launch_plan"]["file_sha256"]
        and binding["runtime_handshake_sha256"] == receipt["runtime_handshake_sha256"]
        and binding["schedule_sha256"] == receipt["schedule_sha256"]
        and binding["checkpoint_sha256"] == receipt["cell"]["checkpoint_sha256"]
        and binding["loaded_checkpoint_bundle"] == checkpoint_bundle
        and binding["dataset_binding_sha256"] == receipt["dataset_binding_sha256"]
        and binding["rollout_output_path"] == artifacts["raw_rollouts"]["path"]
        and binding["instrumented_rollout_output_path"]
        == artifacts["instrumented_rollouts"]["path"]
        and binding["instrument_output_path"] == artifacts["instrument"]["path"]
        and binding["instrument_manifest_sha256"] == receipt["instrument_manifest_sha256"]
        and binding["episode_instrument_sha256"] == receipt["episode_instrument_sha256"]
        and binding["source_bundle_sha256"] == receipt["source_bundle_sha256"],
        "receipt/instrument-binding cross-chain drifted",
    )
    _require(
        binding["analysis_protocol_path"] == artifacts["analysis_protocol"]["path"]
        and binding["analysis_protocol_file_sha256"]
        == artifacts["analysis_protocol"]["file_sha256"]
        and binding["analysis_protocol_sha256"]
        == analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "receipt/instrument-binding analysis protocol drifted",
    )

    raw_records = _load_jsonl(Path(artifacts["raw_rollouts"]["path"]), "raw rollouts")
    raw_by_id: dict[str, dict[str, Any]] = {}
    scheduled_by_id = {str(row["rollout_id"]): row for row in scheduled_rows}
    for index, record in enumerate(raw_records):
        rollout_id = record.get("rollout_id")
        _require(rollout_id in scheduled_by_id, f"raw rollout id {rollout_id!r} is unexpected")
        _require(rollout_id not in raw_by_id, f"raw rollout id {rollout_id!r} is duplicated")
        _validate_raw_episode(
            record,
            scheduled=scheduled_by_id[str(rollout_id)],
            instrument=instrument,
            robot_contract_sha256=receipt["robot_contract_readback_sha256"],
            index=index,
        )
        raw_by_id[str(rollout_id)] = record
    _require(set(raw_by_id) == set(expected_ids), "raw cell coverage is incomplete")

    enriched_records = []
    for rollout_id in expected_ids:
        enriched = dict(raw_by_id[rollout_id])
        enriched.update(
            {
                "instrument_sha256": instrument_digest,
                "instrument_manifest_sha256": instrument[INSTRUMENT_DIGEST_FIELD],
                "instrument_sidecar_file_sha256": artifacts["instrument"]["file_sha256"],
                "instrument_binding_sha256": binding[INSTRUMENT_BINDING_DIGEST_FIELD],
                "measurement_family_sha256": episode_measurement_family_sha256(family),
                "measurement_family_manifest_sha256": family[MEASUREMENT_FAMILY_DIGEST_FIELD],
            }
        )
        enriched_records.append(enriched)
    instrumented_path = Path(artifacts["instrumented_rollouts"]["path"])
    _require(
        instrumented_path.read_bytes() == _canonical_instrumented_bytes(enriched_records),
        "instrumented JSONL is not the deterministic enrichment of immutable raw",
    )
    _require(
        _load_jsonl(instrumented_path, "instrumented rollouts") == enriched_records,
        "instrumented JSONL semantic content drifted",
    )
    return receipt, instrument, enriched_records


def collect_scientific_rollouts(
    *,
    receipt_paths: Sequence[str | Path],
    schedule_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    analysis_protocol_path: str | Path,
    analysis_protocol_lock_path: str | Path,
    expected_analysis_protocol_lock_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Deep-validate and collect one receipt per exact frozen schedule cell."""

    root = Path(repo_root).expanduser().resolve()
    protocol_candidate = Path(analysis_protocol_path).expanduser()
    _require(protocol_candidate.is_absolute(), "analysis protocol path must be absolute")
    _require(not protocol_candidate.is_symlink(), "analysis protocol path may not be a symlink")
    protocol_path = protocol_candidate.resolve()
    _require(
        protocol_path == protocol_candidate and protocol_path.is_file(),
        "analysis protocol path must be canonical and existing",
    )
    protocol_file_sha256 = file_sha256(protocol_path)
    _require(
        _load_json_object(protocol_path, "analysis protocol") == dict(analysis_protocol),
        "analysis protocol bytes differ from supplied payload",
    )
    validate_split_manifest(split_manifest)
    validate_reference_length_inventory(
        reference_length_inventory,
        split_manifest=split_manifest,
        verify_digest=True,
        verify_source_files=True,
    )
    validate_rollout_schedule(
        schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        verify_digest=True,
    )
    validate_analysis_protocol(
        analysis_protocol,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
    )
    protocol_lock_path, protocol_lock, locked_protocol_path, locked_protocol = (
        load_analysis_protocol_lock(analysis_protocol_lock_path, repo_root=root)
    )
    _require(
        _sha256(
            expected_analysis_protocol_lock_sha256,
            "expected analysis protocol lock",
        )
        == protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "analysis protocol lock differs from the independently supplied preregistration digest",
    )
    _require(
        locked_protocol_path == protocol_path
        and locked_protocol == dict(analysis_protocol)
        and protocol_lock["schedule_lock"]["schedule_sha256"]
        == schedule_manifest["schedule_sha256"],
        "analysis protocol preregistration lock drifted",
    )
    _require(schedule_manifest.get("scientific_use") is True, "schedule is not scientific")
    ordered_cells, rows_by_token = _expected_schedule_cells(schedule_manifest)
    registered_cells = [execution_cell_binding(protocol_lock, cell=cell) for cell in ordered_cells]
    registered_receipt_paths = {
        Path(record["rollout_binding_output_path"]) for record in registered_cells
    }
    paths = []
    for raw_path in receipt_paths:
        expanded = Path(raw_path).expanduser()
        absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
        _require(not absolute.is_symlink(), f"receipt path may not be a symlink: {absolute}")
        resolved = absolute.resolve()
        _require(
            absolute == resolved,
            f"receipt path must be canonical and traverse no symlinks: {absolute}",
        )
        paths.append(resolved)
    _require(len(paths) == len(set(paths)), "receipt paths contain duplicates")
    _require(
        len(paths) == len(ordered_cells),
        "receipt count does not match the frozen execution-cell count",
    )
    _require(
        set(paths) == registered_receipt_paths,
        "receipt paths do not exactly match the preregistered one-attempt cell registry",
    )

    paths_by_token: dict[str, Path] = {}
    for path in paths:
        receipt = _load_json_object(path, "rollout receipt")
        validate_rollout_binding(receipt)
        token = _cell_token(receipt["cell"])
        _require(token in rows_by_token, "receipt names an unscheduled execution cell")
        _require(token not in paths_by_token, "multiple receipts claim one execution cell")
        paths_by_token[token] = path
    _require(set(paths_by_token) == set(rows_by_token), "receipts do not exactly cover schedule")

    receipts = []
    instruments: dict[str, dict[str, Any]] = {}
    episodes: list[dict[str, Any]] = []
    family: dict[str, Any] | None = None
    robot_contract_digests: set[str] = set()
    dataset_binding_digests: set[str] = set()
    for cell in ordered_cells:
        token = _cell_token(cell)
        path = paths_by_token[token]
        receipt, instrument, cell_episodes = _validate_cell_receipt(
            path,
            expected_cell=cell,
            scheduled_rows=rows_by_token[token],
            schedule_manifest=schedule_manifest,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
            analysis_protocol=analysis_protocol,
            analysis_protocol_path=protocol_path,
            analysis_protocol_file_sha256=protocol_file_sha256,
            repo_root=root,
        )
        instrument_digest = receipt["episode_instrument_sha256"]
        _require(instrument_digest not in instruments, "one instrument digest is reused by cells")
        validate_instrument_against_analysis_protocol(instrument, analysis_protocol)
        instruments[instrument_digest] = instrument
        if family is None:
            family = dict(receipt["measurement_family"])
        else:
            _require(receipt["measurement_family"] == family, "cell family digests differ")
        robot_contract_digests.add(receipt["robot_contract_readback_sha256"])
        dataset_binding_digests.add(receipt["dataset_binding_sha256"])
        receipts.append(
            {
                "cell": dict(cell),
                "receipt_path": str(path),
                "receipt_file_sha256": file_sha256(path),
                "rollout_binding_sha256": receipt[ROLLOUT_BINDING_DIGEST_FIELD],
                "receipt": receipt,
            }
        )
        episodes.extend(cell_episodes)
    assert family is not None
    _require(
        len(robot_contract_digests) == 1,
        "scientific cells do not share one verified robot contract",
    )
    _require(
        len(dataset_binding_digests) == 1,
        "scientific cells do not share one frozen paired-dataset binding",
    )
    execution_root = Path(protocol_lock["execution_registry"]["execution_root"])
    expected_cell_directories = {Path(record["cell_directory"]) for record in registered_cells}
    actual_execution_entries = set(execution_root.iterdir())
    _require(
        actual_execution_entries == expected_cell_directories
        and all(path.is_dir() and not path.is_symlink() for path in actual_execution_entries),
        "execution root contains missing, extra, or symlinked cell attempts",
    )
    for record in registered_cells:
        expected_artifacts = {
            Path(record[field])
            for field in (
                "plan_output_path",
                "rollout_output_path",
                "instrumented_rollout_output_path",
                "instrument_output_path",
                "runtime_handshake_output_path",
                "instrument_binding_output_path",
                "rollout_binding_output_path",
            )
        }
        cell_directory = Path(record["cell_directory"])
        actual_artifacts = set(cell_directory.iterdir())
        _require(
            actual_artifacts == expected_artifacts
            and all(path.is_file() and not path.is_symlink() for path in actual_artifacts),
            "preregistered cell directory contains a partial, duplicate, or extra attempt artifact",
        )
    runtime_storage_root = Path(protocol_lock["execution_registry"]["runtime_storage_root"])
    expected_cache_tokens = {
        _load_json_object(
            Path(item["receipt"]["artifacts"]["launch_plan"]["path"]),
            "launch plan",
        )["launch_environment"]["launch_cache_token"]
        for item in receipts
    }
    _require(
        {path.name for path in runtime_storage_root.iterdir()} == expected_cache_tokens
        and all(path.is_dir() and not path.is_symlink() for path in runtime_storage_root.iterdir()),
        "runtime storage root contains missing, extra, or symlinked cell cache attempts",
    )

    scheduled_by_id = {str(row["rollout_id"]): row for row in schedule_manifest["rollouts"]}
    from gear_sonic.research.lace.atlas import _validate_scientific_instruments

    validated_family, family_digest, validated_instruments = _validate_scientific_instruments(
        family,
        instruments,
        episodes=episodes,
        expected_schedule_sha256=schedule_manifest["schedule_sha256"],
        scheduled_by_rollout_id=scheduled_by_id,
    )
    episode_by_id = {str(record["rollout_id"]): record for record in episodes}
    ordered_episodes = [
        episode_by_id[str(row["rollout_id"])] for row in schedule_manifest["rollouts"]
    ]
    collection: dict[str, Any] = {
        "kind": ROLLOUT_COLLECTION_KIND,
        "schema_version": ROLLOUT_COLLECTION_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "schedule_sha256": schedule_manifest["schedule_sha256"],
        "split_sha256": split_manifest["split_sha256"],
        "split_selection_sha256": split_manifest["selection_sha256"],
        "reference_length_inventory_sha256": reference_length_inventory[
            REFERENCE_LENGTH_DIGEST_FIELD
        ],
        "analysis_protocol": dict(analysis_protocol),
        "analysis_protocol_path": str(protocol_path),
        "analysis_protocol_file_sha256": protocol_file_sha256,
        ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "analysis_protocol_lock": protocol_lock,
        "analysis_protocol_lock_path": str(protocol_lock_path),
        "analysis_protocol_lock_file_sha256": file_sha256(protocol_lock_path),
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "measurement_family": validated_family,
        "measurement_family_sha256": family_digest,
        "cell_instruments": validated_instruments,
        "receipts": receipts,
        "episodes": ordered_episodes,
        "cell_count": len(receipts),
        "rollout_count": len(ordered_episodes),
    }
    collection[ROLLOUT_COLLECTION_DIGEST_FIELD] = canonical_sha256(
        collection,
        digest_field=ROLLOUT_COLLECTION_DIGEST_FIELD,
    )
    validate_rollout_collection(
        collection,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        analysis_protocol=analysis_protocol,
        analysis_protocol_path=protocol_path,
        analysis_protocol_lock_path=protocol_lock_path,
        expected_analysis_protocol_lock_sha256=expected_analysis_protocol_lock_sha256,
        repo_root=repo_root,
    )
    return collection


def validate_rollout_collection(
    raw: Mapping[str, Any],
    *,
    schedule_manifest: Mapping[str, Any] | None = None,
    split_manifest: Mapping[str, Any] | None = None,
    reference_length_inventory: Mapping[str, Any] | None = None,
    analysis_protocol: Mapping[str, Any] | None = None,
    analysis_protocol_path: str | Path | None = None,
    analysis_protocol_lock_path: str | Path | None = None,
    expected_analysis_protocol_lock_sha256: str | None = None,
    verify_artifacts: bool = False,
    repo_root: str | Path | None = None,
) -> None:
    """Validate a stored collection, with exact schedule proof when supplied."""

    _require(isinstance(raw, Mapping), "rollout collection must be a mapping")
    collection = dict(raw)
    _require(set(collection) == _COLLECTION_FIELDS, "rollout collection fields are invalid")
    _require(collection.get("kind") == ROLLOUT_COLLECTION_KIND, "collection kind is invalid")
    _require(
        collection.get("schema_version") == ROLLOUT_COLLECTION_SCHEMA_VERSION,
        "collection schema_version is invalid",
    )
    _require(
        collection.get("artifact_mode") == "scientific"
        and collection.get("scientific_use") is True,
        "rollout collection is not scientific",
    )
    for field in (
        "schedule_sha256",
        "split_sha256",
        "split_selection_sha256",
        "reference_length_inventory_sha256",
        "analysis_protocol_file_sha256",
        ANALYSIS_PROTOCOL_DIGEST_FIELD,
        "analysis_protocol_lock_file_sha256",
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
        "measurement_family_sha256",
        ROLLOUT_COLLECTION_DIGEST_FIELD,
    ):
        _sha256(collection.get(field), f"rollout_collection.{field}")
    family = collection.get("measurement_family")
    embedded_protocol = collection.get("analysis_protocol")
    _require(isinstance(embedded_protocol, Mapping), "collection analysis protocol missing")
    validate_analysis_protocol(embedded_protocol)
    _require(
        collection[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        == embedded_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "collection analysis protocol digest drifted",
    )
    stored_protocol_path = Path(str(collection.get("analysis_protocol_path", "")))
    _require(
        stored_protocol_path.is_absolute()
        and not stored_protocol_path.is_symlink()
        and stored_protocol_path.resolve() == stored_protocol_path
        and stored_protocol_path.is_file(),
        "collection analysis protocol path is invalid",
    )
    _require(
        file_sha256(stored_protocol_path) == collection["analysis_protocol_file_sha256"]
        and _load_json_object(stored_protocol_path, "analysis protocol") == dict(embedded_protocol),
        "collection analysis protocol bytes drifted",
    )
    embedded_protocol_lock = collection.get("analysis_protocol_lock")
    _require(
        isinstance(embedded_protocol_lock, Mapping),
        "collection analysis protocol preregistration lock missing",
    )
    stored_protocol_lock_path = Path(str(collection.get("analysis_protocol_lock_path", "")))
    _require(
        stored_protocol_lock_path.is_absolute()
        and not stored_protocol_lock_path.is_symlink()
        and stored_protocol_lock_path.resolve() == stored_protocol_lock_path
        and stored_protocol_lock_path.is_file(),
        "collection analysis protocol lock path is invalid",
    )
    loaded_lock_path, loaded_lock, loaded_protocol_path, loaded_protocol = (
        load_analysis_protocol_lock(
            stored_protocol_lock_path,
            repo_root=repo_root if repo_root is not None else Path.cwd(),
        )
    )
    _require(
        loaded_lock_path == stored_protocol_lock_path
        and loaded_lock == dict(embedded_protocol_lock)
        and loaded_protocol_path == stored_protocol_path
        and loaded_protocol == dict(embedded_protocol)
        and file_sha256(stored_protocol_lock_path)
        == collection["analysis_protocol_lock_file_sha256"]
        and loaded_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
        == collection[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "collection analysis protocol preregistration lock drifted",
    )
    instruments = collection.get("cell_instruments")
    _require(isinstance(family, Mapping), "collection measurement family is missing")
    _require(isinstance(instruments, Mapping) and instruments, "collection instruments missing")
    for digest, instrument in instruments.items():
        _sha256(digest, "collection instrument key")
        _require(isinstance(instrument, Mapping), "collection instrument is invalid")
        validate_scientific_instrument(instrument)
        validate_instrument_against_analysis_protocol(instrument, embedded_protocol)
        _require(episode_instrument_sha256(instrument) == digest, "instrument key drifted")
    validate_measurement_family(family, instruments=list(instruments.values()))
    _require(
        episode_measurement_family_sha256(family) == collection["measurement_family_sha256"],
        "collection measurement family digest drifted",
    )
    receipts = collection.get("receipts")
    episodes = collection.get("episodes")
    _require(isinstance(receipts, list) and receipts, "collection receipts are missing")
    _require(isinstance(episodes, list) and episodes, "collection episodes are missing")
    _require(collection.get("cell_count") == len(receipts), "collection cell count drifted")
    _require(collection.get("rollout_count") == len(episodes), "collection rollout count drifted")
    receipt_paths: set[str] = set()
    receipt_digests: set[str] = set()
    stored_dataset_binding_digests: set[str] = set()
    for index, item in enumerate(receipts):
        _require(
            isinstance(item, Mapping) and set(item) == _COLLECTED_RECEIPT_FIELDS,
            f"collection receipt {index} is invalid",
        )
        path = item.get("receipt_path")
        _require(isinstance(path, str) and Path(path).is_absolute(), "receipt path is invalid")
        _require(path not in receipt_paths, "collection receipt path is duplicated")
        receipt_paths.add(path)
        _sha256(item.get("receipt_file_sha256"), "collection receipt file sha256")
        receipt = item.get("receipt")
        _require(isinstance(receipt, Mapping), "embedded receipt is invalid")
        validate_rollout_binding(receipt)
        _require(item.get("cell") == receipt["cell"], "collected receipt cell drifted")
        digest = receipt[ROLLOUT_BINDING_DIGEST_FIELD]
        _require(item.get("rollout_binding_sha256") == digest, "receipt digest drifted")
        _require(digest not in receipt_digests, "collection receipt digest is duplicated")
        receipt_digests.add(digest)
        _require(receipt["measurement_family"] == family, "receipt family drifted")
        stored_dataset_binding_digests.add(
            _sha256(receipt.get("dataset_binding_sha256"), "receipt dataset binding")
        )
        _require(
            receipt["episode_instrument_sha256"] in instruments,
            "receipt instrument is absent from collection",
        )
    _require(
        len(stored_dataset_binding_digests) == 1,
        "stored collection does not share one paired-dataset binding",
    )
    _require(
        collection[ROLLOUT_COLLECTION_DIGEST_FIELD]
        == canonical_sha256(collection, digest_field=ROLLOUT_COLLECTION_DIGEST_FIELD),
        "rollout_collection_sha256 mismatch",
    )

    supplied = (
        schedule_manifest is not None,
        split_manifest is not None,
        reference_length_inventory is not None,
        analysis_protocol is not None,
        analysis_protocol_path is not None,
        analysis_protocol_lock_path is not None,
        expected_analysis_protocol_lock_sha256 is not None,
    )
    _require(all(supplied) or not any(supplied), "collection external validators are all-or-none")
    if not all(supplied):
        _require(
            not verify_artifacts,
            "artifact verification requires schedule, split, and inventory",
        )
        return
    assert schedule_manifest is not None
    assert split_manifest is not None
    assert reference_length_inventory is not None
    assert analysis_protocol is not None
    assert analysis_protocol_path is not None
    assert analysis_protocol_lock_path is not None
    assert expected_analysis_protocol_lock_sha256 is not None
    external_protocol_path = Path(analysis_protocol_path).expanduser()
    _require(
        external_protocol_path.is_absolute()
        and not external_protocol_path.is_symlink()
        and external_protocol_path.resolve() == stored_protocol_path,
        "external analysis protocol path drifted",
    )
    validate_analysis_protocol(
        analysis_protocol,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
    )
    _require(dict(analysis_protocol) == embedded_protocol, "external analysis protocol drifted")
    external_lock_path = Path(analysis_protocol_lock_path).expanduser()
    _require(
        external_lock_path.is_absolute()
        and not external_lock_path.is_symlink()
        and external_lock_path.resolve() == stored_protocol_lock_path,
        "external analysis protocol lock path drifted",
    )
    _require(
        _sha256(
            expected_analysis_protocol_lock_sha256,
            "expected analysis protocol lock",
        )
        == embedded_protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "external preregistration digest does not match the analysis protocol lock",
    )
    validate_split_manifest(split_manifest)
    validate_reference_length_inventory(
        reference_length_inventory,
        split_manifest=split_manifest,
        verify_digest=True,
    )
    validate_rollout_schedule(
        schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        verify_digest=True,
    )
    _require(
        collection["schedule_sha256"] == schedule_manifest["schedule_sha256"], "schedule drift"
    )
    _require(collection["split_sha256"] == split_manifest["split_sha256"], "split drift")
    _require(
        collection["split_selection_sha256"] == split_manifest["selection_sha256"],
        "split selection drift",
    )
    _require(
        collection["reference_length_inventory_sha256"]
        == reference_length_inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "reference-length inventory drift",
    )
    scheduled_by_id = {str(row["rollout_id"]): row for row in schedule_manifest["rollouts"]}
    expected_order = list(scheduled_by_id)
    _require(
        [episode.get("rollout_id") for episode in episodes] == expected_order,
        "collection episodes are not in frozen schedule order",
    )
    ordered_cells, rows_by_token = _expected_schedule_cells(schedule_manifest)
    _require(
        [item["cell"] for item in receipts] == ordered_cells,
        "collection receipts are not in frozen execution-cell order",
    )
    _require(
        [item["receipt_path"] for item in receipts]
        == [
            execution_cell_binding(embedded_protocol_lock, cell=cell)["rollout_binding_output_path"]
            for cell in ordered_cells
        ],
        "stored collection receipt paths drifted from preregistered execution cells",
    )
    receipt_instruments: set[str] = set()
    episode_by_id = {str(episode.get("rollout_id")): episode for episode in episodes}
    _require(
        len(episode_by_id) == len(episodes),
        "collection episodes contain duplicate rollout ids",
    )
    robot_contract_digests: set[str] = set()
    for item in receipts:
        receipt = item["receipt"]
        token = _cell_token(item["cell"])
        _require(token in rows_by_token, "collection receipt cell is not scheduled")
        _require(
            receipt["rollout_ids"] == [str(row["rollout_id"]) for row in rows_by_token[token]],
            "collection receipt rollout ids drifted from its frozen cell",
        )
        instrument_digest = receipt["episode_instrument_sha256"]
        _require(
            instrument_digest not in receipt_instruments,
            "collection reuses one instrument across execution cells",
        )
        receipt_instruments.add(instrument_digest)
        instrument = instruments[instrument_digest]
        robot_contract_digest = receipt["robot_contract_readback_sha256"]
        robot_contract_digests.add(robot_contract_digest)
        for local_index, scheduled in enumerate(rows_by_token[token]):
            rollout_id = str(scheduled["rollout_id"])
            _require(rollout_id in episode_by_id, "collection receipt episode is missing")
            _validate_raw_episode(
                episode_by_id[rollout_id],
                scheduled=scheduled,
                instrument=instrument,
                robot_contract_sha256=robot_contract_digest,
                index=local_index,
                allow_instrument_fields=True,
                instrument_file_sha256=receipt["artifacts"]["instrument"]["file_sha256"],
                instrument_binding_sha256=receipt["instrument_binding_sha256"],
                family=family,
            )
    _require(
        receipt_instruments == set(instruments),
        "collection receipts and instruments do not have one-to-one cell coverage",
    )
    _require(
        len(robot_contract_digests) == 1,
        "stored collection does not share one verified robot contract",
    )
    from gear_sonic.research.lace.atlas import _validate_scientific_instruments

    _validate_scientific_instruments(
        family,
        instruments,
        episodes=episodes,
        expected_schedule_sha256=schedule_manifest["schedule_sha256"],
        scheduled_by_rollout_id=scheduled_by_id,
    )
    if verify_artifacts:
        _require(repo_root is not None, "artifact verification requires repo_root")
        recollected = collect_scientific_rollouts(
            receipt_paths=[item["receipt_path"] for item in receipts],
            schedule_manifest=schedule_manifest,
            split_manifest=split_manifest,
            reference_length_inventory=reference_length_inventory,
            analysis_protocol=analysis_protocol,
            analysis_protocol_path=stored_protocol_path,
            analysis_protocol_lock_path=stored_protocol_lock_path,
            expected_analysis_protocol_lock_sha256=(expected_analysis_protocol_lock_sha256),
            repo_root=repo_root,
        )
        _require(
            recollected == collection,
            "stored collection differs from deep receipt-driven recollection",
        )


def build_scientific_rollout_manifest(
    collection: Mapping[str, Any],
    *,
    schedule_manifest: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    reference_length_inventory: Mapping[str, Any],
    analysis_protocol: Mapping[str, Any],
    analysis_protocol_path: str | Path,
    analysis_protocol_lock_path: str | Path,
    expected_analysis_protocol_lock_sha256: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Assemble the exact build-ready schema-v4 rollout manifest on CPU."""

    validate_rollout_collection(
        collection,
        schedule_manifest=schedule_manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
        analysis_protocol=analysis_protocol,
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_lock_path=analysis_protocol_lock_path,
        expected_analysis_protocol_lock_sha256=expected_analysis_protocol_lock_sha256,
        repo_root=repo_root,
    )
    episodes = [dict(record) for record in collection["episodes"]]
    mechanism_names = list(DEFAULT_MECHANISMS)
    normalizer = fit_d_atlas_normalizer(
        episodes,
        mechanism_names=mechanism_names,
    )
    manifest: dict[str, Any] = {
        "kind": "lace_probe_rollouts",
        "schema_version": 4,
        "artifact_mode": "scientific",
        "data_origin": SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
        "split_sha256": split_manifest["split_sha256"],
        "mechanism_names": mechanism_names,
        "probe_policies": list(schedule_manifest["probe_policies"]),
        "domain_randomization_seeds": list(schedule_manifest["domain_randomization_seeds"]),
        "rollout_schedule": None,
        "selected_motion_keys": list(schedule_manifest["selected_motion_keys"]),
        "signature_config": dict(SCIENTIFIC_SIGNATURE_CONFIG),
        "normalizer": normalizer,
        "analysis_protocol": dict(analysis_protocol),
        "analysis_protocol_path": str(Path(analysis_protocol_path).expanduser().resolve()),
        "analysis_protocol_file_sha256": collection["analysis_protocol_file_sha256"],
        ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "analysis_protocol_lock": dict(collection["analysis_protocol_lock"]),
        "analysis_protocol_lock_path": collection["analysis_protocol_lock_path"],
        "analysis_protocol_lock_file_sha256": collection["analysis_protocol_lock_file_sha256"],
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: collection[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "measurement_family": None,
        "cell_instruments": {},
        "rollout_collection": dict(collection),
        "episodes": episodes,
    }
    from gear_sonic.research.lace.atlas import build_atlas_manifest

    # The atlas builder is the authoritative wrapper validator. Constructing and
    # discarding the deterministic atlas here prevents the collector CLI from
    # publishing an operationally unusable intermediate wrapper.
    build_atlas_manifest(
        manifest,
        split_manifest,
        schedule_manifest,
        reference_length_inventory,
        repo_root=repo_root,
    )
    return manifest
