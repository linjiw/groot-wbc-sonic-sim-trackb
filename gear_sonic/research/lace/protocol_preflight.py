"""Outcome-free live bootstrap for the frozen LACE analysis protocol.

The scientific protocol contains simulator-manager payloads that cannot be
honestly authored from repository constants.  A preflight therefore creates
the exact manager environment for one frozen schedule cell, captures those
payloads while the recorder still has zero rows, writes a protocol candidate,
and exits before the eval entrypoint's first explicit reset, policy
construction, checkpoint-weight load, or rollout step.

The resulting receipt is procedural provenance, not an independent timing
anchor.  A scientific launch must additionally bind the candidate and receipt
in an externally anchored preregistration lock.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    build_analysis_protocol,
    validate_analysis_protocol,
)
from gear_sonic.research.lace.instrument import (
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    SOURCE_BUNDLE_DIGEST_FIELD,
    validate_source_bundle,
)
from gear_sonic.research.lace.instrument_runtime import (
    _load_json_object,
    capture_live_analysis_protocol_inputs,
    file_sha256,
    write_new_json,
)
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import (
    SCHEDULE_DIGEST_FIELD,
    validate_rollout_schedule,
)
from gear_sonic.research.lace.schema import (
    canonical_sha256,
    validate_split_manifest,
)

PROTOCOL_PREFLIGHT_REQUEST_KIND = "lace_analysis_protocol_live_preflight_request"
PROTOCOL_PREFLIGHT_REQUEST_SCHEMA_VERSION = 1
PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD = "protocol_preflight_request_sha256"
PROTOCOL_PREFLIGHT_RECEIPT_KIND = "lace_analysis_protocol_live_preflight_receipt"
PROTOCOL_PREFLIGHT_RECEIPT_SCHEMA_VERSION = 1
PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD = "protocol_preflight_receipt_sha256"
PROTOCOL_PREFLIGHT_LIFECYCLE = (
    "manager_initialized_zero_recorder_rows_before_eval_explicit_reset_policy_"
    "construction_checkpoint_weight_load_or_rollout_step_v1"
)

_REQUEST_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_claims_allowed",
    "capture_lifecycle",
    "schedule_lock",
    "schedule",
    "split",
    "reference_length_inventory",
    "cell",
    "rollout_ids",
    "checkpoint_bundle",
    "dataset_binding_sha256",
    "source_bundle",
    "git_commit",
    "launch_plan_path",
    "recorder_output_path",
    "analysis_protocol_output_path",
    "preflight_receipt_output_path",
    PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
}
_RECEIPT_FIELDS = {
    "kind",
    "schema_version",
    "artifact_mode",
    "scientific_claims_allowed",
    "capture_lifecycle",
    "request",
    "launch_plan",
    "analysis_protocol",
    "schedule_sha256",
    "cell",
    "rollout_ids",
    "checkpoint_bundle",
    "dataset_binding_sha256",
    "git_commit",
    "source_bundle_sha256",
    "resolved_hydra_config",
    "resolved_hydra_config_sha256",
    "live_inputs",
    "live_inputs_sha256",
    "recorder_record_count_before",
    "recorder_record_count_after",
    "recorder_output_absent_before",
    "recorder_output_absent_after",
    PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD,
}
_ARTIFACT_FIELDS = {"path", "file_sha256", "self_sha256"}
_CELL_FIELDS = {
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "repeat_index",
}
_CHECKPOINT_FIELDS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "config_path",
    "config_sha256",
}
_LIVE_INPUT_FIELDS = {
    "probe_thresholds",
    "probe_thresholds_sha256",
    "termination_predicates",
    "termination_predicates_sha256",
    "sensor_semantics",
    "score_window_config",
    "domain_randomization_config",
    "resolved_recorder_config",
    "recorder_term_type",
    "recorder_record_count",
    "num_envs",
    "step_dt_seconds",
    "environment_fingerprint",
}
_ENVIRONMENT_FINGERPRINT_FIELDS = {
    "schema_version",
    "python",
    "platform",
    "modules",
    "cuda",
    "scientific_cache_environment",
    "scientific_cache_environment_semantics",
    "process_executable",
    "process_argv",
    "pythonpath",
    "runtime",
}
_ENVIRONMENT_RUNTIME_FIELDS = {
    "environment_type",
    "termination_manager_type",
    "event_manager_type",
    "device",
    "num_envs",
    "step_dt_seconds",
}
_PREFLIGHT_CONFIG_FIELDS = (
    "lace_protocol_preflight_request_path",
    "lace_protocol_preflight_request_file_sha256",
    PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def protocol_preflight_binding_from_config(
    config: Mapping[str, Any],
) -> tuple[str, str, str, str] | None:
    """Return an exact preflight binding without confusing normal plan use.

    ``lace_launch_plan_path`` is shared with ordinary scientific evaluation;
    only one of the three preflight-exclusive fields requests this lifecycle.
    """

    _require(isinstance(config, Mapping), "LACE config must be a mapping")
    exclusive = tuple(config.get(field) for field in _PREFLIGHT_CONFIG_FIELDS)
    if not any(value is not None for value in exclusive):
        return None
    launch_plan_path = config.get("lace_launch_plan_path")
    binding = (*exclusive, launch_plan_path)
    _require(
        all(isinstance(value, str) and value for value in binding),
        "LACE protocol preflight binding is incomplete",
    )
    return binding


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _git_commit(value: Any) -> str:
    _require(isinstance(value, str) and len(value) == 40, "git_commit must be full length")
    _require(value == value.lower(), "git_commit must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("git_commit must be hexadecimal") from error
    return value


def _canonical_file(path_value: str | Path, name: str) -> Path:
    candidate = Path(path_value).expanduser()
    _require(candidate.is_absolute(), f"{name} must be absolute")
    _require(not candidate.is_symlink(), f"{name} may not be a symlink")
    resolved = candidate.resolve()
    _require(candidate == resolved and resolved.is_file(), f"{name} is not canonical")
    return resolved


def _new_json_path(path_value: str | Path, name: str) -> Path:
    candidate = Path(path_value).expanduser()
    _require(candidate.is_absolute(), f"{name} must be absolute")
    _require(not candidate.is_symlink(), f"{name} may not be a symlink")
    resolved = candidate.resolve()
    _require(candidate == resolved, f"{name} must be canonical")
    _require(resolved.suffix == ".json", f"{name} must use .json")
    _require(not resolved.exists(), f"{name} already exists: {resolved}")
    return resolved


def _artifact(
    path_value: str | Path,
    *,
    self_sha256: str,
    name: str,
) -> dict[str, str]:
    path = _canonical_file(path_value, name)
    return {
        "path": str(path),
        "file_sha256": file_sha256(path),
        "self_sha256": _sha256(self_sha256, f"{name} self digest"),
    }


def _validate_artifact(
    raw: Any,
    *,
    name: str,
    verify_file: bool,
) -> Path:
    _require(
        isinstance(raw, Mapping) and set(raw) == _ARTIFACT_FIELDS,
        f"{name} binding is invalid",
    )
    _sha256(raw.get("file_sha256"), f"{name} file digest")
    _sha256(raw.get("self_sha256"), f"{name} self digest")
    candidate = Path(str(raw.get("path", ""))).expanduser()
    _require(candidate.is_absolute(), f"{name} path must be absolute")
    if not verify_file:
        return candidate
    path = _canonical_file(candidate, f"{name} path")
    _require(file_sha256(path) == raw["file_sha256"], f"{name} file bytes drifted")
    return path


def _validate_checkpoint_bundle(raw: Any, *, verify_files: bool) -> dict[str, str]:
    _require(
        isinstance(raw, Mapping) and set(raw) == _CHECKPOINT_FIELDS,
        "preflight checkpoint bundle is invalid",
    )
    bundle = {name: str(raw[name]) for name in _CHECKPOINT_FIELDS}
    for field in ("checkpoint_sha256", "config_sha256"):
        _sha256(bundle[field], f"checkpoint bundle {field}")
    if verify_files:
        for path_field, digest_field in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("config_path", "config_sha256"),
        ):
            path = _canonical_file(bundle[path_field], f"checkpoint bundle {path_field}")
            _require(
                file_sha256(path) == bundle[digest_field],
                f"checkpoint bundle {path_field} bytes drifted",
            )
            bundle[path_field] = str(path)
    return bundle


def _positive_finite_number(value: Any, name: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0,
        f"{name} must be finite and positive",
    )
    return float(value)


def _validate_live_inputs(raw: Any) -> dict[str, Any]:
    """Validate the complete, outcome-free live-input capture payload."""

    _require(isinstance(raw, Mapping), "preflight live inputs must be a mapping")
    live = dict(raw)
    _require(set(live) == _LIVE_INPUT_FIELDS, "preflight live-input fields are invalid")
    for payload_name, digest_name in (
        ("probe_thresholds", "probe_thresholds_sha256"),
        ("termination_predicates", "termination_predicates_sha256"),
    ):
        payload = live[payload_name]
        _require(
            isinstance(payload, Mapping) and bool(payload),
            f"preflight {payload_name} must be a non-empty mapping",
        )
        _require(
            _sha256(live[digest_name], digest_name) == canonical_sha256(payload),
            f"preflight {digest_name} does not bind {payload_name}",
        )
    for name in (
        "sensor_semantics",
        "score_window_config",
        "domain_randomization_config",
        "resolved_recorder_config",
    ):
        _require(
            isinstance(live[name], Mapping) and bool(live[name]),
            f"preflight {name} must be a non-empty mapping",
        )
    _require(
        isinstance(live["recorder_term_type"], str)
        and bool(live["recorder_term_type"])
        and ":" in live["recorder_term_type"],
        "preflight recorder term type is invalid",
    )
    _require(
        live["recorder_record_count"] == 0 and not isinstance(live["recorder_record_count"], bool),
        "preflight recorder count must be exactly zero",
    )
    _require(
        isinstance(live["num_envs"], int)
        and not isinstance(live["num_envs"], bool)
        and live["num_envs"] > 0,
        "preflight num_envs must be a positive integer",
    )
    step_dt = _positive_finite_number(live["step_dt_seconds"], "preflight step_dt_seconds")
    window = live["score_window_config"]
    _require(
        set(window) == {"rule", "score_window_seconds", "timestep_seconds", "sample_count_rule"}
        and _positive_finite_number(
            window["score_window_seconds"],
            "preflight score-window seconds",
        )
        > 0.0
        and _positive_finite_number(
            window["timestep_seconds"],
            "preflight score-window timestep",
        )
        == step_dt,
        "preflight score window is not bound to the live timestep",
    )
    _require(
        float(window["score_window_seconds"])
        == float(live["probe_thresholds"].get("score_window_seconds", -1.0)),
        "preflight score window is not bound to the live threshold payload",
    )
    environment = live["environment_fingerprint"]
    _require(
        isinstance(environment, Mapping) and set(environment) == _ENVIRONMENT_FINGERPRINT_FIELDS,
        "preflight environment fingerprint fields are invalid",
    )
    _require(environment["schema_version"] == 1, "preflight environment schema is invalid")
    for name in ("python", "platform", "cuda"):
        _require(
            isinstance(environment[name], Mapping) and bool(environment[name]),
            f"preflight environment {name} is invalid",
        )
    _require(
        isinstance(environment["modules"], list) and bool(environment["modules"]),
        "preflight environment modules are invalid",
    )
    _require(
        isinstance(environment["process_executable"], str)
        and bool(environment["process_executable"])
        and isinstance(environment["process_argv"], list)
        and bool(environment["process_argv"])
        and all(isinstance(value, str) and value for value in environment["process_argv"])
        and isinstance(environment["pythonpath"], str)
        and bool(environment["pythonpath"]),
        "preflight process environment is invalid",
    )
    cache = environment["scientific_cache_environment"]
    _require(
        isinstance(cache, Mapping)
        and set(cache) == set(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS)
        and all(isinstance(cache[key], str) and cache[key] for key in cache),
        "preflight scientific cache environment is invalid",
    )
    _require(
        environment["scientific_cache_environment_semantics"]
        == SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "preflight scientific cache semantics are invalid",
    )
    runtime = environment["runtime"]
    _require(
        isinstance(runtime, Mapping) and set(runtime) == _ENVIRONMENT_RUNTIME_FIELDS,
        "preflight environment runtime fields are invalid",
    )
    for name in (
        "environment_type",
        "termination_manager_type",
        "event_manager_type",
        "device",
    ):
        _require(
            isinstance(runtime[name], str) and bool(runtime[name]),
            f"preflight environment runtime {name} is invalid",
        )
    _require(
        runtime["num_envs"] == live["num_envs"]
        and not isinstance(runtime["num_envs"], bool)
        and _positive_finite_number(
            runtime["step_dt_seconds"],
            "preflight environment runtime timestep",
        )
        == step_dt,
        "preflight environment runtime coordinates drifted",
    )
    return live


def build_protocol_preflight_request(
    *,
    schedule_lock_path: str | Path,
    schedule_path: str | Path,
    schedule_manifest: Mapping[str, Any],
    split_path: str | Path,
    split_manifest: Mapping[str, Any],
    reference_length_inventory_path: str | Path,
    reference_length_inventory: Mapping[str, Any],
    cell: Mapping[str, Any],
    rollout_ids: list[str],
    checkpoint_bundle: Mapping[str, Any],
    dataset_binding_sha256: str,
    source_bundle: Mapping[str, Any],
    git_commit: str,
    launch_plan_path: str | Path,
    recorder_output_path: str | Path,
    analysis_protocol_output_path: str | Path,
    preflight_receipt_output_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Build an explicit zero-outcome preflight request before env creation."""

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
    validate_source_bundle(source_bundle, repo_root=repo_root)
    normalized_cell = dict(cell)
    _require(set(normalized_cell) == _CELL_FIELDS, "preflight cell fields are invalid")
    _require(
        isinstance(rollout_ids, list)
        and bool(rollout_ids)
        and all(isinstance(item, str) and item for item in rollout_ids)
        and len(rollout_ids) == len(set(rollout_ids)),
        "preflight rollout ids are invalid",
    )
    schedule_lock = _canonical_file(schedule_lock_path, "schedule lock")
    schedule = _canonical_file(schedule_path, "schedule")
    split = _canonical_file(split_path, "split")
    inventory = _canonical_file(reference_length_inventory_path, "reference inventory")
    bundle = _validate_checkpoint_bundle(checkpoint_bundle, verify_files=True)
    plan_path = Path(launch_plan_path).expanduser().resolve()
    recorder_path = Path(recorder_output_path).expanduser().resolve()
    _require(plan_path.is_absolute() and plan_path.suffix == ".json", "plan path is invalid")
    _require(
        recorder_path.is_absolute() and recorder_path.suffix == ".jsonl",
        "recorder output path is invalid",
    )
    _require(not recorder_path.exists(), "preflight recorder output already exists")
    protocol_output = _new_json_path(analysis_protocol_output_path, "protocol output")
    receipt_output = _new_json_path(preflight_receipt_output_path, "preflight receipt")
    request: dict[str, Any] = {
        "kind": PROTOCOL_PREFLIGHT_REQUEST_KIND,
        "schema_version": PROTOCOL_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "artifact_mode": "outcome_free_live_protocol_preflight",
        "scientific_claims_allowed": False,
        "capture_lifecycle": PROTOCOL_PREFLIGHT_LIFECYCLE,
        "schedule_lock": {
            "path": str(schedule_lock),
            "file_sha256": file_sha256(schedule_lock),
            "self_sha256": schedule_manifest[SCHEDULE_DIGEST_FIELD],
        },
        "schedule": _artifact(
            schedule,
            self_sha256=schedule_manifest[SCHEDULE_DIGEST_FIELD],
            name="schedule",
        ),
        "split": _artifact(
            split,
            self_sha256=split_manifest["split_sha256"],
            name="split",
        ),
        "reference_length_inventory": _artifact(
            inventory,
            self_sha256=reference_length_inventory[REFERENCE_LENGTH_DIGEST_FIELD],
            name="reference inventory",
        ),
        "cell": normalized_cell,
        "rollout_ids": list(rollout_ids),
        "checkpoint_bundle": bundle,
        "dataset_binding_sha256": _sha256(
            dataset_binding_sha256,
            "dataset binding",
        ),
        "source_bundle": dict(source_bundle),
        "git_commit": _git_commit(git_commit),
        "launch_plan_path": str(plan_path),
        "recorder_output_path": str(recorder_path),
        "analysis_protocol_output_path": str(protocol_output),
        "preflight_receipt_output_path": str(receipt_output),
    }
    request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD] = canonical_sha256(
        request,
        digest_field=PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
    )
    validate_protocol_preflight_request(request, repo_root=repo_root, verify_files=True)
    return request


def validate_protocol_preflight_request(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_files: bool = True,
) -> None:
    _require(isinstance(raw, Mapping), "preflight request must be a mapping")
    request = dict(raw)
    _require(set(request) == _REQUEST_FIELDS, "preflight request fields are invalid")
    _require(
        request["kind"] == PROTOCOL_PREFLIGHT_REQUEST_KIND
        and request["schema_version"] == PROTOCOL_PREFLIGHT_REQUEST_SCHEMA_VERSION
        and request["artifact_mode"] == "outcome_free_live_protocol_preflight"
        and request["scientific_claims_allowed"] is False
        and request["capture_lifecycle"] == PROTOCOL_PREFLIGHT_LIFECYCLE,
        "preflight request identity is invalid",
    )
    _require(set(request["cell"]) == _CELL_FIELDS, "preflight request cell is invalid")
    _validate_checkpoint_bundle(request["checkpoint_bundle"], verify_files=verify_files)
    _sha256(request["dataset_binding_sha256"], "preflight dataset binding")
    _git_commit(request["git_commit"])
    validate_source_bundle(
        request["source_bundle"],
        repo_root=repo_root if verify_files else None,
    )
    _require(
        request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD]
        == canonical_sha256(request, digest_field=PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD),
        "protocol_preflight_request_sha256 mismatch",
    )
    schedule_path = _validate_artifact(
        request["schedule"], name="schedule", verify_file=verify_files
    )
    split_path = _validate_artifact(request["split"], name="split", verify_file=verify_files)
    inventory_path = _validate_artifact(
        request["reference_length_inventory"],
        name="reference inventory",
        verify_file=verify_files,
    )
    _validate_artifact(
        request["schedule_lock"],
        name="schedule lock",
        verify_file=verify_files,
    )
    if verify_files:
        from gear_sonic.research.lace.schedule_batch_loader import (
            load_locked_rollout_schedule,
        )

        locked = load_locked_rollout_schedule(
            request["schedule_lock"]["path"],
            repo_root=repo_root,
        )
        schedule = _load_json_object(schedule_path, "preflight schedule")
        split = _load_json_object(split_path, "preflight split")
        inventory = _load_json_object(inventory_path, "preflight reference inventory")
        validate_split_manifest(split)
        validate_reference_length_inventory(inventory, split_manifest=split, verify_digest=True)
        validate_rollout_schedule(
            schedule,
            split_manifest=split,
            reference_length_inventory=inventory,
            verify_digest=True,
        )
        _require(
            schedule[SCHEDULE_DIGEST_FIELD] == request["schedule"]["self_sha256"],
            "preflight schedule self digest drifted",
        )
        _require(
            split["split_sha256"] == request["split"]["self_sha256"],
            "preflight split self digest drifted",
        )
        _require(
            inventory[REFERENCE_LENGTH_DIGEST_FIELD]
            == request["reference_length_inventory"]["self_sha256"],
            "preflight inventory self digest drifted",
        )
        _require(
            locked.schedule_path == schedule_path
            and locked.split_path == split_path
            and locked.reference_length_inventory_path == inventory_path
            and locked.manifest == schedule
            and locked.split_manifest == split
            and locked.reference_length_inventory == inventory,
            "preflight request differs from its independently locked schedule inputs",
        )
        scheduled_rows = [
            row
            for row in schedule["rollouts"]
            if all(row[field] == request["cell"][field] for field in _CELL_FIELDS)
        ]
        _require(
            bool(scheduled_rows)
            and [row["rollout_id"] for row in scheduled_rows] == request["rollout_ids"]
            and request["checkpoint_bundle"]["checkpoint_sha256"]
            == request["cell"]["checkpoint_sha256"],
            "preflight request does not identify one exact frozen schedule cell",
        )
    for field, suffix in (
        ("launch_plan_path", ".json"),
        ("analysis_protocol_output_path", ".json"),
        ("preflight_receipt_output_path", ".json"),
        ("recorder_output_path", ".jsonl"),
    ):
        path = Path(str(request[field])).expanduser()
        _require(path.is_absolute() and path.resolve() == path, f"{field} is not canonical")
        _require(path.suffix == suffix, f"{field} suffix is invalid")


def validate_protocol_preflight_receipt(
    raw: Mapping[str, Any],
    *,
    repo_root: str | Path,
    verify_files: bool = True,
) -> None:
    _require(isinstance(raw, Mapping), "preflight receipt must be a mapping")
    receipt = dict(raw)
    _require(set(receipt) == _RECEIPT_FIELDS, "preflight receipt fields are invalid")
    _require(
        receipt["kind"] == PROTOCOL_PREFLIGHT_RECEIPT_KIND
        and receipt["schema_version"] == PROTOCOL_PREFLIGHT_RECEIPT_SCHEMA_VERSION
        and receipt["artifact_mode"] == "outcome_free_live_protocol_preflight"
        and receipt["scientific_claims_allowed"] is False
        and receipt["capture_lifecycle"] == PROTOCOL_PREFLIGHT_LIFECYCLE,
        "preflight receipt identity is invalid",
    )
    for name in ("request", "launch_plan", "analysis_protocol"):
        _validate_artifact(receipt[name], name=name, verify_file=verify_files)
    for field in (
        "schedule_sha256",
        "dataset_binding_sha256",
        "source_bundle_sha256",
        "resolved_hydra_config_sha256",
        "live_inputs_sha256",
        PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD,
    ):
        _sha256(receipt[field], field)
    _git_commit(receipt["git_commit"])
    _validate_checkpoint_bundle(receipt["checkpoint_bundle"], verify_files=verify_files)
    _require(set(receipt["cell"]) == _CELL_FIELDS, "preflight receipt cell is invalid")
    _require(
        receipt["recorder_record_count_before"] == 0
        and receipt["recorder_record_count_after"] == 0
        and receipt["recorder_output_absent_before"] is True
        and receipt["recorder_output_absent_after"] is True,
        "preflight receipt contains recorder outcomes",
    )
    _require(
        receipt["live_inputs_sha256"] == canonical_sha256(receipt["live_inputs"]),
        "preflight live-input digest drifted",
    )
    live_inputs = _validate_live_inputs(receipt["live_inputs"])
    _require(
        isinstance(receipt["resolved_hydra_config"], Mapping)
        and receipt["resolved_hydra_config_sha256"]
        == canonical_sha256(receipt["resolved_hydra_config"]),
        "preflight resolved Hydra config digest drifted",
    )
    _require(
        receipt[PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD]
        == canonical_sha256(receipt, digest_field=PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD),
        "protocol_preflight_receipt_sha256 mismatch",
    )
    if not verify_files:
        return
    request = _load_json_object(receipt["request"]["path"], "preflight request")
    validate_protocol_preflight_request(request, repo_root=repo_root, verify_files=True)
    plan = _load_json_object(receipt["launch_plan"]["path"], "preflight launch plan")
    _require(
        plan.get("launch_plan_sha256")
        == canonical_sha256(plan, digest_field="launch_plan_sha256")
        == receipt["launch_plan"]["self_sha256"],
        "preflight receipt launch plan digest drifted",
    )
    _require(
        plan.get("protocol_preflight")
        == {
            "request_path": receipt["request"]["path"],
            "request_file_sha256": receipt["request"]["file_sha256"],
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: receipt["request"]["self_sha256"],
        }
        and plan.get("schedule_sha256") == request["schedule"]["self_sha256"]
        and plan.get("cell") == request["cell"]
        and plan.get("rollout_ids") == request["rollout_ids"]
        and plan.get("checkpoint_bundle") == request["checkpoint_bundle"]
        and canonical_sha256(plan.get("dataset_binding")) == request["dataset_binding_sha256"],
        "preflight receipt launch plan causal inputs drifted",
    )
    _require(
        receipt["cell"] == request["cell"]
        and receipt["rollout_ids"] == request["rollout_ids"]
        and receipt["checkpoint_bundle"] == request["checkpoint_bundle"]
        and receipt["dataset_binding_sha256"] == request["dataset_binding_sha256"],
        "preflight receipt causal fields drifted from its request",
    )
    resolved_config = receipt["resolved_hydra_config"]
    failure_config = (
        resolved_config.get("manager_env", {}).get("recorders", {}).get("failure_atlas")
    )
    motion_config = resolved_config.get("manager_env", {}).get("commands", {}).get("motion", {})
    motion_library = motion_config.get("motion_lib_cfg", {})
    _require(
        resolved_config.get("checkpoint") == request["checkpoint_bundle"]["checkpoint_path"]
        and resolved_config.get("lace_scientific_instrument_required") is False
        and resolved_config.get("lace_protocol_preflight_request_path")
        == receipt["request"]["path"]
        and resolved_config.get("lace_protocol_preflight_request_file_sha256")
        == receipt["request"]["file_sha256"]
        and resolved_config.get(PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD)
        == receipt["request"]["self_sha256"]
        and resolved_config.get("lace_launch_plan_path") == request["launch_plan_path"]
        and isinstance(failure_config, Mapping)
        and failure_config.get("output_path") == request["recorder_output_path"]
        and motion_config.get("atlas_probe_schedule_sha256") == request["schedule"]["self_sha256"]
        and motion_library.get("motion_file") == plan["dataset_binding"]["robot"]["motion_root"]
        and motion_library.get("smpl_motion_file")
        == plan["dataset_binding"]["smpl"]["motion_root"],
        "preflight resolved Hydra config is not bound to request/plan inputs",
    )
    _require(
        live_inputs["resolved_recorder_config"] == dict(failure_config)
        and live_inputs["num_envs"] == len(request["rollout_ids"]),
        "preflight live recorder inputs drifted from resolved Hydra config",
    )
    environment = live_inputs["environment_fingerprint"]
    launch_environment = plan.get("launch_environment")
    command = plan.get("command")
    _require(
        isinstance(launch_environment, Mapping)
        and isinstance(command, list)
        and len(command) >= 2
        and all(isinstance(value, str) and value for value in command)
        and environment["process_executable"] == command[0]
        and environment["process_argv"] == command[1:]
        and environment["pythonpath"] == launch_environment.get("PYTHONPATH")
        and environment["scientific_cache_environment"]
        == launch_environment.get("scientific_cache_environment")
        and environment["scientific_cache_environment_semantics"]
        == launch_environment.get("scientific_cache_environment_semantics"),
        "preflight environment fingerprint drifted from the launch plan",
    )
    protocol = _load_json_object(receipt["analysis_protocol"]["path"], "analysis protocol")
    schedule = _load_json_object(request["schedule"]["path"], "preflight schedule")
    split = _load_json_object(request["split"]["path"], "preflight split")
    inventory = _load_json_object(
        request["reference_length_inventory"]["path"],
        "preflight reference inventory",
    )
    validate_analysis_protocol(
        protocol,
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
    )
    expected_protocol = build_analysis_protocol(
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        probe_thresholds=live_inputs["probe_thresholds"],
        score_window_config=live_inputs["score_window_config"],
        sensor_semantics=live_inputs["sensor_semantics"],
        termination_predicates=live_inputs["termination_predicates"],
        domain_randomization_config=live_inputs["domain_randomization_config"],
        git_commit=request["git_commit"],
        source_bundle_sha256=request["source_bundle"][SOURCE_BUNDLE_DIGEST_FIELD],
    )
    _require(
        receipt["request"]["self_sha256"] == request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD]
        and receipt["analysis_protocol"]["self_sha256"] == protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD]
        and receipt["schedule_sha256"] == protocol["schedule_sha256"]
        and receipt["source_bundle_sha256"] == protocol["source_bundle_sha256"]
        and receipt["git_commit"] == protocol["git_commit"],
        "preflight receipt protocol/request identity drifted",
    )
    _require(
        protocol == expected_protocol,
        "preflight protocol is not the deterministic output of captured live inputs",
    )


def build_protocol_preflight_receipt(
    *,
    request_path: str | Path,
    launch_plan_path: str | Path,
    live_inputs: Mapping[str, Any],
    resolved_hydra_config: Mapping[str, Any],
    repo_root: str | Path,
) -> dict[str, Any]:
    """Publish receipt-last from already captured zero-row live inputs.

    This low-level builder does not capture a simulator.  Production callers
    must use :func:`execute_live_protocol_preflight`; this split exists so the
    receipt and lock graph remain CPU-verifiable without Isaac Lab.
    """

    request_file = _canonical_file(request_path, "preflight request")
    plan_file = _canonical_file(launch_plan_path, "preflight launch plan")
    request = _load_json_object(request_file, "preflight request")
    validate_protocol_preflight_request(request, repo_root=repo_root, verify_files=True)
    plan = _load_json_object(plan_file, "preflight launch plan")
    protocol_path = _canonical_file(
        request["analysis_protocol_output_path"],
        "preflight analysis protocol",
    )
    protocol = _load_json_object(protocol_path, "preflight analysis protocol")
    validated_live_inputs = _validate_live_inputs(live_inputs)
    _require(
        isinstance(resolved_hydra_config, Mapping) and bool(resolved_hydra_config),
        "preflight receipt requires a non-empty resolved Hydra config",
    )
    recorder_output = Path(request["recorder_output_path"])
    _require(not recorder_output.exists(), "preflight recorder output exists")
    receipt: dict[str, Any] = {
        "kind": PROTOCOL_PREFLIGHT_RECEIPT_KIND,
        "schema_version": PROTOCOL_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "artifact_mode": "outcome_free_live_protocol_preflight",
        "scientific_claims_allowed": False,
        "capture_lifecycle": PROTOCOL_PREFLIGHT_LIFECYCLE,
        "request": {
            "path": str(request_file),
            "file_sha256": file_sha256(request_file),
            "self_sha256": request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD],
        },
        "launch_plan": {
            "path": str(plan_file),
            "file_sha256": file_sha256(plan_file),
            "self_sha256": plan["launch_plan_sha256"],
        },
        "analysis_protocol": {
            "path": str(protocol_path),
            "file_sha256": file_sha256(protocol_path),
            "self_sha256": protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        },
        "schedule_sha256": protocol["schedule_sha256"],
        "cell": request["cell"],
        "rollout_ids": request["rollout_ids"],
        "checkpoint_bundle": request["checkpoint_bundle"],
        "dataset_binding_sha256": request["dataset_binding_sha256"],
        "git_commit": request["git_commit"],
        "source_bundle_sha256": request["source_bundle"][SOURCE_BUNDLE_DIGEST_FIELD],
        "resolved_hydra_config": dict(resolved_hydra_config),
        "resolved_hydra_config_sha256": canonical_sha256(resolved_hydra_config),
        "live_inputs": validated_live_inputs,
        "live_inputs_sha256": canonical_sha256(validated_live_inputs),
        "recorder_record_count_before": 0,
        "recorder_record_count_after": 0,
        "recorder_output_absent_before": True,
        "recorder_output_absent_after": True,
    }
    receipt[PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=PROTOCOL_PREFLIGHT_RECEIPT_DIGEST_FIELD,
    )
    validate_protocol_preflight_receipt(receipt, repo_root=repo_root, verify_files=True)
    write_new_json(request["preflight_receipt_output_path"], receipt)
    return receipt


def execute_live_protocol_preflight(
    *,
    request_path: str | Path,
    expected_request_file_sha256: str,
    launch_plan_path: str | Path,
    repo_root: str | Path,
    resolved_hydra_config: Mapping[str, Any],
    wrapped_env: Any,
) -> dict[str, Any]:
    """Capture live inputs, publish a protocol candidate, then receipt-last."""

    request_file = _canonical_file(request_path, "preflight request")
    _require(
        file_sha256(request_file)
        == _sha256(
            expected_request_file_sha256,
            "expected preflight request file digest",
        ),
        "preflight request file bytes drifted",
    )
    request = _load_json_object(request_file, "preflight request")
    validate_protocol_preflight_request(request, repo_root=repo_root, verify_files=True)
    plan_file = _canonical_file(launch_plan_path, "preflight launch plan")
    _require(str(plan_file) == request["launch_plan_path"], "preflight plan path drifted")
    plan = _load_json_object(plan_file, "preflight launch plan")
    _require(
        plan.get("launch_plan_sha256") == canonical_sha256(plan, digest_field="launch_plan_sha256"),
        "preflight launch plan digest drifted",
    )
    request_binding = plan.get("protocol_preflight")
    _require(
        isinstance(request_binding, Mapping)
        and request_binding
        == {
            "request_path": str(request_file),
            "request_file_sha256": file_sha256(request_file),
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: request[
                PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
            ],
        },
        "preflight launch plan request binding drifted",
    )
    _require(
        plan.get("schedule_sha256") == request["schedule"]["self_sha256"]
        and plan.get("cell") == request["cell"]
        and plan.get("rollout_ids") == request["rollout_ids"]
        and plan.get("checkpoint_bundle") == request["checkpoint_bundle"]
        and canonical_sha256(plan.get("dataset_binding")) == request["dataset_binding_sha256"],
        "preflight launch plan causal inputs drifted",
    )
    _require(
        resolved_hydra_config.get("lace_protocol_preflight_request_path") == str(request_file)
        and resolved_hydra_config.get("lace_protocol_preflight_request_file_sha256")
        == file_sha256(request_file)
        and resolved_hydra_config.get(PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD)
        == request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD]
        and resolved_hydra_config.get("lace_launch_plan_path") == str(plan_file)
        and resolved_hydra_config.get("lace_scientific_instrument_required") is False,
        "resolved Hydra preflight binding drifted",
    )
    recorder_output = Path(request["recorder_output_path"])
    _require(not recorder_output.exists(), "preflight recorder output exists before capture")
    live_inputs = capture_live_analysis_protocol_inputs(
        wrapped_env=wrapped_env,
        resolved_hydra_config=resolved_hydra_config,
    )
    raw_env = wrapped_env.env
    command = raw_env.command_manager.get_term(str(live_inputs["sensor_semantics"]["command_name"]))
    batch = getattr(command, "atlas_probe_batch", None)
    _require(batch is not None, "preflight live motion command lacks frozen atlas batch")
    _require(
        batch.schedule_sha256 == request["schedule"]["self_sha256"]
        and batch.checkpoint_sha256 == request["checkpoint_bundle"]["checkpoint_sha256"]
        and list(batch.rollout_ids) == request["rollout_ids"],
        "preflight live schedule/checkpoint/rollout identity drifted",
    )
    schedule = _load_json_object(request["schedule"]["path"], "preflight schedule")
    split = _load_json_object(request["split"]["path"], "preflight split")
    inventory = _load_json_object(
        request["reference_length_inventory"]["path"],
        "preflight reference inventory",
    )
    protocol = build_analysis_protocol(
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        probe_thresholds=live_inputs["probe_thresholds"],
        score_window_config=live_inputs["score_window_config"],
        sensor_semantics=live_inputs["sensor_semantics"],
        termination_predicates=live_inputs["termination_predicates"],
        domain_randomization_config=live_inputs["domain_randomization_config"],
        git_commit=request["git_commit"],
        source_bundle_sha256=request["source_bundle"][SOURCE_BUNDLE_DIGEST_FIELD],
    )
    write_new_json(request["analysis_protocol_output_path"], protocol)
    term = raw_env.recorder_manager._terms["failure_atlas"]
    record_count_after = getattr(term, "_record_count", None)
    _require(record_count_after == 0, "preflight recorder emitted an outcome")
    _require(not recorder_output.exists(), "preflight recorder output appeared during capture")
    return build_protocol_preflight_receipt(
        request_path=request_file,
        launch_plan_path=plan_file,
        live_inputs=live_inputs,
        resolved_hydra_config=resolved_hydra_config,
        repo_root=repo_root,
    )
