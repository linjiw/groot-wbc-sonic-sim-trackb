from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    REFERENCE_LENGTH_KIND,
    REFERENCE_LENGTH_SCHEMA_VERSION,
    RESAMPLING_RULE,
    sonic_target_frame_count,
)
from gear_sonic.research.lace.schedule import build_rollout_schedule, derive_runtime_rng_seed
from gear_sonic.research.lace.schedule_batch_loader import (
    build_eval_hydra_overrides,
    load_locked_rollout_schedule,
    select_locked_atlas_probe_cell_for_filtered_library,
    sha256_file,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "scripts/research/run_lace_atlas_probe_cell.py"
CLI_SPEC = importlib.util.spec_from_file_location("lace_atlas_probe_cell_cli", CLI_PATH)
assert CLI_SPEC is not None and CLI_SPEC.loader is not None
CLI_MODULE = importlib.util.module_from_spec(CLI_SPEC)
CLI_SPEC.loader.exec_module(CLI_MODULE)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _inventory(
    split: dict[str, Any],
    selected_motion_keys: list[str],
) -> dict[str, Any]:
    split_records = {record["motion_key"]: record for record in split["motions"]}
    motions = []
    for index, motion_key in enumerate(selected_motion_keys):
        source_frames = 10 + index
        source_path = Path(split_records[motion_key]["robot_path"])
        motions.append(
            {
                "motion_key": motion_key,
                "split_robot_path": split_records[motion_key]["robot_path"],
                "source_path": str(source_path),
                "source_file_sha256": sha256_file(source_path),
                "source_num_frames": source_frames,
                "frame_axis": "root_trans_offset.shape[0]",
                "source_fps": {"numerator": 30, "denominator": 1},
                "target_num_frames": sonic_target_frame_count(source_frames, 30, 50),
            }
        )
    inventory = {
        "kind": REFERENCE_LENGTH_KIND,
        "schema_version": REFERENCE_LENGTH_SCHEMA_VERSION,
        "artifact_mode": "pilot",
        "scientific_use": False,
        "pilot_status": "non_scientific_subset_for_contract_or_runtime_validation_only",
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_atlas",
        "selected_motion_keys": selected_motion_keys,
        "selection_complete_for_d_atlas": False,
        "motion_count": len(selected_motion_keys),
        "target_fps": 50,
        "runtime_contract": {
            "target_fps": 50,
            "sim_fps": 50,
            "motion_fps_scale": {"numerator": 1, "denominator": 1},
            "max_len": -1,
            "reference_num_steps_equals_target_num_frames": True,
            "float32_runtime_equivalence_scope": "not_claimed_for_generic_pilot_rates",
        },
        "resampling_rule": dict(RESAMPLING_RULE),
        "dataset_provenance": {
            "materialized_manifest": None,
            "materialized_manifest_sha256": None,
            "paired_dataset_sha256": None,
        },
        "source_file_set_sha256": canonical_sha256(
            {
                "files": [
                    {
                        "motion_key": record["motion_key"],
                        "source_file_sha256": record["source_file_sha256"],
                    }
                    for record in motions
                ]
            }
        ),
        "motions": motions,
    }
    inventory[REFERENCE_LENGTH_DIGEST_FIELD] = canonical_sha256(
        inventory,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    return inventory


def _fixture(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    robot_root = tmp_path / "robot_filtered"
    smpl_root = tmp_path / "smpl"
    robot_root.mkdir()
    smpl_root.mkdir()
    records = [
        {
            "motion_key": f"motion_{index:02d}__A{index:03d}",
            "release_filter_key": f"motion_{index:02d}__A{index:03d}.pkl",
            "source_group_id": f"actor_{index:03d}",
            "robot_path": str(robot_root / f"motion_{index:02d}__A{index:03d}.pkl"),
            "duration_source_frames": 80 + index,
            "stratum": f"duration_{index % 3}",
        }
        for index in range(25)
    ]
    split = build_source_disjoint_split(records, seed=71)
    atlas_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    selected = atlas_keys[:2]
    for motion_key in selected:
        (robot_root / f"{motion_key}.pkl").write_bytes(f"robot fixture {motion_key}\n".encode())
        (smpl_root / f"{motion_key}.pkl").write_bytes(f"smpl fixture {motion_key}\n".encode())
    inventory = _inventory(split, selected)
    checkpoint = tmp_path / "weak.pt"
    checkpoint.write_bytes(b"fixture checkpoint bytes\n")
    (tmp_path / "config.yaml").write_text("fixture: sonic-config\n", encoding="utf-8")
    checkpoint_digest = sha256_file(checkpoint)
    schedule = build_rollout_schedule(
        split,
        reference_length_inventory=inventory,
        probe_policies=[
            {"id": "weak", "checkpoint_sha256": checkpoint_digest},
            {"id": "strong", "checkpoint_sha256": "b" * 64},
        ],
        domain_randomization_seeds=[101, 202],
        phase_targets=[
            {"phase_id": "start", "target_fraction": 0.0},
            {"phase_id": "mid", "target_fraction": 0.5},
        ],
        repeats=2,
    )
    schedule_spec = {
        "schema_version": 2,
        "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "probe_policies": [
            {"id": "weak", "checkpoint_sha256": checkpoint_digest},
            {"id": "strong", "checkpoint_sha256": "b" * 64},
        ],
        "domain_randomization_seeds": [101, 202],
        "phase_targets": [
            {"phase_id": "start", "target_fraction": 0.0},
            {"phase_id": "mid", "target_fraction": 0.5},
        ],
        "repeats": 2,
    }

    split_path = tmp_path / "split.json"
    inventory_path = tmp_path / "inventory.json"
    schedule_path = tmp_path / "schedule.json"
    schedule_spec_path = tmp_path / "schedule_spec.json"
    _write_json(split_path, split)
    _write_json(inventory_path, inventory)
    _write_json(schedule_path, schedule)
    _write_json(schedule_spec_path, schedule_spec)
    lock = {
        "schema_version": 1,
        "kind": "lace_probe_rollout_schedule_lock",
        "scientific_use": False,
        "artifact": {
            "path": str(schedule_path),
            "file_sha256": sha256_file(schedule_path),
            "schedule_sha256": schedule["schedule_sha256"],
            "schedule_schema_version": 2,
        },
        "inputs": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": sha256_file(split_path),
            "schedule_spec": str(schedule_spec_path),
            "schedule_spec_sha256": sha256_file(schedule_spec_path),
            "reference_length_inventory": str(inventory_path),
            "reference_length_inventory_file_sha256": sha256_file(inventory_path),
            "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        },
    }
    lock_path = tmp_path / "schedule_lock.json"
    _write_json(lock_path, lock)
    return {
        "atlas_keys": selected,
        "checkpoint": checkpoint,
        "checkpoint_digest": checkpoint_digest,
        "inventory": inventory,
        "lock": lock,
        "lock_path": lock_path,
        "robot_root": robot_root,
        "schedule": schedule,
        "schedule_path": schedule_path,
        "smpl_root": smpl_root,
    }


def _rewrite_schedule_and_lock(
    fixture: dict[str, Any],
    schedule: dict[str, Any],
    *,
    recompute_schedule_digest: bool,
) -> None:
    if recompute_schedule_digest:
        schedule["schedule_sha256"] = canonical_sha256(
            schedule,
            digest_field="schedule_sha256",
        )
    _write_json(fixture["schedule_path"], schedule)
    lock = deepcopy(fixture["lock"])
    lock["artifact"]["file_sha256"] = sha256_file(fixture["schedule_path"])
    if recompute_schedule_digest:
        lock["artifact"]["schedule_sha256"] = schedule["schedule_sha256"]
    _write_json(fixture["lock_path"], lock)


def _selection(fixture: dict[str, Any], *, num_envs: int | None = None):
    locked = load_locked_rollout_schedule(fixture["lock_path"])
    return select_locked_atlas_probe_cell_for_filtered_library(
        locked,
        probe_policy_id="weak",
        domain_randomization_seed=101,
        phase_id="mid",
        repeat_index=1,
        expected_checkpoint_sha256=fixture["checkpoint_digest"],
        num_envs=len(fixture["atlas_keys"]) if num_envs is None else num_envs,
    )


def test_locked_loader_selects_every_motion_in_one_exact_cell(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    selection = _selection(fixture)

    assert selection.batch.num_envs == len(fixture["atlas_keys"])
    assert selection.batch.motion_keys == tuple(fixture["atlas_keys"])
    assert selection.batch.motion_ids == tuple(range(len(fixture["atlas_keys"])))
    assert selection.batch.probe_policy_id == "weak"
    assert selection.batch.domain_randomization_seed == 101
    assert selection.batch.phase_id == "mid"
    assert selection.batch.repeat_index == 1
    assert selection.batch.runtime_rng_seed == derive_runtime_rng_seed(101, "mid", 1)
    assert [row["motion_key"] for row in selection.assignments] == fixture["atlas_keys"]


def test_loader_fails_on_schedule_bytes_or_recomputed_row_order(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["schedule_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact.file_sha256 mismatch"):
        load_locked_rollout_schedule(fixture["lock_path"])

    fixture = _fixture(tmp_path / "reordered")
    schedule = deepcopy(fixture["schedule"])
    schedule["rollouts"][0], schedule["rollouts"][1] = (
        schedule["rollouts"][1],
        schedule["rollouts"][0],
    )
    _rewrite_schedule_and_lock(fixture, schedule, recompute_schedule_digest=True)
    with pytest.raises(ValueError, match="full Cartesian coverage and order"):
        load_locked_rollout_schedule(fixture["lock_path"])


def test_loader_fails_on_policy_checkpoint_and_num_env_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    locked = load_locked_rollout_schedule(fixture["lock_path"])

    with pytest.raises(ValueError, match="does not match probe policy"):
        select_locked_atlas_probe_cell_for_filtered_library(
            locked,
            probe_policy_id="weak",
            domain_randomization_seed=101,
            phase_id="mid",
            repeat_index=1,
            expected_checkpoint_sha256="f" * 64,
            num_envs=len(fixture["atlas_keys"]),
        )
    with pytest.raises(ValueError, match="does not match selected cell size"):
        _selection(fixture, num_envs=len(fixture["atlas_keys"]) + 1)


def test_loader_fails_on_heterogeneous_runtime_seed_even_if_rehashed(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    schedule = deepcopy(fixture["schedule"])
    row = next(
        record
        for record in schedule["rollouts"]
        if record["probe_policy_id"] == "weak"
        and record["domain_randomization_seed"] == 101
        and record["phase_id"] == "mid"
        and record["repeat_index"] == 1
    )
    row["runtime_rng_seed"] += 1
    identity = {
        "identity_version": 1,
        **{key: value for key, value in row.items() if key not in {"rollout_id", "partition"}},
    }
    row["rollout_id"] = f"lace-rollout:{canonical_sha256(identity)}"
    _rewrite_schedule_and_lock(fixture, schedule, recompute_schedule_digest=True)

    with pytest.raises(ValueError, match="runtime_rng_seed"):
        load_locked_rollout_schedule(fixture["lock_path"])


def test_hydra_integration_emits_exact_assignments_and_frozen_runtime_seed(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    selection = _selection(fixture)
    rollout_output = tmp_path / "rollouts" / "cell.jsonl"
    rollout_output.parent.mkdir()

    overrides = build_eval_hydra_overrides(
        selection,
        checkpoint_path=fixture["checkpoint"],
        rollout_output_path=rollout_output,
        robot_motion_root=fixture["robot_root"],
        smpl_motion_root=fixture["smpl_root"],
        use_dummy_smpl=False,
    )

    assignment_override = next(
        item
        for item in overrides
        if item.startswith("++manager_env.commands.motion.atlas_probe_assignments=")
    )
    encoded_assignments = assignment_override.split("=", 1)[1]
    assert encoded_assignments.startswith("[{checkpoint_sha256:")
    assert encoded_assignments.count("{checkpoint_sha256:") == selection.batch.num_envs
    for rollout_id in selection.batch.rollout_ids:
        assert json.dumps(rollout_id) in encoded_assignments
    assert f"++num_envs={selection.batch.num_envs}" in overrides
    assert f"++seed={selection.batch.runtime_rng_seed}" in overrides
    assert f"checkpoint={json.dumps(str(fixture['checkpoint']))}" in overrides
    assert not any(item.startswith("+checkpoint=") for item in overrides)
    assert "+manager_env/recorders=lace_atlas" in overrides
    assert "++manager_env.commands.motion.atlas_probe_mode=true" in overrides
    assert "++headless=true" in overrides
    assert "++manager_env.config.terrain_type=plane" in overrides
    assert "++manager_env.observations.policy.enable_corruption=false" in overrides
    assert "++manager_env.observations.tokenizer.enable_corruption=false" in overrides
    assert "++use_encoder=g1" in overrides
    assert "++eval_callbacks=[]" in overrides


def test_generated_overrides_dry_compose_with_base_eval(tmp_path: Path) -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    fixture = _fixture(tmp_path)
    selection = _selection(fixture)
    rollout_output = tmp_path / "rollouts" / "cell.jsonl"
    rollout_output.parent.mkdir()
    overrides = build_eval_hydra_overrides(
        selection,
        checkpoint_path=fixture["checkpoint"],
        rollout_output_path=rollout_output,
        robot_motion_root=fixture["robot_root"],
        smpl_motion_root=fixture["smpl_root"],
        use_dummy_smpl=False,
    )

    with initialize_config_dir(
        version_base=None,
        config_dir=str((REPO_ROOT / "gear_sonic/config").resolve()),
    ):
        config = compose(config_name="base_eval", overrides=list(overrides))

    assert config.checkpoint == str(fixture["checkpoint"])
    assert config.num_envs == selection.batch.num_envs
    assert config.seed == selection.batch.runtime_rng_seed
    assert len(config.manager_env.commands.motion.atlas_probe_assignments) == config.num_envs
    assert config.manager_env.commands.motion.atlas_probe_mode is True
    assert config.manager_env.recorders.failure_atlas.enabled is True
    assert config.manager_env.config.terrain_type == "plane"
    assert config.manager_env.observations.policy.enable_corruption is False
    assert config.manager_env.observations.tokenizer.enable_corruption is False


def test_cli_writes_plan_and_rejects_conflicting_manual_overrides(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    rollout_output = tmp_path / "outputs" / "cell.jsonl"
    plan_output = tmp_path / "outputs" / "cell-plan.json"
    argv = [
        "--schedule-lock",
        str(fixture["lock_path"]),
        "--probe-policy-id",
        "weak",
        "--domain-randomization-seed",
        "101",
        "--phase-id",
        "mid",
        "--repeat-index",
        "1",
        "--checkpoint",
        str(fixture["checkpoint"]),
        "--num-envs",
        str(len(fixture["atlas_keys"])),
        "--robot-motion-root",
        str(fixture["robot_root"]),
        "--smpl-motion-root",
        str(fixture["smpl_root"]),
        "--rollout-output",
        str(rollout_output),
        "--plan-output",
        str(plan_output),
    ]

    assert CLI_MODULE.main(argv) == 0
    plan = json.loads(plan_output.read_text(encoding="utf-8"))
    assert plan["scientific_use"] is False
    assert plan["instrument_runtime"] == {
        "instrument_output_path": None,
        "required": False,
        "state": "forbidden_for_non_scientific_contract_smoke",
    }
    assert "++lace_scientific_instrument_required=false" in plan["hydra_overrides"]
    assert plan["num_envs"] == len(fixture["atlas_keys"])
    assert plan["cell"]["probe_policy_id"] == "weak"
    assert plan["cell"]["runtime_rng_seed"] == derive_runtime_rng_seed(101, "mid", 1)
    assert plan["launch_plan_sha256"] == canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    assert plan["checkpoint_bundle"] == {
        "checkpoint_path": str(fixture["checkpoint"]),
        "checkpoint_sha256": fixture["checkpoint_digest"],
        "config_path": str(fixture["checkpoint"].parent / "config.yaml"),
        "config_sha256": sha256_file(fixture["checkpoint"].parent / "config.yaml"),
    }
    pythonpath_entries = plan["launch_environment"]["PYTHONPATH"].split(os.pathsep)
    assert pythonpath_entries[0] == str(REPO_ROOT)
    assert len(pythonpath_entries) == len(set(pythonpath_entries))
    assert (
        plan["launch_environment"]["semantics"]
        == "repository_root_first_preserve_inherited_unique_entries_v1"
    )
    cache_environment = plan["launch_environment"]["scientific_cache_environment"]
    assert set(cache_environment) == set(CLI_MODULE.SCIENTIFIC_CACHE_ENVIRONMENT_KEYS)
    assert plan["launch_environment"]["scientific_cache_environment_semantics"] == (
        CLI_MODULE.SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS
    )
    expected_cache_root = rollout_output.parent / ".lace-runtime"
    assert all(
        Path(path).is_dir() and Path(path).is_relative_to(expected_cache_root)
        for path in cache_environment.values()
    )
    assert all(
        "/.cache/" not in path and not path.startswith("/root/")
        for path in cache_environment.values()
    )
    assert plan["instrumentation_invariants"] == {
        "eval_callbacks": [],
        "headless": True,
        "max_render_steps": 11,
        "native_adaptive_sampling": False,
        "policy_enable_corruption": False,
        "render_results": False,
        "run_once": True,
        "terrain_type": "plane",
        "tokenizer_enable_corruption": False,
        "use_encoder": "g1",
    }
    assert not rollout_output.exists()

    with pytest.raises(ValueError, match="conflicts with a frozen atlas field"):
        CLI_MODULE.main([*argv, "--", "++num_envs=99"])
    with pytest.raises(ValueError, match="may not claim an instrument"):
        CLI_MODULE.main([*argv, "--instrument-output", str(tmp_path / "instrument.json")])


def test_scientific_cli_validates_preregistration_before_creating_attempt_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gear_sonic.research.lace.analysis_protocol_lock as protocol_lock_module

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint\n")
    (tmp_path / "config.yaml").write_text("model: fixture\n", encoding="utf-8")
    rollout_output = tmp_path / "attempt" / "cell.jsonl"
    plan_output = tmp_path / "attempt" / "plan.json"
    runtime_root = Path("/data") / f"lace-invalid-lock-{tmp_path.name}"
    assert not rollout_output.parent.exists()
    assert not runtime_root.exists()

    locked = SimpleNamespace(manifest={"scientific_use": True})
    selection = SimpleNamespace(locked_schedule=locked)
    monkeypatch.setattr(CLI_MODULE, "load_locked_rollout_schedule", lambda *args, **kwargs: locked)
    monkeypatch.setattr(
        CLI_MODULE,
        "select_locked_atlas_probe_cell_for_filtered_library",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        CLI_MODULE,
        "build_launch_plan",
        lambda *args, **kwargs: {
            "scientific_use": True,
            "schedule_sha256": "a" * 64,
            "cell": {
                "probe_policy_id": "weak",
                "checkpoint_sha256": sha256_file(checkpoint),
            },
        },
    )

    def reject_lock(*args: object, **kwargs: object) -> object:
        raise ValueError("invalid preregistration lock")

    monkeypatch.setattr(protocol_lock_module, "load_analysis_protocol_lock", reject_lock)
    arguments = [
        "--schedule-lock",
        str(tmp_path / "schedule-lock.json"),
        "--probe-policy-id",
        "weak",
        "--domain-randomization-seed",
        "101",
        "--phase-id",
        "start",
        "--repeat-index",
        "0",
        "--checkpoint",
        str(checkpoint),
        "--num-envs",
        "1",
        "--robot-motion-root",
        str(tmp_path),
        "--smpl-motion-root",
        str(tmp_path),
        "--rollout-output",
        str(rollout_output),
        "--plan-output",
        str(plan_output),
        "--instrument-output",
        str(tmp_path / "attempt" / "instrument.json"),
        "--analysis-protocol-lock",
        str(tmp_path / "protocol-lock.json"),
        "--expected-analysis-protocol-lock-sha256",
        "b" * 64,
        "--runtime-storage-root",
        str(runtime_root),
    ]
    with pytest.raises(ValueError, match="invalid preregistration lock"):
        CLI_MODULE.main(arguments)
    assert not rollout_output.parent.exists()
    assert not runtime_root.exists()
    with pytest.raises(ValueError, match="scientific launch is NO-GO"):
        CLI_MODULE.main([*arguments, "--launch"])
    assert not rollout_output.parent.exists()
    assert not runtime_root.exists()


def test_repo_first_pythonpath_removes_duplicates_without_dropping_inherited_entries() -> None:
    inherited = os.pathsep.join(("/opt/first", str(REPO_ROOT), "/opt/second", "/opt/first"))
    assert CLI_MODULE._repo_first_pythonpath(inherited).split(os.pathsep) == [
        str(REPO_ROOT),
        "/opt/first",
        "/opt/second",
    ]


def test_scientific_eval_entrypoint_is_exact_source_bound_canonical_file(
    tmp_path: Path,
) -> None:
    canonical = REPO_ROOT / "gear_sonic/eval_agent_trl.py"
    assert (
        CLI_MODULE._validated_eval_entrypoint(
            canonical,
            scientific_use=True,
        )
        == canonical.resolve()
    )
    alternate = tmp_path / "eval.py"
    alternate.write_text("raise SystemExit(0)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source-bound canonical"):
        CLI_MODULE._validated_eval_entrypoint(alternate, scientific_use=True)
    assert (
        CLI_MODULE._validated_eval_entrypoint(
            alternate,
            scientific_use=False,
        )
        == alternate.resolve()
    )


def test_cli_checkpoint_preflight_requires_companion_sonic_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "missing_config" / "last.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights only\n")
    with pytest.raises(ValueError, match="must contain SONIC config.yaml"):
        CLI_MODULE._checkpoint_bundle(checkpoint.resolve())


def test_scientific_runtime_paths_are_new_distinct_and_never_forced(tmp_path: Path) -> None:
    instrument = tmp_path / "instrument.json"
    runtime_paths = CLI_MODULE._instrument_runtime_paths(instrument)
    CLI_MODULE._validate_new_scientific_paths(
        plan_output=tmp_path / "plan.json",
        rollout_output=tmp_path / "rollout.jsonl",
        runtime_paths=runtime_paths,
        force=False,
    )

    with pytest.raises(ValueError, match="may not use --force"):
        CLI_MODULE._validate_new_scientific_paths(
            plan_output=tmp_path / "plan.json",
            rollout_output=tmp_path / "rollout.jsonl",
            runtime_paths=runtime_paths,
            force=True,
        )
    with pytest.raises(ValueError, match="must be distinct"):
        CLI_MODULE._validate_new_scientific_paths(
            plan_output=runtime_paths["handshake"],
            rollout_output=tmp_path / "rollout.jsonl",
            runtime_paths=runtime_paths,
            force=False,
        )

    runtime_paths["binding"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="binding already exists"):
        CLI_MODULE._validate_new_scientific_paths(
            plan_output=tmp_path / "plan.json",
            rollout_output=tmp_path / "rollout.jsonl",
            runtime_paths=runtime_paths,
            force=False,
        )


def test_loader_and_cli_remain_cpu_safe() -> None:
    loader_source = (REPO_ROOT / "gear_sonic/research/lace/schedule_batch_loader.py").read_text(
        encoding="utf-8"
    )
    cli_source = CLI_PATH.read_text(encoding="utf-8")

    assert "import torch" not in loader_source
    assert "import isaaclab" not in loader_source
    assert "import torch" not in cli_source
    assert "import isaaclab" not in cli_source
