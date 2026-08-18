from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from gear_sonic.research.lace.fixed_distribution import (
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
)
from gear_sonic.research.lace.intervention_plan import (
    INTERVENTION_PLAN_DIGEST_FIELD,
    INTERVENTION_PROTOCOL_DIGEST_FIELD,
    build_intervention_plan,
)
from gear_sonic.research.lace.panels import build_representation_blind_panels
from gear_sonic.research.lace.rq1_training import (
    ATTEMPT_CLAIM_DIGEST_FIELD,
    ATTEMPT_CLAIM_KIND,
    ATTEMPT_CLAIM_SCHEMA_VERSION,
    EXECUTION_COMPLETION_DIGEST_FIELD,
    EXECUTION_COMPLETION_KIND,
    EXECUTION_COMPLETION_SCHEMA_VERSION,
    PLAN_DIGEST_FIELD,
    PLAN_KIND,
    PLAN_SCHEMA_VERSION,
    RQ1RuntimeAccounting,
    RQ1TrainingError,
    assert_rq1_trainer_callbacks,
    build_rq1_training_plan,
    build_rq1_training_receipt,
    dry_compose_rq1_training_plan,
    file_sha256,
    load_json_object,
    strict_load_rq1_state_dict,
    torch_state_dict_record,
    validate_attempt_claim,
    validate_execution_completion,
    validate_gpu_isolation_monitor,
    validate_rq1_training_plan,
    validate_rq1_training_receipt,
    validate_runtime_metrics,
    write_new_json,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split
from gear_sonic.research.lace.throughput import materialize_partition_subset

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/robotixx/miniconda3/envs/env_isaaclab/bin/python")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_checkpoint_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha(path),
    }


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _file_sha(path),
    }


def _fixture_initialization_report(plan: dict) -> dict:
    policy = torch_state_dict_record({"fixture.weight": torch.tensor([1.0])})
    value = torch_state_dict_record({"fixture.value": torch.tensor([2.0])})
    report = {
        "kind": "lace_rq1_model_only_initialization_report",
        "schema_version": 1,
        "checkpoint_path": plan["initialization"]["checkpoint_path"],
        "checkpoint_sha256": plan["initialization"]["checkpoint_sha256"],
        "resume": False,
        "policy_source_key": "policy_state_dict",
        "policy_strict": True,
        "policy_checkpoint_state": policy,
        "policy_post_load_state": deepcopy(policy),
        "value_source_key": "value_state_dict",
        "value_strict": True,
        "value_checkpoint_state": value,
        "value_post_load_state": deepcopy(value),
        "optimizer_state_restored": False,
        "lr_scheduler_state_restored": False,
        "environment_state_restored": False,
        "trainer_state_restored": False,
        "optimizer_state_entry_count_before_training": 0,
    }
    report["initialization_report_sha256"] = canonical_sha256(report)
    return report


def _fixture_attempt_claim(plan: dict, *, publish: bool) -> dict:
    plan_path = Path(plan["plan_path"])
    plan_file_sha256 = _file_sha(plan_path) if plan_path.is_file() else "e" * 64
    claim = {
        "kind": ATTEMPT_CLAIM_KIND,
        "schema_version": ATTEMPT_CLAIM_SCHEMA_VERSION,
        "frozen": True,
        "cell_id": plan["cell_id"],
        "attempt_id": "attempt-001",
        "plan_path": str(plan_path.resolve()),
        "plan_file_sha256": plan_file_sha256,
        PLAN_DIGEST_FIELD: plan[PLAN_DIGEST_FIELD],
        "command_sha256": plan["command_sha256"],
        "cwd": plan["launch_cwd"],
        "argv": plan["argv"],
        "environment": plan["launch_environment"],
        "launcher_pid": 1234,
        "launcher_process_group_id": 1234,
        "preflight_completed_at_unix_ns": 1_000_000_000,
        "deep_plan_reconstruction_pass": True,
        "dataset_bytes_preflight_pass": True,
        "source_asset_bytes_preflight_pass": True,
        "claimed_at_utc": "1970-01-01T00:00:02Z",
        "claimed_at_unix_ns": 2_000_000_000,
        "authority_scope": "local_atomic_no_clobber_not_external_worm",
    }
    claim[ATTEMPT_CLAIM_DIGEST_FIELD] = canonical_sha256(
        claim,
        digest_field=ATTEMPT_CLAIM_DIGEST_FIELD,
    )
    if publish:
        write_new_json(plan["runtime_artifacts"]["attempt_claim"], claim)
    return claim


def _fixture_draw_report(plan: dict, draw_counts: list[int]) -> dict:
    arm = plan["arm_contract"]
    total = sum(draw_counts)
    report = {
        "kind": "lace_fixed_motion_distribution_draw_report",
        "schema_version": 1,
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD: arm[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD],
        "fixed_distribution_config_sha256": arm["fixed_distribution_config_sha256"],
        "arm_id": plan["arm_id"],
        INTERVENTION_PLAN_DIGEST_FIELD: arm[INTERVENTION_PLAN_DIGEST_FIELD],
        "plan_lock_file_sha256": arm["plan_lock_file_sha256"],
        "distribution_sha256": arm["distribution_sha256"],
        "motion_keys": arm["motion_keys"],
        "motion_count": arm["motion_count"],
        "draw_counts": draw_counts,
        "total_draw_count": total,
        "counter_sum": total,
        "exact_invariants_pass": True,
        "accounting_scope": arm["accounting_scope"],
    }
    report[FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD] = canonical_sha256(
        report,
        digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    )
    return report


def _locked_inputs(tmp_path: Path) -> tuple[Path, dict, Path]:
    sources = tmp_path / "sources"
    robot_root = sources / "robot"
    smpl_root = sources / "smpl"
    robot_root.mkdir(parents=True)
    smpl_root.mkdir()
    motions = []
    for group_index in range(48):
        for variant in range(1 + group_index % 2):
            key = f"motion_{group_index:02d}_{variant}__A{group_index:03d}"
            robot = robot_root / f"{key}.pkl"
            smpl = smpl_root / f"{key}.pkl"
            robot.write_bytes(f"robot:{key}".encode())
            smpl.write_bytes(f"smpl:{key}".encode())
            motions.append(
                {
                    "motion_key": key,
                    "release_filter_key": f"{key}.pkl",
                    "source_group_id": f"actor_{group_index:03d}",
                    "duration_source_frames": 100 + group_index + variant,
                    "stratum": f"q{group_index % 4}",
                    "robot_path": str(robot),
                    "smpl_path": str(smpl),
                }
            )
    split = build_source_disjoint_split(motions, seed=81)
    panels = build_representation_blind_panels(split, panel_count=4, seed=27)
    motion_count = sum(row["partition"] == "D_curriculum" for row in split["motions"])
    protocol = {
        "kind": "lace_rq1_intervention_protocol",
        "schema_version": 1,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_curriculum",
        "expected_motion_count": motion_count,
        "expected_panel_count": 4,
        "panel_seed": 27,
        "base_distribution_method": "uniform_over_canonically_ordered_motions_v1",
        "sequence_length_agnostic": True,
        "target_kl_nats": 0.03,
        "max_probability_ratio": 3.0,
        "kl_tolerance": 1e-12,
        "probability_tolerance": 1e-12,
        "bisection_iterations": 100,
        "maximum_added_exposure_range": 0.02,
        "interpretation": "RQ1 runner unit-test dose",
    }
    protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD] = canonical_sha256(
        protocol,
        digest_field=INTERVENTION_PROTOCOL_DIGEST_FIELD,
    )
    intervention_plan = build_intervention_plan(split, panels, protocol)
    inputs = tmp_path / "locked"
    split_path = inputs / "split.json"
    panels_path = inputs / "panels.json"
    intervention_protocol_path = inputs / "intervention_protocol.json"
    intervention_plan_path = inputs / "intervention_plan.json"
    _write_json(split_path, split)
    _write_json(panels_path, panels)
    _write_json(intervention_protocol_path, protocol)
    _write_json(intervention_plan_path, intervention_plan)
    lock = {
        "kind": "lace_rq1_intervention_plan_artifact_lock",
        "schema_version": 1,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "artifact": {
            "path": str(intervention_plan_path),
            "file_sha256": _file_sha(intervention_plan_path),
            INTERVENTION_PLAN_DIGEST_FIELD: intervention_plan[INTERVENTION_PLAN_DIGEST_FIELD],
        },
        "inputs": {
            "split_manifest": str(split_path),
            "split_manifest_sha256": _file_sha(split_path),
            "split_sha256": split["split_sha256"],
            "split_selection_sha256": split["selection_sha256"],
            "panel_manifest": str(panels_path),
            "panel_manifest_sha256": _file_sha(panels_path),
            "panel_sha256": panels["panel_sha256"],
            "intervention_protocol": str(intervention_protocol_path),
            "intervention_protocol_sha256": _file_sha(intervention_protocol_path),
            INTERVENTION_PROTOCOL_DIGEST_FIELD: protocol[INTERVENTION_PROTOCOL_DIGEST_FIELD],
        },
    }
    lock_path = inputs / "intervention_plan_lock.json"
    _write_json(lock_path, lock)
    subset_root = tmp_path / "datasets/d_curriculum"
    materialize_partition_subset(
        split_path,
        partition="D_curriculum",
        destination=subset_root,
    )
    return lock_path, intervention_plan, subset_root


def _training_protocol(tmp_path: Path, *, arm_index: int | None = None) -> tuple[dict, Path]:
    lock_path, intervention_plan, subset_root = _locked_inputs(tmp_path)
    init = tmp_path / "checkpoints/init"
    init.mkdir(parents=True)
    checkpoint = init / "last.pt"
    checkpoint_config = init / "config.yaml"
    checkpoint.write_bytes(b"model-only-init")
    checkpoint_config.write_text("profile: fixture\n", encoding="utf-8")
    arm_id = (
        "base" if arm_index is None else intervention_plan["interventions"][arm_index]["panel_id"]
    )
    protocol = {
        "kind": "lace_rq1_training_protocol",
        "schema_version": 2,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "run_id": f"fixture-{arm_id}",
        "arm_id": arm_id,
        "repo_root": str(REPO_ROOT),
        "storage_root": str(tmp_path),
        "output_root": str(tmp_path / "runs"),
        "tmp_root": str(tmp_path / "tmp"),
        "python_executable": str(PYTHON),
        "train_entrypoint": "gear_sonic/train_agent_trl.py",
        "experiment": "manager/universal_token/g1_only/lace_lite_s",
        "seed": 7,
        "fixed_distribution": {
            "plan_lock_path": str(lock_path),
            "plan_lock_file_sha256": _file_sha(lock_path),
        },
        "dataset": {
            "partition": "D_curriculum",
            "split_manifest": str(tmp_path / "locked/split.json"),
            "subset_root": str(subset_root),
            "expected_motion_count": intervention_plan["motion_count"],
        },
        "initialization": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _file_sha(checkpoint),
            "checkpoint_config_path": str(checkpoint_config),
            "checkpoint_config_sha256": _file_sha(checkpoint_config),
            "mode": "model_only",
            "resume": False,
        },
        "training": {
            "num_envs_per_rank": 4,
            "world_size": 1,
            "rollout_steps_per_iteration": 2,
            "iterations": 2,
            "ppo_epochs": 2,
            "minibatches_per_epoch": 2,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "decimation": 4,
            "terrain_type": "plane",
        },
        "gpu_isolation": {
            "device_index": 0,
            "device_uuid": "GPU-fixture-5090",
            "device_name": "NVIDIA GeForce RTX 5090",
            "driver_version": "fixture-driver",
            "total_memory_mib": 32607,
            "minimum_free_preflight_mib": 28672,
            "monitor_poll_seconds": 1.0,
        },
    }
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    protocol_path = tmp_path / "rq1_training_protocol.json"
    _write_json(protocol_path, protocol)
    return protocol, protocol_path


def _build_plan(tmp_path: Path, *, arm_index: int | None = None) -> tuple[dict, Path]:
    protocol, protocol_path = _training_protocol(tmp_path, arm_index=arm_index)
    plan_path = tmp_path / "manifests/rq1_plan.json"
    plan = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=plan_path,
    )
    return plan, plan_path


def _complete_metrics(plan: dict) -> dict:
    plan_path = Path(plan["plan_path"])
    if not plan_path.exists():
        write_new_json(plan_path, plan)
    claim_path = Path(plan["runtime_artifacts"]["attempt_claim"])
    claim = (
        load_json_object(claim_path)
        if claim_path.exists()
        else _fixture_attempt_claim(plan, publish=True)
    )
    accounting = RQ1RuntimeAccounting(plan)
    motion_count = plan["arm_contract"]["motion_count"]
    for step in range(4):
        ids = [(step * 4 + offset) % motion_count for offset in range(4)]
        accounting.record_control_step(ids, [0, 0, 0, 0], [0, 0, 0, 0])
    for _ in range(2):
        accounting.record_optimizer_iteration(
            optimizer_step_attempts=4,
            optimizer_nonfinite_skips=0,
            optimizer_accelerator_skips=0,
            sync_boundaries=4,
            successful_parameter_updates=4,
            synchronized_update_skips=0,
        )
    draw_counts = [0] * motion_count
    for index in range(plan["training_semantics"]["num_envs_per_rank"]):
        draw_counts[index] += 1
    draw_report = _fixture_draw_report(plan, draw_counts)
    checkpoint_path = Path(plan["runtime_artifacts"]["final_checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        checkpoint_path.write_bytes(b"fixture-final-checkpoint")
    return accounting.build_runtime_metrics(
        draw_report,
        final_checkpoint=_final_checkpoint_record(checkpoint_path),
        attempt_claim=claim,
        initialization_report=_fixture_initialization_report(plan),
    )


def _save_composed_config(plan: dict, path: Path) -> None:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        config_dir=str(REPO_ROOT / "gear_sonic/config"),
        version_base="1.1",
    ):
        config = compose(config_name="base", overrides=plan["argv"][2:])
    config.algo.trl.output_dir = str(Path(plan["output_dir"]))
    config.multi_gpu = False
    OmegaConf.save(config, path)


def _publish_completion_authority(plan: dict) -> dict:
    artifacts = plan["runtime_artifacts"]
    claim_path = Path(artifacts["attempt_claim"])
    claim = load_json_object(claim_path)
    contract = plan["gpu_isolation_contract"]
    sample_specs = [
        ("preflight", 2_500_000_000, 1_000, 30_000, []),
        (
            "continuous",
            3_500_000_000,
            14_000,
            18_000,
            [{"pid": 1235, "process_group_id": 1235, "used_memory_mib": 12_000}],
        ),
        (
            "continuous",
            4_500_000_000,
            14_500,
            17_000,
            [{"pid": 1236, "process_group_id": 1235, "used_memory_mib": 12_500}],
        ),
        ("postflight", 5_500_000_000, 1_000, 30_000, []),
    ]
    samples = []
    previous_sample_sha256 = None
    for index, (phase, observed_at, used_mib, free_mib, processes) in enumerate(sample_specs):
        raw_query = f"fixture nvidia-smi sample {index}"
        sample = {
            "sequence_index": index,
            "phase": phase,
            "observed_at_unix_ns": observed_at,
            "device_index": contract["device_index"],
            "device_uuid": contract["device_uuid"],
            "device_name": contract["device_name"],
            "driver_version": contract["driver_version"],
            "total_memory_mib": contract["total_memory_mib"],
            "used_mib": used_mib,
            "free_mib": free_mib,
            "compute_processes": processes,
            "raw_query": raw_query,
            "raw_query_sha256": hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
            "previous_sample_sha256": previous_sample_sha256,
        }
        sample["sample_sha256"] = canonical_sha256(
            sample,
            digest_field="sample_sha256",
        )
        previous_sample_sha256 = sample["sample_sha256"]
        samples.append(sample)
    monitor = {
        "kind": "lace_rq1_gpu_isolation_monitor",
        "schema_version": 2,
        "frozen": True,
        "cell_id": plan["cell_id"],
        "attempt_id": claim["attempt_id"],
        "gpu_isolation_contract_sha256": contract["gpu_isolation_contract_sha256"],
        "device_index": contract["device_index"],
        "device_uuid": contract["device_uuid"],
        "device_name": contract["device_name"],
        "driver_version": contract["driver_version"],
        "total_memory_mib": contract["total_memory_mib"],
        "minimum_free_preflight_mib": contract["minimum_free_preflight_mib"],
        "monitor_poll_seconds": contract["monitor_poll_seconds"],
        "authorized_child_pid": 1235,
        "authorized_process_group_id": 1235,
        "raw_samples": samples,
        "exclusive_preflight_pass": True,
        "continuous_monitor_pass": True,
        "postflight_pass": True,
        "foreign_compute_process_observations": 0,
        "foreign_memory_peak_mib": 0,
        "sample_count": len(samples),
        "minimum_observed_free_mib": 17_000,
        "peak_used_mib": 14_500,
        "maximum_observed_poll_gap_seconds": 1.0,
        "started_at_unix_ns": 2_500_000_000,
        "ended_at_unix_ns": 5_500_000_000,
        "raw_sample_chain_sha256": canonical_sha256(
            {"sample_sha256": [sample["sample_sha256"] for sample in samples]}
        ),
    }
    monitor["gpu_monitor_sha256"] = canonical_sha256(
        monitor,
        digest_field="gpu_monitor_sha256",
    )
    monitor_path = Path(artifacts["gpu_isolation_monitor"])
    write_new_json(monitor_path, monitor)
    claim_artifact = _artifact_record(claim_path)
    claim_artifact[ATTEMPT_CLAIM_DIGEST_FIELD] = claim[ATTEMPT_CLAIM_DIGEST_FIELD]
    completion = {
        "kind": EXECUTION_COMPLETION_KIND,
        "schema_version": EXECUTION_COMPLETION_SCHEMA_VERSION,
        "frozen": True,
        "cell_id": plan["cell_id"],
        "attempt_id": claim["attempt_id"],
        "attempt_claim": claim_artifact,
        PLAN_DIGEST_FIELD: plan[PLAN_DIGEST_FIELD],
        "plan_file_sha256": _file_sha(Path(plan["plan_path"])),
        "command_sha256": plan["command_sha256"],
        "cwd": plan["launch_cwd"],
        "argv": plan["argv"],
        "environment": plan["launch_environment"],
        "child_pid": 1235,
        "child_process_group_id": 1235,
        "started_at_utc": "1970-01-01T00:00:03Z",
        "ended_at_utc": "1970-01-01T00:00:05Z",
        "started_at_unix_ns": 3_000_000_000,
        "ended_at_unix_ns": 5_000_000_000,
        "exit_code": 0,
        "term_signal": None,
        "oom_killed": False,
        "foreign_process_contamination": False,
        "process_log": _artifact_record(Path(artifacts["process_log"])),
        "runtime_metrics": _artifact_record(Path(artifacts["runtime_metrics"])),
        "final_checkpoint": _artifact_record(Path(artifacts["final_checkpoint"])),
        "gpu_isolation_monitor": _artifact_record(monitor_path),
    }
    completion[EXECUTION_COMPLETION_DIGEST_FIELD] = canonical_sha256(
        completion,
        digest_field=EXECUTION_COMPLETION_DIGEST_FIELD,
    )
    write_new_json(artifacts["execution_completion"], completion)
    return completion


def _rehash_gpu_monitor(monitor: dict) -> None:
    previous_sample_sha256 = None
    sample_digests = []
    for index, sample in enumerate(monitor["raw_samples"]):
        sample["sequence_index"] = index
        sample["raw_query_sha256"] = hashlib.sha256(sample["raw_query"].encode("utf-8")).hexdigest()
        sample["previous_sample_sha256"] = previous_sample_sha256
        sample["sample_sha256"] = canonical_sha256(
            sample,
            digest_field="sample_sha256",
        )
        previous_sample_sha256 = sample["sample_sha256"]
        sample_digests.append(sample["sample_sha256"])
    monitor["sample_count"] = len(monitor["raw_samples"])
    monitor["minimum_observed_free_mib"] = min(
        sample["free_mib"] for sample in monitor["raw_samples"]
    )
    monitor["peak_used_mib"] = max(sample["used_mib"] for sample in monitor["raw_samples"])
    times = [sample["observed_at_unix_ns"] for sample in monitor["raw_samples"]]
    monitor["maximum_observed_poll_gap_seconds"] = max(
        (right - left) / 1_000_000_000 for left, right in zip(times[:-1], times[1:], strict=True)
    )
    monitor["started_at_unix_ns"] = times[0]
    monitor["ended_at_unix_ns"] = times[-1]
    monitor["raw_sample_chain_sha256"] = canonical_sha256({"sample_sha256": sample_digests})
    monitor["gpu_monitor_sha256"] = canonical_sha256(
        monitor,
        digest_field="gpu_monitor_sha256",
    )


def _completed_run(tmp_path: Path) -> tuple[dict, Path, Path]:
    plan, plan_path = _build_plan(tmp_path)
    write_new_json(plan_path, plan)
    _fixture_attempt_claim(plan, publish=True)
    output = Path(plan["output_dir"])
    (output / ".hydra").mkdir(parents=True)
    _save_composed_config(plan, output / "config.yaml")
    (output / "last.pt").write_bytes(b"final-checkpoint")
    (output / ".hydra/train.log").write_text("train complete\n", encoding="utf-8")
    (output / "process.log").write_text("process exit 0\n", encoding="utf-8")
    metrics = _complete_metrics(plan)
    write_new_json(output / "rq1_runtime_metrics.json", metrics)
    _publish_completion_authority(plan)
    return plan, plan_path, output


def test_plan_deep_resolves_one_arm_and_cpu_composes_without_sampler_bypasses(
    tmp_path: Path,
) -> None:
    plan, _ = _build_plan(tmp_path, arm_index=0)

    validate_rq1_training_plan(plan)
    readback = dry_compose_rq1_training_plan(plan)

    assert plan["arm_id"].startswith("source_panel_")
    assert plan["dataset"]["motion_count"] == plan["arm_contract"]["motion_count"]
    assert plan["dataset"]["motion_count"] != 233
    assert all(readback["training_modes"].values())
    assert "im_resample" not in readback["callback_names"]
    assert plan["training_semantics"]["initialization_mode"] == "model_only_resume_false"


def test_plan_rejects_rehashed_protocol_arm_not_present_in_external_lock(tmp_path: Path) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    protocol["arm_id"] = "source_panel_99"
    protocol["run_id"] = "fixture-invalid-arm"
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)

    with pytest.raises((RQ1TrainingError, ValueError), match="unknown|unique panel"):
        build_rq1_training_plan(
            protocol,
            protocol_path=protocol_path,
            plan_path=tmp_path / "manifests/invalid.json",
        )


def test_plan_rejects_alternate_valid_split_with_same_curriculum_motions(tmp_path: Path) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    alternate_split = tmp_path / "alternate/split.json"
    original_split = Path(protocol["dataset"]["split_manifest"])
    _write_json(alternate_split, load_json_object(original_split))
    alternate_subset = tmp_path / "datasets/alternate_d_curriculum"
    materialize_partition_subset(
        alternate_split,
        partition="D_curriculum",
        destination=alternate_subset,
    )
    protocol["dataset"]["split_manifest"] = str(alternate_split)
    protocol["dataset"]["subset_root"] = str(alternate_subset)
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)

    with pytest.raises(RQ1TrainingError, match="split path differs"):
        build_rq1_training_plan(
            protocol,
            protocol_path=protocol_path,
            plan_path=tmp_path / "manifests/alternate-split.json",
        )


def _synthetic_plan(motion_count: int = 1996) -> dict:
    keys = [f"motion_{index:04d}" for index in range(motion_count)]
    probabilities = [1.0 / motion_count] * motion_count
    panels = []
    for panel_index in range(4):
        panels.append(
            {
                "panel_id": f"source_panel_{panel_index:02d}",
                "motion_keys": keys[panel_index::4],
            }
        )
    plan = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "run_id": "fixture-synthetic",
        "cell_id": "rq1-" + "a" * 32,
        "arm_id": "base",
        "launch_contract_sha256": "b" * 64,
        "plan_path": "/fixture/plan.json",
        "initialization": {
            "checkpoint_path": "/fixture/init.pt",
            "checkpoint_sha256": "f" * 64,
        },
        "arm_contract": {
            "motion_keys": keys,
            "motion_count": motion_count,
            "base_probabilities": probabilities,
            "probabilities": probabilities,
            "panels": panels,
            FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD: "c" * 64,
            "fixed_distribution_config_sha256": "d" * 64,
            INTERVENTION_PLAN_DIGEST_FIELD: "e" * 64,
            "plan_lock_file_sha256": "f" * 64,
            "distribution_sha256": "1" * 64,
            "accounting_scope": (
                "motion_draw_counts_only_v1;occupancy_and_optimizer_updates_runner_required"
            ),
        },
        "training_semantics": {
            "num_envs_per_rank": 2,
            "world_size": 1,
            "iterations": 1,
            "rollout_steps_per_iteration": 1,
            "ppo_epochs": 1,
            "minibatches_per_epoch": 1,
            "local_minibatch_size": 2,
            "per_device_train_batch_size": 2,
            "num_microbatches_per_minibatch": 1,
            "gradient_accumulation_steps": 1,
            "decimation": 4,
            "planned_control_transitions": 2,
            "planned_physics_substeps": 8,
            "physics_substeps_semantics": (
                "derived_control_transitions_times_live_verified_decimation_not_directly_observed"
            ),
            "planned_optimizer_opportunities": 1,
            "planned_synchronized_parameter_updates": 1,
        },
        "launch_cwd": "/fixture",
        "launch_environment": {},
        "argv": ["python", "train.py", "+exp=fixture"],
        "expected_saved_config": {},
        "expected_saved_config_sha256": canonical_sha256({}),
    }
    plan["command_sha256"] = canonical_sha256(
        {
            "cwd": plan["launch_cwd"],
            "environment": plan["launch_environment"],
            "argv": plan["argv"],
        }
    )
    plan[PLAN_DIGEST_FIELD] = canonical_sha256(plan, digest_field=PLAN_DIGEST_FIELD)
    return plan


def test_accounting_supports_headline_1996_and_attributes_terminal_to_pre_step_motion() -> None:
    plan = _synthetic_plan()
    accounting = RQ1RuntimeAccounting(plan)
    accounting.record_control_step([0, 1995], [1, 1], [0, 1])
    accounting.record_optimizer_iteration(
        optimizer_step_attempts=1,
        optimizer_nonfinite_skips=0,
        optimizer_accelerator_skips=0,
        sync_boundaries=1,
        successful_parameter_updates=1,
        synchronized_update_skips=0,
    )
    draw_counts = [0] * 1996
    draw_counts[0] = 1
    draw_counts[1995] = 1
    draw_report = _fixture_draw_report(plan, draw_counts)

    metrics = accounting.build_runtime_metrics(
        draw_report,
        final_checkpoint={
            "path": "/fixture/final.pt",
            "bytes": 1,
            "sha256": "d" * 64,
        },
        attempt_claim=_fixture_attempt_claim(plan, publish=False),
        initialization_report=_fixture_initialization_report(plan),
    )

    rows = metrics["control_accounting"]["per_motion"]
    assert len(rows) == 1996
    assert rows[0]["termination_count"] == 1
    assert rows[1995]["timeout_count"] == 1
    assert metrics["control_accounting"]["total_control_transitions"] == 2
    assert sum(row["draw_count"] for row in metrics["control_accounting"]["per_panel"]) == 2


def test_runtime_metrics_fail_closed_on_any_optimizer_skip() -> None:
    plan = _synthetic_plan(4)
    accounting = RQ1RuntimeAccounting(plan)
    accounting.record_control_step([0, 1], [0, 0], [0, 0])
    accounting.record_optimizer_iteration(
        optimizer_step_attempts=0,
        optimizer_nonfinite_skips=1,
        optimizer_accelerator_skips=0,
        sync_boundaries=1,
        successful_parameter_updates=0,
        synchronized_update_skips=1,
    )
    draw_report = _fixture_draw_report(plan, [1, 1, 0, 0])

    with pytest.raises(RQ1TrainingError, match="not every planned optimizer step"):
        accounting.build_runtime_metrics(
            draw_report,
            final_checkpoint={
                "path": "/fixture/final.pt",
                "bytes": 1,
                "sha256": "d" * 64,
            },
            attempt_claim=_fixture_attempt_claim(plan, publish=False),
            initialization_report=_fixture_initialization_report(plan),
        )


def test_completed_receipt_binds_plan_config_checkpoints_dataset_sources_logs_and_metrics(
    tmp_path: Path,
) -> None:
    plan, plan_path, output = _completed_run(tmp_path)

    receipt = build_rq1_training_receipt(plan, plan_path=plan_path)
    validate_rq1_training_receipt(receipt, plan)

    assert receipt["exact_budget_pass"] is True
    assert set(receipt["artifacts"]) >= {
        "plan",
        "protocol",
        "attempt_claim",
        "resolved_config",
        "initial_checkpoint",
        "initial_checkpoint_config",
        "final_checkpoint",
        "hydra_log",
        "process_log",
        "runtime_metrics",
        "gpu_isolation_monitor",
        "execution_completion",
        "dataset_manifest",
        "split_manifest",
    }
    omitted = deepcopy(receipt)
    del omitted["artifacts"]["final_checkpoint"]
    omitted["receipt_sha256"] = canonical_sha256(
        omitted,
        digest_field="receipt_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="deterministic reconstruction"):
        validate_rq1_training_receipt(omitted, plan)

    (output / "process.log").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RQ1TrainingError, match="artifact bytes drifted"):
        validate_rq1_training_receipt(receipt, plan)


def test_receipt_rejects_fully_rehashed_reward_config_mutation(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    plan, plan_path, output = _completed_run(tmp_path)
    config_path = output / "config.yaml"
    config = OmegaConf.load(config_path)
    config.manager_env.rewards.feet_acc.weight = 123456.0
    OmegaConf.save(config, config_path)

    with pytest.raises(RQ1TrainingError, match="exact composed config"):
        build_rq1_training_receipt(plan, plan_path=plan_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exit_code", 137, "exit zero"),
        ("cwd", "/tmp/foreign", "cwd mismatch"),
        ("argv", ["python", "foreign.py"], "argv mismatch"),
        ("environment", {"PATH": "/foreign"}, "environment mismatch"),
    ],
)
def test_execution_completion_rejects_rehashed_exit_cwd_argv_or_environment(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    plan, _, _ = _completed_run(tmp_path)
    completion = load_json_object(plan["runtime_artifacts"]["execution_completion"])
    completion[field] = value
    completion[EXECUTION_COMPLETION_DIGEST_FIELD] = canonical_sha256(
        completion,
        digest_field=EXECUTION_COMPLETION_DIGEST_FIELD,
    )

    with pytest.raises(RQ1TrainingError, match=message):
        validate_execution_completion(completion, plan)


def test_execution_chronology_rejects_rehashed_inverted_utc_and_unix_times(
    tmp_path: Path,
) -> None:
    plan, _, _ = _completed_run(tmp_path)
    completion = load_json_object(plan["runtime_artifacts"]["execution_completion"])
    completion["started_at_utc"] = "1970-01-01T00:00:06Z"
    completion["started_at_unix_ns"] = 6_000_000_000
    completion["ended_at_utc"] = "1970-01-01T00:00:04Z"
    completion["ended_at_unix_ns"] = 4_000_000_000
    completion[EXECUTION_COMPLETION_DIGEST_FIELD] = canonical_sha256(
        completion,
        digest_field=EXECUTION_COMPLETION_DIGEST_FIELD,
    )

    with pytest.raises(RQ1TrainingError, match="timing precedes"):
        validate_execution_completion(completion, plan)

    claim = _fixture_attempt_claim(plan, publish=False)
    claim["claimed_at_utc"] = "1970-01-01T00:00:03Z"
    claim[ATTEMPT_CLAIM_DIGEST_FIELD] = canonical_sha256(
        claim,
        digest_field=ATTEMPT_CLAIM_DIGEST_FIELD,
    )
    with pytest.raises(RQ1TrainingError, match="UTC and unix-nanosecond"):
        validate_attempt_claim(claim, plan, require_files=False)


def test_gpu_monitor_rejects_fully_rehashed_identity_isolation_memory_and_cadence_attacks(
    tmp_path: Path,
) -> None:
    plan, _, _ = _completed_run(tmp_path)
    monitor_path = Path(plan["runtime_artifacts"]["gpu_isolation_monitor"])
    original = load_json_object(monitor_path)

    foreign = deepcopy(original)
    foreign["raw_samples"][1]["compute_processes"].append(
        {"pid": 9999, "process_group_id": 9999, "used_memory_mib": 512}
    )
    _rehash_gpu_monitor(foreign)
    with pytest.raises(RQ1TrainingError, match="foreign GPU contamination"):
        validate_gpu_isolation_monitor(
            foreign,
            plan,
            attempt_id="attempt-001",
            child_pid=1235,
            child_process_group_id=1235,
        )

    low_free = deepcopy(original)
    low_free["raw_samples"][0]["free_mib"] = 28_000
    _rehash_gpu_monitor(low_free)
    with pytest.raises(RQ1TrainingError, match="below the frozen threshold"):
        validate_gpu_isolation_monitor(
            low_free,
            plan,
            attempt_id="attempt-001",
            child_pid=1235,
            child_process_group_id=1235,
        )

    cadence_gap = deepcopy(original)
    cadence_gap["raw_samples"][2]["observed_at_unix_ns"] = 7_000_000_000
    cadence_gap["raw_samples"][3]["observed_at_unix_ns"] = 8_000_000_000
    _rehash_gpu_monitor(cadence_gap)
    with pytest.raises(RQ1TrainingError, match="poll cadence"):
        validate_gpu_isolation_monitor(
            cadence_gap,
            plan,
            attempt_id="attempt-001",
            child_pid=1235,
            child_process_group_id=1235,
        )

    wrong_device = deepcopy(original)
    wrong_device["device_uuid"] = "GPU-foreign-5090"
    for sample in wrong_device["raw_samples"]:
        sample["device_uuid"] = "GPU-foreign-5090"
    _rehash_gpu_monitor(wrong_device)
    with pytest.raises(RQ1TrainingError, match="device_uuid differs"):
        validate_gpu_isolation_monitor(
            wrong_device,
            plan,
            attempt_id="attempt-001",
            child_pid=1235,
            child_process_group_id=1235,
        )

    wrong_process_group = deepcopy(original)
    wrong_process_group["authorized_process_group_id"] = 1236
    _rehash_gpu_monitor(wrong_process_group)
    with pytest.raises(RQ1TrainingError, match="authorized child process identity"):
        validate_gpu_isolation_monitor(
            wrong_process_group,
            plan,
            attempt_id="attempt-001",
            child_pid=1235,
            child_process_group_id=1235,
        )


def test_plan_requires_pristine_output_tmp_plan_and_attempt_targets(tmp_path: Path) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    output_dir = Path(protocol["output_root"]) / protocol["run_id"]
    output_dir.mkdir(parents=True)
    (output_dir / "last.pt").write_bytes(b"aborted-prior-attempt")

    with pytest.raises(RQ1TrainingError, match="output directory must not exist"):
        build_rq1_training_plan(
            protocol,
            protocol_path=protocol_path,
            plan_path=tmp_path / "manifests/freshness.json",
        )


def test_local_attempt_claim_blocks_run_id_retry_within_same_storage_root(tmp_path: Path) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    first_path = tmp_path / "manifests/first.json"
    first = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=first_path,
    )
    write_new_json(first_path, first)
    _fixture_attempt_claim(first, publish=True)

    protocol["run_id"] = "fixture-renamed-retry"
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)
    with pytest.raises(RQ1TrainingError, match="attempt claim must not exist"):
        build_rq1_training_plan(
            protocol,
            protocol_path=protocol_path,
            plan_path=tmp_path / "manifests/retry.json",
        )


def test_copied_identical_intervention_lock_keeps_path_independent_cell_identity(
    tmp_path: Path,
) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    first = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=tmp_path / "manifests/first.json",
    )
    original_lock = Path(protocol["fixed_distribution"]["plan_lock_path"])
    copied_lock = tmp_path / "copied-lock/intervention_plan_lock.json"
    copied_lock.parent.mkdir(parents=True)
    copied_lock.write_bytes(original_lock.read_bytes())
    protocol["run_id"] = "fixture-copied-lock-path"
    protocol["fixed_distribution"]["plan_lock_path"] = str(copied_lock)
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)
    second_path = tmp_path / "manifests/second.json"
    second = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=second_path,
    )

    assert second["cell_id"] == first["cell_id"]
    assert (
        second["runtime_artifacts"]["attempt_claim"] == first["runtime_artifacts"]["attempt_claim"]
    )
    write_new_json(first["runtime_artifacts"]["attempt_claim"], {"advisory_claim": True})
    with pytest.raises(RQ1TrainingError, match="attempt claim must not exist"):
        build_rq1_training_plan(
            protocol,
            protocol_path=protocol_path,
            plan_path=second_path,
        )


def test_storage_root_relocation_does_not_create_external_attempt_authority(
    tmp_path: Path,
) -> None:
    original_storage = tmp_path / "original-storage"
    protocol, protocol_path = _training_protocol(original_storage)
    first = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=original_storage / "manifests/first.json",
    )
    write_new_json(first["plan_path"], first)
    _fixture_attempt_claim(first, publish=True)

    protocol["run_id"] = "fixture-relocated-storage-root"
    protocol["storage_root"] = str(tmp_path)
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)
    second = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=tmp_path / "manifests/second.json",
    )

    assert second["cell_id"] == first["cell_id"]
    assert (
        second["runtime_artifacts"]["attempt_claim"] != first["runtime_artifacts"]["attempt_claim"]
    )
    assert second["launch_policy"]["local_attempt_claim_scope"].startswith("advisory_")
    assert second["launch_policy"]["canonical_attempt_authority"].startswith("absent_external_")
    assert second["launch_policy"]["scientific_launch_ready"] is False


def test_matched_cell_identity_binds_stable_effective_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_path = _training_protocol(tmp_path)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    first = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=tmp_path / "manifests/omp-1.json",
    )

    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    protocol["run_id"] = "fixture-omp-64"
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    _write_json(protocol_path, protocol)
    second = build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=tmp_path / "manifests/omp-64.json",
    )

    first_environment = first["matched_launch_environment"]
    second_environment = second["matched_launch_environment"]
    assert first_environment["variables"]["OMP_NUM_THREADS"] == "1"
    assert second_environment["variables"]["OMP_NUM_THREADS"] == "64"
    assert first_environment["variables"]["TMPDIR"] == "${RQ1_TMP_DIR}"
    assert second_environment["variables"]["TMPDIR"] == "${RQ1_TMP_DIR}"
    assert first["matched_training_config_sha256"] != second["matched_training_config_sha256"]
    assert first["cell_id"] != second["cell_id"]


def test_runtime_rejects_rehashed_false_zero_draw_and_impossible_accounting(
    tmp_path: Path,
) -> None:
    plan, _ = _build_plan(tmp_path)
    metrics = _complete_metrics(plan)

    false_draw = deepcopy(metrics)
    false_draw["draw_report"]["exact_invariants_pass"] = False
    false_draw["draw_report"][FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD] = canonical_sha256(
        false_draw["draw_report"],
        digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    )
    false_draw["runtime_metrics_sha256"] = canonical_sha256(
        false_draw,
        digest_field="runtime_metrics_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="exact invariants"):
        validate_runtime_metrics(false_draw, plan)

    zero_draw = deepcopy(metrics)
    zero_draw["draw_report"]["draw_counts"] = [0] * plan["arm_contract"]["motion_count"]
    zero_draw["draw_report"]["total_draw_count"] = 0
    zero_draw["draw_report"]["counter_sum"] = 0
    zero_draw["draw_report"][FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD] = canonical_sha256(
        zero_draw["draw_report"],
        digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    )
    zero_draw["runtime_metrics_sha256"] = canonical_sha256(
        zero_draw,
        digest_field="runtime_metrics_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="initial environment assignments"):
        validate_runtime_metrics(zero_draw, plan)

    bad_sync = deepcopy(metrics)
    bad_sync["optimizer_accounting"]["sync_boundaries"] = 0
    bad_sync["runtime_metrics_sha256"] = canonical_sha256(
        bad_sync,
        digest_field="runtime_metrics_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="sync boundaries"):
        validate_runtime_metrics(bad_sync, plan)

    bad_completion = deepcopy(metrics)
    row = bad_completion["control_accounting"]["per_motion"][0]
    row["episode_completion_count"] = row["control_step_occupancy"] + 1
    row["termination_count"] = row["episode_completion_count"]
    bad_completion["runtime_metrics_sha256"] = canonical_sha256(
        bad_completion,
        digest_field="runtime_metrics_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="completions exceed occupancy"):
        validate_runtime_metrics(bad_completion, plan)


def test_rq1_strict_state_load_rejects_deleted_or_extra_checkpoint_keys() -> None:
    module = torch.nn.Linear(3, 2)
    state = module.state_dict()
    strict_load_rq1_state_dict(module, state, label="fixture")

    deleted = dict(state)
    deleted.pop("bias")
    with pytest.raises(RQ1TrainingError, match="strict RQ1 fixture"):
        strict_load_rq1_state_dict(module, deleted, label="fixture")

    extra = dict(state)
    extra["foreign"] = torch.ones(1)
    with pytest.raises(RQ1TrainingError, match="strict RQ1 fixture"):
        strict_load_rq1_state_dict(module, extra, label="fixture")


def test_rehashed_robot_asset_inventory_mutation_invalidates_plan(tmp_path: Path) -> None:
    plan, _ = _build_plan(tmp_path)
    tampered = deepcopy(plan)
    target = next(
        row
        for row in tampered["robot_asset_inventory"]
        if row["path"].endswith("g1_29dof_rev_1_0.xml")
    )
    target["sha256"] = "0" * 64
    tampered["robot_asset_inventory_sha256"] = canonical_sha256(
        {"assets": tampered["robot_asset_inventory"]}
    )
    tampered[PLAN_DIGEST_FIELD] = canonical_sha256(tampered, digest_field=PLAN_DIGEST_FIELD)

    with pytest.raises(RQ1TrainingError, match="deterministic deep reconstruction"):
        validate_rq1_training_plan(tampered)


def test_no_clobber_and_callback_composition_guards(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    write_new_json(destination, {"value": 1})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_new_json(destination, {"value": 2})
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_new_json(dangling, {"value": 3})

    callback = type("RQ1TrainingCallback", (), {})()
    saver = type("ModelSaveCallback", (), {})()
    assert_rq1_trainer_callbacks([saver, callback], resume=False)
    forbidden = type("ImResampleCallback", (), {})()
    with pytest.raises(RQ1TrainingError, match="ImResampleCallback"):
        assert_rq1_trainer_callbacks([saver, callback, forbidden], resume=False)
    with pytest.raises(RQ1TrainingError, match="resume=False"):
        assert_rq1_trainer_callbacks([saver, callback], resume=True)


def test_runtime_metrics_validation_detects_signed_exposure_or_count_tamper(
    tmp_path: Path,
) -> None:
    plan, _ = _build_plan(tmp_path)
    metrics = _complete_metrics(plan)
    validate_runtime_metrics(metrics, plan)
    tampered = deepcopy(metrics)
    tampered["control_accounting"]["per_motion"][0]["realized_signed_exposure_delta_vs_p0"] += 0.25
    tampered["runtime_metrics_sha256"] = canonical_sha256(
        tampered,
        digest_field="runtime_metrics_sha256",
    )
    with pytest.raises(RQ1TrainingError, match="deterministic"):
        validate_runtime_metrics(tampered, plan)


def test_checkpoint_or_source_byte_drift_invalidates_deep_plan(tmp_path: Path) -> None:
    plan, _ = _build_plan(tmp_path)
    checkpoint = Path(plan["initialization"]["checkpoint_path"])
    checkpoint.write_bytes(b"changed")

    with pytest.raises(RQ1TrainingError, match="checkpoint bytes drifted"):
        validate_rq1_training_plan(plan)


def test_file_sha256_requires_regular_non_symlink_file(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    link = tmp_path / "link"
    link.symlink_to(target)

    assert file_sha256(target) == _file_sha(target)
    # resolve() deliberately canonicalizes an explicit caller path before the
    # final no-follow open; dataset symlinks are separately verified by target.
    assert file_sha256(link) == _file_sha(target)


@pytest.mark.parametrize("invalid", ['{"x":NaN}', '{"x":1e999}', '{"x":1,"x":2}'])
def test_json_loader_rejects_nonfinite_overflow_and_duplicate_keys(
    tmp_path: Path,
    invalid: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(invalid, encoding="utf-8")

    with pytest.raises(RQ1TrainingError):
        load_json_object(path)
