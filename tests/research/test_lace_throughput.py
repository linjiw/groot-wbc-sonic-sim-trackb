from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gear_sonic.research.lace import throughput
from gear_sonic.research.lace.lite_init import (
    LiteInitializationError,
    materialize_lite_initialization,
    verify_lite_initialization,
)
from gear_sonic.research.lace.schema import (
    canonical_sha256 as schema_canonical_sha256,
    split_selection_sha256,
)
from gear_sonic.research.lace.throughput import (
    EXPECTED_ENV_COUNTS,
    ThroughputLaunchError,
    ThroughputProtocolError,
    aggregate_throughput_plan,
    build_throughput_plan,
    canonical_sha256,
    expected_optimizer_schedule,
    launch_throughput_cell,
    load_json_object,
    materialize_partition_subset,
    query_gpu_preflight,
    resolve_materialization_targets,
    summarize_cell,
    validate_plan,
    verify_partition_subset,
    write_new_json,
)
from gear_sonic.trl.callbacks import throughput_benchmark_callback as callback_module
from gear_sonic.trl.callbacks.throughput_benchmark_callback import (
    ThroughputBenchmarkCallback,
)
from scripts.research.run_lace_throughput import main as throughput_main

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADLINE_PROTOCOL_PATH = (
    REPO_ROOT / "configs/research/lace/throughput_lite_s_headline_scale4950_v1.json"
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_protocol(tmp_path: Path, *, with_checkpoint: bool = True) -> tuple[dict, Path]:
    storage = tmp_path / "storage"
    storage.mkdir()
    sources = storage / "sources"
    sources.mkdir()
    motions = []
    partitions = (
        "D_curriculum",
        "D_curriculum",
        "D_atlas",
        "D_geometry",
        "D_controller",
        "D_test",
    )
    for index, partition in enumerate(partitions):
        key = f"motion_{index}"
        robot = sources / f"{key}.robot.pkl"
        smpl = sources / f"{key}.smpl.pkl"
        # The production validator requires the filename stem to equal the key.
        robot_dir = sources / "robot"
        smpl_dir = sources / "smpl"
        robot_dir.mkdir(exist_ok=True)
        smpl_dir.mkdir(exist_ok=True)
        robot = robot_dir / f"{key}.pkl"
        smpl = smpl_dir / f"{key}.pkl"
        robot.write_bytes(f"robot-{key}".encode())
        smpl.write_bytes(f"smpl-{key}".encode())
        motions.append(
            {
                "motion_key": key,
                "source_group_id": f"source_{index}",
                "partition": partition,
                "robot_path": str(robot),
                "smpl_path": str(smpl),
            }
        )
    split = {
        "schema_version": 1,
        "kind": "lace_source_disjoint_split",
        "seed": 7,
        "ratios": {},
        "grouping_rule": "fixture unique source",
        "partition_summary": {
            name: {"motion_count": partitions.count(name)}
            for name in ("D_atlas", "D_curriculum", "D_geometry", "D_controller", "D_test")
        },
        "motions": motions,
    }
    split["selection_sha256"] = split_selection_sha256(split)
    split["split_sha256"] = schema_canonical_sha256(split, digest_field="split_sha256")
    split_path = storage / "split.json"
    _write_json(split_path, split)
    subset_root = storage / "datasets" / "d_curriculum"
    materialize_partition_subset(
        split_path,
        partition="D_curriculum",
        destination=subset_root,
    )

    checkpoint_dir = storage / "checkpoints" / "init"
    checkpoint_path = checkpoint_dir / "last.pt"
    checkpoint_config = checkpoint_dir / "config.yaml"
    if with_checkpoint:
        materialize_lite_initialization(
            repo_root=REPO_ROOT,
            destination=checkpoint_dir,
            seed=7,
        )
    protocol = {
        "schema_version": 1,
        "kind": "lace_sonic_lite_throughput_protocol",
        "repo_root": str(REPO_ROOT),
        "storage_root": str(storage),
        "output_root": str(storage / "throughput"),
        "tmp_root": str(storage / "tmp"),
        "run_id": "fixture_v1",
        "python_executable": sys.executable,
        "train_entrypoint": "gear_sonic/train_agent_trl.py",
        "experiment": "manager/universal_token/g1_only/lace_lite_s",
        "seed": 7,
        "env_counts": list(EXPECTED_ENV_COUNTS),
        "dataset": {
            "partition": "D_curriculum",
            "split_manifest": str(split_path),
            "subset_root": str(subset_root),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "config_path": str(checkpoint_config),
        },
        "benchmark": {
            "warmup_iterations": 5,
            "timed_iterations": 20,
            "num_steps_per_env": 24,
            "decimation": 4,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "gradient_accumulation_steps": 1,
        },
        "launch": {
            "gpu_index": 0,
            "max_external_gpu_memory_mib": 0,
            "minimum_free_gpu_memory_mib": 4096,
            "cell_timeout_seconds": 60,
            "monitor_poll_seconds": 0.25,
        },
        "selection_rule": {
            "eligibility": "all_gates_pass",
            "metric": "sustained_end_to_end_control_transitions_per_second",
            "direction": "maximize",
            "tie_breaker": "smaller_num_envs",
            "cuda_memory_safety_margin_mib": 4096,
        },
        "workload": {
            "resident_motion_count": 2,
            "resident_motion_order": "lexicographic_motion_key",
            "all_motions_loaded_required": True,
            "terrain_type": "plane",
        },
    }
    protocol_path = storage / "protocol.json"
    _write_json(protocol_path, protocol)
    return protocol, protocol_path


def _metric_row(plan: dict, cell: dict, iteration: int, *, seconds: float) -> dict:
    benchmark = plan["benchmark_contract"]["benchmark"]
    schedule = cell["cell_training_contract"]["optimizer_schedule"]
    transitions = cell["num_envs"] * benchmark["num_steps_per_env"]
    dataset = plan["benchmark_contract"]["dataset"]
    return {
        "schema_version": 1,
        "kind": "lace_sonic_lite_throughput_iteration",
        "benchmark_contract_sha256": plan["benchmark_contract_sha256"],
        "cell_contract_sha256": cell["cell_contract_sha256"],
        "cell_id": cell["cell_id"],
        "iteration": iteration,
        "phase": "warmup" if iteration <= benchmark["warmup_iterations"] else "timed",
        "num_envs": cell["num_envs"],
        "num_steps_per_env": benchmark["num_steps_per_env"],
        "decimation": benchmark["decimation"],
        "control_transitions": transitions,
        "physics_substeps": transitions * benchmark["decimation"],
        "collection_seconds": seconds * 0.6,
        "learn_seconds": seconds * 0.3,
        "trainer_iteration_seconds": seconds * 0.9,
        "end_to_end_iteration_seconds": seconds,
        "collection_control_transitions_per_second": transitions / (seconds * 0.6),
        "collection_physics_substeps_per_second": (
            transitions * benchmark["decimation"] / (seconds * 0.6)
        ),
        "trainer_control_transitions_per_second": transitions / (seconds * 0.9),
        "end_to_end_control_transitions_per_second": transitions / seconds,
        "end_to_end_physics_substeps_per_second": (transitions * benchmark["decimation"] / seconds),
        "end_to_end_iterations_per_hour": 3600.0 / seconds,
        "dataset_subset_sha256": dataset["subset_sha256"],
        "resident_motion_count": dataset["resident_motion_count"],
        "resident_unique_motion_count": dataset["resident_motion_count"],
        "resident_universe_motion_count": dataset["resident_motion_count"],
        "resident_all_motions_loaded": True,
        "resident_motion_order_sha256": dataset["resident_motion_order_sha256"],
        "resident_motion_set_sha256": dataset["resident_motion_set_sha256"],
        "terrain_type": "plane",
        "resident_dataset_identity_attested": True,
        "optimizer_step_attempts": schedule["optimizer_expected_attempts"],
        "optimizer_nonfinite_skips": 0,
        "optimizer_accelerator_skips": 0,
        "optimizer_expected_attempts": schedule["optimizer_expected_attempts"],
        "sync_boundaries": schedule["synchronized_expected_updates"],
        "synchronized_parameter_updates": schedule["synchronized_expected_updates"],
        "synchronized_update_skips": 0,
        "synchronized_expected_updates": schedule["synchronized_expected_updates"],
        "reset_count": 2,
        "termination_count": 1,
        "timeout_count": 1,
        "reset_rate_per_control_transition": 2 / transitions,
        "termination_rate_per_control_transition": 1 / transitions,
        "timeout_rate_per_control_transition": 1 / transitions,
        "termination_count_definition": "done_and_not_timeout_timeout_precedence",
        "numerical_failure_count": 0,
        "process_rss_bytes": 2 * 1024**3,
        "process_peak_rss_bytes": 3 * 1024**3,
        "output_dir_bytes": 1000,
        "output_dir_growth_bytes": 500,
        "cuda_allocated_bytes": 10 * 1024**3,
        "cuda_reserved_bytes": 12 * 1024**3,
        "cuda_peak_allocated_bytes": 13 * 1024**3,
        "cuda_peak_reserved_bytes": 14 * 1024**3,
        "cuda_total_bytes": 32 * 1024**3,
        "cuda_global_free_bytes": 18 * 1024**3,
        "cuda_global_total_bytes": 32 * 1024**3,
    }


def _launch_result(
    plan: dict,
    cell: dict,
    *,
    rows: list[dict] | None = None,
    failure_class: str | None = None,
) -> dict:
    output = Path(cell["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    process_log = Path(cell["process_log"])
    process_log.write_text("fixture process log\n", encoding="utf-8")
    metrics_path = Path(cell["metrics_jsonl"])
    if rows is not None:
        metrics_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    metrics = (
        {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": throughput.file_sha256(metrics_path),
            "row_count": len(throughput.load_iteration_rows(metrics_path)),
        }
        if metrics_path.is_file()
        else None
    )
    result = {
        "schema_version": 1,
        "kind": "lace_sonic_lite_throughput_launch_result",
        "plan_sha256": plan["plan_sha256"],
        "benchmark_contract_sha256": plan["benchmark_contract_sha256"],
        "cell_contract_sha256": cell["cell_contract_sha256"],
        "command_sha256": cell["command_sha256"],
        "cell_id": cell["cell_id"],
        "num_envs": cell["num_envs"],
        "returncode": 1 if failure_class is not None else 0,
        "failure_class": failure_class,
        "timed_out": False,
        "elapsed_seconds": 1.0,
        "external_process_peak_rss_bytes": 3 * 1024**3,
        "process_tree_peak_rss_bytes": 3 * 1024**3,
        "process_tree_peak_process_count": 2,
        "final_output_dir_bytes": 0,
        "final_tmp_dir_bytes": 0,
        "process_log": {
            "path": str(process_log),
            "bytes": process_log.stat().st_size,
            "sha256": throughput.file_sha256(process_log),
        },
        "metrics": metrics,
        "metrics_error": None if metrics is not None else "fixture failed before metrics",
        "gpu_preflight": {
            "index": 0,
            "uuid": "GPU-fixture",
            "name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "total_mib": 32768,
            "used_mib": 0,
            "free_mib": 32768,
            "pstate": "P8",
            "temperature_c": 30,
            "graphics_clock_mhz": 210,
            "memory_clock_mhz": 405,
            "compute_processes": [],
            "external_compute_memory_mib": 0,
        },
        "gpu_monitor": {
            "sample_count": 1,
            "minimum_free_mib": 18 * 1024,
            "peak_used_mib": 14 * 1024,
            "peak_unexpected_memory_mib": 0,
            "first_unexpected_processes": None,
            "observed_pstates": ["P2"],
            "maximum_temperature_c": 55,
            "graphics_clock_range_mhz": [1800, 2100],
            "memory_clock_range_mhz": [9000, 10501],
            "failure": None,
        },
        "gpu_postflight": {
            "index": 0,
            "uuid": "GPU-fixture",
            "name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "total_mib": 32768,
            "compute_processes": [],
        },
        "runtime_versions": plan["benchmark_contract"]["runtime_versions"],
    }
    result["launch_result_sha256"] = canonical_sha256(result)
    return result


def test_optimizer_schedule_distinguishes_attempts_and_synchronized_updates() -> None:
    assert expected_optimizer_schedule(
        num_ppo_epochs=5,
        num_mini_batches=4,
        num_micro_batches=2,
        gradient_accumulation_steps=2,
    ) == {
        "optimizer_expected_attempts": 40,
        "synchronized_expected_updates": 20,
    }
    with pytest.raises(ThroughputProtocolError, match="divide exactly"):
        expected_optimizer_schedule(
            num_ppo_epochs=1,
            num_mini_batches=1,
            num_micro_batches=3,
            gradient_accumulation_steps=2,
        )


def test_partition_subset_is_exact_paired_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol, protocol_path = _fixture_protocol(tmp_path)
    dataset = protocol["dataset"]
    verified = verify_partition_subset(
        dataset["subset_root"],
        split_manifest_path=dataset["split_manifest"],
        partition="D_curriculum",
    )
    assert verified["motion_count"] == 2
    assert verified["resident_motion_count"] == 2
    assert verified["resident_unique_motion_count"] == 2
    assert verified["resident_motion_keys"] == ["motion_0", "motion_1"]
    assert len(list(Path(verified["robot_dir"]).glob("*.pkl"))) == 2

    with pytest.raises(FileExistsError, match="already exists"):
        materialize_partition_subset(
            dataset["split_manifest"],
            partition="D_curriculum",
            destination=dataset["subset_root"],
        )

    target = next(Path(verified["robot_dir"]).glob("*.pkl"))
    target.unlink()
    target.symlink_to(protocol_path)
    with pytest.raises(ThroughputProtocolError, match="target drifted"):
        verify_partition_subset(
            dataset["subset_root"],
            split_manifest_path=dataset["split_manifest"],
            partition="D_curriculum",
        )


def test_subset_rejects_split_self_hash_and_source_group_leakage(tmp_path: Path) -> None:
    protocol, _ = _fixture_protocol(tmp_path)
    original = load_json_object(protocol["dataset"]["split_manifest"])

    broken_digest = deepcopy(original)
    broken_digest["tampered_without_updating_split_sha256"] = True
    broken_path = tmp_path / "broken-digest.json"
    _write_json(broken_path, broken_digest)
    with pytest.raises(ThroughputProtocolError, match="split_sha256"):
        throughput.build_partition_subset_manifest(broken_path, partition="D_curriculum")

    leaked = deepcopy(original)
    leaked["motions"][2]["source_group_id"] = leaked["motions"][0]["source_group_id"]
    leaked["selection_sha256"] = split_selection_sha256(leaked)
    leaked["split_sha256"] = schema_canonical_sha256(leaked, digest_field="split_sha256")
    leaked_path = tmp_path / "leaked.json"
    _write_json(leaked_path, leaked)
    with pytest.raises(ThroughputProtocolError, match="leaks across"):
        throughput.build_partition_subset_manifest(leaked_path, partition="D_curriculum")


def test_plan_binds_same_profile_data_checkpoint_and_schedule(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    validate_plan(plan)

    assert plan["launch_ready"] is True
    assert [cell["num_envs"] for cell in plan["cells"]] == list(EXPECTED_ENV_COUNTS)
    assert len({cell["command_sha256"] for cell in plan["cells"]}) == 4
    assert len({cell["cell_contract_sha256"] for cell in plan["cells"]}) == 4
    common = plan["benchmark_contract"]
    assert common["dataset"]["motion_count"] == 2
    assert common["dataset"]["resident_motion_count"] == 2
    assert common["dataset"]["resident_unique_motion_count"] == 2
    assert common["dataset"]["resident_motion_keys"] == ["motion_0", "motion_1"]
    assert common["workload"] == protocol["workload"]
    assert common["profile"]["actor"] == 1_227_514
    assert common["checkpoint"]["checkpoint_sha256"]
    assert common["checkpoint"]["initialization"]["environment_constructed"] is False
    assert all(
        cell["cell_training_contract"]["optimizer_schedule"]
        == {
            "optimizer_expected_attempts": 20,
            "synchronized_expected_updates": 20,
        }
        for cell in plan["cells"]
    )
    assert all(
        str(cell["output_dir"]).startswith(str(tmp_path / "storage")) for cell in plan["cells"]
    )
    assert all("+callbacks=throughput_benchmark" in cell["argv"] for cell in plan["cells"])
    expected_argv = {
        "++manager_env.config.terrain_type=plane",
        "++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load=2",
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
        f"++throughput_benchmark.dataset_subset_sha256={common['dataset']['subset_sha256']}",
        "++throughput_benchmark.expected_resident_motion_count=2",
        "++throughput_benchmark.expected_resident_motion_order_sha256="
        f"{common['dataset']['resident_motion_order_sha256']}",
        "++throughput_benchmark.expected_resident_motion_set_sha256="
        f"{common['dataset']['resident_motion_set_sha256']}",
        "++throughput_benchmark.expected_terrain_type=plane",
    }
    assert all(expected_argv.issubset(set(cell["argv"])) for cell in plan["cells"])
    cache_keys = (
        "CUDA_CACHE_PATH",
        "ISAACLAB_USD_CACHE_DIR",
        "TMPDIR",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
    )
    assert all(
        str(cell["launch_environment"][key]).startswith(str(tmp_path / "storage"))
        for cell in plan["cells"]
        for key in cache_keys
    )
    assert all("HOME" not in cell["launch_environment"] for cell in plan["cells"])

    rebuilt = build_throughput_plan(protocol, protocol_path=path)
    assert rebuilt == plan


def test_expected_dataset_identity_is_enforced(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    verified = verify_partition_subset(
        protocol["dataset"]["subset_root"],
        split_manifest_path=protocol["dataset"]["split_manifest"],
        partition="D_curriculum",
    )
    identity_keys = (
        "split_manifest_file_sha256",
        "split_sha256",
        "selection_sha256",
        "subset_manifest_file_sha256",
        "subset_sha256",
        "motion_count",
        "resident_motion_count",
        "resident_unique_motion_count",
        "resident_motion_order_sha256",
        "resident_motion_set_sha256",
    )
    protocol["dataset"]["expected_identity"] = {key: verified[key] for key in identity_keys}
    _write_json(path, protocol)
    assert build_throughput_plan(protocol, protocol_path=path)["launch_ready"] is True

    drifted = deepcopy(protocol)
    drifted["dataset"]["expected_identity"]["subset_sha256"] = "0" * 64
    _write_json(path, drifted)
    with pytest.raises(ThroughputProtocolError, match="dataset.expected_identity"):
        build_throughput_plan(drifted, protocol_path=path)


def test_plan_is_unlaunchable_until_initial_checkpoint_is_bound(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path, with_checkpoint=False)
    plan = build_throughput_plan(protocol, protocol_path=path)
    assert plan["launch_ready"] is False
    assert len(plan["readiness_errors"]) == 3
    with pytest.raises(ThroughputLaunchError, match="not launch-ready"):
        launch_throughput_cell(plan, num_envs=128)


def test_plan_only_cli_never_launches_and_refuses_overwrite(tmp_path: Path) -> None:
    _, protocol_path = _fixture_protocol(tmp_path)
    plan_path = tmp_path / "plan.json"
    assert throughput_main(["--protocol", str(protocol_path), "--plan", str(plan_path)]) == 0
    plan = load_json_object(plan_path)
    assert plan["launch_ready"] is True
    assert all(not Path(cell["output_dir"]).exists() for cell in plan["cells"])
    with pytest.raises(FileExistsError, match="overwrite"):
        throughput_main(["--protocol", str(protocol_path), "--plan", str(plan_path)])


def test_materialization_targets_cannot_escape_storage(tmp_path: Path) -> None:
    protocol, protocol_path = _fixture_protocol(tmp_path, with_checkpoint=False)
    outside = tmp_path / "outside-init"
    protocol["checkpoint"] = {
        "path": str(outside / "last.pt"),
        "config_path": str(outside / "config.yaml"),
    }
    _write_json(protocol_path, protocol)
    with pytest.raises(ThroughputProtocolError, match="strictly beneath storage_root"):
        resolve_materialization_targets(protocol, protocol_path=protocol_path)


def test_materialize_init_is_pre_env_deterministic_and_tamper_evident(
    tmp_path: Path,
) -> None:
    first = tmp_path / "init-a"
    second = tmp_path / "init-b"
    receipt_a = materialize_lite_initialization(
        repo_root=REPO_ROOT,
        destination=first,
        seed=23,
    )
    receipt_b = materialize_lite_initialization(
        repo_root=REPO_ROOT,
        destination=second,
        seed=23,
    )
    assert receipt_a["environment_constructed"] is False
    assert receipt_a["construction_order"] == "seed_then_actor_critic_without_environment"
    assert receipt_a["policy"]["trainable_parameters"] == 1_227_514
    assert receipt_a["value"]["trainable_parameters"] == 1_171_329
    assert receipt_a["policy"]["state_sha256"] == receipt_b["policy"]["state_sha256"]
    assert receipt_a["value"]["state_sha256"] == receipt_b["value"]["state_sha256"]
    assert receipt_a["model_spec_sha256"] == receipt_b["model_spec_sha256"]
    verify_lite_initialization(first, repo_root=REPO_ROOT, expected_seed=23)
    with pytest.raises(FileExistsError, match="already exists"):
        materialize_lite_initialization(
            repo_root=REPO_ROOT,
            destination=first,
            seed=23,
        )

    with (first / "last.pt").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(LiteInitializationError, match="byte count"):
        verify_lite_initialization(first, repo_root=REPO_ROOT, expected_seed=23)


def test_materialize_init_cli_makes_missing_plan_ready(tmp_path: Path) -> None:
    protocol, protocol_path = _fixture_protocol(tmp_path, with_checkpoint=False)
    assert throughput_main(["--protocol", str(protocol_path), "--materialize-init"]) == 0
    assert throughput_main(["--protocol", str(protocol_path), "--verify-init"]) == 0
    plan = build_throughput_plan(protocol, protocol_path=protocol_path)
    assert plan["launch_ready"] is True


def test_callback_records_exact_accumulation_and_resource_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "cell" / "throughput_metrics.jsonl"
    monkeypatch.setattr(callback_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(callback_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(callback_module.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(
        callback_module,
        "_cuda_snapshot",
        lambda: {
            "cuda_allocated_bytes": 1,
            "cuda_reserved_bytes": 2,
            "cuda_peak_allocated_bytes": 3,
            "cuda_peak_reserved_bytes": 4,
            "cuda_total_bytes": 32 * 1024**3,
            "cuda_global_free_bytes": 18 * 1024**3,
            "cuda_global_total_bytes": 32 * 1024**3,
        },
    )
    ticks = iter(float(index) for index in range(1, 30))
    monkeypatch.setattr(callback_module.time, "monotonic", lambda: next(ticks))
    callback = ThroughputBenchmarkCallback(
        output_path=str(output),
        benchmark_contract_sha256="a" * 64,
        cell_contract_sha256="b" * 64,
        cell_id="env_0128",
        num_envs=128,
        num_steps_per_env=24,
        decimation=4,
        warmup_iterations=5,
        timed_iterations=20,
        dataset_subset_sha256="c" * 64,
        expected_resident_motion_count=2,
        expected_resident_motion_order_sha256="d" * 64,
        expected_resident_motion_set_sha256="e" * 64,
        expected_terrain_type="plane",
    )
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    control = object()
    callback.on_train_begin(None, state, control)
    logs = {
        "collection_time": 0.4,
        "learn_time": 0.5,
        "benchmark/optimizer_step_attempts": 40,
        "benchmark/optimizer_nonfinite_skips": 0,
        "benchmark/optimizer_accelerator_skips": 0,
        "benchmark/optimizer_expected_attempts": 40,
        "benchmark/sync_boundaries": 20,
        "benchmark/synchronized_parameter_updates": 20,
        "benchmark/synchronized_update_skips": 0,
        "benchmark/synchronized_expected_updates": 20,
        "benchmark/reset_count": 3,
        "benchmark/termination_count": 2,
        "benchmark/timeout_count": 1,
        "benchmark/resident_motion_count": 2,
        "benchmark/resident_unique_motion_count": 2,
        "benchmark/resident_universe_motion_count": 2,
        "benchmark/resident_all_motions_loaded": True,
        "benchmark/resident_motion_order_sha256": "d" * 64,
        "benchmark/resident_motion_set_sha256": "e" * 64,
        "benchmark/terrain_type": "plane",
    }
    state.global_step = 1
    reordered = dict(logs)
    reordered["benchmark/resident_motion_order_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="resident motion order"):
        callback.on_log(None, state, control, logs=reordered)
    nonboolean = dict(logs)
    nonboolean["benchmark/resident_all_motions_loaded"] = "true"
    with pytest.raises(RuntimeError, match="must be boolean"):
        callback.on_log(None, state, control, logs=nonboolean)
    for iteration in range(1, 26):
        state.global_step = iteration
        callback.on_log(None, state, control, logs=dict(logs))
    callback.on_train_end(None, state, control)

    rows = throughput.load_iteration_rows(output)
    assert len(rows) == 25
    assert [row["phase"] for row in rows].count("timed") == 20
    assert rows[-1]["optimizer_step_attempts"] == 40
    assert rows[-1]["synchronized_parameter_updates"] == 20
    assert rows[-1]["termination_count_definition"] == "done_and_not_timeout_timeout_precedence"
    assert rows[-1]["physics_substeps"] == 128 * 24 * 4
    assert rows[-1]["dataset_subset_sha256"] == "c" * 64
    assert rows[-1]["resident_motion_count"] == 2
    assert rows[-1]["resident_motion_order_sha256"] == "d" * 64
    assert rows[-1]["resident_motion_set_sha256"] == "e" * 64
    assert rows[-1]["resident_all_motions_loaded"] is True
    assert rows[-1]["resident_dataset_identity_attested"] is True
    assert rows[-1]["terrain_type"] == "plane"


def test_callback_rejects_existing_output_and_inexact_optimizer_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "metrics.jsonl"
    output.write_text("existing\n")
    monkeypatch.setattr(callback_module.torch.cuda, "is_available", lambda: True)
    callback = ThroughputBenchmarkCallback(
        output_path=str(output),
        benchmark_contract_sha256="a" * 64,
        cell_contract_sha256="b" * 64,
        cell_id="env_0128",
        num_envs=128,
        num_steps_per_env=24,
        decimation=4,
        warmup_iterations=5,
        timed_iterations=20,
        dataset_subset_sha256="c" * 64,
        expected_resident_motion_count=2,
        expected_resident_motion_order_sha256="d" * 64,
        expected_resident_motion_set_sha256="e" * 64,
        expected_terrain_type="plane",
    )
    state = SimpleNamespace(is_world_process_zero=True, global_step=0)
    with pytest.raises(FileExistsError, match="resume/overwrite"):
        callback.on_train_begin(None, state, object())


def test_cell_summary_and_selection_use_end_to_end_rate_and_memory_gate(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    # Iterations/hour falls with N_env, while transitions/s rises.  The frozen
    # selector must choose throughput, not the mechanically small-N-biased
    # iteration rate.  N=1024 is then excluded by its memory safety margin.
    seconds_by_env = {128: 1.0, 256: 1.5, 512: 2.0, 1024: 2.5}
    for cell in plan["cells"]:
        output = Path(cell["output_dir"])
        output.mkdir(parents=True)
        rows = [
            _metric_row(plan, cell, iteration, seconds=seconds_by_env[cell["num_envs"]])
            for iteration in range(1, 26)
        ]
        if cell["num_envs"] == 1024:
            for row in rows:
                row["cuda_peak_reserved_bytes"] = 30 * 1024**3
                # PyTorch reserved memory alone would appear safe if Isaac's
                # non-Torch allocations were ignored; global free is decisive.
                row["cuda_global_free_bytes"] = 2 * 1024**3
        Path(cell["metrics_jsonl"]).write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        _write_json(Path(cell["launch_result"]), _launch_result(plan, cell))

    report = aggregate_throughput_plan(plan)
    assert report["complete"] is True
    assert report["selected_num_envs"] == 512
    failed = next(row for row in report["cells"] if row["num_envs"] == 1024)
    assert "cuda_memory_safety_margin_failed" in failed["ineligibility_reasons"]
    assert failed["minimum_cuda_global_free_bytes"] == 2 * 1024**3
    selected = next(row for row in report["cells"] if row["num_envs"] == 512)
    smallest = next(row for row in report["cells"] if row["num_envs"] == 128)
    assert (
        selected["sustained_end_to_end_iterations_per_hour"]
        < smallest["sustained_end_to_end_iterations_per_hour"]
    )
    assert (
        selected["sustained_end_to_end_control_transitions_per_second"]
        > smallest["sustained_end_to_end_control_transitions_per_second"]
    )
    assert selected["optimizer_step_attempts"] == 20 * 20
    assert selected["synchronized_parameter_updates"] == 20 * 20


def test_summary_rejects_skipped_update_and_disjoint_reset_drift(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    cell = plan["cells"][0]
    rows = [_metric_row(plan, cell, iteration, seconds=1.0) for iteration in range(1, 26)]
    rows[-1]["optimizer_nonfinite_skips"] = 1
    rows[-1]["optimizer_step_attempts"] -= 1
    rows[-1]["termination_count"] = 2
    rows[-1]["decimation"] = 3
    rows[-1]["resident_motion_count"] = 1
    rows[-1]["resident_all_motions_loaded"] = False
    rows[-1]["resident_motion_order_sha256"] = "f" * 64
    rows[-1]["dataset_subset_sha256"] = "0" * 64
    summary = summarize_cell(
        plan,
        cell,
        launch_result=_launch_result(plan, cell, rows=rows),
        rows=rows,
    )
    assert summary["eligible"] is False
    assert "optimizer_nonfinite_skip" in summary["ineligibility_reasons"]
    assert "reset_counter_mismatch" in summary["ineligibility_reasons"]
    assert "metric_contract_mismatch:decimation" in summary["ineligibility_reasons"]
    assert "metric_contract_mismatch:resident_motion_count" in summary["ineligibility_reasons"]
    assert (
        "metric_contract_mismatch:resident_all_motions_loaded" in summary["ineligibility_reasons"]
    )
    assert (
        "metric_contract_mismatch:resident_motion_order_sha256" in summary["ineligibility_reasons"]
    )
    assert "metric_contract_mismatch:dataset_subset_sha256" in summary["ineligibility_reasons"]


def test_oom_before_metrics_is_terminal_and_skips_larger_cells(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    first = plan["cells"][0]
    result = _launch_result(plan, first, failure_class="oom")
    _write_json(Path(first["launch_result"]), result)

    report = aggregate_throughput_plan(plan)
    assert report["complete"] is True
    assert report["stopped_after_oom_cell_id"] == first["cell_id"]
    assert report["selected_num_envs"] is None
    assert report["cells"][0]["failure_class"] == "oom"
    assert all(
        item["ineligibility_reasons"] == [f"not_run_after_prior_oom:{first['cell_id']}"]
        for item in report["cells"][1:]
    )


def test_gpu_preflight_parses_external_process_memory() -> None:
    outputs = iter(
        (
            "0, GPU-abcd, NVIDIA RTX 5090, 590.1, 32768, 9000, 23768, " "P2, 55, 2100, 10501\n",
            "GPU-abcd, 123, 7000\nGPU-other, 999, 100\nGPU-abcd, 456, 500\n",
        )
    )

    def fake_run(argv, **kwargs):  # noqa: ANN001, ANN003
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        return subprocess.CompletedProcess(argv, 0, stdout=next(outputs), stderr="")

    result = query_gpu_preflight(0, run=fake_run)
    assert result["total_mib"] == 32768
    assert result["name"] == "NVIDIA RTX 5090"
    assert result["driver_version"] == "590.1"
    assert result["pstate"] == "P2"
    assert result["external_compute_memory_mib"] == 7500
    assert result["compute_processes"] == [
        {"pid": 123, "used_memory_mib": 7000},
        {"pid": 456, "used_memory_mib": 500},
    ]


def test_launch_refuses_existing_output_and_out_of_order_before_gpu_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    first = plan["cells"][0]
    Path(first["output_dir"]).mkdir(parents=True)
    monkeypatch.setattr(
        throughput,
        "query_gpu_preflight",
        lambda index: pytest.fail("GPU query must not run after local preflight fails"),
    )
    with pytest.raises(ThroughputLaunchError, match="resume/overwrite"):
        launch_throughput_cell(plan, num_envs=128)

    Path(first["output_dir"]).rmdir()
    with pytest.raises(ThroughputLaunchError, match="strict ascending"):
        launch_throughput_cell(plan, num_envs=256)


def test_launch_continuously_rejects_foreign_gpu_process_cpu_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)

    def gpu_snapshot(apps: list[dict[str, int]]) -> dict:
        return {
            "index": 0,
            "uuid": "GPU-fixture",
            "name": "Fixture GPU",
            "driver_version": "fixture-driver",
            "total_mib": 32768,
            "used_mib": sum(app["used_memory_mib"] for app in apps),
            "free_mib": 32768 - sum(app["used_memory_mib"] for app in apps),
            "pstate": "P2",
            "temperature_c": 50,
            "graphics_clock_mhz": 2000,
            "memory_clock_mhz": 10000,
            "compute_processes": apps,
            "external_compute_memory_mib": sum(app["used_memory_mib"] for app in apps),
        }

    snapshots = iter(
        [
            gpu_snapshot([]),
            gpu_snapshot([{"pid": 999_999, "used_memory_mib": 777}]),
            gpu_snapshot([]),
        ]
    )
    monkeypatch.setattr(throughput, "query_gpu_preflight", lambda index: next(snapshots))
    monkeypatch.setattr(
        throughput,
        "_process_tree_snapshot",
        lambda pid: {"pids": [pid], "process_count": 1, "rss_bytes": 1234},
    )
    monkeypatch.setattr(throughput, "_pid_in_process_group", lambda pid, pgid: False)
    monkeypatch.setattr(throughput.os, "killpg", lambda pid, signal: None)
    monkeypatch.setattr(
        throughput,
        "_runtime_versions",
        lambda executable: plan["benchmark_contract"]["runtime_versions"],
    )

    class FakeProcess:
        pid = 424_242

        def __init__(self) -> None:
            self.returncode = None

        def poll(self):  # noqa: ANN201
            return self.returncode

        def wait(self, timeout=None):  # noqa: ANN001, ANN201, ARG002
            self.returncode = -15
            return self.returncode

    monkeypatch.setattr(throughput.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    result = launch_throughput_cell(plan, num_envs=128)
    assert result["failure_class"] == "gpu_contamination"
    assert result["gpu_monitor"]["sample_count"] == 1
    assert result["gpu_monitor"]["peak_unexpected_memory_mib"] == 777
    assert result["gpu_monitor"]["first_unexpected_processes"] == [
        {"pid": 999_999, "used_memory_mib": 777}
    ]
    throughput.validate_launch_result(plan, plan["cells"][0], result)


def test_plan_and_result_hashes_fail_closed(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    tampered = deepcopy(plan)
    tampered["cells"][0]["num_envs"] = 129
    with pytest.raises(ThroughputProtocolError, match="self-hash"):
        validate_plan(tampered)

    artifact = tmp_path / "artifact.json"
    write_new_json(artifact, {"a": 1})
    with pytest.raises(FileExistsError, match="overwrite"):
        write_new_json(artifact, {"a": 2})
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})


def test_rehashed_plan_tampering_and_dataset_drift_fail_before_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    monkeypatch.setattr(
        throughput,
        "query_gpu_preflight",
        lambda index: pytest.fail("GPU query must follow every bound-file check"),
    )

    tampered = deepcopy(plan)
    tampered["cells"][0]["launch_environment"]["LOGURU_LEVEL"] = "DEBUG"
    command = {
        "environment": tampered["cells"][0]["launch_environment"],
        "argv": tampered["cells"][0]["argv"],
    }
    tampered["cells"][0]["command_sha256"] = canonical_sha256(command)
    tampered.pop("plan_sha256")
    tampered["plan_sha256"] = canonical_sha256(tampered)
    with pytest.raises(ThroughputLaunchError, match="fresh protocol rebuild"):
        launch_throughput_cell(tampered, num_envs=128)

    source = Path(load_json_object(path)["dataset"]["split_manifest"])
    split = load_json_object(source)
    robot_path = Path(split["motions"][0]["robot_path"])
    robot_path.write_bytes(b"drifted after planning")
    with pytest.raises(ThroughputLaunchError, match="rebuild|subset"):
        launch_throughput_cell(plan, num_envs=128)


def test_iteration_loader_rejects_nonfinite_json(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(ThroughputProtocolError, match="non-finite JSON constant"):
        throughput.load_iteration_rows(metrics)


def test_aggregate_rejects_metrics_tampered_after_launch_receipt(tmp_path: Path) -> None:
    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    cell = plan["cells"][0]
    rows = [_metric_row(plan, cell, iteration, seconds=1.0) for iteration in range(1, 26)]
    result = _launch_result(plan, cell, rows=rows)
    _write_json(Path(cell["launch_result"]), result)
    with Path(cell["metrics_jsonl"]).open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ThroughputProtocolError, match="metrics byte count"):
        aggregate_throughput_plan(plan)


def test_callback_hydra_group_composes_without_isaac() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(REPO_ROOT / "gear_sonic/config"), version_base="1.1"):
        config = compose(
            config_name="base",
            overrides=[
                "+exp=manager/universal_token/g1_only/lace_lite_s",
                "+callbacks=throughput_benchmark",
                f"++throughput_benchmark.contract_sha256={'a' * 64}",
                f"++throughput_benchmark.cell_contract_sha256={'b' * 64}",
                "++throughput_benchmark.cell_id=env_0128",
                "++throughput_benchmark.warmup_iterations=5",
                "++throughput_benchmark.timed_iterations=20",
                f"++throughput_benchmark.dataset_subset_sha256={'c' * 64}",
                "++throughput_benchmark.expected_resident_motion_count=2",
                f"++throughput_benchmark.expected_resident_motion_order_sha256={'d' * 64}",
                f"++throughput_benchmark.expected_resident_motion_set_sha256={'e' * 64}",
                "++throughput_benchmark.expected_terrain_type=plane",
            ],
        )
    callback = OmegaConf.to_container(config.callbacks.throughput_benchmark, resolve=False)
    assert callback["num_steps_per_env"] == "${algo.config.num_steps_per_env}"
    assert callback["decimation"] == "${manager_env.config.decimation}"


def test_exact_planned_command_hydra_composes_without_isaac(tmp_path: Path) -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    protocol, path = _fixture_protocol(tmp_path)
    plan = build_throughput_plan(protocol, protocol_path=path)
    cell = plan["cells"][0]
    with initialize_config_dir(config_dir=str(REPO_ROOT / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=cell["argv"][2:])
    assert config.resume is False
    assert config.manager_env.config.decimation == 4
    assert config.algo.trl.per_device_train_batch_size == 32
    assert config.algo.config.num_learning_iterations == 25
    assert config.callbacks.throughput_benchmark.cell_id == "env_0128"
    assert config.manager_env.config.terrain_type == "plane"
    assert config.manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load == 2
    assert config.manager_env.commands.motion.motion_lib_cfg.sort_motion_keys is True
    assert config.callbacks.throughput_benchmark.dataset_subset_sha256 == (
        plan["benchmark_contract"]["dataset"]["subset_sha256"]
    )
    assert config.callbacks.throughput_benchmark.expected_resident_motion_count == 2
    assert config.callbacks.throughput_benchmark.expected_resident_motion_order_sha256 == (
        plan["benchmark_contract"]["dataset"]["resident_motion_order_sha256"]
    )
    assert config.callbacks.throughput_benchmark.expected_resident_motion_set_sha256 == (
        plan["benchmark_contract"]["dataset"]["resident_motion_set_sha256"]
    )
    assert config.callbacks.throughput_benchmark.expected_terrain_type == "plane"


def test_headline_protocol_reuses_schedule_and_freezes_full_workload() -> None:
    headline = load_json_object(HEADLINE_PROTOCOL_PATH)
    pilot = load_json_object(REPO_ROOT / "configs/research/lace/throughput_lite_s_scale512_v1.json")

    assert headline["benchmark"] == pilot["benchmark"]
    assert headline["checkpoint"]["path"] == pilot["checkpoint"]["path"]
    assert headline["checkpoint"]["config_path"] == pilot["checkpoint"]["config_path"]
    assert headline["experiment"] == pilot["experiment"]
    assert headline["seed"] == pilot["seed"]
    assert headline["env_counts"] == list(EXPECTED_ENV_COUNTS)
    assert headline["workload"] == {
        "all_motions_loaded_required": True,
        "resident_motion_count": 1996,
        "resident_motion_order": "lexicographic_motion_key",
        "terrain_type": "plane",
    }
    assert headline["selection_rule"] == pilot["selection_rule"]
    assert headline["launch"] == pilot["launch"]
    assert headline["dataset"]["expected_identity"]["motion_count"] == 1996
    assert headline["dataset"]["expected_identity"]["resident_motion_count"] == 1996
    assert headline["dataset"]["expected_identity"]["resident_unique_motion_count"] == 1996


def test_headline_plan_verifies_1996_residents_and_dry_composes_every_cell() -> None:
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    protocol = load_json_object(HEADLINE_PROTOCOL_PATH)
    required_paths = (
        Path(protocol["dataset"]["split_manifest"]),
        Path(protocol["dataset"]["subset_root"]) / "subset_manifest.json",
        Path(protocol["checkpoint"]["path"]),
        Path(protocol["checkpoint"]["config_path"]),
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        pytest.skip("headline /data artifacts are not installed: " + ", ".join(map(str, missing)))

    plan = build_throughput_plan(protocol, protocol_path=HEADLINE_PROTOCOL_PATH)
    validate_plan(plan)
    assert plan["launch_ready"] is True
    dataset = plan["benchmark_contract"]["dataset"]
    assert dataset["motion_count"] == 1996
    assert dataset["resident_motion_count"] == 1996
    assert dataset["resident_unique_motion_count"] == 1996
    assert len(dataset["resident_motion_keys"]) == 1996
    assert dataset["resident_motion_keys"] == sorted(dataset["resident_motion_keys"])
    assert dataset["resident_motion_order_sha256"] == (
        protocol["dataset"]["expected_identity"]["resident_motion_order_sha256"]
    )
    assert dataset["resident_motion_set_sha256"] == (
        protocol["dataset"]["expected_identity"]["resident_motion_set_sha256"]
    )

    for cell in plan["cells"]:
        with initialize_config_dir(
            config_dir=str(REPO_ROOT / "gear_sonic/config"),
            version_base="1.1",
        ):
            config = compose(config_name="base", overrides=cell["argv"][2:])
        assert config.num_envs == cell["num_envs"]
        assert config.resume is False
        assert config.manager_env.config.terrain_type == "plane"
        assert (
            config.manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load == 1996
        )
        assert config.manager_env.commands.motion.motion_lib_cfg.sort_motion_keys is True
        assert config.manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable is False
        assert config.callbacks.throughput_benchmark.expected_resident_motion_count == 1996
        assert config.callbacks.throughput_benchmark.expected_resident_motion_order_sha256 == (
            dataset["resident_motion_order_sha256"]
        )
        assert config.callbacks.throughput_benchmark.expected_resident_motion_set_sha256 == (
            dataset["resident_motion_set_sha256"]
        )


def test_resident_motion_identity_binds_order_and_rejects_duplicates() -> None:
    ordered = throughput.resident_motion_identity(["motion_a", "motion_b"])
    reordered = throughput.resident_motion_identity(["motion_b", "motion_a"])
    duplicated = throughput.resident_motion_identity(["motion_a", "motion_a"])

    assert ordered["resident_motion_order_sha256"] != reordered["resident_motion_order_sha256"]
    assert ordered["resident_motion_set_sha256"] == reordered["resident_motion_set_sha256"]
    assert duplicated["resident_motion_count"] == 2
    assert duplicated["resident_unique_motion_count"] == 1
