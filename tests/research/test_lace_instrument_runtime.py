from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.research.lace.analysis_protocol import build_analysis_protocol
from gear_sonic.research.lace.analysis_protocol_lock import (
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    build_analysis_protocol_lock,
    execution_cell_binding,
    validate_analysis_protocol_lock,
)
from gear_sonic.research.lace.instrument import (
    MEASUREMENT_FAMILY_DIGEST_FIELD,
    RUNTIME_RNG_SEED_SEMANTICS,
    TERMINATION_FRESHNESS_SEMANTICS,
    TERMINATION_MULTI_HOT_SEMANTICS,
    episode_measurement_family_sha256,
    validate_scientific_instrument,
)
from gear_sonic.research.lace.instrument_collection import (
    ROLLOUT_COLLECTION_DIGEST_FIELD,
    collect_scientific_rollouts,
    validate_rollout_collection,
)
from gear_sonic.research.lace.instrument_runtime import (
    INSTRUMENT_BINDING_DIGEST_FIELD,
    ROLLOUT_BINDING_DIGEST_FIELD,
    RUNTIME_HANDSHAKE_DIGEST_FIELD,
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    build_plan_binding,
    build_runtime_handshake,
    build_source_bundle,
    discover_scientific_source_paths,
    file_sha256,
    finalize_scientific_rollout,
    materialize_instrument_from_payloads,
    validate_runtime_handshake,
    write_new_json,
)
from gear_sonic.research.lace.probes import ProbeThresholds, compute_episode_probe
from gear_sonic.research.lace.protocol_preflight import (
    PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
    build_protocol_preflight_receipt,
    build_protocol_preflight_request,
    execute_live_protocol_preflight,
    protocol_preflight_binding_from_config,
    validate_protocol_preflight_receipt,
)
from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    REFERENCE_LENGTH_KIND,
    REFERENCE_LENGTH_SCHEMA_VERSION,
    RESAMPLING_RULE,
    sonic_target_frame_count,
)
from gear_sonic.research.lace.schedule import (
    DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
    build_rollout_schedule,
)
from gear_sonic.research.lace.schedule_batch_loader import (
    rebuild_eval_hydra_overrides_from_launch_plan,
)
from gear_sonic.research.lace.schema import (
    ATLAS_V1_INTERVAL_EVENT_POLICY,
    ROBOT_CONTRACT_READBACK_KIND,
    ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
    RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
    RUNTIME_REALIZATION_KIND,
    RUNTIME_REALIZATION_SCHEMA_VERSION,
    TERMINATION_TRACE_ALGORITHM,
    TERMINATION_TRACE_KIND,
    TERMINATION_TRACE_SCHEMA_VERSION,
    canonical_sha256,
)
from gear_sonic.research.lace.split import build_source_disjoint_split
from scripts.research.collect_lace_scientific_cells import main as collection_cli_main


def _termination_contract() -> dict:
    configs = [
        {
            "term_name": "anchor_pos",
            "callable": "test:anchor_pos",
            "time_out": False,
            "params": {"threshold": 0.5},
        },
        {
            "term_name": "time_out",
            "callable": "test:time_out",
            "time_out": True,
            "params": {},
        },
    ]
    return {
        "kind": TERMINATION_TRACE_KIND,
        "schema_version": TERMINATION_TRACE_SCHEMA_VERSION,
        "algorithm": TERMINATION_TRACE_ALGORITHM,
        "manager_type": "test:TerminationManager",
        "manager_compute_source_sha256": "a" * 64,
        "instrument_compute_source_sha256": "b" * 64,
        "term_names": ["anchor_pos", "time_out"],
        "time_out_flags": [False, True],
        "term_config_sha256": canonical_sha256({"terms": configs}),
        "term_configs": configs,
        "legacy_term_dones_semantics": "last_trigger_wins_stale_rows_preserved",
        "raw_trace_semantics": "ordered_independent_values_from_single_manager_evaluation",
        "freshness_semantics": TERMINATION_FRESHNESS_SEMANTICS,
    }


def _probe_thresholds() -> dict:
    return asdict(ProbeThresholds())


def _score_window_config() -> dict:
    return {
        "rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
        "score_window_seconds": 2.0,
        "timestep_seconds": 0.02,
        "sample_count_rule": ("max(1,floor(score_window_seconds/timestep_seconds+1e-12))"),
    }


def _sensor_semantics() -> dict:
    return {
        "command_name": "motion",
        "robot_name": "robot",
        "joint_action_name": "joint_pos",
        "contact_sensor_name": "contact",
        "foot_body_names": ["left_foot", "right_foot"],
        "contact_force_threshold": 1.0,
        "ground_normal_axis": 2,
        "actual_contact": "latest_net_forces_w_history_norm_threshold",
        "reference_contact": "sonic_feet_channel_ground_height_rule",
        "foot_slip": "link_origin_velocity_tangent_to_configured_plane",
        "torque": "requested_action_target_minus_joint_state_and_applied_joint_effort",
        "joint_limits": "live_soft_joint_position_limits",
    }


def _event_config() -> dict:
    return {
        "interval_events_instrumented": False,
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
    }


def _preflight_environment_fingerprint(plan: dict, *, num_envs: int) -> dict:
    return {
        "schema_version": 1,
        "python": {
            "version": "3.11",
            "implementation": "CPython",
            "executable": plan["command"][0],
        },
        "platform": {"system": "Linux"},
        "modules": [{"module": "fixture"}],
        "cuda": {"available": False},
        "scientific_cache_environment": plan["launch_environment"]["scientific_cache_environment"],
        "scientific_cache_environment_semantics": plan["launch_environment"][
            "scientific_cache_environment_semantics"
        ],
        "process_executable": plan["command"][0],
        "process_argv": plan["command"][1:],
        "pythonpath": plan["launch_environment"]["PYTHONPATH"],
        "runtime": {
            "environment_type": "test:ManagerEnv",
            "termination_manager_type": "test:TerminationManager",
            "event_manager_type": "test:EventManager",
            "device": "cpu",
            "num_envs": num_envs,
            "step_dt_seconds": 0.02,
        },
    }


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "gear_sonic" / "research" / "lace").mkdir(parents=True)
    (root / "scripts" / "research").mkdir(parents=True)
    (root / "gear_sonic" / "research" / "lace" / "probe.py").write_text(
        "probe = 1\n", encoding="utf-8"
    )
    (root / "gear_sonic" / "config.yaml").write_text("seed: 1\n", encoding="utf-8")
    (root / "gear_sonic" / "eval_agent_trl.py").write_text(
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    (root / "gear_sonic" / "robot.usd").write_bytes(b"usd robot fixture\n")
    (root / "gear_sonic" / "collision.STL").write_bytes(b"collision fixture\n")
    bytecode = root / "gear_sonic" / "__pycache__" / "probe.cpython-311.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"generated and deliberately unbound\n")
    (root / "scripts" / "research" / "run_lace_atlas_probe_cell.py").write_text(
        "runner = 1\n", encoding="utf-8"
    )
    (root / "scripts" / "research" / "build_lace_instrument.py").write_text(
        "builder = 1\n", encoding="utf-8"
    )
    return root


def _overwrite_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _overwrite_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _planned_runtime(tmp_path: Path) -> dict:
    root = _repo(tmp_path)

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "model.pt"
    checkpoint_path.write_bytes(b"exact checkpoint bytes\n")
    checkpoint_config_path = checkpoint_dir / "config.yaml"
    checkpoint_config_path.write_text("model: fixture\n", encoding="utf-8")
    checkpoint_sha256 = file_sha256(checkpoint_path)

    robot_root = tmp_path / "robot-motions"
    smpl_root = tmp_path / "smpl-motions"
    robot_root.mkdir()
    smpl_root.mkdir()
    split_records = [
        {
            "motion_key": f"motion_{index:02d}__A{index:03d}",
            "release_filter_key": f"motion_{index:02d}__A{index:03d}.pkl",
            "robot_path": str(robot_root / f"motion_{index:02d}__A{index:03d}.pkl"),
            "source_group_id": f"actor_{index:03d}",
            "duration_source_frames": 7 + index,
            "stratum": f"kind_{index % 2}",
        }
        for index in range(110)
    ]
    split = build_source_disjoint_split(split_records, seed=9)
    motion_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    assert motion_keys
    split_by_key = {record["motion_key"]: record for record in split["motions"]}
    robot_files = []
    smpl_files = []
    inventory_motions = []
    for index, motion_key in enumerate(motion_keys):
        robot_path = robot_root / f"{motion_key}.pkl"
        smpl_path = smpl_root / f"{motion_key}.pkl"
        robot_path.write_bytes(f"robot {motion_key}\n".encode())
        smpl_path.write_bytes(f"smpl {motion_key}\n".encode())
        source_frames = 7 + index
        target_frames = sonic_target_frame_count(source_frames, 30, 50)
        robot_files.append(
            {
                "motion_key": motion_key,
                "path": str(robot_path),
                "sha256": file_sha256(robot_path),
            }
        )
        smpl_files.append(
            {
                "motion_key": motion_key,
                "path": str(smpl_path),
                "sha256": file_sha256(smpl_path),
            }
        )
        inventory_motions.append(
            {
                "motion_key": motion_key,
                "split_robot_path": split_by_key[motion_key]["robot_path"],
                "source_path": str(robot_path),
                "source_file_sha256": file_sha256(robot_path),
                "source_num_frames": source_frames,
                "frame_axis": "root_trans_offset.shape[0]",
                "source_fps": {"numerator": 30, "denominator": 1},
                "target_num_frames": target_frames,
            }
        )

    paired_dataset_sha256 = canonical_sha256({"robot": robot_files, "smpl": smpl_files})
    materialized_manifest = {
        "kind": "fixture_materialized_paired_dataset",
        "schema_version": 1,
        "output": {
            "motion_count": len(motion_keys),
            "motion_keys": motion_keys,
            "robot_dir": robot_root.name,
            "smpl_dir": smpl_root.name,
            "paired_dataset_sha256": paired_dataset_sha256,
            "variants": [
                {
                    "motion_key": motion_key,
                    "robot": {
                        "path": f"{robot_root.name}/{motion_key}.pkl",
                        "sha256": robot_files[index]["sha256"],
                        "frames": inventory_motions[index]["source_num_frames"],
                        "fps": 30.0,
                    },
                    "smpl": {
                        "path": f"{smpl_root.name}/{motion_key}.pkl",
                        "sha256": smpl_files[index]["sha256"],
                        "frames": inventory_motions[index]["target_num_frames"],
                        "fps": 50.0,
                        "original_frames": inventory_motions[index]["source_num_frames"],
                        "original_fps": 30.0,
                    },
                }
                for index, motion_key in enumerate(motion_keys)
            ],
        },
    }
    materialized_path = write_new_json(
        tmp_path / "materialized-dataset.json",
        materialized_manifest,
    )
    split = build_source_disjoint_split(
        split_records,
        seed=9,
        dataset={
            "materialized_manifest": str(materialized_path),
            "materialized_manifest_sha256": file_sha256(materialized_path),
            "paired_dataset_sha256": paired_dataset_sha256,
        },
    )
    rebound_motion_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    assert rebound_motion_keys == motion_keys

    inventory = {
        "kind": REFERENCE_LENGTH_KIND,
        "schema_version": REFERENCE_LENGTH_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "pilot_status": None,
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_atlas",
        "selected_motion_keys": motion_keys,
        "selection_complete_for_d_atlas": True,
        "motion_count": len(motion_keys),
        "target_fps": 50,
        "runtime_contract": {
            "target_fps": 50,
            "sim_fps": 50,
            "motion_fps_scale": {"numerator": 1, "denominator": 1},
            "max_len": -1,
            "reference_num_steps_equals_target_num_frames": True,
            "float32_runtime_equivalence_scope": (
                "locked_30_to_50_hz_release_cohort_with_materialized_pair_cross_check"
            ),
        },
        "resampling_rule": dict(RESAMPLING_RULE),
        "dataset_provenance": {
            "materialized_manifest": str(materialized_path),
            "materialized_manifest_sha256": file_sha256(materialized_path),
            "paired_dataset_sha256": paired_dataset_sha256,
        },
        "source_file_set_sha256": canonical_sha256(
            {
                "files": [
                    {
                        "motion_key": row["motion_key"],
                        "source_file_sha256": row["source_file_sha256"],
                    }
                    for row in inventory_motions
                ]
            }
        ),
        "motions": inventory_motions,
    }
    inventory[REFERENCE_LENGTH_DIGEST_FIELD] = canonical_sha256(
        inventory,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    schedule = build_rollout_schedule(
        split,
        reference_length_inventory=inventory,
        probe_policies=[{"id": "medium", "checkpoint_sha256": checkpoint_sha256}],
        domain_randomization_seeds=[17],
        phase_targets=[{"phase_id": "start", "target_fraction": 0.0}],
        repeats=1,
    )
    assignments = list(schedule["rollouts"])
    rollout_ids = [row["rollout_id"] for row in assignments]
    schedule_spec = {
        "schema_version": 2,
        "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "probe_policies": [{"id": "medium", "checkpoint_sha256": checkpoint_sha256}],
        "domain_randomization_seeds": [17],
        "phase_targets": [{"phase_id": "start", "target_fraction": 0.0}],
        "repeats": 1,
    }

    inputs = tmp_path / "locked-inputs"
    inputs.mkdir()
    schedule_path = write_new_json(inputs / "schedule.json", schedule)
    schedule_spec_path = write_new_json(inputs / "schedule-spec.json", schedule_spec)
    split_path = write_new_json(inputs / "split.json", split)
    inventory_path = write_new_json(inputs / "inventory.json", inventory)
    schedule_lock_path = write_new_json(
        inputs / "schedule-lock.json",
        {
            "schema_version": 1,
            "kind": "lace_probe_rollout_schedule_lock",
            "scientific_use": True,
            "artifact": {
                "path": str(schedule_path),
                "file_sha256": file_sha256(schedule_path),
                "schedule_sha256": schedule["schedule_sha256"],
                "schedule_schema_version": 2,
            },
            "inputs": {
                "split_manifest": str(split_path),
                "split_manifest_sha256": file_sha256(split_path),
                "schedule_spec": str(schedule_spec_path),
                "schedule_spec_sha256": file_sha256(schedule_spec_path),
                "reference_length_inventory": str(inventory_path),
                "reference_length_inventory_file_sha256": file_sha256(inventory_path),
                "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
            },
        },
    )

    source_paths = discover_scientific_source_paths(root)
    assert source_paths == (
        "gear_sonic/collision.STL",
        "gear_sonic/config.yaml",
        "gear_sonic/eval_agent_trl.py",
        "gear_sonic/research/lace/probe.py",
        "gear_sonic/robot.usd",
        "scripts/research/build_lace_instrument.py",
        "scripts/research/run_lace_atlas_probe_cell.py",
    )
    source_bundle = build_source_bundle(root, source_paths)
    checkpoint_bundle = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(checkpoint_config_path),
        "config_sha256": file_sha256(checkpoint_config_path),
    }
    dataset_binding = {
        "robot": {
            "motion_root": str(robot_root),
            "files": robot_files,
            "selected_file_set_sha256": canonical_sha256({"files": robot_files}),
        },
        "smpl": {
            "mode": "real",
            "motion_root": str(smpl_root),
            "files": smpl_files,
            "selected_file_set_sha256": canonical_sha256({"files": smpl_files}),
        },
    }
    cell = {
        "probe_policy_id": "medium",
        "checkpoint_sha256": checkpoint_sha256,
        "domain_randomization_seed": 17,
        "runtime_rng_seed": assignments[0]["runtime_rng_seed"],
        "phase_id": "start",
        "target_fraction": 0.0,
        "repeat_index": 0,
    }
    analysis_protocol = build_analysis_protocol(
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        probe_thresholds=_probe_thresholds(),
        score_window_config=_score_window_config(),
        sensor_semantics=_sensor_semantics(),
        termination_predicates=_termination_contract(),
        domain_randomization_config=_event_config(),
        git_commit="9" * 40,
        source_bundle_sha256=source_bundle["source_bundle_sha256"],
    )
    analysis_protocol_path = inputs / "analysis-protocol.json"
    preflight_request_path = inputs / "analysis-protocol.preflight-request.json"
    preflight_receipt_path = inputs / "analysis-protocol.preflight-receipt.json"
    preflight_plan_path = inputs / "analysis-protocol.preflight-plan.json"
    preflight_request = build_protocol_preflight_request(
        schedule_lock_path=schedule_lock_path,
        schedule_path=schedule_path,
        schedule_manifest=schedule,
        split_path=split_path,
        split_manifest=split,
        reference_length_inventory_path=inventory_path,
        reference_length_inventory=inventory,
        cell=cell,
        rollout_ids=rollout_ids,
        checkpoint_bundle=checkpoint_bundle,
        dataset_binding_sha256=canonical_sha256(dataset_binding),
        source_bundle=source_bundle,
        git_commit="9" * 40,
        launch_plan_path=preflight_plan_path,
        recorder_output_path=inputs / "preflight-raw.jsonl",
        analysis_protocol_output_path=analysis_protocol_path,
        preflight_receipt_output_path=preflight_receipt_path,
        repo_root=root,
    )
    write_new_json(preflight_request_path, preflight_request)
    preflight_plan = {
        "schedule_sha256": schedule["schedule_sha256"],
        "cell": cell,
        "rollout_ids": rollout_ids,
        "checkpoint_bundle": checkpoint_bundle,
        "dataset_binding": dataset_binding,
        "protocol_preflight": {
            "request_path": str(preflight_request_path),
            "request_file_sha256": file_sha256(preflight_request_path),
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: preflight_request[
                PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
            ],
        },
    }
    preflight_cache = {
        key: str(tmp_path / "preflight-cache" / key.lower())
        for key in SCIENTIFIC_CACHE_ENVIRONMENT_KEYS
    }
    preflight_plan["command"] = [
        str(Path(sys.executable).resolve()),
        str(root / "gear_sonic" / "eval_agent_trl.py"),
        "++lace_protocol_preflight=true",
    ]
    preflight_plan["launch_environment"] = {
        "PYTHONPATH": str(root),
        "scientific_cache_environment": preflight_cache,
        "scientific_cache_environment_semantics": SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    }
    preflight_plan["launch_plan_sha256"] = canonical_sha256(
        preflight_plan,
        digest_field="launch_plan_sha256",
    )
    write_new_json(preflight_plan_path, preflight_plan)
    write_new_json(analysis_protocol_path, analysis_protocol)
    preflight_failure_config = {
        "enabled": True,
        "allow_append_existing": False,
        "output_path": preflight_request["recorder_output_path"],
    }
    preflight_resolved_config = {
        "checkpoint": checkpoint_bundle["checkpoint_path"],
        "lace_scientific_instrument_required": False,
        "lace_protocol_preflight_request_path": str(preflight_request_path),
        "lace_protocol_preflight_request_file_sha256": file_sha256(preflight_request_path),
        PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: preflight_request[
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
        ],
        "lace_launch_plan_path": str(preflight_plan_path),
        "manager_env": {
            "recorders": {"failure_atlas": preflight_failure_config},
            "commands": {
                "motion": {
                    "atlas_probe_schedule_sha256": schedule["schedule_sha256"],
                    "motion_lib_cfg": {
                        "motion_file": dataset_binding["robot"]["motion_root"],
                        "smpl_motion_file": dataset_binding["smpl"]["motion_root"],
                    },
                }
            },
        },
    }
    preflight_live_inputs = {
        "probe_thresholds": _probe_thresholds(),
        "probe_thresholds_sha256": canonical_sha256(_probe_thresholds()),
        "termination_predicates": _termination_contract(),
        "termination_predicates_sha256": canonical_sha256(_termination_contract()),
        "sensor_semantics": _sensor_semantics(),
        "score_window_config": _score_window_config(),
        "domain_randomization_config": _event_config(),
        "resolved_recorder_config": preflight_failure_config,
        "recorder_term_type": "test:FailureAtlasRecorder",
        "recorder_record_count": 0,
        "num_envs": len(rollout_ids),
        "step_dt_seconds": 0.02,
        "environment_fingerprint": _preflight_environment_fingerprint(
            preflight_plan,
            num_envs=len(rollout_ids),
        ),
    }
    build_protocol_preflight_receipt(
        request_path=preflight_request_path,
        launch_plan_path=preflight_plan_path,
        live_inputs=preflight_live_inputs,
        resolved_hydra_config=preflight_resolved_config,
        repo_root=root,
    )
    execution_root = tmp_path / "executions"
    execution_root.mkdir()
    runtime_storage_root = tmp_path / "runtime-cache"
    runtime_storage_root.mkdir()
    analysis_protocol_lock = build_analysis_protocol_lock(
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_preflight_receipt_path=preflight_receipt_path,
        schedule_lock_path=schedule_lock_path,
        checkpoint_config_paths={"medium": checkpoint_config_path},
        execution_root=execution_root,
        runtime_storage_root=runtime_storage_root,
        repo_root=root,
    )
    analysis_protocol_lock_path = write_new_json(
        inputs / "analysis-protocol-lock.json",
        analysis_protocol_lock,
    )
    registered_cell = execution_cell_binding(
        analysis_protocol_lock,
        cell=cell,
    )
    outputs = Path(registered_cell["cell_directory"])
    outputs.mkdir()
    handshake = build_runtime_handshake(
        schedule_sha256=schedule["schedule_sha256"],
        checkpoint_sha256=checkpoint_sha256,
        probe_policy_id="medium",
        rollout_ids=rollout_ids,
        rollout_output_path=registered_cell["rollout_output_path"],
        instrumented_rollout_output_path=registered_cell["instrumented_rollout_output_path"],
        launch_plan_path=registered_cell["plan_output_path"],
        instrument_output_path=registered_cell["instrument_output_path"],
        instrument_binding_output_path=registered_cell["instrument_binding_output_path"],
        rollout_binding_output_path=registered_cell["rollout_binding_output_path"],
        analysis_protocol_path=analysis_protocol_path,
        git_commit="9" * 40,
        repo_root=root,
        source_paths=source_paths,
    )
    handshake_path = write_new_json(
        registered_cell["runtime_handshake_output_path"],
        handshake,
    )
    plan_binding = build_plan_binding(handshake_path, handshake)
    cache_environment: dict[str, str] = {}
    plan = {
        "kind": "lace_atlas_probe_launch_plan",
        "schema_version": 1,
        "scientific_use": True,
        "schedule_sha256": handshake["schedule_sha256"],
        "schedule_lock_path": str(schedule_lock_path),
        "schedule_path": str(schedule_path),
        "schedule_file_sha256": file_sha256(schedule_path),
        "schedule_spec_path": str(schedule_spec_path),
        "cell": cell,
        "num_envs": len(rollout_ids),
        "motion_keys": motion_keys,
        "reference_num_steps": [row["reference_num_steps"] for row in assignments],
        "rollout_ids": rollout_ids,
        "atlas_probe_assignments": assignments,
        "checkpoint_path": str(checkpoint_path),
        "rollout_output_path": handshake["rollout_output_path"],
        "checkpoint_bundle": checkpoint_bundle,
        "dataset_binding": dataset_binding,
        "instrument_runtime": plan_binding,
        "instrumentation_invariants": {
            "headless": True,
            "terrain_type": "plane",
            "render_results": False,
            "policy_enable_corruption": False,
            "tokenizer_enable_corruption": False,
            "use_encoder": "g1",
            "eval_callbacks": [],
            "run_once": True,
            "max_render_steps": max(
                row["reference_num_steps"] - row["start_step"] for row in assignments
            )
            + 2,
            "native_adaptive_sampling": False,
        },
        "eval_entrypoint": str((root / "gear_sonic/eval_agent_trl.py").resolve()),
        "hydra_overrides": [],
        "passthrough_hydra_args": [],
        "command": [],
        "launch_environment": {
            "PYTHONPATH": str(root),
            "semantics": "repository_root_first_preserve_inherited_unique_entries_v1",
            "scientific_cache_environment": cache_environment,
            "scientific_cache_environment_semantics": (SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS),
            "launch_cache_token": None,
        },
        "analysis_protocol_path": str(analysis_protocol_path),
        "analysis_protocol_file_sha256": file_sha256(analysis_protocol_path),
        "analysis_protocol_sha256": analysis_protocol["analysis_protocol_sha256"],
        "analysis_protocol_lock_path": str(analysis_protocol_lock_path),
        "analysis_protocol_lock_file_sha256": file_sha256(analysis_protocol_lock_path),
        "analysis_protocol_lock_sha256": analysis_protocol_lock[
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
    }
    cache_token = canonical_sha256(
        {
            "schedule_sha256": plan["schedule_sha256"],
            "cell": plan["cell"],
            "plan_output": registered_cell["plan_output_path"],
            "rollout_output": plan["rollout_output_path"],
        }
    )
    suffixes = (
        "tmp",
        "xdg-cache",
        "isaaclab-usd-cache",
        "cuda-cache",
        "torch-home",
        "omni-user-cache",
    )
    for key, suffix in zip(SCIENTIFIC_CACHE_ENVIRONMENT_KEYS, suffixes, strict=True):
        path = runtime_storage_root / cache_token / suffix
        path.mkdir(parents=True)
        cache_environment[key] = str(path)
    plan["launch_environment"]["launch_cache_token"] = cache_token
    runtime_overrides = [
        "++lace_scientific_instrument_required=true",
        "++lace_instrument_handshake_path=" + json.dumps(str(handshake_path)),
        "++lace_instrument_handshake_file_sha256=" + file_sha256(handshake_path),
        "++lace_launch_plan_path=" + json.dumps(registered_cell["plan_output_path"]),
        "++lace_analysis_protocol_path=" + json.dumps(str(analysis_protocol_path)),
        "++lace_analysis_protocol_file_sha256=" + file_sha256(analysis_protocol_path),
        "++lace_analysis_protocol_sha256=" + analysis_protocol["analysis_protocol_sha256"],
        "++lace_analysis_protocol_lock_path=" + json.dumps(str(analysis_protocol_lock_path)),
        "++lace_analysis_protocol_lock_file_sha256=" + file_sha256(analysis_protocol_lock_path),
        "++lace_analysis_protocol_lock_sha256="
        + analysis_protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
    ]
    plan["hydra_overrides"] = [
        *rebuild_eval_hydra_overrides_from_launch_plan(plan),
        *runtime_overrides,
    ]
    plan["command"] = [
        str(Path(sys.executable).resolve()),
        plan["eval_entrypoint"],
        *plan["hydra_overrides"],
    ]
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    write_new_json(registered_cell["plan_output_path"], plan)
    return {
        "root": root,
        "outputs": outputs,
        "handshake": handshake,
        "handshake_path": handshake_path,
        "plan": plan,
        "checkpoint_bundle": plan["checkpoint_bundle"],
        "split": split,
        "inventory": inventory,
        "schedule": schedule,
        "schedule_path": schedule_path,
        "split_path": split_path,
        "inventory_path": inventory_path,
        "analysis_protocol": analysis_protocol,
        "analysis_protocol_path": analysis_protocol_path,
        "analysis_protocol_lock": analysis_protocol_lock,
        "analysis_protocol_lock_path": analysis_protocol_lock_path,
        "registered_cell": registered_cell,
        "runtime_storage_root": runtime_storage_root,
    }


def _materialize(fixture: dict) -> object:
    termination = _termination_contract()
    assignments = fixture["plan"]["atlas_probe_assignments"]
    failure_config = {
        "enabled": True,
        "allow_append_existing": False,
        "output_path": fixture["handshake"]["rollout_output_path"],
    }
    resolved_config = {
        "checkpoint": fixture["checkpoint_bundle"]["checkpoint_path"],
        "num_envs": len(assignments),
        "seed": assignments[0]["runtime_rng_seed"],
        "headless": True,
        "run_eval_loop": True,
        "run_once": True,
        "use_encoder": "g1",
        "eval_callbacks": [],
        "max_render_steps": max(
            row["reference_num_steps"] - row["start_step"] for row in assignments
        )
        + 2,
        "lace_scientific_instrument_required": True,
        "lace_instrument_handshake_path": str(fixture["handshake_path"]),
        "lace_instrument_handshake_file_sha256": file_sha256(fixture["handshake_path"]),
        "lace_launch_plan_path": str(fixture["outputs"] / "plan.json"),
        "lace_analysis_protocol_path": str(fixture["analysis_protocol_path"]),
        "lace_analysis_protocol_file_sha256": file_sha256(fixture["analysis_protocol_path"]),
        "lace_analysis_protocol_sha256": fixture["analysis_protocol"]["analysis_protocol_sha256"],
        "lace_analysis_protocol_lock_path": str(fixture["analysis_protocol_lock_path"]),
        "lace_analysis_protocol_lock_file_sha256": file_sha256(
            fixture["analysis_protocol_lock_path"]
        ),
        "lace_analysis_protocol_lock_sha256": fixture["analysis_protocol_lock"][
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        "timestamp": "20260814_010203",
        "manager_env": {
            "config": {"terrain_type": "plane", "render_results": False},
            "commands": {
                "motion": {
                    "atlas_probe_mode": True,
                    "atlas_probe_schedule_sha256": fixture["handshake"]["schedule_sha256"],
                    "atlas_probe_assignments": assignments,
                    "filter_motion_keys": fixture["plan"]["motion_keys"],
                    "motion_lib_cfg": {
                        "adaptive_sampling": {"enable": False},
                        "filter_motion_keys": fixture["plan"]["motion_keys"],
                        "motion_file": fixture["plan"]["dataset_binding"]["robot"]["motion_root"],
                        "smpl_motion_file": fixture["plan"]["dataset_binding"]["smpl"][
                            "motion_root"
                        ],
                    },
                }
            },
            "observations": {
                "policy": {"enable_corruption": False},
                "tokenizer": {"enable_corruption": False},
            },
            "recorders": {"failure_atlas": failure_config},
        },
    }
    event_config = _event_config()
    return materialize_instrument_from_payloads(
        handshake_path=fixture["handshake_path"],
        expected_handshake_file_sha256=file_sha256(fixture["handshake_path"]),
        launch_plan_path=fixture["outputs"] / "plan.json",
        repo_root=fixture["root"],
        resolved_hydra_config=resolved_config,
        environment_fingerprint={
            "python": "3.11",
            "isaaclab": "2.3.2",
            "scientific_cache_environment": fixture["plan"]["launch_environment"][
                "scientific_cache_environment"
            ],
            "scientific_cache_environment_semantics": (SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS),
            "process_executable": fixture["plan"]["command"][0],
            "process_argv": fixture["plan"]["command"][1:],
            "pythonpath": fixture["plan"]["launch_environment"]["PYTHONPATH"],
        },
        probe_thresholds=_probe_thresholds(),
        recorder_config={
            "resolved_hydra_term": failure_config,
            "runtime": {
                "num_envs": len(assignments),
                "step_dt_seconds": 0.02,
                "effective_probe_thresholds_sha256": canonical_sha256(_probe_thresholds()),
                "cell_identity": {
                    "schedule_sha256": fixture["handshake"]["schedule_sha256"],
                    "checkpoint_sha256": fixture["handshake"]["checkpoint_sha256"],
                    "probe_policy_id": "medium",
                    "rollout_ids": list(fixture["handshake"]["rollout_ids"]),
                },
            },
        },
        sensor_semantics=_sensor_semantics(),
        termination_predicates=termination,
        score_window_config=_score_window_config(),
        domain_randomization_config=event_config,
        loaded_checkpoint_bundle=fixture["checkpoint_bundle"],
        verify_git_head=False,
    )


def _record(rollout_id: str, materialized: object) -> dict:
    instrument = materialized.instrument
    contract = deepcopy(instrument["termination_predicates"])
    scheduled = next(
        item
        for item in materialized.launch_plan["atlas_probe_assignments"]
        if item["rollout_id"] == rollout_id
    )
    event_config = {
        "interval_events_instrumented": False,
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
    }
    realization = {
        "kind": RUNTIME_REALIZATION_KIND,
        "schema_version": RUNTIME_REALIZATION_SCHEMA_VERSION,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "runtime_rng_seed_semantics": RUNTIME_RNG_SEED_SEMANTICS,
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
        "resolved_event_configuration": event_config,
        "resolved_event_configuration_sha256": canonical_sha256(event_config),
        "ordered_names": {},
        "realized_parameters": {},
        "post_reset_state": {},
        "scheduled_reference": {
            "identity": {
                "schedule_sha256": instrument["schedule_sha256"],
                "motion_key": scheduled["motion_key"],
                "domain_randomization_seed": scheduled["domain_randomization_seed"],
                "runtime_rng_seed": scheduled["runtime_rng_seed"],
                "phase_id": scheduled["phase_id"],
                "target_fraction": scheduled["target_fraction"],
                "realized_fraction": scheduled["realized_fraction"],
                "reference_start_step": scheduled["start_step"],
                "reference_num_steps": scheduled["reference_num_steps"],
                "repeat_index": scheduled["repeat_index"],
            },
            "state": {},
        },
    }
    robot_contract = {
        "kind": ROBOT_CONTRACT_READBACK_KIND,
        "schema_version": ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "ordered_body_names": ["pelvis"],
        "ordered_joint_names": ["joint"],
        "source_properties": {
            "joint_names": "robot.joint_names",
            "joint_pos_limits": "robot.data.joint_pos_limits[0]",
            "soft_joint_pos_limits": "robot.data.soft_joint_pos_limits[0]",
            "joint_vel_limits": "robot.data.joint_vel_limits[0]",
            "soft_joint_vel_limits": "robot.data.soft_joint_vel_limits[0]",
        },
        "limits": {
            "joint_pos_limits": {
                "dtype": "float64",
                "shape": [1, 2],
                "values": [[-1.0, 1.0]],
            },
            "soft_joint_pos_limits": {
                "dtype": "float64",
                "shape": [1, 2],
                "values": [[-0.9, 0.9]],
            },
            "joint_vel_limits": {
                "dtype": "float64",
                "shape": [1],
                "values": [10.0],
            },
            "soft_joint_vel_limits": {
                "dtype": "float64",
                "shape": [1],
                "values": [9.0],
            },
        },
        "environment_invariance_verified": True,
    }
    probe_result = compute_episode_probe(
        reference_contacts=np.asarray([[False], [True]]),
        actual_contacts=np.asarray([[True], [True]]),
        foot_tangential_speed=np.asarray([[0.2], [0.3]]),
        base_translation_error=np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]),
        base_orientation_error=np.asarray([0.2, 0.3]),
        base_tilt=np.asarray([0.4, 0.5]),
        requested_torque=np.asarray([[2.0], [2.0]]),
        applied_torque=np.asarray([[1.0], [1.0]]),
        effort_limits=np.asarray([[1.0], [1.0]]),
        reference_joint_position=np.asarray([[0.0], [0.0]]),
        joint_position=np.asarray([[0.9], [0.95]]),
        joint_soft_lower_limits=np.asarray([[-1.0], [-1.0]]),
        joint_soft_upper_limits=np.asarray([[1.0], [1.0]]),
        local_pose_error=np.asarray([[0.2], [0.3]]),
        fall_mask=np.asarray([False, True]),
        failure_mask=np.asarray([False, True]),
        timestep_seconds=0.02,
    )
    probe_episode = dict(probe_result.episode_diagnostics)
    reference_start = scheduled["start_step"]
    reference_steps = scheduled["reference_num_steps"]
    reference_denominator = max(reference_steps - 1, 1)
    first_failure_index = probe_episode["first_failure_index"]
    reference_failure_step = reference_start + first_failure_index
    reference_end_step = min(reference_start + 1, reference_steps - 1)
    probe_episode.update(
        {
            "reference_start_step": reference_start,
            "reference_num_steps": reference_steps,
            "reference_end_step": reference_end_step,
            "reference_progress_at_end": reference_end_step / reference_denominator,
            "reference_failure_step": reference_failure_step,
            "reference_progress_to_failure": reference_failure_step / reference_denominator,
            "failure_progress_censored": False,
        }
    )
    probe_payload = {
        "scores": dict(probe_result.mechanism_scores),
        "onsets": dict(probe_result.onset_times_seconds),
        "onset_unit": "seconds",
        "diagnostics": {name: dict(values) for name, values in probe_result.diagnostics.items()},
        "episode_diagnostics": probe_episode,
    }
    record = {
        **scheduled,
        "rollout_id": rollout_id,
        "schedule_sha256": instrument["schedule_sha256"],
        "checkpoint_sha256": scheduled["checkpoint_sha256"],
        "reference_start_step": scheduled["start_step"],
        "initial_phase": scheduled["realized_fraction"],
        "schedule_entry_id": rollout_id,
        "policy_id": scheduled["probe_policy_id"],
        "domain_randomization_seed_semantics": DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "runtime_rng_seed_readback": scheduled["runtime_rng_seed"],
        "runtime_rng_seed_readback_source": "env.cfg.seed",
        "runtime_rng_seed_semantics": RUNTIME_RNG_SEED_SEMANTICS,
        "scientific_runtime_ready": True,
        "scientific_runtime_blockers": [],
        "completion_reason": "episode_end",
        "termination_multi_hot_available": True,
        "termination_semantics": TERMINATION_MULTI_HOT_SEMANTICS,
        "probe_thresholds": instrument["probe_thresholds"],
        "probe_thresholds_sha256": instrument["probe_thresholds_sha256"],
        "domain_randomization_realization": realization,
        "domain_randomization_realization_sha256": canonical_sha256(realization),
        "buffering_semantics": "bounded_full_episode_cpu_smoke",
        "buffered_step_count": 2,
        "max_episode_steps": reference_steps,
        "foot_slip_velocity_proxy": ("link_origin_velocity_tangent_to_configured_plane"),
        "ground_normal_axis": 2,
        "failed": True,
        "mechanism_scores": dict(probe_result.mechanism_scores),
        "probe": probe_payload,
        "robot_contract_readback": robot_contract,
        "robot_contract_readback_sha256": canonical_sha256(robot_contract),
        "termination_terms": {
            "anchor_pos": {"occurred": True},
            "time_out": {"occurred": False},
        },
        "termination_multi_hot": {
            "term_names": list(contract["term_names"]),
            "time_out_flags": list(contract["time_out_flags"]),
            "values": [[False, False], [True, False]],
            "trace_generations": [1, 2],
            "common_step_counters": [1, 2],
            "contract": contract,
            "contract_sha256": instrument["termination_predicates_sha256"],
        },
    }
    record.pop("start_step")
    return record


def test_live_protocol_preflight_is_zero_outcome_and_receipt_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gear_sonic.research.lace.instrument_runtime as runtime_module
    import gear_sonic.research.lace.isaac_recorder as recorder_module

    fixture = _planned_runtime(tmp_path / "fixture")
    preflight_root = tmp_path / "preflight"
    preflight_root.mkdir()
    request_path = preflight_root / "protocol.preflight-request.json"
    plan_path = preflight_root / "protocol.preflight-plan.json"
    protocol_path = preflight_root / "protocol.json"
    receipt_path = preflight_root / "protocol.preflight-receipt.json"
    raw_path = preflight_root / "must-remain-absent.jsonl"
    source_paths = discover_scientific_source_paths(fixture["root"])
    source_bundle = build_source_bundle(fixture["root"], source_paths)
    request = build_protocol_preflight_request(
        schedule_lock_path=Path(fixture["schedule_path"]).with_name("schedule-lock.json"),
        schedule_path=fixture["schedule_path"],
        schedule_manifest=fixture["schedule"],
        split_path=fixture["split_path"],
        split_manifest=fixture["split"],
        reference_length_inventory_path=fixture["inventory_path"],
        reference_length_inventory=fixture["inventory"],
        cell=fixture["plan"]["cell"],
        rollout_ids=fixture["plan"]["rollout_ids"],
        checkpoint_bundle=fixture["checkpoint_bundle"],
        dataset_binding_sha256=canonical_sha256(fixture["plan"]["dataset_binding"]),
        source_bundle=source_bundle,
        git_commit="9" * 40,
        launch_plan_path=plan_path,
        recorder_output_path=raw_path,
        analysis_protocol_output_path=protocol_path,
        preflight_receipt_output_path=receipt_path,
        repo_root=fixture["root"],
    )
    write_new_json(request_path, request)
    plan = {
        "schedule_sha256": fixture["schedule"]["schedule_sha256"],
        "cell": fixture["plan"]["cell"],
        "rollout_ids": fixture["plan"]["rollout_ids"],
        "checkpoint_bundle": fixture["checkpoint_bundle"],
        "dataset_binding": fixture["plan"]["dataset_binding"],
        "protocol_preflight": {
            "request_path": str(request_path),
            "request_file_sha256": file_sha256(request_path),
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: request[
                PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
            ],
        },
    }
    plan["command"] = [
        str(Path(sys.executable).resolve()),
        str(fixture["root"] / "gear_sonic" / "eval_agent_trl.py"),
        "++lace_protocol_preflight=true",
    ]
    plan["launch_environment"] = {
        "PYTHONPATH": str(fixture["root"]),
        "scientific_cache_environment": fixture["plan"]["launch_environment"][
            "scientific_cache_environment"
        ],
        "scientific_cache_environment_semantics": SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    }
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    write_new_json(plan_path, plan)

    termination = _termination_contract()
    thresholds = _probe_thresholds()
    bindings = SimpleNamespace(
        command_name="motion",
        robot_name="robot",
        joint_action_name="joint_pos",
        contact_sensor_name="contact",
        foot_body_names=("left_foot", "right_foot"),
        contact_force_threshold=1.0,
        ground_normal_axis=2,
    )
    term = SimpleNamespace(
        _enabled=True,
        _record_count=0,
        _termination_trace_contract=termination,
        _termination_trace_contract_sha256=canonical_sha256(termination),
        _assembler=SimpleNamespace(
            _threshold_record=thresholds,
            _threshold_sha256=canonical_sha256(thresholds),
        ),
        _bindings=bindings,
    )
    batch = SimpleNamespace(
        schedule_sha256=fixture["schedule"]["schedule_sha256"],
        checkpoint_sha256=fixture["checkpoint_bundle"]["checkpoint_sha256"],
        rollout_ids=tuple(fixture["plan"]["rollout_ids"]),
    )
    command = SimpleNamespace(atlas_probe_batch=batch)
    raw_env = SimpleNamespace(
        recorder_manager=SimpleNamespace(_terms={"failure_atlas": term}),
        command_manager=SimpleNamespace(get_term=lambda name: command),
        termination_manager=SimpleNamespace(),
        event_manager=SimpleNamespace(),
        device="cpu",
        num_envs=len(fixture["plan"]["rollout_ids"]),
        step_dt=0.02,
    )
    wrapped_env = SimpleNamespace(env=raw_env)
    monkeypatch.setattr(
        runtime_module,
        "capture_environment_fingerprint",
        lambda env: _preflight_environment_fingerprint(
            plan,
            num_envs=len(fixture["plan"]["rollout_ids"]),
        ),
    )
    monkeypatch.setattr(
        recorder_module,
        "capture_resolved_event_configuration",
        lambda manager: _event_config(),
    )
    resolved_config = {
        "checkpoint": fixture["checkpoint_bundle"]["checkpoint_path"],
        "lace_scientific_instrument_required": False,
        "lace_protocol_preflight_request_path": str(request_path),
        "lace_protocol_preflight_request_file_sha256": file_sha256(request_path),
        PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD],
        "lace_launch_plan_path": str(plan_path),
        "manager_env": {
            "commands": {
                "motion": {
                    "atlas_probe_schedule_sha256": fixture["schedule"]["schedule_sha256"],
                    "motion_lib_cfg": {
                        "motion_file": fixture["plan"]["dataset_binding"]["robot"]["motion_root"],
                        "smpl_motion_file": fixture["plan"]["dataset_binding"]["smpl"][
                            "motion_root"
                        ],
                    },
                }
            },
            "recorders": {
                "failure_atlas": {
                    "enabled": True,
                    "allow_append_existing": False,
                    "output_path": str(raw_path),
                }
            },
        },
    }
    receipt = execute_live_protocol_preflight(
        request_path=request_path,
        expected_request_file_sha256=file_sha256(request_path),
        launch_plan_path=plan_path,
        repo_root=fixture["root"],
        resolved_hydra_config=resolved_config,
        wrapped_env=wrapped_env,
    )
    assert protocol_path.is_file()
    assert receipt_path.is_file()
    assert not raw_path.exists()
    assert term._record_count == 0
    assert receipt["recorder_record_count_before"] == 0
    assert receipt["recorder_record_count_after"] == 0
    validate_protocol_preflight_receipt(
        receipt,
        repo_root=fixture["root"],
        verify_files=True,
    )

    forged = deepcopy(receipt)
    forged["live_inputs"]["sensor_semantics"]["command_name"] = "forged"
    forged["live_inputs_sha256"] = canonical_sha256(forged["live_inputs"])
    forged["protocol_preflight_receipt_sha256"] = canonical_sha256(
        forged,
        digest_field="protocol_preflight_receipt_sha256",
    )
    with pytest.raises(ValueError, match="deterministic output"):
        validate_protocol_preflight_receipt(
            forged,
            repo_root=fixture["root"],
            verify_files=True,
        )

    causal_forges = []
    forged_cell = deepcopy(receipt)
    forged_cell["cell"]["probe_policy_id"] = "forged-policy"
    causal_forges.append(forged_cell)
    forged_rollouts = deepcopy(receipt)
    forged_rollouts["rollout_ids"] = ["forged-rollout"]
    causal_forges.append(forged_rollouts)
    forged_checkpoint = deepcopy(receipt)
    forged_checkpoint["checkpoint_bundle"] = {
        "checkpoint_path": receipt["checkpoint_bundle"]["config_path"],
        "checkpoint_sha256": receipt["checkpoint_bundle"]["config_sha256"],
        "config_path": receipt["checkpoint_bundle"]["checkpoint_path"],
        "config_sha256": receipt["checkpoint_bundle"]["checkpoint_sha256"],
    }
    causal_forges.append(forged_checkpoint)
    forged_dataset = deepcopy(receipt)
    forged_dataset["dataset_binding_sha256"] = "f" * 64
    causal_forges.append(forged_dataset)
    for forged_causal in causal_forges:
        forged_causal["protocol_preflight_receipt_sha256"] = canonical_sha256(
            forged_causal,
            digest_field="protocol_preflight_receipt_sha256",
        )
        with pytest.raises(ValueError, match="causal fields drifted"):
            validate_protocol_preflight_receipt(
                forged_causal,
                repo_root=fixture["root"],
                verify_files=True,
            )

    forged_config = deepcopy(receipt)
    forged_config["resolved_hydra_config"]["lace_protocol_preflight_request_path"] = str(
        preflight_root / "forged-request.json"
    )
    forged_config["resolved_hydra_config_sha256"] = canonical_sha256(
        forged_config["resolved_hydra_config"]
    )
    forged_config["protocol_preflight_receipt_sha256"] = canonical_sha256(
        forged_config,
        digest_field="protocol_preflight_receipt_sha256",
    )
    with pytest.raises(ValueError, match="resolved Hydra config"):
        validate_protocol_preflight_receipt(
            forged_config,
            repo_root=fixture["root"],
            verify_files=True,
        )

    live_input_forges = []
    forged_extra = deepcopy(receipt)
    forged_extra["live_inputs"]["posthoc"] = True
    live_input_forges.append((forged_extra, "live-input fields"))
    forged_threshold_digest = deepcopy(receipt)
    forged_threshold_digest["live_inputs"]["probe_thresholds_sha256"] = "f" * 64
    live_input_forges.append((forged_threshold_digest, "does not bind probe_thresholds"))
    forged_record_count = deepcopy(receipt)
    forged_record_count["live_inputs"]["recorder_record_count"] = 1
    live_input_forges.append((forged_record_count, "recorder count"))
    forged_argv = deepcopy(receipt)
    forged_argv["live_inputs"]["environment_fingerprint"]["process_argv"] = ["forged-entrypoint.py"]
    live_input_forges.append((forged_argv, "environment fingerprint drifted"))
    for forged_live, message in live_input_forges:
        forged_live["live_inputs_sha256"] = canonical_sha256(forged_live["live_inputs"])
        forged_live["protocol_preflight_receipt_sha256"] = canonical_sha256(
            forged_live,
            digest_field="protocol_preflight_receipt_sha256",
        )
        with pytest.raises(ValueError, match=message):
            validate_protocol_preflight_receipt(
                forged_live,
                repo_root=fixture["root"],
                verify_files=True,
            )


def test_preflight_detection_does_not_intercept_normal_scientific_plan_binding() -> None:
    assert (
        protocol_preflight_binding_from_config(
            {
                "lace_launch_plan_path": "/data/lace/executions/cell/launch-plan.json",
                "lace_scientific_instrument_required": True,
            }
        )
        is None
    )
    with pytest.raises(ValueError, match="binding is incomplete"):
        protocol_preflight_binding_from_config(
            {
                "lace_launch_plan_path": "/data/lace/preflight/launch-plan.json",
                "lace_protocol_preflight_request_path": "/data/lace/preflight/request.json",
            }
        )
    binding = protocol_preflight_binding_from_config(
        {
            "lace_launch_plan_path": "/data/lace/preflight/launch-plan.json",
            "lace_protocol_preflight_request_path": "/data/lace/preflight/request.json",
            "lace_protocol_preflight_request_file_sha256": "a" * 64,
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: "b" * 64,
        }
    )
    assert binding == (
        "/data/lace/preflight/request.json",
        "a" * 64,
        "b" * 64,
        "/data/lace/preflight/launch-plan.json",
    )


def test_protocol_lock_requires_exact_live_preflight_receipt(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    missing = deepcopy(fixture["analysis_protocol_lock"])
    missing.pop("analysis_protocol_preflight")
    missing[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD] = canonical_sha256(
        missing,
        digest_field=ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="fields are invalid"):
        validate_analysis_protocol_lock(
            missing,
            repo_root=fixture["root"],
            verify_files=True,
        )

    receipt_path = Path(fixture["analysis_protocol_lock"]["analysis_protocol_preflight"]["path"])
    receipt_path.write_bytes(receipt_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="preflight receipt file digest drifted"):
        validate_analysis_protocol_lock(
            fixture["analysis_protocol_lock"],
            repo_root=fixture["root"],
            verify_files=True,
        )


def test_runtime_handshake_materializes_and_binds_complete_rollout(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    validate_scientific_instrument(materialized.instrument, repo_root=fixture["root"])
    assert materialized.binding[INSTRUMENT_BINDING_DIGEST_FIELD] == canonical_sha256(
        materialized.binding,
        digest_field=INSTRUMENT_BINDING_DIGEST_FIELD,
    )

    # Deliberately write completion order opposite the frozen handshake order.
    output = fixture["outputs"] / "cell.jsonl"
    with output.open("x", encoding="utf-8") as handle:
        for rollout_id in reversed(fixture["handshake"]["rollout_ids"]):
            handle.write(json.dumps(_record(rollout_id, materialized), sort_keys=True) + "\n")
    raw_bytes = output.read_bytes()
    receipt = finalize_scientific_rollout(materialized)

    assert output.read_bytes() == raw_bytes
    instrumented = fixture["outputs"] / "cell.instrumented.jsonl"
    rows = [json.loads(line) for line in instrumented.read_text(encoding="utf-8").splitlines()]
    assert [row["rollout_id"] for row in rows] == fixture["handshake"]["rollout_ids"]
    assert {row["instrument_sha256"] for row in rows} == {materialized.episode_instrument_sha256}
    receipt_payload = json.loads(
        Path(fixture["registered_cell"]["rollout_binding_output_path"]).read_text(encoding="utf-8")
    )
    assert receipt[ROLLOUT_BINDING_DIGEST_FIELD] == canonical_sha256(
        receipt_payload,
        digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
    )
    assert receipt["artifacts"]["raw_rollouts"]["file_sha256"] == file_sha256(output)
    assert receipt["artifacts"]["instrumented_rollouts"]["file_sha256"] == file_sha256(instrumented)


def test_runtime_rejects_source_or_plan_drift_before_instrument_write(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    (fixture["root"] / "gear_sonic" / "research" / "lace" / "probe.py").write_text(
        "probe = 2\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="digest drifted"):
        _materialize(fixture)
    assert not (fixture["outputs"] / "instrument.json").exists()

    fixture = _planned_runtime(tmp_path / "plan-drift")
    plan = deepcopy(fixture["plan"])
    plan["rollout_ids"].reverse()
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    (fixture["outputs"] / "plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rollout ids drifted"):
        _materialize(fixture)


@pytest.mark.parametrize(
    "tamper",
    [
        "kind",
        "schema_version",
        "reference_num_steps",
        "instrumentation_invariants",
        "eval_entrypoint",
        "hydra_overrides",
        "command",
        "pythonpath",
        "pythonpath_semantics",
        "passthrough",
    ],
)
def test_materialization_reconstructs_every_scientific_launch_plan_field(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _planned_runtime(tmp_path)
    plan = deepcopy(fixture["plan"])
    if tamper == "kind":
        plan["kind"] = "forged-plan"
    elif tamper == "schema_version":
        plan["schema_version"] += 1
    elif tamper == "reference_num_steps":
        plan["reference_num_steps"][0] += 1
    elif tamper == "instrumentation_invariants":
        plan["instrumentation_invariants"]["max_render_steps"] += 1
    elif tamper == "eval_entrypoint":
        plan["eval_entrypoint"] = str(fixture["root"] / "forged_eval.py")
    elif tamper == "hydra_overrides":
        plan["hydra_overrides"][0] += "-forged"
    elif tamper == "command":
        plan["command"][0] = "/forged/python"
    elif tamper == "pythonpath":
        plan["launch_environment"]["PYTHONPATH"] = "/forged/pythonpath"
    elif tamper == "pythonpath_semantics":
        plan["launch_environment"]["semantics"] = "forged"
    else:
        plan["passthrough_hydra_args"] = ["++forged=true"]
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _overwrite_json(fixture["outputs"] / "plan.json", plan)
    with pytest.raises(ValueError):
        _materialize(fixture)


@pytest.mark.parametrize("tamper", ["cache_token", "cache_root"])
def test_materialization_rejects_forged_scientific_cache_coordinates(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _planned_runtime(tmp_path)
    plan = deepcopy(fixture["plan"])
    environment = plan["launch_environment"]
    if tamper == "cache_token":
        environment["launch_cache_token"] = "f" * 64
    else:
        key = SCIENTIFIC_CACHE_ENVIRONMENT_KEYS[0]
        forged = tmp_path / "forged-root" / environment["launch_cache_token"] / "tmp"
        forged.mkdir(parents=True)
        environment["scientific_cache_environment"][key] = str(forged.resolve())
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _overwrite_json(fixture["outputs"] / "plan.json", plan)
    with pytest.raises(ValueError, match="cache"):
        _materialize(fixture)


def test_incomplete_rollout_remains_uninstrumented_and_fails_closed(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    output = fixture["outputs"] / "cell.jsonl"
    raw_record = _record(fixture["handshake"]["rollout_ids"][0], materialized)
    original = json.dumps(raw_record, sort_keys=True) + "\n"
    output.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="incomplete"):
        finalize_scientific_rollout(materialized)
    assert output.read_text(encoding="utf-8") == original
    assert "instrument_sha256" not in json.loads(original)


def test_finalizer_rejects_trace_tampering_or_existing_receipt_without_rewrite(
    tmp_path: Path,
) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    output = fixture["outputs"] / "cell.jsonl"
    records = [
        _record(rollout_id, materialized) for rollout_id in fixture["handshake"]["rollout_ids"]
    ]
    records[0]["termination_multi_hot"]["contract"] = deepcopy(
        records[0]["termination_multi_hot"]["contract"]
    )
    records[0]["termination_multi_hot"]["contract"]["manager_type"] = "tampered:Manager"
    records[0]["termination_multi_hot"]["contract_sha256"] = canonical_sha256(
        records[0]["termination_multi_hot"]["contract"]
    )
    original = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    output.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="termination contract drifted"):
        finalize_scientific_rollout(materialized)
    assert output.read_text(encoding="utf-8") == original

    records[0] = _record(fixture["handshake"]["rollout_ids"][0], materialized)
    original = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    output.write_text(original, encoding="utf-8")
    Path(fixture["registered_cell"]["rollout_binding_output_path"]).write_text(
        "{}\n", encoding="utf-8"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite runtime artifact"):
        finalize_scientific_rollout(materialized)
    assert output.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_probe", "probe fields"),
        ("score_out_of_range", r"must be in \[0, 1\]"),
        ("score_mismatch", "probe scores drifted"),
        ("joint_score_rewrite", "does not match frozen diagnostic reconstruction"),
        ("failed_mismatch", "probe failed status drifted"),
        ("diagnostics_window", "score-window indices are inconsistent"),
    ],
)
def test_finalizer_rejects_probe_outcome_tampering(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    records = [
        _record(rollout_id, materialized) for rollout_id in fixture["handshake"]["rollout_ids"]
    ]
    target = records[0]
    if tamper == "missing_probe":
        target.pop("probe")
    elif tamper == "score_out_of_range":
        target["mechanism_scores"]["contact_timing"] = 1.1
        target["probe"]["scores"]["contact_timing"] = 1.1
    elif tamper == "score_mismatch":
        target["mechanism_scores"]["contact_timing"] = 0.123
    elif tamper == "joint_score_rewrite":
        target["mechanism_scores"]["base_drift"] = 0.123
        target["probe"]["scores"]["base_drift"] = 0.123
    elif tamper == "failed_mismatch":
        target["probe"]["episode_diagnostics"]["failed"] = False
    else:
        target["probe"]["episode_diagnostics"]["score_window_num_samples"] = 1
    output = fixture["outputs"] / "cell.jsonl"
    output.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        finalize_scientific_rollout(materialized)


def test_handshake_rejects_tampering_and_output_reuse(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    tampered = deepcopy(fixture["handshake"])
    tampered["checkpoint_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="runtime_handshake_sha256 mismatch"):
        validate_runtime_handshake(tampered, repo_root=fixture["root"])

    (fixture["outputs"] / "instrument.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        validate_runtime_handshake(
            fixture["handshake"],
            repo_root=fixture["root"],
            require_new_outputs=True,
        )


def test_runtime_rejects_rehashed_operator_selected_checkpoint_config(
    tmp_path: Path,
) -> None:
    fixture = _planned_runtime(tmp_path)
    alternate_dir = tmp_path / "alternate-checkpoint"
    alternate_dir.mkdir()
    alternate_config = alternate_dir / "config.yaml"
    alternate_config.write_text("model: outcome-selected-alternative\n", encoding="utf-8")
    plan_path = fixture["outputs"] / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["checkpoint_bundle"]["config_path"] = str(alternate_config)
    plan["checkpoint_bundle"]["config_sha256"] = file_sha256(alternate_config)
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _overwrite_json(plan_path, plan)

    with pytest.raises(ValueError, match="companion config drifted from preregistration"):
        _materialize(fixture)


def test_materialization_recovers_only_exact_pre_rollout_sidecars(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    first = _materialize(fixture)
    second = _materialize(fixture)
    assert second.instrument == first.instrument
    assert second.binding == first.binding

    binding_path = Path(fixture["registered_cell"]["instrument_binding_output_path"])
    binding_path.write_bytes(binding_path.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="deterministic expected artifact"):
        _materialize(fixture)


def test_source_discovery_rejects_symlinks_and_duplicate_extras(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(ValueError, match="duplicate instrument source path"):
        discover_scientific_source_paths(
            root,
            extra_paths=["gear_sonic/research/lace/probe.py"],
        )

    (root / "gear_sonic" / "unsafe.py").symlink_to("/outside/repository.py")
    with pytest.raises(ValueError, match="source closure contains symlink"):
        discover_scientific_source_paths(root)


def test_eval_entrypoint_materializes_before_policy_rollout_and_finalizes_afterward() -> None:
    source = (Path(__file__).resolve().parents[2] / "gear_sonic/eval_agent_trl.py").read_text(
        encoding="utf-8"
    )
    reinit = source.index("    env.reinit_dr()")
    materialize = source.index("        lace_runtime_instrument = materialize_live_instrument(")
    rollout = source.index('    if config.get("run_eval_loop", True):')
    finalize = source.index("        lace_rollout_binding = finalize_scientific_rollout(")
    process_exit = source.index('    if simulator_type == "IsaacSim":', finalize)

    assert reinit < materialize < rollout < finalize < process_exit


def _completed_runtime_fixture(tmp_path: Path) -> tuple[dict, object, Path]:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    raw_path = fixture["outputs"] / "cell.jsonl"
    with raw_path.open("x", encoding="utf-8") as handle:
        for rollout_id in reversed(fixture["handshake"]["rollout_ids"]):
            handle.write(json.dumps(_record(rollout_id, materialized), sort_keys=True) + "\n")
    finalize_scientific_rollout(materialized)
    return (
        fixture,
        materialized,
        Path(fixture["registered_cell"]["rollout_binding_output_path"]),
    )


def test_receipt_collector_revalidates_all_bytes_and_freezes_schedule_order(
    tmp_path: Path,
) -> None:
    fixture, materialized, receipt_path = _completed_runtime_fixture(tmp_path)
    collection = collect_scientific_rollouts(
        receipt_paths=[receipt_path],
        schedule_manifest=fixture["schedule"],
        split_manifest=fixture["split"],
        reference_length_inventory=fixture["inventory"],
        analysis_protocol=fixture["analysis_protocol"],
        analysis_protocol_path=fixture["analysis_protocol_path"],
        analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
        expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        repo_root=fixture["root"],
    )
    assert [row["rollout_id"] for row in collection["episodes"]] == [
        row["rollout_id"] for row in fixture["schedule"]["rollouts"]
    ]
    assert collection["cell_count"] == 1
    assert collection["rollout_count"] == len(fixture["schedule"]["rollouts"])
    assert collection[ROLLOUT_COLLECTION_DIGEST_FIELD] == canonical_sha256(
        collection,
        digest_field=ROLLOUT_COLLECTION_DIGEST_FIELD,
    )
    assert set(collection["cell_instruments"]) == {materialized.episode_instrument_sha256}
    validate_rollout_collection(
        collection,
        schedule_manifest=fixture["schedule"],
        split_manifest=fixture["split"],
        reference_length_inventory=fixture["inventory"],
        analysis_protocol=fixture["analysis_protocol"],
        analysis_protocol_path=fixture["analysis_protocol_path"],
        analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
        expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        repo_root=fixture["root"],
    )


def test_collector_rejects_missing_receipt_and_post_receipt_byte_drift(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path / "missing")
    with pytest.raises(ValueError, match="receipt count"):
        collect_scientific_rollouts(
            receipt_paths=[],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )

    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path / "drift")
    raw_path = fixture["outputs"] / "cell.jsonl"
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw_rollouts.*digest drifted"):
        collect_scientific_rollouts(
            receipt_paths=[receipt_path],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )


def test_collector_rejects_receipt_symlink_before_canonicalization(tmp_path: Path) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    linked_receipt = tmp_path / "receipt-link.json"
    linked_receipt.symlink_to(receipt_path)
    with pytest.raises(ValueError, match="may not be a symlink"):
        collect_scientific_rollouts(
            receipt_paths=[linked_receipt],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )


def test_collector_rejects_unregistered_receipt_path_and_orphan_attempt(
    tmp_path: Path,
) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    common = {
        "schedule_manifest": fixture["schedule"],
        "split_manifest": fixture["split"],
        "reference_length_inventory": fixture["inventory"],
        "analysis_protocol": fixture["analysis_protocol"],
        "analysis_protocol_path": fixture["analysis_protocol_path"],
        "analysis_protocol_lock_path": fixture["analysis_protocol_lock_path"],
        "expected_analysis_protocol_lock_sha256": fixture["analysis_protocol_lock"][
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        "repo_root": fixture["root"],
    }
    alternate_receipt = tmp_path / "cherry-picked-receipt.json"
    alternate_receipt.write_bytes(receipt_path.read_bytes())
    with pytest.raises(ValueError, match="one-attempt cell registry"):
        collect_scientific_rollouts(
            receipt_paths=[alternate_receipt],
            **common,
        )

    orphan = (
        Path(fixture["analysis_protocol_lock"]["execution_registry"]["execution_root"])
        / "unregistered-attempt"
    )
    orphan.mkdir()
    with pytest.raises(ValueError, match="extra.*cell attempts"):
        collect_scientific_rollouts(
            receipt_paths=[receipt_path],
            **common,
        )


def test_collector_rejects_fully_rehashed_source_bundle_chain_forgery(tmp_path: Path) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    outputs = fixture["outputs"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    forged_source_digest = "f" * 64
    family = deepcopy(receipt["measurement_family"])
    family["source_bundle_sha256"] = forged_source_digest
    family[MEASUREMENT_FAMILY_DIGEST_FIELD] = canonical_sha256(
        family,
        digest_field=MEASUREMENT_FAMILY_DIGEST_FIELD,
    )
    episode_family_digest = episode_measurement_family_sha256(family)

    binding_path = Path(fixture["registered_cell"]["instrument_binding_output_path"])
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["source_bundle_sha256"] = forged_source_digest
    binding["measurement_family"] = family
    binding["measurement_family_manifest_sha256"] = family[MEASUREMENT_FAMILY_DIGEST_FIELD]
    binding["episode_measurement_family_sha256"] = episode_family_digest
    binding[INSTRUMENT_BINDING_DIGEST_FIELD] = canonical_sha256(
        binding,
        digest_field=INSTRUMENT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(binding_path, binding)

    instrumented_path = outputs / "cell.instrumented.jsonl"
    instrumented_rows = [
        json.loads(line) for line in instrumented_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in instrumented_rows:
        row["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
        row["measurement_family_sha256"] = episode_family_digest
        row["measurement_family_manifest_sha256"] = family[MEASUREMENT_FAMILY_DIGEST_FIELD]
    _overwrite_jsonl(instrumented_path, instrumented_rows)

    receipt["source_bundle_sha256"] = forged_source_digest
    receipt["measurement_family"] = family
    receipt["measurement_family_manifest_sha256"] = family[MEASUREMENT_FAMILY_DIGEST_FIELD]
    receipt["episode_measurement_family_sha256"] = episode_family_digest
    receipt["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
    receipt["artifacts"]["instrument_binding"]["file_sha256"] = file_sha256(binding_path)
    receipt["artifacts"]["instrumented_rollouts"]["file_sha256"] = file_sha256(instrumented_path)
    receipt[ROLLOUT_BINDING_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="source-bundle provenance chain drifted"):
        collect_scientific_rollouts(
            receipt_paths=[receipt_path],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )


def test_collector_rejects_fully_rehashed_handshake_cell_forgery(tmp_path: Path) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    outputs = fixture["outputs"]
    handshake_path = Path(fixture["registered_cell"]["runtime_handshake_output_path"])
    handshake = json.loads(handshake_path.read_text(encoding="utf-8"))
    handshake["probe_policy_id"] = "forged-policy"
    handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD] = canonical_sha256(
        handshake,
        digest_field=RUNTIME_HANDSHAKE_DIGEST_FIELD,
    )
    _overwrite_json(handshake_path, handshake)

    plan_path = outputs / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["instrument_runtime"] = build_plan_binding(handshake_path, handshake)
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _overwrite_json(plan_path, plan)

    binding_path = Path(fixture["registered_cell"]["instrument_binding_output_path"])
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["runtime_handshake_sha256"] = handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD]
    binding["launch_plan_sha256"] = plan["launch_plan_sha256"]
    binding["launch_plan_file_sha256"] = file_sha256(plan_path)
    binding[INSTRUMENT_BINDING_DIGEST_FIELD] = canonical_sha256(
        binding,
        digest_field=INSTRUMENT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(binding_path, binding)

    instrumented_path = outputs / "cell.instrumented.jsonl"
    instrumented_rows = [
        json.loads(line) for line in instrumented_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in instrumented_rows:
        row["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
    _overwrite_jsonl(instrumented_path, instrumented_rows)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runtime_handshake_sha256"] = handshake[RUNTIME_HANDSHAKE_DIGEST_FIELD]
    receipt["launch_plan_sha256"] = plan["launch_plan_sha256"]
    receipt["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
    receipt["artifacts"]["runtime_handshake"]["file_sha256"] = file_sha256(handshake_path)
    receipt["artifacts"]["launch_plan"]["file_sha256"] = file_sha256(plan_path)
    receipt["artifacts"]["instrument_binding"]["file_sha256"] = file_sha256(binding_path)
    receipt["artifacts"]["instrumented_rollouts"]["file_sha256"] = file_sha256(instrumented_path)
    receipt[ROLLOUT_BINDING_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="plan policy drifted"):
        collect_scientific_rollouts(
            receipt_paths=[receipt_path],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )


def test_collector_rejects_fully_rehashed_smpl_only_alternate_root(tmp_path: Path) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    outputs = fixture["outputs"]
    plan_path = outputs / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for group_name in ("smpl",):
        original = plan["dataset_binding"][group_name]
        alternate_root = tmp_path / f"alternate-{group_name}"
        alternate_root.mkdir()
        alternate_files = []
        for record in original["files"]:
            alternate_path = alternate_root / Path(record["path"]).name
            alternate_path.write_bytes(Path(record["path"]).read_bytes())
            alternate_files.append(
                {
                    "motion_key": record["motion_key"],
                    "path": str(alternate_path),
                    "sha256": file_sha256(alternate_path),
                }
            )
        original["motion_root"] = str(alternate_root)
        original["files"] = alternate_files
        original["selected_file_set_sha256"] = canonical_sha256({"files": alternate_files})
    runtime_override_index = next(
        index
        for index, value in enumerate(plan["hydra_overrides"])
        if value == "++lace_scientific_instrument_required=true"
    )
    plan["hydra_overrides"] = [
        *rebuild_eval_hydra_overrides_from_launch_plan(plan),
        *plan["hydra_overrides"][runtime_override_index:],
    ]
    plan["command"] = [*plan["command"][:2], *plan["hydra_overrides"]]
    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _overwrite_json(plan_path, plan)

    binding_path = Path(fixture["registered_cell"]["instrument_binding_output_path"])
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["launch_plan_sha256"] = plan["launch_plan_sha256"]
    binding["launch_plan_file_sha256"] = file_sha256(plan_path)
    binding["dataset_binding_sha256"] = canonical_sha256(plan["dataset_binding"])
    binding[INSTRUMENT_BINDING_DIGEST_FIELD] = canonical_sha256(
        binding,
        digest_field=INSTRUMENT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(binding_path, binding)

    instrumented_path = outputs / "cell.instrumented.jsonl"
    instrumented_rows = [
        json.loads(line) for line in instrumented_path.read_text(encoding="utf-8").splitlines()
    ]
    for row in instrumented_rows:
        row["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
    _overwrite_jsonl(instrumented_path, instrumented_rows)

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["launch_plan_sha256"] = plan["launch_plan_sha256"]
    receipt["dataset_binding_sha256"] = binding["dataset_binding_sha256"]
    receipt["instrument_binding_sha256"] = binding[INSTRUMENT_BINDING_DIGEST_FIELD]
    receipt["artifacts"]["launch_plan"]["file_sha256"] = file_sha256(plan_path)
    receipt["artifacts"]["instrument_binding"]["file_sha256"] = file_sha256(binding_path)
    receipt["artifacts"]["instrumented_rollouts"]["file_sha256"] = file_sha256(instrumented_path)
    receipt[ROLLOUT_BINDING_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
    )
    _overwrite_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="scientific smpl root differs from the frozen"):
        collect_scientific_rollouts(
            receipt_paths=[receipt_path],
            schedule_manifest=fixture["schedule"],
            split_manifest=fixture["split"],
            reference_length_inventory=fixture["inventory"],
            analysis_protocol=fixture["analysis_protocol"],
            analysis_protocol_path=fixture["analysis_protocol_path"],
            analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
            expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            repo_root=fixture["root"],
        )


def test_finalizer_recovers_only_exact_enriched_half_state(tmp_path: Path) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    raw_path = fixture["outputs"] / "cell.jsonl"
    with raw_path.open("x", encoding="utf-8") as handle:
        for rollout_id in fixture["handshake"]["rollout_ids"]:
            handle.write(json.dumps(_record(rollout_id, materialized), sort_keys=True) + "\n")
    finalize_scientific_rollout(materialized)
    receipt_path = Path(fixture["registered_cell"]["rollout_binding_output_path"])
    receipt_path.unlink()
    recovered = finalize_scientific_rollout(materialized)
    assert recovered["rollout_count"] == len(fixture["handshake"]["rollout_ids"])

    receipt_path.unlink()
    instrumented = fixture["outputs"] / "cell.instrumented.jsonl"
    instrumented.write_bytes(instrumented.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="deterministic enrichment"):
        finalize_scientific_rollout(materialized)


@pytest.mark.parametrize(
    "invalid_line",
    [
        '{"rollout_id":"x","rollout_id":"y"}\n',
        '{"rollout_id":"x","value":NaN}\n',
        '{"rollout_id":"x","value":1e999}\n',
        "\n",
    ],
)
def test_finalizer_rejects_ambiguous_or_nonfinite_jsonl(
    tmp_path: Path,
    invalid_line: str,
) -> None:
    fixture = _planned_runtime(tmp_path)
    materialized = _materialize(fixture)
    (fixture["outputs"] / "cell.jsonl").write_text(invalid_line, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key|non-finite|blank"):
        finalize_scientific_rollout(materialized)


def test_collection_cli_uses_explicit_list_refuses_overwrite_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path)
    receipt_list = tmp_path / "receipts.json"
    receipt_list.write_text(json.dumps([str(receipt_path)]) + "\n", encoding="utf-8")
    output = tmp_path / "collection.json"
    arguments = [
        "--receipt-list",
        str(receipt_list),
        "--schedule",
        str(fixture["schedule_path"]),
        "--split",
        str(fixture["split_path"]),
        "--reference-length-inventory",
        str(fixture["inventory_path"]),
        "--analysis-protocol",
        str(fixture["analysis_protocol_path"]),
        "--analysis-protocol-lock",
        str(fixture["analysis_protocol_lock_path"]),
        "--expected-analysis-protocol-lock-sha256",
        fixture["analysis_protocol_lock"][ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        "--repo-root",
        str(fixture["root"]),
        "--output",
        str(output),
    ]
    assert collection_cli_main(arguments) == 0
    wrapper = json.loads(output.read_text(encoding="utf-8"))
    validate_rollout_collection(
        wrapper["rollout_collection"],
        schedule_manifest=fixture["schedule"],
        split_manifest=fixture["split"],
        reference_length_inventory=fixture["inventory"],
        analysis_protocol=fixture["analysis_protocol"],
        analysis_protocol_path=fixture["analysis_protocol_path"],
        analysis_protocol_lock_path=fixture["analysis_protocol_lock_path"],
        expected_analysis_protocol_lock_sha256=fixture["analysis_protocol_lock"][
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        repo_root=fixture["root"],
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        collection_cli_main(arguments)

    schedule_link = tmp_path / "schedule-link.json"
    schedule_link.symlink_to(fixture["schedule_path"])
    linked_arguments = list(arguments)
    linked_arguments[linked_arguments.index("--schedule") + 1] = str(schedule_link)
    linked_arguments[linked_arguments.index("--output") + 1] = str(tmp_path / "other.json")
    with pytest.raises(ValueError, match="may not be a symlink"):
        collection_cli_main(linked_arguments)
