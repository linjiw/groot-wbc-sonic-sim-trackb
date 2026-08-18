from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.research.lace.analysis_protocol import (
    ANALYSIS_PROTOCOL_DIGEST_FIELD,
    SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
    SCIENTIFIC_SIGNATURE_CONFIG,
    build_analysis_protocol,
)
from gear_sonic.research.lace.analysis_protocol_lock import (
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
    build_analysis_protocol_lock,
    execution_cell_binding,
)
from gear_sonic.research.lace.atlas import (
    ROLLOUT_SCHEMA_VERSION,
    SCIENTIFIC_SEED_SEMANTICS,
    build_atlas_manifest,
)
from gear_sonic.research.lace.instrument import (
    MEASUREMENT_FAMILY_DIGEST_FIELD,
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    build_measurement_family,
    build_scientific_instrument,
    episode_instrument_sha256,
    episode_measurement_family_sha256,
)
from gear_sonic.research.lace.instrument_collection import (
    ROLLOUT_COLLECTION_DIGEST_FIELD,
    ROLLOUT_COLLECTION_KIND,
    ROLLOUT_COLLECTION_SCHEMA_VERSION,
)
from gear_sonic.research.lace.instrument_runtime import (
    ROLLOUT_BINDING_DIGEST_FIELD,
    ROLLOUT_BINDING_KIND,
    ROLLOUT_BINDING_SCHEMA_VERSION,
    file_sha256,
    write_new_json,
)
from gear_sonic.research.lace.normalizer import fit_d_atlas_normalizer
from gear_sonic.research.lace.probes import ProbeThresholds, compute_episode_probe
from gear_sonic.research.lace.protocol_preflight import (
    PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
    build_protocol_preflight_receipt,
    build_protocol_preflight_request,
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
    validate_atlas_manifest,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS
from gear_sonic.research.lace.split import build_source_disjoint_split

MECHANISMS = ["contact", "slip"]
SCIENTIFIC_MECHANISMS = list(DEFAULT_MECHANISMS)
REPO_ROOT = Path(__file__).resolve().parents[2]


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
            "environment_type": "fixture:ManagerEnv",
            "termination_manager_type": "fixture:TerminationManager",
            "event_manager_type": "fixture:EventManager",
            "device": "cpu",
            "num_envs": num_envs,
            "step_dt_seconds": 0.02,
        },
    }


def _split() -> dict:
    motions = [
        {
            "motion_key": f"motion_{index:02d}__A{index:03d}",
            "release_filter_key": f"240101/motion_{index:02d}__A{index:03d}.pkl",
            "robot_path": f"/fixture/motion_{index:02d}__A{index:03d}.pkl",
            "stratum": f"kind_{index % 2}",
        }
        for index in range(20)
    ]
    return build_source_disjoint_split(motions, seed=9)


def _rollouts(split: dict) -> dict:
    atlas_motions = [
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    ][:2]
    assert len(atlas_motions) == 2
    policies = [
        {"id": "early", "checkpoint_sha256": "b" * 64},
        {"id": "late", "checkpoint_sha256": "c" * 64},
    ]
    episodes = []
    for motion_index, motion_key in enumerate(atlas_motions):
        for policy in policies:
            for seed in (11, 12):
                failed = seed == 11 or motion_index == 1
                scores = {"contact": float(motion_index == 0), "slip": float(motion_index == 1)}
                episodes.append(
                    {
                        "rollout_id": f"{motion_key}:{policy['id']}:{seed}",
                        "motion_key": motion_key,
                        "probe_policy_id": policy["id"],
                        "domain_randomization_seed": seed,
                        "initial_phase": 0.25,
                        "repeat_index": 0,
                        "partition": "D_atlas",
                        "failed": failed,
                        "mechanism_scores": scores,
                    }
                )
    return {
        "kind": "lace_probe_rollouts",
        "schema_version": 3,
        "artifact_mode": "contract_smoke",
        "data_origin": "synthetic_contract_test",
        "split_sha256": split["split_sha256"],
        "mechanism_names": MECHANISMS,
        "probe_policies": policies,
        "domain_randomization_seeds": [11, 12],
        "rollout_schedule": [
            {"domain_randomization_seed": 11, "initial_phase": 0.25, "repeat_index": 0},
            {"domain_randomization_seed": 12, "initial_phase": 0.25, "repeat_index": 0},
        ],
        "selected_motion_keys": atlas_motions,
        "signature_config": {
            "minimum_resolved_failures": 1,
            "mechanism_dirichlet_prior": 0.5,
            "credible_interval_level": 0.95,
        },
        "normalizer": {
            "kind": "identity_contract_smoke",
            "frozen": True,
            "fit_partition": "D_atlas",
            "mechanism_scales": {"contact": 1.0, "slip": 1.0},
        },
        "episodes": episodes,
    }


def _scientific_inventory(split: dict, atlas_motions: list[str]) -> dict:
    split_records = {record["motion_key"]: record for record in split["motions"]}
    motions = []
    for motion_key in atlas_motions:
        source_frames = 13
        motions.append(
            {
                "motion_key": motion_key,
                "split_robot_path": split_records[motion_key]["robot_path"],
                "source_path": split_records[motion_key]["robot_path"],
                "source_file_sha256": canonical_sha256({"fixture_motion": motion_key}),
                "source_num_frames": source_frames,
                "frame_axis": "root_trans_offset.shape[0]",
                "source_fps": {"numerator": 30, "denominator": 1},
                "target_num_frames": sonic_target_frame_count(source_frames, 30, 50),
            }
        )
    inventory = {
        "kind": REFERENCE_LENGTH_KIND,
        "schema_version": REFERENCE_LENGTH_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "pilot_status": None,
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "partition": "D_atlas",
        "selected_motion_keys": atlas_motions,
        "selection_complete_for_d_atlas": True,
        "motion_count": len(atlas_motions),
        "target_fps": 50,
        "runtime_contract": {
            "target_fps": 50,
            "sim_fps": 50,
            "motion_fps_scale": {"numerator": 1, "denominator": 1},
            "max_len": -1,
            "reference_num_steps_equals_target_num_frames": True,
            "float32_runtime_equivalence_scope": (
                "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
            ),
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
                        "motion_key": row["motion_key"],
                        "source_file_sha256": row["source_file_sha256"],
                    }
                    for row in motions
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


def _scientific_runtime_realization(row: dict, schedule_sha256: str) -> dict:
    event_configuration = {
        "mode_order": ["startup", "reset"],
        "modes": [
            {"mode": "startup", "terms": []},
            {"mode": "reset", "terms": []},
        ],
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
        "interval_events_instrumented": False,
    }
    return {
        "kind": RUNTIME_REALIZATION_KIND,
        "schema_version": RUNTIME_REALIZATION_SCHEMA_VERSION,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "runtime_rng_seed_semantics": SCIENTIFIC_SEED_SEMANTICS,
        "interval_event_policy": ATLAS_V1_INTERVAL_EVENT_POLICY,
        "resolved_event_configuration_sha256": canonical_sha256(event_configuration),
        "resolved_event_configuration": event_configuration,
        "ordered_names": {
            "body_names": ["pelvis"],
            "joint_names": ["hip"],
            "action_joint_names": ["hip"],
            "reference_body_names": ["pelvis"],
        },
        "realized_parameters": {"masses": [1.0]},
        "post_reset_state": {"root_position_w": [0.0, 0.0, 1.0]},
        "scheduled_reference": {
            "identity": {
                "schedule_sha256": schedule_sha256,
                "motion_key": row["motion_key"],
                "domain_randomization_seed": row["domain_randomization_seed"],
                "runtime_rng_seed": row["runtime_rng_seed"],
                "runtime_rng_seed_readback": row["runtime_rng_seed"],
                "phase_id": row["phase_id"],
                "target_fraction": row["target_fraction"],
                "realized_fraction": row["realized_fraction"],
                "reference_start_step": row["start_step"],
                "reference_num_steps": row["reference_num_steps"],
                "repeat_index": row["repeat_index"],
            },
            "state": {"command_time_step": 0},
        },
    }


def _scientific_termination_trace() -> dict:
    # Deliberately nonalphabetic: JSONL emission uses sort_keys=True, while
    # raw matrix columns retain this explicit manager order.
    term_names = ["time_out", "anchor_pos"]
    time_out_flags = [True, False]
    term_configs = [
        {
            "term_name": "time_out",
            "callable": "isaaclab.envs.mdp.terminations:time_out",
            "time_out": True,
            "params": {},
        },
        {
            "term_name": "anchor_pos",
            "callable": "gear_sonic.envs.manager_env.mdp.terminations:bad_anchor_pos",
            "time_out": False,
            "params": {"threshold": 0.5},
        },
    ]
    contract = {
        "kind": TERMINATION_TRACE_KIND,
        "schema_version": TERMINATION_TRACE_SCHEMA_VERSION,
        "algorithm": TERMINATION_TRACE_ALGORITHM,
        "manager_type": "isaaclab.managers.termination_manager:TerminationManager",
        "manager_compute_source_sha256": "a" * 64,
        "instrument_compute_source_sha256": "b" * 64,
        "term_names": term_names,
        "time_out_flags": time_out_flags,
        "term_configs": term_configs,
        "legacy_term_dones_semantics": "last_trigger_wins_stale_rows_preserved",
        "raw_trace_semantics": "ordered_independent_values_from_single_manager_evaluation",
        "freshness_semantics": "one_compute_one_consume_common_step_counter_bound",
    }
    contract["term_config_sha256"] = canonical_sha256({"terms": contract["term_configs"]})
    return {
        "termination_semantics": "instrumented_single_evaluation_ordered_raw_boolean_matrix",
        "termination_multi_hot_available": True,
        "termination_terms": {
            "anchor_pos": {
                "occurred": True,
                "incidence_rate": 0.5,
                "onset_index": 1,
                "onset_time_seconds": 0.02,
            },
            "time_out": {
                "occurred": False,
                "incidence_rate": 0.0,
                "onset_index": None,
                "onset_time_seconds": None,
            },
        },
        "buffered_step_count": 2,
        "termination_multi_hot": {
            "term_names": term_names,
            "time_out_flags": time_out_flags,
            "values": [[False, False], [False, True]],
            "trace_generations": [1, 2],
            "common_step_counters": [1, 2],
            "contract": contract,
            "contract_sha256": canonical_sha256(contract),
        },
    }


def _scientific_thresholds() -> dict:
    return asdict(ProbeThresholds())


def _scientific_score_window() -> dict:
    return {
        "rule": "fixed_window_ending_at_first_failure_or_censored_end_v1",
        "score_window_seconds": ProbeThresholds().score_window_seconds,
        "timestep_seconds": 0.02,
        "sample_count_rule": "max(1,floor(score_window_seconds/timestep_seconds+1e-12))",
    }


def _scientific_sensor_semantics() -> dict:
    return {
        "command_name": "motion",
        "robot_name": "robot",
        "joint_action_name": "joint_pos",
        "contact_sensor_name": "contact_forces",
        "foot_body_names": ["left_foot", "right_foot"],
        "contact_force_threshold": 1.0,
        "ground_normal_axis": 2,
        "actual_contact": "latest_net_forces_w_history_norm_threshold",
        "reference_contact": "sonic_feet_channel_ground_height_rule",
        "foot_slip": "link_origin_velocity_tangent_to_configured_plane",
        "torque": "requested_action_target_minus_joint_state_and_applied_joint_effort",
        "joint_limits": "live_soft_joint_position_limits",
    }


def _scientific_probe(row: dict) -> dict:
    result = compute_episode_probe(
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
    episode = dict(result.episode_diagnostics)
    reference_start = row["start_step"]
    reference_steps = row["reference_num_steps"]
    denominator = max(reference_steps - 1, 1)
    failure_index = int(episode["first_failure_index"])
    reference_failure = reference_start + failure_index
    reference_end = min(reference_start + episode["num_steps"] - 1, reference_steps - 1)
    episode.update(
        {
            "reference_start_step": reference_start,
            "reference_num_steps": reference_steps,
            "reference_end_step": reference_end,
            "reference_progress_at_end": reference_end / denominator,
            "reference_failure_step": reference_failure,
            "reference_progress_to_failure": reference_failure / denominator,
            "failure_progress_censored": False,
        }
    )
    return {
        "scores": dict(result.mechanism_scores),
        "onsets": dict(result.onset_times_seconds),
        "onset_unit": "seconds",
        "diagnostics": {name: dict(values) for name, values in result.diagnostics.items()},
        "episode_diagnostics": episode,
    }


def _synthetic_receipt_collection(
    *,
    split: dict,
    inventory: dict,
    schedule: dict,
    family: dict,
    cell_instruments: dict,
    instrument_by_rollout_id: dict,
    episodes: list[dict],
    analysis_protocol: dict,
    analysis_protocol_path: Path,
    analysis_protocol_lock: dict,
    analysis_protocol_lock_path: Path,
) -> dict:
    artifact_names = (
        "runtime_handshake",
        "launch_plan",
        "checkpoint",
        "checkpoint_config",
        "instrument",
        "instrument_binding",
        "raw_rollouts",
        "instrumented_rollouts",
        "schedule_lock",
        "schedule",
        "schedule_spec",
        "split_manifest",
        "reference_length_inventory",
        "analysis_protocol",
    )
    cell_fields = (
        "probe_policy_id",
        "checkpoint_sha256",
        "domain_randomization_seed",
        "runtime_rng_seed",
        "phase_id",
        "target_fraction",
        "repeat_index",
    )
    rows_by_cell: dict[tuple, list[dict]] = {}
    for row in schedule["rollouts"]:
        key = tuple(row[field] for field in cell_fields)
        rows_by_cell.setdefault(key, []).append(row)
    receipts = []
    episode_by_id = {episode["rollout_id"]: episode for episode in episodes}
    for cell_index, rows in enumerate(rows_by_cell.values()):
        cell = {field: rows[0][field] for field in cell_fields}
        instrument = instrument_by_rollout_id[rows[0]["rollout_id"]]
        instrument_digest = episode_instrument_sha256(instrument)
        artifacts = {
            name: {
                "path": f"/synthetic-scientific-cell-{cell_index}/{name}.json",
                "file_sha256": canonical_sha256({"cell_index": cell_index, "artifact": name}),
            }
            for name in artifact_names
        }
        artifacts["analysis_protocol"] = {
            "path": str(analysis_protocol_path),
            "file_sha256": file_sha256(analysis_protocol_path),
        }
        receipt = {
            "kind": ROLLOUT_BINDING_KIND,
            "schema_version": ROLLOUT_BINDING_SCHEMA_VERSION,
            "scientific_use": True,
            "cell": cell,
            "schedule_sha256": schedule["schedule_sha256"],
            "split_sha256": split["split_sha256"],
            "split_selection_sha256": split["selection_sha256"],
            "rollout_ids": [row["rollout_id"] for row in rows],
            "rollout_count": len(rows),
            "rollout_order": "frozen_handshake_rollout_id_order",
            "launch_plan_sha256": canonical_sha256({"plan": cell_index}),
            "runtime_handshake_sha256": canonical_sha256({"handshake": cell_index}),
            "instrument_binding_sha256": canonical_sha256({"binding": cell_index}),
            "instrument_manifest_sha256": instrument["instrument_manifest_sha256"],
            "episode_instrument_sha256": instrument_digest,
            "measurement_family": family,
            "measurement_family_manifest_sha256": family[MEASUREMENT_FAMILY_DIGEST_FIELD],
            "episode_measurement_family_sha256": episode_measurement_family_sha256(family),
            "source_bundle_sha256": instrument["source_bundle_sha256"],
            "dataset_binding_sha256": canonical_sha256({"dataset": "fixture"}),
            "robot_contract_readback_sha256": episode_by_id[rows[0]["rollout_id"]][
                "robot_contract_readback_sha256"
            ],
            "analysis_protocol_path": str(analysis_protocol_path),
            "analysis_protocol_file_sha256": file_sha256(analysis_protocol_path),
            ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
            "artifacts": artifacts,
        }
        receipt[ROLLOUT_BINDING_DIGEST_FIELD] = canonical_sha256(
            receipt,
            digest_field=ROLLOUT_BINDING_DIGEST_FIELD,
        )
        for row in rows:
            episode_by_id[row["rollout_id"]].update(
                {
                    "instrument_manifest_sha256": instrument["instrument_manifest_sha256"],
                    "instrument_sidecar_file_sha256": artifacts["instrument"]["file_sha256"],
                    "instrument_binding_sha256": receipt["instrument_binding_sha256"],
                }
            )
        receipts.append(
            {
                "cell": cell,
                "receipt_path": execution_cell_binding(
                    analysis_protocol_lock,
                    cell=cell,
                )["rollout_binding_output_path"],
                "receipt_file_sha256": canonical_sha256({"receipt_file": cell_index}),
                "rollout_binding_sha256": receipt[ROLLOUT_BINDING_DIGEST_FIELD],
                "receipt": receipt,
            }
        )
    collection = {
        "kind": ROLLOUT_COLLECTION_KIND,
        "schema_version": ROLLOUT_COLLECTION_SCHEMA_VERSION,
        "artifact_mode": "scientific",
        "scientific_use": True,
        "schedule_sha256": schedule["schedule_sha256"],
        "split_sha256": split["split_sha256"],
        "split_selection_sha256": split["selection_sha256"],
        "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "analysis_protocol": analysis_protocol,
        "analysis_protocol_path": str(analysis_protocol_path),
        "analysis_protocol_file_sha256": file_sha256(analysis_protocol_path),
        ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
        "analysis_protocol_lock": analysis_protocol_lock,
        "analysis_protocol_lock_path": str(analysis_protocol_lock_path),
        "analysis_protocol_lock_file_sha256": file_sha256(analysis_protocol_lock_path),
        ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: analysis_protocol_lock[
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
        ],
        "measurement_family": family,
        "measurement_family_sha256": episode_measurement_family_sha256(family),
        "cell_instruments": cell_instruments,
        "receipts": receipts,
        "episodes": episodes,
        "cell_count": len(receipts),
        "rollout_count": len(episodes),
    }
    collection[ROLLOUT_COLLECTION_DIGEST_FIELD] = canonical_sha256(
        collection,
        digest_field=ROLLOUT_COLLECTION_DIGEST_FIELD,
    )
    return collection


def _scientific_rollouts(split: dict, tmp_path: Path) -> tuple[dict, dict, dict]:
    atlas_motions = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    checkpoint_paths = {}
    checkpoint_config_paths = {}
    policies = []
    for policy_id in ("early", "late"):
        checkpoint_directory = tmp_path / f"{policy_id}-checkpoint"
        checkpoint_directory.mkdir()
        checkpoint_path = checkpoint_directory / "model.pt"
        checkpoint_path.write_bytes(f"{policy_id} checkpoint\n".encode())
        config_path = write_new_json(
            checkpoint_directory / "config.json",
            {"probe_policy_id": policy_id},
        )
        checkpoint_paths[policy_id] = checkpoint_path
        checkpoint_config_paths[policy_id] = config_path
        policies.append({"id": policy_id, "checkpoint_sha256": file_sha256(checkpoint_path)})
    inventory = _scientific_inventory(split, atlas_motions)
    schedule = build_rollout_schedule(
        split,
        reference_length_inventory=inventory,
        probe_policies=policies,
        domain_randomization_seeds=[11],
        phase_targets=[{"phase_id": "start", "target_fraction": 0.0}],
        repeats=3,
    )
    locked_inputs = tmp_path / "locked-inputs"
    schedule_path = write_new_json(locked_inputs / "schedule.json", schedule)
    split_path = write_new_json(locked_inputs / "split.json", split)
    inventory_path = write_new_json(locked_inputs / "inventory.json", inventory)
    schedule_spec = {
        "schema_version": 2,
        "reference_length_inventory_sha256": inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "probe_policies": policies,
        "domain_randomization_seeds": [11],
        "phase_targets": [{"phase_id": "start", "target_fraction": 0.0}],
        "repeats": 3,
    }
    schedule_spec_path = write_new_json(
        locked_inputs / "schedule-spec.json",
        schedule_spec,
    )
    schedule_lock_path = write_new_json(
        locked_inputs / "schedule-lock.json",
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
    rows_by_cell: dict[tuple, list[dict]] = {}
    for row in schedule["rollouts"]:
        cell = (
            row["probe_policy_id"],
            row["domain_randomization_seed"],
            row["phase_id"],
            row["repeat_index"],
        )
        rows_by_cell.setdefault(cell, []).append(row)

    cell_instruments = {}
    instrument_by_rollout_id = {}
    measurement_family = None
    event_configuration = _scientific_runtime_realization(
        schedule["rollouts"][0], schedule["schedule_sha256"]
    )["resolved_event_configuration"]
    termination_contract = _scientific_termination_trace()["termination_multi_hot"]["contract"]
    thresholds = _scientific_thresholds()
    threshold_digest = canonical_sha256(thresholds)
    score_window = _scientific_score_window()
    sensor_semantics = _scientific_sensor_semantics()
    for cell_index, rows in enumerate(rows_by_cell.values()):
        first_row = rows[0]
        output_path = f"/data/lace/cell-{cell_index}.jsonl"
        expected_motion_keys = [row["motion_key"] for row in rows]
        max_render_steps = max(row["reference_num_steps"] - row["start_step"] for row in rows) + 2
        recorder_term = {
            "enabled": True,
            "allow_append_existing": False,
            "output_path": output_path,
        }
        cache_suffixes = (
            "tmp",
            "xdg-cache",
            "isaaclab-usd-cache",
            "cuda-cache",
            "torch-home",
            "omni-user-cache",
        )
        cache_environment = {
            key: f"/data/lace/runtime/cell-{cell_index}/{suffix}"
            for key, suffix in zip(
                SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
                cache_suffixes,
                strict=True,
            )
        }
        instrument = build_scientific_instrument(
            schedule_sha256=schedule["schedule_sha256"],
            probe_thresholds=thresholds,
            recorder_config={
                "resolved_hydra_term": recorder_term,
                "runtime": {
                    "term_type": "fixture:LaceIsaacRecorderTerm",
                    "num_envs": len(rows),
                    "step_dt_seconds": 0.02,
                    "effective_probe_thresholds_sha256": threshold_digest,
                    "cell_identity": {
                        "schedule_sha256": schedule["schedule_sha256"],
                        "checkpoint_sha256": first_row["checkpoint_sha256"],
                        "probe_policy_id": first_row["probe_policy_id"],
                        "rollout_ids": [row["rollout_id"] for row in rows],
                    },
                },
            },
            resolved_hydra_config={
                "checkpoint": str(checkpoint_paths[first_row["probe_policy_id"]]),
                "seed": first_row["runtime_rng_seed"],
                "num_envs": len(rows),
                "headless": True,
                "run_eval_loop": True,
                "run_once": True,
                "use_encoder": "g1",
                "eval_callbacks": [],
                "lace_scientific_instrument_required": True,
                "max_render_steps": max_render_steps,
                "manager_env": {
                    "config": {"terrain_type": "plane", "render_results": False},
                    "commands": {
                        "motion": {
                            "atlas_probe_mode": True,
                            "atlas_probe_schedule_sha256": schedule["schedule_sha256"],
                            "atlas_probe_assignments": rows,
                            "filter_motion_keys": expected_motion_keys,
                            "motion_lib_cfg": {
                                "adaptive_sampling": {"enable": False},
                                "filter_motion_keys": expected_motion_keys,
                                "motion_file": "/data/lace/robot",
                                "smpl_motion_file": "/data/lace/smpl",
                            },
                        }
                    },
                    "observations": {
                        "policy": {"enable_corruption": False},
                        "tokenizer": {"enable_corruption": False},
                    },
                    "recorders": {"failure_atlas": recorder_term},
                },
            },
            environment_fingerprint={
                "isaaclab": "2.3.2",
                "python": "3.11",
                "process_argv": [
                    "/fixture/gear_sonic/eval_agent_trl.py",
                    f"++lace_cell_index={cell_index}",
                ],
                "scientific_cache_environment": cache_environment,
                "scientific_cache_environment_semantics": (SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS),
            },
            sensor_semantics=sensor_semantics,
            termination_predicates=termination_contract,
            score_window_config=score_window,
            domain_randomization_config=event_configuration,
            git_commit="9" * 40,
            repo_root=REPO_ROOT,
            source_paths=["tests/research/test_lace_atlas.py"],
        )
        family = build_measurement_family(instrument)
        if measurement_family is None:
            measurement_family = family
        else:
            assert family == measurement_family
        instrument_sha256 = episode_instrument_sha256(instrument)
        cell_instruments[instrument_sha256] = instrument
        for row in rows:
            instrument_by_rollout_id[row["rollout_id"]] = instrument

    assert measurement_family is not None
    first_instrument = next(iter(cell_instruments.values()))
    analysis_protocol = build_analysis_protocol(
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        probe_thresholds=thresholds,
        score_window_config=score_window,
        sensor_semantics=sensor_semantics,
        termination_predicates=termination_contract,
        domain_randomization_config=event_configuration,
        git_commit="9" * 40,
        source_bundle_sha256=first_instrument["source_bundle_sha256"],
    )
    analysis_protocol_path = tmp_path / "analysis-protocol.json"
    first_cell_rows = next(iter(rows_by_cell.values()))
    first_cell = {
        field: first_cell_rows[0][field]
        for field in (
            "probe_policy_id",
            "checkpoint_sha256",
            "domain_randomization_seed",
            "runtime_rng_seed",
            "phase_id",
            "target_fraction",
            "repeat_index",
        )
    }
    first_policy = first_cell["probe_policy_id"]
    preflight_request_path = tmp_path / "analysis-protocol.preflight-request.json"
    preflight_receipt_path = tmp_path / "analysis-protocol.preflight-receipt.json"
    preflight_plan_path = tmp_path / "analysis-protocol.preflight-plan.json"
    preflight_dataset_binding = {
        "robot": {"motion_root": "/data/lace/robot"},
        "smpl": {"motion_root": "/data/lace/smpl"},
    }
    preflight_request = build_protocol_preflight_request(
        schedule_lock_path=schedule_lock_path,
        schedule_path=schedule_path,
        schedule_manifest=schedule,
        split_path=split_path,
        split_manifest=split,
        reference_length_inventory_path=inventory_path,
        reference_length_inventory=inventory,
        cell=first_cell,
        rollout_ids=[row["rollout_id"] for row in first_cell_rows],
        checkpoint_bundle={
            "checkpoint_path": str(checkpoint_paths[first_policy]),
            "checkpoint_sha256": file_sha256(checkpoint_paths[first_policy]),
            "config_path": str(checkpoint_config_paths[first_policy]),
            "config_sha256": file_sha256(checkpoint_config_paths[first_policy]),
        },
        dataset_binding_sha256=canonical_sha256(preflight_dataset_binding),
        source_bundle=first_instrument["source_bundle"],
        git_commit="9" * 40,
        launch_plan_path=preflight_plan_path,
        recorder_output_path=tmp_path / "analysis-protocol.preflight-raw.jsonl",
        analysis_protocol_output_path=analysis_protocol_path,
        preflight_receipt_output_path=preflight_receipt_path,
        repo_root=REPO_ROOT,
    )
    write_new_json(preflight_request_path, preflight_request)
    preflight_plan = {
        "schedule_sha256": schedule["schedule_sha256"],
        "cell": first_cell,
        "rollout_ids": [row["rollout_id"] for row in first_cell_rows],
        "checkpoint_bundle": preflight_request["checkpoint_bundle"],
        "dataset_binding": preflight_dataset_binding,
        "protocol_preflight": {
            "request_path": str(preflight_request_path),
            "request_file_sha256": file_sha256(preflight_request_path),
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: preflight_request[
                PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
            ],
        },
    }
    preflight_cache = {
        key: f"/data/lace/preflight-cache/{key.lower()}"
        for key in SCIENTIFIC_CACHE_ENVIRONMENT_KEYS
    }
    preflight_plan["command"] = [
        "/usr/bin/python3",
        str(REPO_ROOT / "gear_sonic" / "eval_agent_trl.py"),
        "++lace_protocol_preflight=true",
    ]
    preflight_plan["launch_environment"] = {
        "PYTHONPATH": str(REPO_ROOT),
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
        "checkpoint": preflight_request["checkpoint_bundle"]["checkpoint_path"],
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
                        "motion_file": preflight_dataset_binding["robot"]["motion_root"],
                        "smpl_motion_file": preflight_dataset_binding["smpl"]["motion_root"],
                    },
                }
            },
        },
    }
    build_protocol_preflight_receipt(
        request_path=preflight_request_path,
        launch_plan_path=preflight_plan_path,
        live_inputs={
            "probe_thresholds": thresholds,
            "probe_thresholds_sha256": canonical_sha256(thresholds),
            "termination_predicates": termination_contract,
            "termination_predicates_sha256": canonical_sha256(termination_contract),
            "sensor_semantics": sensor_semantics,
            "score_window_config": score_window,
            "domain_randomization_config": event_configuration,
            "resolved_recorder_config": preflight_failure_config,
            "recorder_term_type": "fixture:LaceIsaacRecorderTerm",
            "recorder_record_count": 0,
            "num_envs": len(first_cell_rows),
            "step_dt_seconds": 0.02,
            "environment_fingerprint": _preflight_environment_fingerprint(
                preflight_plan,
                num_envs=len(first_cell_rows),
            ),
        },
        resolved_hydra_config=preflight_resolved_config,
        repo_root=REPO_ROOT,
    )
    execution_root = tmp_path / "scientific-executions"
    execution_root.mkdir()
    runtime_storage_root = tmp_path / "scientific-runtime-cache"
    runtime_storage_root.mkdir()
    analysis_protocol_lock = build_analysis_protocol_lock(
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_preflight_receipt_path=preflight_receipt_path,
        schedule_lock_path=schedule_lock_path,
        checkpoint_config_paths=checkpoint_config_paths,
        execution_root=execution_root,
        runtime_storage_root=runtime_storage_root,
        repo_root=REPO_ROOT,
    )
    analysis_protocol_lock_path = write_new_json(
        tmp_path / "analysis-protocol-lock.json",
        analysis_protocol_lock,
    )
    measurement_family_sha256 = episode_measurement_family_sha256(measurement_family)
    episodes = []
    robot_contract = {
        "kind": ROBOT_CONTRACT_READBACK_KIND,
        "schema_version": ROBOT_CONTRACT_READBACK_SCHEMA_VERSION,
        "capture_lifecycle": RUNTIME_REALIZATION_CAPTURE_LIFECYCLE,
        "ordered_body_names": ["pelvis"],
        "ordered_joint_names": ["hip"],
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
    for row in schedule["rollouts"]:
        instrument = instrument_by_rollout_id[row["rollout_id"]]
        realization = _scientific_runtime_realization(row, schedule["schedule_sha256"])
        probe = _scientific_probe(row)
        episode = {
            **row,
            **_scientific_termination_trace(),
            "reference_start_step": row["start_step"],
            "initial_phase": row["realized_fraction"],
            "schedule_entry_id": row["rollout_id"],
            "schedule_sha256": schedule["schedule_sha256"],
            "policy_id": row["probe_policy_id"],
            "completion_reason": "episode_end",
            "scientific_runtime_ready": True,
            "scientific_runtime_blockers": [],
            "domain_randomization_seed_semantics": DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
            "runtime_rng_seed_semantics": SCIENTIFIC_SEED_SEMANTICS,
            "runtime_rng_seed_readback": row["runtime_rng_seed"],
            "runtime_rng_seed_readback_source": "env.cfg.seed",
            "domain_randomization_realization": realization,
            "domain_randomization_realization_sha256": canonical_sha256(realization),
            "instrument_sha256": episode_instrument_sha256(instrument),
            "measurement_family_sha256": measurement_family_sha256,
            "measurement_family_manifest_sha256": measurement_family[
                MEASUREMENT_FAMILY_DIGEST_FIELD
            ],
            "probe_thresholds_sha256": instrument["probe_thresholds_sha256"],
            "probe_thresholds": instrument["probe_thresholds"],
            "termination_multi_hot_available": True,
            "robot_contract_readback": robot_contract,
            "robot_contract_readback_sha256": canonical_sha256(robot_contract),
            "failed": True,
            "mechanism_scores": dict(probe["scores"]),
            "probe": probe,
            "buffering_semantics": "bounded_full_episode_cpu_smoke",
            "max_episode_steps": row["reference_num_steps"],
            "foot_slip_velocity_proxy": ("link_origin_velocity_tangent_to_configured_plane"),
            "ground_normal_axis": 2,
        }
        episodes.append(episode)
    rollout_collection = _synthetic_receipt_collection(
        split=split,
        inventory=inventory,
        schedule=schedule,
        family=measurement_family,
        cell_instruments=cell_instruments,
        instrument_by_rollout_id=instrument_by_rollout_id,
        episodes=episodes,
        analysis_protocol=analysis_protocol,
        analysis_protocol_path=analysis_protocol_path,
        analysis_protocol_lock=analysis_protocol_lock,
        analysis_protocol_lock_path=analysis_protocol_lock_path,
    )
    normalizer = fit_d_atlas_normalizer(
        episodes,
        mechanism_names=SCIENTIFIC_MECHANISMS,
        minimum_positive_observations=20,
    )
    return (
        {
            "kind": "lace_probe_rollouts",
            "schema_version": ROLLOUT_SCHEMA_VERSION,
            "artifact_mode": "scientific",
            "data_origin": SCIENTIFIC_ROLLOUT_DATA_ORIGIN,
            "split_sha256": split["split_sha256"],
            "mechanism_names": SCIENTIFIC_MECHANISMS,
            "probe_policies": policies,
            "domain_randomization_seeds": [11],
            "rollout_schedule": None,
            "selected_motion_keys": atlas_motions,
            "signature_config": dict(SCIENTIFIC_SIGNATURE_CONFIG),
            "normalizer": normalizer,
            "measurement_family": None,
            "cell_instruments": {},
            "instrument": None,
            "analysis_protocol": analysis_protocol,
            "analysis_protocol_path": str(analysis_protocol_path),
            "analysis_protocol_file_sha256": file_sha256(analysis_protocol_path),
            ANALYSIS_PROTOCOL_DIGEST_FIELD: analysis_protocol[ANALYSIS_PROTOCOL_DIGEST_FIELD],
            "analysis_protocol_lock": analysis_protocol_lock,
            "analysis_protocol_lock_path": str(analysis_protocol_lock_path),
            "analysis_protocol_lock_file_sha256": file_sha256(analysis_protocol_lock_path),
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: analysis_protocol_lock[
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
            ],
            "rollout_collection": rollout_collection,
            "episodes": episodes,
        },
        schedule,
        inventory,
    )


def _sync_collection_episodes(rollouts: dict) -> None:
    collection = rollouts["rollout_collection"]
    collection["episodes"] = deepcopy(rollouts["episodes"])
    collection["rollout_count"] = len(collection["episodes"])
    collection[ROLLOUT_COLLECTION_DIGEST_FIELD] = canonical_sha256(
        collection,
        digest_field=ROLLOUT_COLLECTION_DIGEST_FIELD,
    )


def test_atlas_builder_is_deterministic_and_validates_complete_pairing() -> None:
    split = _split()
    rollouts = _rollouts(split)

    first = build_atlas_manifest(rollouts, split)
    second = build_atlas_manifest(rollouts, split)

    assert first == second
    assert first["motion_count"] == 2
    assert first["rollout_count"] == 8
    assert first["scientific_use"] is False
    assert first["selection_complete_for_d_atlas"] is False
    assert first["atlas_sha256"] == canonical_sha256(first, digest_field="atlas_sha256")
    validate_atlas_manifest(first)


def test_atlas_builder_rejects_non_common_random_numbers() -> None:
    split = _split()
    rollouts = _rollouts(split)
    rollouts["episodes"][4]["initial_phase"] = 0.75

    with pytest.raises(ValueError, match="declared rollout_schedule"):
        build_atlas_manifest(rollouts, split)


def test_atlas_builder_rejects_missing_dr_seed_coverage() -> None:
    split = _split()
    rollouts = _rollouts(split)
    rollouts["episodes"] = [
        episode
        for episode in rollouts["episodes"]
        if not (episode["probe_policy_id"] == "late" and episode["domain_randomization_seed"] == 12)
    ]

    with pytest.raises(ValueError, match="declared rollout_schedule"):
        build_atlas_manifest(rollouts, split)


def test_atlas_builder_rejects_motion_from_controller_or_test_partition() -> None:
    split = _split()
    rollouts = _rollouts(split)
    forbidden = next(record for record in split["motions"] if record["partition"] == "D_test")
    mutated = deepcopy(rollouts)
    mutated["episodes"][0]["motion_key"] = forbidden["motion_key"]

    with pytest.raises(ValueError, match="not D_atlas"):
        build_atlas_manifest(mutated, split)


def test_atlas_applies_declared_normalizer_scales() -> None:
    split = _split()
    rollouts = _rollouts(split)
    rollouts["normalizer"]["mechanism_scales"] = {"contact": 2.0, "slip": 1.0}

    atlas = build_atlas_manifest(rollouts, split)

    assert all(signature["mechanism_scales"] == [2.0, 1.0] for signature in atlas["signatures"])


def test_scientific_atlas_requires_and_binds_independent_exact_schedule(tmp_path: Path) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)

    with pytest.raises(ValueError, match="requires --schedule"):
        build_atlas_manifest(rollouts, split)
    with pytest.raises(ValueError, match="requires --reference-length-inventory"):
        build_atlas_manifest(rollouts, split, schedule)

    wrong_inventory = deepcopy(inventory)
    wrong_inventory["motions"][0]["source_num_frames"] = 14
    wrong_inventory["motions"][0]["target_num_frames"] = sonic_target_frame_count(14, 30, 50)
    wrong_inventory[REFERENCE_LENGTH_DIGEST_FIELD] = canonical_sha256(
        wrong_inventory,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="binding does not match the supplied inventory"):
        build_atlas_manifest(rollouts, split, schedule, wrong_inventory)

    atlas = build_atlas_manifest(rollouts, split, schedule, inventory)

    assert atlas["scientific_use"] is True
    assert atlas["rollout_schedule"] is None
    assert atlas["rollout_schedule_sha256"] == schedule["schedule_sha256"]
    assert atlas["rollout_schedule_summary"]["rollout_count"] == len(schedule["rollouts"])
    assert atlas["measurement_family"]["schedule_sha256"] == schedule["schedule_sha256"]
    assert atlas["cell_instrument_count"] == 6
    assert len(atlas["cell_instruments"]) == 6
    assert len({episode["instrument_sha256"] for episode in rollouts["episodes"]}) == 6
    validate_atlas_manifest(atlas)


def test_scientific_atlas_rejects_runtime_tuple_or_schedule_coverage_drift(
    tmp_path: Path,
) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    drifted = deepcopy(rollouts)
    drifted["episodes"][0]["runtime_rng_seed"] ^= 1
    drifted["normalizer"] = fit_d_atlas_normalizer(
        drifted["episodes"],
        mechanism_names=SCIENTIFIC_MECHANISMS,
        minimum_positive_observations=20,
    )
    _sync_collection_episodes(drifted)
    with pytest.raises(ValueError, match="runtime_rng_seed does not match"):
        build_atlas_manifest(drifted, split, schedule, inventory)

    missing = deepcopy(rollouts)
    missing["episodes"].pop()
    missing["normalizer"] = fit_d_atlas_normalizer(
        missing["episodes"],
        mechanism_names=SCIENTIFIC_MECHANISMS,
        minimum_positive_observations=20,
    )
    _sync_collection_episodes(missing)
    with pytest.raises(ValueError, match="frozen schedule"):
        build_atlas_manifest(missing, split, schedule, inventory)


def test_scientific_atlas_rejects_policy_dependent_runtime_realization(tmp_path: Path) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    drifted = deepcopy(rollouts)
    first = drifted["episodes"][0]
    first["domain_randomization_realization"]["post_reset_state"]["root_position_w"][0] = 0.1
    first["domain_randomization_realization_sha256"] = canonical_sha256(
        first["domain_randomization_realization"]
    )
    _sync_collection_episodes(drifted)

    with pytest.raises(ValueError, match="policy-dependent runtime realization"):
        build_atlas_manifest(drifted, split, schedule, inventory)


def test_scientific_atlas_rejects_missing_or_tampered_multi_hot_trace(tmp_path: Path) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    missing = deepcopy(rollouts)
    del missing["episodes"][0]["termination_multi_hot"]
    _sync_collection_episodes(missing)

    with pytest.raises(ValueError, match="termination_multi_hot must be a mapping"):
        build_atlas_manifest(missing, split, schedule, inventory)

    tampered = deepcopy(rollouts)
    tampered["episodes"][0]["termination_multi_hot"]["contract"][
        "manager_compute_source_sha256"
    ] = ("d" * 64)
    _sync_collection_episodes(tampered)
    with pytest.raises(ValueError, match="contract_sha256 mismatch"):
        build_atlas_manifest(tampered, split, schedule, inventory)


def test_scientific_atlas_accepts_sorted_json_diagnostics_independent_of_column_order(
    tmp_path: Path,
) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    round_tripped = json.loads(json.dumps(rollouts, allow_nan=False, sort_keys=True))
    episode = round_tripped["episodes"][0]

    assert episode["termination_multi_hot"]["term_names"] == ["time_out", "anchor_pos"]
    assert list(episode["termination_terms"]) == ["anchor_pos", "time_out"]

    atlas = build_atlas_manifest(round_tripped, split, schedule, inventory)

    assert atlas["scientific_use"] is True


def test_scientific_atlas_rejects_all_false_or_skipped_termination_trace(
    tmp_path: Path,
) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    all_false = deepcopy(rollouts)
    episode = all_false["episodes"][0]
    episode["failed"] = False
    episode["termination_multi_hot"]["values"] = [[False, False], [False, False]]
    for term in episode["termination_terms"].values():
        term["occurred"] = False
    _sync_collection_episodes(all_false)
    with pytest.raises(ValueError, match="final row has no episode-ending predicate"):
        build_atlas_manifest(all_false, split, schedule, inventory)

    skipped = deepcopy(rollouts)
    episode = skipped["episodes"][0]
    episode["termination_multi_hot"]["values"] = [[False, False], [False, True]]
    episode["termination_multi_hot"]["trace_generations"] = [1, 3]
    episode["termination_multi_hot"]["common_step_counters"] = [1, 2]
    episode["buffered_step_count"] = 2
    _sync_collection_episodes(skipped)
    with pytest.raises(ValueError, match="trace_generations must be consecutive"):
        build_atlas_manifest(skipped, split, schedule, inventory)


def test_stored_atlas_validator_deeply_rejects_rehashed_invalid_instrument(
    tmp_path: Path,
) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    atlas = build_atlas_manifest(rollouts, split, schedule, inventory)
    tampered = deepcopy(atlas)
    old_key = next(iter(tampered["cell_instruments"]))
    instrument = tampered["cell_instruments"].pop(old_key)
    instrument["termination_multi_hot_available"] = False
    instrument["instrument_manifest_sha256"] = canonical_sha256(
        instrument,
        digest_field="instrument_manifest_sha256",
    )
    tampered["cell_instruments"][canonical_sha256(instrument)] = instrument
    tampered["atlas_sha256"] = canonical_sha256(tampered, digest_field="atlas_sha256")

    with pytest.raises(ValueError, match="verified independent termination multi-hot"):
        validate_atlas_manifest(tampered)


def test_stored_atlas_rejects_rehashed_receipt_to_instrument_swap(tmp_path: Path) -> None:
    split = _split()
    rollouts, schedule, inventory = _scientific_rollouts(split, tmp_path)
    atlas = build_atlas_manifest(rollouts, split, schedule, inventory)
    tampered = deepcopy(atlas)
    first, second = tampered["rollout_receipts"][:2]
    first["episode_instrument_sha256"], second["episode_instrument_sha256"] = (
        second["episode_instrument_sha256"],
        first["episode_instrument_sha256"],
    )
    tampered["atlas_sha256"] = canonical_sha256(tampered, digest_field="atlas_sha256")

    with pytest.raises(ValueError, match="mapped to the wrong instrument"):
        validate_atlas_manifest(tampered)
