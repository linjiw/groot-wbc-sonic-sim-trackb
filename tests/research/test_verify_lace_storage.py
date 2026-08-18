from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.research.lace.analysis_protocol_lock import (
    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
)
from gear_sonic.research.lace.atlas import build_atlas_manifest
from gear_sonic.research.lace.instrument_collection import (
    build_scientific_rollout_manifest,
    collect_scientific_rollouts,
)
from gear_sonic.research.lace.intervention_plan import build_intervention_plan
from gear_sonic.research.lace.panels import build_representation_blind_panels
from gear_sonic.research.lace.reference_lengths import build_reference_length_inventory
from gear_sonic.research.lace.schedule import build_rollout_schedule
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split
from scripts.research.build_bones_seed_official_cohort import build_cohort
from scripts.research.build_bones_seed_paired_manifest import (
    build_manifest as build_paired_manifest,
)
from scripts.research.verify_lace_storage import (
    ReadinessError,
    sha256_file,
    verify_artifact_lock,
    verify_readiness,
    verify_storage_roots,
)
from tests.research.test_lace_instrument_runtime import _completed_runtime_fixture


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_protocol_lock_digest(paths: dict[str, Path]) -> str:
    lock = json.loads(paths["protocol_lock"].read_text(encoding="utf-8"))
    return str(lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD])


def _storage_config(tmp_path: Path) -> Path:
    storage_root = tmp_path / "storage"
    dataset_root = tmp_path / "dataset"
    artifact_roots = {
        "atlases": storage_root / "atlases",
        "manifests": storage_root / "manifests",
    }
    for path in (storage_root, dataset_root, *artifact_roots.values()):
        path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "storage.json"
    _write_json(
        config_path,
        {
            "schema_version": 1,
            "storage_root": str(storage_root),
            "dataset_root": str(dataset_root),
            "artifact_roots": {name: str(path) for name, path in artifact_roots.items()},
        },
    )
    return config_path


def _split_manifest() -> dict:
    motions = [
        {
            "motion_key": f"walk_{index:03d}__A{index:03d}",
            "release_filter_key": f"240101/walk_{index:03d}__A{index:03d}.pkl",
            "duration_source_frames": 100 + index,
            "stratum": f"kind_{index % 3}",
        }
        for index in range(30)
    ]
    return build_source_disjoint_split(motions, seed=17, dataset={"name": "fixture"})


def _split_lock(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_path = tmp_path / "storage/manifests/split.json"
    split = _split_manifest()
    _write_json(artifact_path, split)

    cohort_path = tmp_path / "cohort.json"
    _write_json(cohort_path, {"kind": "fixture_cohort", "motions": ["walk"]})
    materialized_path = tmp_path / "dataset/dataset_manifest.json"
    paired_digest = "a" * 64
    _write_json(
        materialized_path,
        {"kind": "fixture_materialized", "output": {"paired_dataset_sha256": paired_digest}},
    )

    lock_path = tmp_path / "split_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "kind": "lace_split_artifact_lock",
            "artifact": {
                "path": str(artifact_path),
                "file_sha256": sha256_file(artifact_path),
                "selection_sha256": split["selection_sha256"],
                "split_sha256": split["split_sha256"],
            },
            "inputs": {
                "cohort_manifest": str(cohort_path),
                "cohort_manifest_sha256": sha256_file(cohort_path),
                "materialized_manifest": str(materialized_path),
                "materialized_manifest_sha256": sha256_file(materialized_path),
                "paired_dataset_sha256": paired_digest,
            },
        },
    )
    return lock_path, artifact_path, cohort_path, materialized_path


def _write_headline_pair(robot_dir: Path, smpl_dir: Path, key: str) -> None:
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    frames = 2
    joblib.dump(
        {
            key: {
                "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
                "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
                "dof": np.zeros((frames, 29), dtype=np.float32),
                "root_rot": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (frames, 1)),
                "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
                "fps": 30,
            }
        },
        robot_dir / f"{key}.pkl",
    )
    joblib.dump(
        {
            "pose_aa": np.zeros((3, 72), dtype=np.float32),
            "transl": np.zeros((3, 3), dtype=np.float32),
            "smpl_joints": np.zeros((3, 24, 3), dtype=np.float32),
            "fps": 50.0,
            "original_pose_aa": np.zeros((frames, 72), dtype=np.float32),
            "original_fps": 30.0,
        },
        smpl_dir / f"{key}.pkl",
    )


def _headline_split_lock(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    metadata_path = tmp_path / "metadata.csv"
    fieldnames = [
        "move_name",
        "filename",
        "move_duration_frames",
        "package",
        "category",
        "is_mirror",
        "move_g1_path",
    ]
    rows = []
    for index in range(10):
        key = f"walk_{index:03d}__A{index:03d}"
        rows.append(
            {
                "move_name": key,
                "filename": key,
                "move_duration_frames": "8",
                "package": "Locomotion",
                "category": f"Category {index % 2}",
                "is_mirror": "false",
                "move_g1_path": f"g1/csv/session/{key}.csv",
            }
        )
    with metadata_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    cohort = build_cohort(metadata_path, size=10, seed=37, duration_bins=2)
    cohort_path = tmp_path / "cohort.json"
    _write_json(cohort_path, cohort)
    g1_members = tmp_path / "g1-members.txt"
    smpl_members = tmp_path / "smpl-members.txt"
    g1_members.write_text(
        "".join(f"{record['g1_archive_member']}\n" for record in cohort["motions"]),
        encoding="utf-8",
    )
    smpl_members.write_text(
        "".join(f"{record['smpl_archive_member']}\n" for record in cohort["motions"]),
        encoding="utf-8",
    )

    dataset_root = tmp_path / "dataset"
    robot_dir = dataset_root / "robot_filtered"
    smpl_dir = dataset_root / "smpl_filtered"
    for record in cohort["motions"]:
        _write_headline_pair(robot_dir, smpl_dir, record["motion_key"])
    materialized = build_paired_manifest(cohort_path, robot_dir, smpl_dir)
    source_lock_path = tmp_path / "source-lock.json"
    g1_archive = {"bytes": 0, "sha256": "1" * 64}
    smpl_parts = [
        {
            "path": f"bones_seed_smpl/bones_seed_smpl.tar.part_a{chr(97 + index)}",
            "bytes": index,
            "sha256": f"{index + 2:x}" * 64,
        }
        for index in range(7)
    ]
    source_lock = {
        "schema_version": 1,
        "bones_seed": {
            "repo_id": cohort["source"]["repo_id"],
            "repo_type": "dataset",
            "revision": cohort["source"]["revision"],
            "files": {
                "g1.tar.gz": g1_archive,
                "metadata/seed_metadata_v004.csv": {
                    "bytes": metadata_path.stat().st_size,
                    "sha256": sha256_file(metadata_path),
                },
            },
        },
        "smpl": {"repo_id": "fixture", "revision": "2" * 40, "parts": smpl_parts},
    }
    _write_json(source_lock_path, source_lock)
    materialized["materialization"] = {
        "schema_version": 1,
        "generator": "scripts/research/materialize_bones_seed_official_cohort.py",
        "source_lock": {
            "path": str(source_lock_path),
            "sha256": sha256_file(source_lock_path),
        },
        "archives": {
            "g1": {"path": "g1.tar.gz", **g1_archive},
            "smpl_parts": smpl_parts,
            "all_full_archives_verified_before_extraction": True,
        },
        "member_manifests": {
            "g1_sha256": sha256_file(g1_members),
            "smpl_sha256": sha256_file(smpl_members),
            "exact_cohort_order": True,
        },
        "robot_transform": {
            "implementation": "fixture",
            "source_fps": 120,
            "target_fps": 30,
            "stride": 4,
        },
        "selective_regular_files_only": True,
        "source_lock_schema_version": 1,
    }
    materialized_path = dataset_root / "dataset_manifest.json"
    _write_json(materialized_path, materialized)

    dataset = {
        "cohort_manifest": str(cohort_path),
        "cohort_manifest_sha256": sha256_file(cohort_path),
        "dataset_root": str(dataset_root),
        "materialized_manifest": str(materialized_path),
        "materialized_manifest_sha256": sha256_file(materialized_path),
        "motion_count": 10,
        "paired_dataset_sha256": materialized["output"]["paired_dataset_sha256"],
    }
    variants = {record["motion_key"]: record for record in materialized["output"]["variants"]}
    enriched = []
    for record in cohort["motions"]:
        variant = variants[record["motion_key"]]
        enriched.append(
            {
                **record,
                "robot_path": str(dataset_root / variant["robot"]["path"]),
                "smpl_path": str(dataset_root / variant["smpl"]["path"]),
                "available_modalities": ["g1", "smpl"],
            }
        )
    split = build_source_disjoint_split(enriched, seed=41, dataset=dataset)
    split_path = tmp_path / "split.json"
    _write_json(split_path, split)
    preview = build_source_disjoint_split(cohort["motions"], seed=41)
    preview_summary = {
        name: {
            field: preview["partition_summary"][name][field]
            for field in ("motion_count", "source_group_count")
        }
        for name in preview["partition_summary"]
    }
    disposition = (
        "A prior unrelated contract smoke is ineligible and was not consulted for this grid."
    )
    protocol = {
        "schema_version": 2,
        "kind": "lace_headline_cohort_selection_protocol",
        "frozen": True,
        "scientific_use": True,
        "declared_before_headline_candidate_policy_outcomes": True,
        "outcome_access_permitted": False,
        "prior_rollout_disposition": disposition,
        "source": {
            "repo_id": cohort["source"]["repo_id"],
            "revision": cohort["source"]["revision"],
            "metadata_path": str(metadata_path),
            "metadata_bytes": metadata_path.stat().st_size,
            "metadata_sha256": sha256_file(metadata_path),
            "published_motion_count": 10,
            "eligible_motion_count": 10,
            "eligibility_rule": cohort["eligibility"]["rule"],
        },
        "selection": {
            "builder": "scripts/research/build_bones_seed_official_cohort.py",
            "builder_sha256": sha256_file(
                repo_root / "scripts/research/build_bones_seed_official_cohort.py"
            ),
            "filter_implementation": "gear_sonic/data_process/filter_and_copy_bones_data.py",
            "filter_implementation_sha256": sha256_file(
                repo_root / "gear_sonic/data_process/filter_and_copy_bones_data.py"
            ),
            "seed": 37,
            "duration_bins": 2,
            "stratification": cohort["selection"]["stratification"],
            "tie_breaker": cohort["selection"]["tie_breaker"],
            "candidate_size_grid": {
                "start_inclusive": 10,
                "stop_inclusive": 10,
                "step": 1,
            },
            "pre_outcome_constraints": {
                "minimum_d_atlas_motion_count": preview_summary["D_atlas"]["motion_count"],
                "maximum_d_atlas_motion_count": preview_summary["D_atlas"]["motion_count"],
                "minimum_d_geometry_plus_d_test_source_groups": sum(
                    preview_summary[name]["source_group_count"] for name in ("D_geometry", "D_test")
                ),
            },
            "selection_rule": "unique_grid_candidate_satisfying_all_pre_outcome_constraints",
            "selected_size": 10,
            "selection_sha256": cohort["selection"]["selection_sha256"],
        },
        "split_preview": {
            "builder": "gear_sonic.research.lace.split:build_source_disjoint_split",
            "builder_sha256": sha256_file(repo_root / "gear_sonic/research/lace/split.py"),
            "seed": 41,
            "source_group_count": 10,
            "partition_summary": preview_summary,
            "d_geometry_plus_d_test_source_groups": sum(
                preview_summary[name]["source_group_count"] for name in ("D_geometry", "D_test")
            ),
            "note": "fixture",
        },
        "artifacts": {
            "cohort_path": str(cohort_path),
            "cohort_file_sha256": sha256_file(cohort_path),
            "g1_member_list_path": str(g1_members),
            "g1_member_list_file_sha256": sha256_file(g1_members),
            "smpl_member_list_path": str(smpl_members),
            "smpl_member_list_file_sha256": sha256_file(smpl_members),
            "materializer_path": "scripts/research/materialize_bones_seed_official_cohort.py",
            "materializer_file_sha256": sha256_file(
                repo_root / "scripts/research/materialize_bones_seed_official_cohort.py"
            ),
            "paired_manifest_builder_path": "scripts/research/build_bones_seed_paired_manifest.py",
            "paired_manifest_builder_file_sha256": sha256_file(
                repo_root / "scripts/research/build_bones_seed_paired_manifest.py"
            ),
        },
        "scope": "fixture",
    }
    protocol["protocol_sha256"] = canonical_sha256(protocol, digest_field="protocol_sha256")
    protocol_path = tmp_path / "selection-protocol.json"
    _write_json(protocol_path, protocol)
    summary = {
        name: {
            field: split["partition_summary"][name][field]
            for field in ("motion_count", "source_group_count")
        }
        for name in split["partition_summary"]
    }
    lock = {
        "schema_version": 2,
        "kind": "lace_split_artifact_lock",
        "scientific_use": True,
        "declared_before_headline_candidate_policy_outcomes": True,
        "prior_rollout_disposition": disposition,
        "seed": 41,
        "artifact": {
            "path": str(split_path),
            "file_sha256": sha256_file(split_path),
            "selection_sha256": split["selection_sha256"],
            "split_sha256": split["split_sha256"],
        },
        "inputs": {
            "selection_protocol": str(protocol_path),
            "selection_protocol_sha256": sha256_file(protocol_path),
            "selection_protocol_self_sha256": protocol["protocol_sha256"],
            "cohort_manifest": str(cohort_path),
            "cohort_manifest_sha256": sha256_file(cohort_path),
            "materialized_manifest": str(materialized_path),
            "materialized_manifest_sha256": sha256_file(materialized_path),
            "paired_dataset_sha256": materialized["output"]["paired_dataset_sha256"],
        },
        "partition_summary": summary,
        "scope": "fixture",
    }
    lock_path = tmp_path / "headline-split-lock.json"
    _write_json(lock_path, lock)
    return lock_path, {
        "split": split_path,
        "protocol": protocol_path,
        "cohort": cohort_path,
        "materialized": materialized_path,
        "dataset_root": dataset_root,
    }


def _atlas_lock(tmp_path: Path) -> Path:
    split = _split_manifest()
    split_path = tmp_path / "storage/manifests/atlas_split.json"
    _write_json(split_path, split)
    motion_key = next(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    policy = {"id": "probe", "checkpoint_sha256": "b" * 64}
    rollouts = {
        "kind": "lace_probe_rollouts",
        "schema_version": 3,
        "artifact_mode": "contract_smoke",
        "data_origin": "synthetic_contract_test",
        "split_sha256": split["split_sha256"],
        "mechanism_names": ["contact", "slip"],
        "probe_policies": [policy],
        "domain_randomization_seeds": [11],
        "rollout_schedule": [
            {"domain_randomization_seed": 11, "initial_phase": 0.0, "repeat_index": 0}
        ],
        "selected_motion_keys": [motion_key],
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
        "episodes": [
            {
                "rollout_id": "fixture:probe:11",
                "motion_key": motion_key,
                "probe_policy_id": "probe",
                "domain_randomization_seed": 11,
                "initial_phase": 0.0,
                "repeat_index": 0,
                "partition": "D_atlas",
                "failed": True,
                "mechanism_scores": {"contact": 1.0, "slip": 0.0},
            }
        ],
    }
    atlas = build_atlas_manifest(rollouts, split)
    artifact_path = tmp_path / "storage/atlases/atlas.json"
    _write_json(artifact_path, atlas)
    lock_path = tmp_path / "atlas_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "kind": "lace_atlas_contract_smoke_lock",
            "artifact": {
                "path": str(artifact_path),
                "file_sha256": sha256_file(artifact_path),
                "atlas_sha256": atlas["atlas_sha256"],
            },
            "inputs": {
                "split_manifest": str(split_path),
                "split_manifest_sha256": sha256_file(split_path),
            },
        },
    )
    return lock_path


def _scientific_atlas_lock(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    fixture, _, receipt_path = _completed_runtime_fixture(tmp_path / "scientific-runtime")
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
    wrapper = build_scientific_rollout_manifest(
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
    atlas = build_atlas_manifest(
        wrapper,
        fixture["split"],
        fixture["schedule"],
        fixture["inventory"],
        repo_root=fixture["root"],
    )
    manifest_root = tmp_path / "scientific-lock-inputs"
    paths = {
        "atlas": manifest_root / "atlas.json",
        "rollout": manifest_root / "rollout-wrapper.json",
        "split": manifest_root / "split.json",
        "schedule": manifest_root / "schedule.json",
        "inventory": manifest_root / "inventory.json",
        "protocol": fixture["analysis_protocol_path"],
        "protocol_lock": fixture["analysis_protocol_lock_path"],
    }
    for name, payload in (
        ("atlas", atlas),
        ("rollout", wrapper),
        ("split", fixture["split"]),
        ("schedule", fixture["schedule"]),
        ("inventory", fixture["inventory"]),
    ):
        _write_json(paths[name], payload)
    lock_path = manifest_root / "atlas-lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 2,
            "kind": "lace_atlas_artifact_lock",
            "scientific_use": True,
            "artifact": {
                "path": str(paths["atlas"]),
                "file_sha256": sha256_file(paths["atlas"]),
                "atlas_sha256": atlas["atlas_sha256"],
            },
            "inputs": {
                "rollout_manifest": str(paths["rollout"]),
                "rollout_manifest_sha256": sha256_file(paths["rollout"]),
                "split_manifest": str(paths["split"]),
                "split_manifest_sha256": sha256_file(paths["split"]),
                "schedule_manifest": str(paths["schedule"]),
                "schedule_manifest_sha256": sha256_file(paths["schedule"]),
                "reference_length_inventory": str(paths["inventory"]),
                "reference_length_inventory_file_sha256": sha256_file(paths["inventory"]),
                "reference_length_inventory_sha256": fixture["inventory"]["inventory_sha256"],
                "analysis_protocol": str(paths["protocol"]),
                "analysis_protocol_file_sha256": sha256_file(paths["protocol"]),
                "analysis_protocol_sha256": fixture["analysis_protocol"][
                    "analysis_protocol_sha256"
                ],
                "analysis_protocol_lock": str(paths["protocol_lock"]),
                "analysis_protocol_lock_file_sha256": sha256_file(paths["protocol_lock"]),
                ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD: fixture["analysis_protocol_lock"][
                    ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD
                ],
            },
        },
    )
    paths["repo_root"] = fixture["root"]
    return lock_path, paths


def _panel_lock(tmp_path: Path) -> Path:
    split = _split_manifest()
    split_path = tmp_path / "storage/manifests/panel_split.json"
    _write_json(split_path, split)
    panels = build_representation_blind_panels(split, panel_count=4, seed=23)
    artifact_path = tmp_path / "storage/manifests/panels.json"
    _write_json(artifact_path, panels)
    lock_path = tmp_path / "panel_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "kind": "lace_source_panel_artifact_lock",
            "artifact": {
                "path": str(artifact_path),
                "file_sha256": sha256_file(artifact_path),
                "panel_sha256": panels["panel_sha256"],
                "split_sha256": split["split_sha256"],
                "split_selection_sha256": split["selection_sha256"],
            },
            "inputs": {
                "split_manifest": str(split_path),
                "split_manifest_sha256": sha256_file(split_path),
            },
        },
    )
    return lock_path


def _intervention_plan_lock(tmp_path: Path) -> tuple[Path, Path]:
    split = _split_manifest()
    split_path = tmp_path / "storage/manifests/intervention_split.json"
    _write_json(split_path, split)
    panels = build_representation_blind_panels(split, panel_count=4, seed=23)
    panel_path = tmp_path / "storage/manifests/intervention_panels.json"
    _write_json(panel_path, panels)
    motion_count = sum(record["partition"] == "D_curriculum" for record in split["motions"])
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
        "panel_seed": 23,
        "base_distribution_method": "uniform_over_canonically_ordered_motions_v1",
        "sequence_length_agnostic": True,
        "target_kl_nats": 0.02,
        "max_probability_ratio": 3.0,
        "kl_tolerance": 1e-12,
        "probability_tolerance": 1e-12,
        "bisection_iterations": 100,
        "maximum_added_exposure_range": 0.03,
        "interpretation": "storage verifier fixture",
    }
    protocol["protocol_sha256"] = canonical_sha256(
        protocol,
        digest_field="protocol_sha256",
    )
    protocol_path = tmp_path / "intervention_protocol.json"
    _write_json(protocol_path, protocol)
    plan = build_intervention_plan(split, panels, protocol)
    artifact_path = tmp_path / "storage/manifests/intervention_plan.json"
    _write_json(artifact_path, plan)
    lock_path = tmp_path / "intervention_plan_lock.json"
    _write_json(
        lock_path,
        {
            "schema_version": 1,
            "kind": "lace_rq1_intervention_plan_artifact_lock",
            "scientific_use": True,
            "declared_before_transfer_outcomes": True,
            "artifact": {
                "path": str(artifact_path),
                "file_sha256": sha256_file(artifact_path),
                "intervention_plan_sha256": plan["intervention_plan_sha256"],
            },
            "inputs": {
                "split_manifest": str(split_path),
                "split_manifest_sha256": sha256_file(split_path),
                "panel_manifest": str(panel_path),
                "panel_manifest_sha256": sha256_file(panel_path),
                "panel_sha256": panels["panel_sha256"],
                "intervention_protocol": str(protocol_path),
                "intervention_protocol_sha256": sha256_file(protocol_path),
                "protocol_sha256": protocol["protocol_sha256"],
            },
        },
    )
    return lock_path, artifact_path


def _schedule_lock(tmp_path: Path) -> tuple[Path, dict[str, Path], dict]:
    motion_root = tmp_path / "dataset/robot_filtered"
    motion_root.mkdir(parents=True, exist_ok=True)
    source_records = [
        {
            "motion_key": f"schedule_{index:03d}__A{index:03d}",
            "robot_path": str(motion_root / f"schedule_{index:03d}__A{index:03d}.pkl"),
            "duration_source_frames": 100 + index,
            "stratum": f"kind_{index % 3}",
        }
        for index in range(30)
    ]
    split = build_source_disjoint_split(source_records, seed=31, dataset={"name": "fixture"})
    selected = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )[:2]
    for key, frames in zip(selected, (262, 216), strict=True):
        joblib.dump(
            {
                key: {
                    "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
                    "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
                    "dof": np.zeros((frames, 29), dtype=np.float32),
                    "root_rot": np.zeros((frames, 4), dtype=np.float32),
                    "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
                    "fps": 30,
                }
            },
            motion_root / f"{key}.pkl",
            compress=True,
        )

    split_path = tmp_path / "storage/manifests/schedule_split.json"
    _write_json(split_path, split)
    inventory = build_reference_length_inventory(
        split,
        artifact_mode="pilot",
        selected_motion_keys=selected,
        motion_root=motion_root,
    )
    inventory_path = tmp_path / "storage/manifests/reference_lengths.json"
    _write_json(inventory_path, inventory)
    spec = {
        "schema_version": 2,
        "reference_length_inventory_sha256": inventory["inventory_sha256"],
        "probe_policies": [{"id": "release", "checkpoint_sha256": "d" * 64}],
        "domain_randomization_seeds": [7],
        "phase_targets": [{"phase_id": "start", "target_fraction": 0.0}],
        "repeats": 1,
        "rollout_id_prefix": "fixture-schedule",
    }
    spec_path = tmp_path / "schedule_spec.json"
    _write_json(spec_path, spec)
    schedule = build_rollout_schedule(
        split,
        reference_length_inventory=inventory,
        probe_policies=spec["probe_policies"],
        domain_randomization_seeds=spec["domain_randomization_seeds"],
        phase_targets=spec["phase_targets"],
        repeats=spec["repeats"],
        rollout_id_prefix=spec["rollout_id_prefix"],
    )
    schedule_path = tmp_path / "storage/manifests/schedule.json"
    _write_json(schedule_path, schedule)
    lock_path = tmp_path / "schedule_lock.json"
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
            "schedule_spec": str(spec_path),
            "schedule_spec_sha256": sha256_file(spec_path),
            "reference_length_inventory": str(inventory_path),
            "reference_length_inventory_file_sha256": sha256_file(inventory_path),
            "reference_length_inventory_sha256": inventory["inventory_sha256"],
        },
    }
    _write_json(lock_path, lock)
    return (
        lock_path,
        {
            "split": split_path,
            "spec": spec_path,
            "inventory": inventory_path,
            "schedule": schedule_path,
        },
        split,
    )


def test_split_readiness_verifies_storage_artifact_and_references(tmp_path: Path) -> None:
    config_path = _storage_config(tmp_path)
    lock_path, _, _, _ = _split_lock(tmp_path)

    report = verify_readiness(
        config_path,
        lock_path,
        minimum_free_bytes=0,
        repo_root=tmp_path,
        verify_referenced_manifests=True,
    )

    assert report["ready"] is True
    assert report["artifact"]["artifact_type"] == "split"
    assert len(report["storage"]["roots"]) == 4
    assert {item["name"] for item in report["artifact"]["referenced_manifests"]} == {
        "cohort_manifest",
        "materialized_manifest",
    }


def test_atlas_lock_uses_atlas_validator_and_self_digest(tmp_path: Path) -> None:
    report = verify_artifact_lock(
        _atlas_lock(tmp_path),
        repo_root=tmp_path,
        verify_referenced_manifests=True,
    )

    assert report["artifact_type"] == "atlas"
    assert len(report["self_digests"]["atlas_sha256"]) == 64
    assert [item["name"] for item in report["referenced_manifests"]] == ["split_manifest"]


def test_atlas_reference_must_be_the_split_bound_inside_the_artifact(tmp_path: Path) -> None:
    lock_path = _atlas_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    split_path = Path(lock["inputs"]["split_manifest"])
    replacement = build_source_disjoint_split(
        [
            {
                "motion_key": f"replacement_{index:02d}__A{index:03d}",
                "release_filter_key": f"240101/replacement_{index:02d}__A{index:03d}.pkl",
                "duration": float(index + 1),
                "stratum": f"kind_{index % 3}",
            }
            for index in range(30)
        ],
        seed=99,
        dataset={"name": "replacement"},
    )
    _write_json(split_path, replacement)
    lock["inputs"]["split_manifest_sha256"] = sha256_file(split_path)
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="does not bind the referenced split"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_scientific_atlas_lock_requires_receipt_driven_exact_rebuild(tmp_path: Path) -> None:
    lock_path, paths = _scientific_atlas_lock(tmp_path)

    report = verify_artifact_lock(
        lock_path,
        repo_root=paths["repo_root"],
        expected_analysis_protocol_lock_sha256=_expected_protocol_lock_digest(paths),
    )

    assert report["artifact_type"] == "atlas"
    assert report["referenced_manifests_required"] is True
    assert {item["name"] for item in report["referenced_manifests"]} == {
        "rollout_manifest",
        "split_manifest",
        "schedule_manifest",
        "reference_length_inventory",
        "analysis_protocol",
        "analysis_protocol_lock",
    }


@pytest.mark.parametrize("tamper", ["signature", "normalizer", "collection"])
def test_scientific_atlas_lock_rejects_fully_rehashed_source_or_output_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    lock_path, paths = _scientific_atlas_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if tamper == "signature":
        atlas = json.loads(paths["atlas"].read_text(encoding="utf-8"))
        atlas["signatures"][0]["mechanism_evidence"][0] += 1.0
        atlas["atlas_sha256"] = canonical_sha256(
            atlas,
            digest_field="atlas_sha256",
        )
        _write_json(paths["atlas"], atlas)
        lock["artifact"]["file_sha256"] = sha256_file(paths["atlas"])
        lock["artifact"]["atlas_sha256"] = atlas["atlas_sha256"]
    else:
        wrapper = json.loads(paths["rollout"].read_text(encoding="utf-8"))
        if tamper == "normalizer":
            wrapper["normalizer"]["mechanism_scales"]["base_drift"] = 999.0
        else:
            wrapper["rollout_collection"]["episodes"][0]["probe"]["scores"]["base_drift"] = 0.123
            wrapper["rollout_collection"]["episodes"][0]["mechanism_scores"]["base_drift"] = 0.123
            wrapper["episodes"] = deepcopy(wrapper["rollout_collection"]["episodes"])
            wrapper["rollout_collection"]["rollout_collection_sha256"] = canonical_sha256(
                wrapper["rollout_collection"],
                digest_field="rollout_collection_sha256",
            )
        _write_json(paths["rollout"], wrapper)
        lock["inputs"]["rollout_manifest_sha256"] = sha256_file(paths["rollout"])
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="scientific atlas receipt-driven validation failed"):
        verify_artifact_lock(
            lock_path,
            repo_root=paths["repo_root"],
            expected_analysis_protocol_lock_sha256=_expected_protocol_lock_digest(paths),
        )


def test_scientific_atlas_lock_requires_external_protocol_lock_digest(
    tmp_path: Path,
) -> None:
    lock_path, paths = _scientific_atlas_lock(tmp_path)

    with pytest.raises(ReadinessError, match="independently supplied expected"):
        verify_artifact_lock(lock_path, repo_root=paths["repo_root"])

    with pytest.raises(ReadinessError, match="independently supplied preregistration"):
        verify_artifact_lock(
            lock_path,
            repo_root=paths["repo_root"],
            expected_analysis_protocol_lock_sha256="f" * 64,
        )


def test_panel_lock_binds_representation_blind_panels_to_split(tmp_path: Path) -> None:
    report = verify_artifact_lock(
        _panel_lock(tmp_path),
        repo_root=tmp_path,
        verify_referenced_manifests=True,
    )

    assert report["artifact_type"] == "source_panels"
    assert len(report["self_digests"]["panel_sha256"]) == 64
    assert [item["name"] for item in report["referenced_manifests"]] == ["split_manifest"]


def test_intervention_plan_lock_requires_and_deep_verifies_all_inputs(tmp_path: Path) -> None:
    lock_path, _ = _intervention_plan_lock(tmp_path)

    report = verify_artifact_lock(lock_path, repo_root=tmp_path)

    assert report["artifact_type"] == "rq1_intervention_plan"
    assert report["referenced_manifests_required"] is True
    assert report["referenced_manifests_verified"] is True
    assert [item["name"] for item in report["referenced_manifests"]] == [
        "split_manifest",
        "panel_manifest",
        "intervention_protocol",
    ]


def test_intervention_plan_lock_rejects_rehashed_plan_tamper(tmp_path: Path) -> None:
    lock_path, artifact_path = _intervention_plan_lock(tmp_path)
    plan = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan["interventions"][0]["intervention"]["p_plus"].reverse()
    plan["intervention_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="intervention_plan_sha256",
    )
    _write_json(artifact_path, plan)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifact"]["file_sha256"] = sha256_file(artifact_path)
    lock["artifact"]["intervention_plan_sha256"] = plan["intervention_plan_sha256"]
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="deterministic reconstruction"):
        verify_artifact_lock(lock_path, repo_root=tmp_path)


def test_schedule_lock_verifies_exact_artifact_spec_split_and_length_inventory(
    tmp_path: Path,
) -> None:
    lock_path, _, _ = _schedule_lock(tmp_path)

    report = verify_artifact_lock(
        lock_path,
        repo_root=tmp_path,
    )

    assert report["artifact_type"] == "rollout_schedule"
    assert report["referenced_manifests_required"] is True
    assert report["referenced_manifests_verified"] is True
    assert len(report["self_digests"]["schedule_sha256"]) == 64
    assert [item["name"] for item in report["referenced_manifests"]] == [
        "split_manifest",
        "schedule_spec",
        "reference_length_inventory",
    ]


def test_schedule_lock_rejects_a_rehashed_but_semantically_different_spec(
    tmp_path: Path,
) -> None:
    lock_path, paths, _ = _schedule_lock(tmp_path)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    spec["rollout_id_prefix"] = "different-prefix"
    _write_json(paths["spec"], spec)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"]["schedule_spec_sha256"] = sha256_file(paths["spec"])
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="not the exact deterministic output"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_schedule_lock_requires_all_length_inventory_bindings(tmp_path: Path) -> None:
    lock_path, _, _ = _schedule_lock(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for field in (
        "reference_length_inventory",
        "reference_length_inventory_file_sha256",
        "reference_length_inventory_sha256",
    ):
        lock["inputs"].pop(field)
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="missing required referenced inputs"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_schedule_lock_rejects_valid_inventory_with_different_frame_counts(
    tmp_path: Path,
) -> None:
    lock_path, paths, split = _schedule_lock(tmp_path)
    spec = json.loads(paths["spec"].read_text(encoding="utf-8"))
    original = json.loads(paths["inventory"].read_text(encoding="utf-8"))
    alternate = build_reference_length_inventory(
        split,
        target_fps=25,
        sim_fps=25,
        artifact_mode="pilot",
        selected_motion_keys=original["selected_motion_keys"],
        motion_root=tmp_path / "dataset/robot_filtered",
    )
    _write_json(paths["inventory"], alternate)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"]["reference_length_inventory_file_sha256"] = sha256_file(paths["inventory"])
    lock["inputs"]["reference_length_inventory_sha256"] = alternate["inventory_sha256"]
    spec["reference_length_inventory_sha256"] = alternate["inventory_sha256"]
    _write_json(paths["spec"], spec)
    lock["inputs"]["schedule_spec_sha256"] = sha256_file(paths["spec"])
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="binding does not match the supplied inventory"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_storage_rejects_missing_root_and_insufficient_free_space(tmp_path: Path) -> None:
    config_path = _storage_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = Path(config["artifact_roots"]["atlases"])
    missing.rmdir()

    with pytest.raises(ReadinessError, match="does not exist"):
        verify_storage_roots(config_path, minimum_free_bytes=0)

    missing.mkdir()
    with pytest.raises(ReadinessError, match="insufficient free space"):
        verify_storage_roots(config_path, minimum_free_bytes=10**30)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ('{"schema_version":1,"schema_version":1}', "duplicate JSON key"),
        ('{"schema_version":NaN}', "non-finite JSON constant"),
        ('{"schema_version":Infinity}', "non-finite JSON constant"),
        ('{"schema_version":1e999}', "non-finite number"),
        ("   \n", "blank"),
    ],
)
def test_readiness_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path,
    raw: str,
    message: str,
) -> None:
    config_path = tmp_path / "storage.json"
    config_path.write_text(raw, encoding="utf-8")

    with pytest.raises(ReadinessError, match=message):
        verify_storage_roots(config_path, minimum_free_bytes=0)


def test_lock_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    lock_path, artifact_path, _, _ = _split_lock(tmp_path)
    artifact_path.write_bytes(artifact_path.read_bytes() + b"\n")

    with pytest.raises(ReadinessError, match="artifact.file_sha256 mismatch"):
        verify_artifact_lock(lock_path, repo_root=tmp_path)


def test_lock_rejects_invalid_internal_self_digest_even_when_file_hash_matches(
    tmp_path: Path,
) -> None:
    lock_path, artifact_path, _, _ = _split_lock(tmp_path)
    manifest = json.loads(artifact_path.read_text(encoding="utf-8"))
    manifest["dataset"] = {"name": "tampered"}
    _write_json(artifact_path, manifest)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifact"]["file_sha256"] = sha256_file(artifact_path)
    _write_json(lock_path, lock)

    with pytest.raises(
        ReadinessError, match="split artifact validation failed.*split_sha256 mismatch"
    ):
        verify_artifact_lock(lock_path, repo_root=tmp_path)


def test_referenced_manifest_hash_check_is_optional(tmp_path: Path) -> None:
    lock_path, _, cohort_path, _ = _split_lock(tmp_path)
    original = json.loads(cohort_path.read_text(encoding="utf-8"))
    changed = deepcopy(original)
    changed["motions"].append("tampered")
    _write_json(cohort_path, changed)

    unchecked = verify_artifact_lock(lock_path, repo_root=tmp_path)
    assert unchecked["referenced_manifests_verified"] is False

    with pytest.raises(ReadinessError, match="inputs.cohort_manifest_sha256 mismatch"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_materialized_manifest_is_bound_to_paired_dataset_digest(tmp_path: Path) -> None:
    lock_path, _, _, materialized_path = _split_lock(tmp_path)
    manifest = json.loads(materialized_path.read_text(encoding="utf-8"))
    manifest["output"]["paired_dataset_sha256"] = "c" * 64
    _write_json(materialized_path, manifest)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"]["materialized_manifest_sha256"] = hashlib.sha256(
        materialized_path.read_bytes()
    ).hexdigest()
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="paired_dataset_sha256 does not match"):
        verify_artifact_lock(
            lock_path,
            repo_root=tmp_path,
            verify_referenced_manifests=True,
        )


def test_scientific_headline_split_deep_verifies_references_by_default(
    tmp_path: Path,
) -> None:
    lock_path, _ = _headline_split_lock(tmp_path)

    report = verify_artifact_lock(
        lock_path,
        repo_root=Path(__file__).resolve().parents[2],
        expected_headline_split_lock_file_sha256=sha256_file(lock_path),
    )

    assert report["artifact_type"] == "split"
    assert report["referenced_manifests_required"] is True
    assert report["referenced_manifests_verified"] is True
    assert [record["name"] for record in report["referenced_manifests"]] == [
        "cohort_manifest",
        "materialized_manifest",
        "selection_protocol",
    ]


def test_scientific_headline_split_requires_independent_lock_file_digest(
    tmp_path: Path,
) -> None:
    lock_path, _ = _headline_split_lock(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ReadinessError, match="independently retained expected split-lock"):
        verify_artifact_lock(lock_path, repo_root=repo_root)
    with pytest.raises(ReadinessError, match="differs from the independently retained"):
        verify_artifact_lock(
            lock_path,
            repo_root=repo_root,
            expected_headline_split_lock_file_sha256="f" * 64,
        )


def test_scientific_headline_split_rejects_rehashed_grid_claim(tmp_path: Path) -> None:
    lock_path, paths = _headline_split_lock(tmp_path)
    protocol = json.loads(paths["protocol"].read_text(encoding="utf-8"))
    protocol["selection"]["pre_outcome_constraints"]["minimum_d_atlas_motion_count"] += 1
    protocol["selection"]["pre_outcome_constraints"]["maximum_d_atlas_motion_count"] += 1
    protocol["protocol_sha256"] = canonical_sha256(protocol, digest_field="protocol_sha256")
    _write_json(paths["protocol"], protocol)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"]["selection_protocol_sha256"] = sha256_file(paths["protocol"])
    lock["inputs"]["selection_protocol_self_sha256"] = protocol["protocol_sha256"]
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="unique-grid claim failed"):
        verify_artifact_lock(
            lock_path,
            repo_root=Path(__file__).resolve().parents[2],
            expected_headline_split_lock_file_sha256=sha256_file(lock_path),
        )


def test_scientific_headline_split_rejects_unrelated_valid_input_chain(
    tmp_path: Path,
) -> None:
    lock_path, _ = _headline_split_lock(tmp_path / "primary")
    _, alternate = _headline_split_lock(tmp_path / "alternate")
    alternate_protocol = json.loads(alternate["protocol"].read_text(encoding="utf-8"))
    alternate_materialized = json.loads(alternate["materialized"].read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"].update(
        {
            "selection_protocol": str(alternate["protocol"]),
            "selection_protocol_sha256": sha256_file(alternate["protocol"]),
            "selection_protocol_self_sha256": alternate_protocol["protocol_sha256"],
            "cohort_manifest": str(alternate["cohort"]),
            "cohort_manifest_sha256": sha256_file(alternate["cohort"]),
            "materialized_manifest": str(alternate["materialized"]),
            "materialized_manifest_sha256": sha256_file(alternate["materialized"]),
            "paired_dataset_sha256": alternate_materialized["output"]["paired_dataset_sha256"],
        }
    )
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="split dataset provenance"):
        verify_artifact_lock(
            lock_path,
            repo_root=Path(__file__).resolve().parents[2],
            expected_headline_split_lock_file_sha256=sha256_file(lock_path),
        )


def test_scientific_headline_split_rejects_rehashed_dataset_provenance(
    tmp_path: Path,
) -> None:
    lock_path, paths = _headline_split_lock(tmp_path)
    split = json.loads(paths["split"].read_text(encoding="utf-8"))
    split["dataset"]["dataset_root"] = str(tmp_path / "forged-root")
    split["split_sha256"] = canonical_sha256(split, digest_field="split_sha256")
    _write_json(paths["split"], split)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["artifact"]["file_sha256"] = sha256_file(paths["split"])
    lock["artifact"]["split_sha256"] = split["split_sha256"]
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="split dataset provenance"):
        verify_artifact_lock(
            lock_path,
            repo_root=Path(__file__).resolve().parents[2],
            expected_headline_split_lock_file_sha256=sha256_file(lock_path),
        )


def test_scientific_headline_split_rejects_symlinked_reference(tmp_path: Path) -> None:
    lock_path, paths = _headline_split_lock(tmp_path)
    alias = tmp_path / "protocol-alias.json"
    alias.symlink_to(paths["protocol"])
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["inputs"]["selection_protocol"] = str(alias)
    _write_json(lock_path, lock)

    with pytest.raises(ReadinessError, match="must not contain symlink components"):
        verify_artifact_lock(
            lock_path,
            repo_root=Path(__file__).resolve().parents[2],
            expected_headline_split_lock_file_sha256=sha256_file(lock_path),
        )
