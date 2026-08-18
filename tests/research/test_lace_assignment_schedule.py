from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.research.lace.assignment_schedule import (
    ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS,
    ASSIGNMENT_INVENTORY_DIGEST_FIELD,
    ASSIGNMENT_INVENTORY_KIND,
    ASSIGNMENT_POLICY_ORDER,
    ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
    ASSIGNMENT_SCHEDULE_SPEC_KIND,
    build_assignment_probe_schedule,
    build_assignment_reference_length_inventory,
    deep_validate_assignment_artifacts,
    validate_assignment_probe_schedule,
    validate_assignment_reference_length_inventory,
    validate_assignment_schedule_spec,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cli(relative_path: str, module_name: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


build_inventory_main = _load_cli(
    "scripts/research/build_lace_assignment_reference_lengths.py",
    "lace_assignment_inventory_cli_for_test",
)
build_schedule_main = _load_cli(
    "scripts/research/build_lace_assignment_schedule.py",
    "lace_assignment_schedule_cli_for_test",
)


def _write_motion(path: Path, motion_key: str, *, frames: int, fps: int = 30) -> None:
    payload = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "root_rot": np.zeros((frames, 4), dtype=np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "fps": fps,
    }
    joblib.dump({motion_key: payload}, path, compress=True)


def _fixture(tmp_path: Path) -> tuple[dict, Path]:
    motion_root = tmp_path / "robot_filtered"
    motion_root.mkdir()
    records = []
    for index in range(40):
        motion_key = f"assignment_{index:03d}__A{index:03d}"
        path = motion_root / f"{motion_key}.pkl"
        _write_motion(path, motion_key, frames=8 + index)
        records.append(
            {
                "motion_key": motion_key,
                "source_group_id": f"actor_{index:03d}",
                "robot_path": str(path),
                "duration_source_frames": 8 + index,
                "stratum": f"kind_{index % 4}",
            }
        )
    split = build_source_disjoint_split(
        records,
        seed=17,
        dataset={"name": "assignment-fixture"},
    )
    return split, motion_root


def _schedule_spec(inventory: dict, *, prefix: str = "lace-assignment-fixture") -> dict:
    spec = {
        "kind": ASSIGNMENT_SCHEDULE_SPEC_KIND,
        "schema_version": 1,
        "partition": inventory["partition"],
        "final_open": inventory["final_open"],
        "assignment_only": True,
        "fit_normalizer": False,
        "reference_length_inventory_sha256": inventory[ASSIGNMENT_INVENTORY_DIGEST_FIELD],
        "parent_normalizer_sha256": "a" * 64,
        "analysis_protocol_sha256": "b" * 64,
        "probe_policies": [
            {"id": policy_id, "checkpoint_sha256": str(index + 1) * 64}
            for index, policy_id in enumerate(ASSIGNMENT_POLICY_ORDER)
        ],
        "domain_randomization_seeds": list(ASSIGNMENT_DOMAIN_RANDOMIZATION_SEEDS),
        "phase_targets": [
            {"phase_id": "start", "target_fraction": 0.0},
            {"phase_id": "mid", "target_fraction": 0.5},
        ],
        "repeats": 2,
        "rollout_id_prefix": prefix,
    }
    spec[ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD] = canonical_sha256(
        spec,
        digest_field=ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
    )
    return spec


@pytest.mark.parametrize("partition", ["D_curriculum", "D_geometry"])
def test_full_partition_inventory_is_deterministic_and_source_bound(
    tmp_path: Path,
    partition: str,
) -> None:
    split, motion_root = _fixture(tmp_path)

    first = build_assignment_reference_length_inventory(
        split,
        partition=partition,
        motion_root=motion_root,
    )
    second = build_assignment_reference_length_inventory(
        split,
        partition=partition,
        motion_root=motion_root,
    )

    expected_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == partition
    )
    assert first == second
    assert first["kind"] == ASSIGNMENT_INVENTORY_KIND
    assert first["selected_motion_keys"] == expected_keys
    assert first["motion_count"] == len(expected_keys)
    assert first["selection_complete_for_partition"] is True
    assert first["assignment_only"] is True
    assert first["fit_normalizer"] is False
    assert first["runtime_contract"]["target_fps"] == 50
    assert first["runtime_contract"]["sim_fps"] == 50
    assert first["runtime_contract"]["motion_fps_scale"] == {
        "numerator": 1,
        "denominator": 1,
    }
    assert first["runtime_contract"]["max_len"] == -1
    assert all(
        record["source_fps"] == {"numerator": 30, "denominator": 1} for record in first["motions"]
    )
    validate_assignment_reference_length_inventory(
        first,
        split_manifest=split,
        verify_source_files=True,
        deterministic_rebuild=True,
    )


def test_schedule_has_exact_grid_and_common_random_numbers(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_geometry",
        motion_root=motion_root,
    )
    spec = _schedule_spec(inventory)

    first = build_assignment_probe_schedule(
        split,
        reference_length_inventory=inventory,
        spec=spec,
    )
    second = build_assignment_probe_schedule(
        split,
        reference_length_inventory=inventory,
        spec=spec,
    )

    assert first == second
    assert first["policy_order"] == list(ASSIGNMENT_POLICY_ORDER)
    assert first["domain_randomization_seeds"] == [101, 202]
    assert first["repeat_count"] == 2
    assert first["phase_targets"] == [
        {"phase_id": "start", "target_fraction": 0.0},
        {"phase_id": "mid", "target_fraction": 0.5},
    ]
    assert first["rollout_count"] == inventory["motion_count"] * 3 * 2 * 2 * 2
    assert first["assignment_only"] is True
    assert first["fit_normalizer"] is False
    assert first["parent_normalizer_sha256"] == "a" * 64
    assert first["analysis_protocol_sha256"] == "b" * 64
    assert len({row["rollout_id"] for row in first["rollouts"]}) == first["rollout_count"]
    assert all(row["assignment_only"] is True for row in first["rollouts"])
    assert all(row["fit_normalizer"] is False for row in first["rollouts"])
    assert all(row["parent_normalizer_sha256"] == "a" * 64 for row in first["rollouts"])
    assert all(row["analysis_protocol_sha256"] == "b" * 64 for row in first["rollouts"])

    seeds_by_condition: dict[tuple[int, str, int], set[int]] = {}
    for row in first["rollouts"]:
        condition = (
            row["domain_randomization_seed"],
            row["phase_id"],
            row["repeat_index"],
        )
        seeds_by_condition.setdefault(condition, set()).add(row["runtime_rng_seed"])
    assert len(seeds_by_condition) == 2 * 2 * 2
    assert all(len(values) == 1 for values in seeds_by_condition.values())
    deep_validate_assignment_artifacts(
        split_manifest=split,
        reference_length_inventory=inventory,
        spec=spec,
        schedule=first,
    )


def test_rehashed_partial_partition_inventory_is_rejected(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_geometry",
        motion_root=motion_root,
    )
    tampered = deepcopy(inventory)
    tampered["selected_motion_keys"] = tampered["selected_motion_keys"][:-1]
    tampered["motions"] = tampered["motions"][:-1]
    tampered["motion_count"] -= 1
    tampered["source_file_set_sha256"] = canonical_sha256(
        {
            "files": [
                {
                    "motion_key": row["motion_key"],
                    "source_file_sha256": row["source_file_sha256"],
                }
                for row in tampered["motions"]
            ]
        }
    )
    tampered[ASSIGNMENT_INVENTORY_DIGEST_FIELD] = canonical_sha256(
        tampered,
        digest_field=ASSIGNMENT_INVENTORY_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="exactly cover"):
        validate_assignment_reference_length_inventory(
            tampered,
            split_manifest=split,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("policy_order", "probe policy order"),
        ("seed_grid", "domain_randomization_seeds"),
        ("phase_grid", "phase_targets"),
        ("repeat_grid", "repeats"),
    ],
)
def test_schedule_spec_rejects_grid_drift_after_rehash(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    split, motion_root = _fixture(tmp_path)
    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_geometry",
        motion_root=motion_root,
    )
    spec = _schedule_spec(inventory)
    if mutation == "policy_order":
        spec["probe_policies"] = list(reversed(spec["probe_policies"]))
    elif mutation == "seed_grid":
        spec["domain_randomization_seeds"] = [101, 303]
    elif mutation == "phase_grid":
        spec["phase_targets"][1]["target_fraction"] = 0.75
    else:
        spec["repeats"] = 3
    spec[ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD] = canonical_sha256(
        spec,
        digest_field=ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match=message):
        validate_assignment_schedule_spec(spec)


def test_d_test_requires_explicit_final_open_in_inventory_and_spec(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    with pytest.raises(ValueError, match="D_test is forbidden"):
        build_assignment_reference_length_inventory(
            split,
            partition="D_test",
            motion_root=motion_root,
        )

    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_test",
        final_open=True,
        motion_root=motion_root,
    )
    spec = _schedule_spec(inventory)
    schedule = build_assignment_probe_schedule(
        split,
        reference_length_inventory=inventory,
        spec=spec,
    )
    assert schedule["partition"] == "D_test"
    assert schedule["final_open"] is True

    tampered_spec = deepcopy(spec)
    tampered_spec["final_open"] = False
    tampered_spec[ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD] = canonical_sha256(
        tampered_spec,
        digest_field=ASSIGNMENT_SCHEDULE_SPEC_DIGEST_FIELD,
    )
    with pytest.raises(ValueError, match="D_test is forbidden"):
        validate_assignment_schedule_spec(tampered_spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("assignment_only", False, "assignment_only"),
        ("fit_normalizer", True, "fit_normalizer"),
        ("parent_normalizer_sha256", "c" * 64, "deterministic"),
        ("analysis_protocol_sha256", "d" * 64, "deterministic"),
    ],
)
def test_rehashed_schedule_tampering_is_rejected_against_frozen_spec(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    split, motion_root = _fixture(tmp_path)
    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_geometry",
        motion_root=motion_root,
    )
    spec = _schedule_spec(inventory)
    schedule = build_assignment_probe_schedule(
        split,
        reference_length_inventory=inventory,
        spec=spec,
    )
    tampered = deepcopy(schedule)
    tampered[field] = value
    tampered["schedule_sha256"] = canonical_sha256(
        tampered,
        digest_field="schedule_sha256",
    )
    with pytest.raises(ValueError, match=message):
        validate_assignment_probe_schedule(
            tampered,
            split_manifest=split,
            reference_length_inventory=inventory,
            spec=spec,
        )


def test_source_byte_tampering_is_rejected_even_after_inventory_rehash(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    inventory = build_assignment_reference_length_inventory(
        split,
        partition="D_geometry",
        motion_root=motion_root,
    )
    source_path = Path(inventory["motions"][0]["source_path"])
    with source_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ValueError, match="source file SHA-256 mismatch"):
        validate_assignment_reference_length_inventory(
            inventory,
            split_manifest=split,
            verify_source_files=True,
            deterministic_rebuild=True,
        )


def test_non_30_hz_source_is_rejected(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    geometry_key = next(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_geometry"
    )
    _write_motion(motion_root / f"{geometry_key}.pkl", geometry_key, frames=12, fps=31)
    with pytest.raises(ValueError, match="exactly 30 Hz"):
        build_assignment_reference_length_inventory(
            split,
            partition="D_geometry",
            motion_root=motion_root,
        )


def test_cpu_clis_are_overwrite_safe_and_deep_validate(tmp_path: Path) -> None:
    split, motion_root = _fixture(tmp_path)
    split_path = tmp_path / "split.json"
    inventory_path = tmp_path / "inventory.json"
    spec_path = tmp_path / "spec.json"
    schedule_path = tmp_path / "schedule.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")

    inventory_args = [
        "--split",
        str(split_path),
        "--partition",
        "D_geometry",
        "--motion-root",
        str(motion_root),
        "--output",
        str(inventory_path),
    ]
    assert build_inventory_main(inventory_args) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_inventory_main(inventory_args)
    assert build_inventory_main([*inventory_args, "--force"]) == 0

    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    spec = _schedule_spec(inventory, prefix="lace-assignment-cli")
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    schedule_args = [
        "--split",
        str(split_path),
        "--reference-length-inventory",
        str(inventory_path),
        "--spec",
        str(spec_path),
        "--output",
        str(schedule_path),
    ]
    assert build_schedule_main(schedule_args) == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_schedule_main(schedule_args)
    assert build_schedule_main([*schedule_args, "--force"]) == 0

    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    deep_validate_assignment_artifacts(
        split_manifest=split,
        reference_length_inventory=inventory,
        spec=spec,
        schedule=schedule,
    )
