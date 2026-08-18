from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pytest

from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    RESAMPLING_RULE_ID,
    build_reference_length_inventory,
    reference_num_steps_by_motion,
    sonic_target_frame_count,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split

REPO_ROOT = Path(__file__).resolve().parents[2]
_CLI_PATH = REPO_ROOT / "scripts/research/build_lace_reference_lengths.py"
_CLI_SPEC = importlib.util.spec_from_file_location("lace_reference_lengths_cli", _CLI_PATH)
assert _CLI_SPEC is not None and _CLI_SPEC.loader is not None
_CLI_MODULE = importlib.util.module_from_spec(_CLI_SPEC)
_CLI_SPEC.loader.exec_module(_CLI_MODULE)
build_lengths_main = _CLI_MODULE.main


def _write_motion(path: Path, key: str, *, frames: int, fps: int | float) -> None:
    payload = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "root_rot": np.zeros((frames, 4), dtype=np.float32),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "fps": fps,
    }
    joblib.dump({key: payload}, path, compress=True)


def _split_and_files(tmp_path: Path) -> tuple[dict, Path, list[str]]:
    motion_root = tmp_path / "robot_filtered"
    motion_root.mkdir()
    records = []
    for index in range(30):
        key = f"motion_{index:03d}__A{index:03d}"
        records.append(
            {
                "motion_key": key,
                "robot_path": str(motion_root / f"{key}.pkl"),
                "duration_source_frames": 100 + index,
                "stratum": f"kind_{index % 3}",
            }
        )
    split = build_source_disjoint_split(records, seed=17, dataset={"name": "fixture"})
    atlas_keys = sorted(
        record["motion_key"] for record in split["motions"] if record["partition"] == "D_atlas"
    )
    fps_cycle: tuple[int | float, ...] = (30,)
    for index, key in enumerate(atlas_keys):
        _write_motion(
            motion_root / f"{key}.pkl",
            key,
            frames=8 + index,
            fps=fps_cycle[index % len(fps_cycle)],
        )
    return split, motion_root, atlas_keys


def _split_with_materialized_provenance(tmp_path: Path) -> tuple[dict, Path, Path]:
    motion_root = tmp_path / "materialized/robot_filtered"
    motion_root.mkdir(parents=True)
    records = []
    variants = []
    for index in range(30):
        key = f"paired_{index:03d}__A{index:03d}"
        frames = 10 + index
        path = motion_root / f"{key}.pkl"
        _write_motion(path, key, frames=frames, fps=30)
        records.append(
            {
                "motion_key": key,
                "robot_path": str(path),
                "duration_source_frames": frames,
                "stratum": f"kind_{index % 3}",
            }
        )
        variants.append(
            {
                "motion_key": key,
                "robot": {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "frames": frames,
                    "fps": 30.0,
                },
                "smpl": {
                    "frames": sonic_target_frame_count(frames, 30, 50),
                    "fps": 50.0,
                    "original_frames": frames,
                    "original_fps": 30.0,
                },
            }
        )
    materialized_path = tmp_path / "materialized/dataset_manifest.json"
    paired_digest = "a" * 64
    materialized_path.write_text(
        json.dumps(
            {
                "output": {
                    "paired_dataset_sha256": paired_digest,
                    "variants": variants,
                }
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    split = build_source_disjoint_split(
        records,
        seed=17,
        dataset={
            "materialized_manifest": str(materialized_path),
            "materialized_manifest_sha256": hashlib.sha256(
                materialized_path.read_bytes()
            ).hexdigest(),
            "paired_dataset_sha256": paired_digest,
        },
    )
    return split, motion_root, materialized_path


@pytest.mark.parametrize(
    ("source_frames", "source_fps", "target_fps", "expected"),
    [
        (262, 30, 50, 435),
        (216, 30, 50, 359),
        (8, 25, 50, 14),
        (300, 29.97, 50, math.ceil(Fraction(299 * 5000, 2997))),
        (2, 120, 50, 1),
        # SONIC skips interpolation at equal FPS; the raw final frame is retained.
        (8, 50, 50, 8),
    ],
)
def test_exact_exclusive_end_frame_count(
    source_frames: int,
    source_fps: int | float,
    target_fps: int,
    expected: int,
) -> None:
    assert sonic_target_frame_count(source_frames, source_fps, target_fps) == expected


def test_exact_count_matches_torch_arange_on_sonic_cohort_endpoint_cases() -> None:
    torch = pytest.importorskip("torch")

    # The official materialized G1 cohort is 30 Hz. Include exact-integer and
    # fractional endpoint ratios as well as both real pilot lengths.
    for source_frames in (2, 4, 7, 8, 216, 262, 1000, 1048):
        duration = (source_frames - 1) / 30
        torch_count = len(torch.arange(0, duration, 1 / 50, dtype=torch.float32))
        assert sonic_target_frame_count(source_frames, 30, 50) == torch_count

    # At equal FPS fk_batch does not call arange at all.  Do not assert a
    # float32 arange length here: endpoint rounding is version/platform
    # sensitive and is irrelevant to SONIC's executed equal-rate branch.
    assert sonic_target_frame_count(8, 50, 50) == 8


def test_exact_rational_rule_does_not_inherit_float32_endpoint_artifact() -> None:
    torch = pytest.importorskip("torch")

    # Mathematically [0, 0.28) at 0.02 spacing has 14 samples. Torch 2.x can
    # materialize 15 because duration rounds above the exclusive endpoint.
    torch_count = len(torch.arange(0, 7 / 25, 1 / 50, dtype=torch.float32))
    assert torch_count == 15
    assert sonic_target_frame_count(8, 25, 50) == 14


def test_builds_deterministic_scientific_inventory_bound_to_split_and_files(
    tmp_path: Path,
) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)

    first = build_reference_length_inventory(split, motion_root=motion_root)
    second = build_reference_length_inventory(split, motion_root=motion_root)

    assert first == second
    assert first["scientific_use"] is True
    assert first["selection_complete_for_d_atlas"] is True
    assert first["selected_motion_keys"] == atlas_keys
    assert first["split_sha256"] == split["split_sha256"]
    assert first["split_selection_sha256"] == split["selection_sha256"]
    assert first["resampling_rule"]["id"] == RESAMPLING_RULE_ID
    assert first[REFERENCE_LENGTH_DIGEST_FIELD] == canonical_sha256(
        first,
        digest_field=REFERENCE_LENGTH_DIGEST_FIELD,
    )
    assert reference_num_steps_by_motion(first) == {
        record["motion_key"]: record["target_num_frames"] for record in first["motions"]
    }
    validate_reference_length_inventory(
        first,
        split_manifest=split,
        verify_source_files=True,
    )
    assert first["runtime_contract"] == {
        "target_fps": 50,
        "sim_fps": 50,
        "motion_fps_scale": {"numerator": 1, "denominator": 1},
        "max_len": -1,
        "reference_num_steps_equals_target_num_frames": True,
        "float32_runtime_equivalence_scope": (
            "exact_30_to_50_hz_scientific_contract_without_materialized_pair_provenance"
        ),
    }


def test_scientific_inventory_binds_and_cross_checks_materialized_provenance(
    tmp_path: Path,
) -> None:
    split, motion_root, materialized_path = _split_with_materialized_provenance(tmp_path)

    inventory = build_reference_length_inventory(split, motion_root=motion_root)

    assert inventory["dataset_provenance"] == {
        "materialized_manifest": str(materialized_path),
        "materialized_manifest_sha256": hashlib.sha256(materialized_path.read_bytes()).hexdigest(),
        "paired_dataset_sha256": "a" * 64,
    }
    assert inventory["runtime_contract"]["float32_runtime_equivalence_scope"] == (
        "locked_30_to_50_hz_release_cohort_with_materialized_pair_cross_check"
    )
    validate_reference_length_inventory(
        inventory,
        split_manifest=split,
        verify_source_files=True,
    )

    materialized_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="materialized manifest file SHA-256"):
        validate_reference_length_inventory(inventory, verify_source_files=True)


def test_pilot_requires_explicit_strict_subset_and_is_non_scientific(tmp_path: Path) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)
    selected = atlas_keys[:2]

    pilot = build_reference_length_inventory(
        split,
        artifact_mode="pilot",
        selected_motion_keys=selected,
        motion_root=motion_root,
    )

    assert pilot["artifact_mode"] == "pilot"
    assert pilot["scientific_use"] is False
    assert pilot["selection_complete_for_d_atlas"] is False
    assert "non_scientific" in pilot["pilot_status"]
    assert pilot["selected_motion_keys"] == selected
    assert pilot["runtime_contract"]["float32_runtime_equivalence_scope"] == (
        "not_claimed_for_generic_pilot_rates"
    )

    with pytest.raises(ValueError, match="explicit selected_motion_keys"):
        build_reference_length_inventory(
            split,
            artifact_mode="pilot",
            motion_root=motion_root,
        )


def test_scientific_contract_rejects_runtime_drift_but_pilot_allows_generic_rate(
    tmp_path: Path,
) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)
    selected = atlas_keys[:2]
    generic_key = selected[0]
    _write_motion(motion_root / f"{generic_key}.pkl", generic_key, frames=8, fps=25)

    with pytest.raises(ValueError, match="exactly 30 Hz"):
        build_reference_length_inventory(split, motion_root=motion_root)

    pilot = build_reference_length_inventory(
        split,
        artifact_mode="pilot",
        selected_motion_keys=selected,
        motion_root=motion_root,
    )
    assert pilot["runtime_contract"]["float32_runtime_equivalence_scope"] == (
        "not_claimed_for_generic_pilot_rates"
    )
    with pytest.raises(ValueError, match="target_fps=sim_fps=50"):
        build_reference_length_inventory(
            split,
            target_fps=25,
            sim_fps=25,
            motion_root=motion_root,
        )
    with pytest.raises(ValueError, match="motion_fps_scale must be exactly 1"):
        build_reference_length_inventory(
            split,
            motion_fps_scale=2,
            motion_root=motion_root,
        )
    with pytest.raises(ValueError, match="max_len must be exactly -1"):
        build_reference_length_inventory(split, max_len=10, motion_root=motion_root)
    with pytest.raises(ValueError, match="strict D_atlas subset"):
        build_reference_length_inventory(
            split,
            artifact_mode="pilot",
            selected_motion_keys=atlas_keys,
            motion_root=motion_root,
        )
    with pytest.raises(ValueError, match="full D_atlas"):
        build_reference_length_inventory(
            split,
            artifact_mode="scientific",
            selected_motion_keys=selected,
            motion_root=motion_root,
        )


def test_rejects_ambiguous_key_fps_and_frame_axis_contracts(tmp_path: Path) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)
    key = atlas_keys[0]
    path = motion_root / f"{key}.pkl"
    payload = joblib.load(path)[key]

    joblib.dump({key: payload, "unexpected": payload}, path)
    with pytest.raises(ValueError, match="exactly one key"):
        build_reference_length_inventory(split, motion_root=motion_root)

    payload.pop("fps")
    joblib.dump({key: payload}, path)
    with pytest.raises(ValueError, match=r"\.fps must be explicit"):
        build_reference_length_inventory(split, motion_root=motion_root)

    payload["fps"] = 30
    payload["pose_aa"] = np.zeros((payload["root_trans_offset"].shape[0] + 1, 30, 3))
    joblib.dump({key: payload}, path)
    with pytest.raises(ValueError, match="frame axis does not match"):
        build_reference_length_inventory(split, motion_root=motion_root)


def test_validator_detects_count_digest_split_and_source_file_drift(tmp_path: Path) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)
    inventory = build_reference_length_inventory(split, motion_root=motion_root)

    wrong_count = deepcopy(inventory)
    wrong_count["motions"][0]["target_num_frames"] += 1
    with pytest.raises(ValueError, match="violates the exact resampling rule"):
        validate_reference_length_inventory(wrong_count, verify_digest=False)

    stale_digest = deepcopy(inventory)
    stale_digest[REFERENCE_LENGTH_DIGEST_FIELD] = "0" * 64
    with pytest.raises(ValueError, match="inventory_sha256 mismatch"):
        validate_reference_length_inventory(stale_digest)

    replacement_split = build_source_disjoint_split(
        [
            {
                "motion_key": f"replacement_{index:03d}__A{index:03d}",
                "robot_path": str(motion_root / f"replacement_{index:03d}__A{index:03d}.pkl"),
                "stratum": f"kind_{index % 3}",
            }
            for index in range(30)
        ],
        seed=29,
    )
    with pytest.raises(ValueError, match="split_sha256 binding mismatch"):
        validate_reference_length_inventory(inventory, split_manifest=replacement_split)

    source_path = motion_root / f"{atlas_keys[0]}.pkl"
    source_path.write_bytes(source_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="source file SHA-256 mismatch"):
        validate_reference_length_inventory(inventory, verify_source_files=True)


def test_cpu_cli_writes_canonical_inventory_and_refuses_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    split, motion_root, atlas_keys = _split_and_files(tmp_path)
    split_path = tmp_path / "split.json"
    output_path = tmp_path / "artifacts/reference-lengths.json"
    split_path.write_text(json.dumps(split), encoding="utf-8")
    selected = atlas_keys[:2]
    arguments = [
        "--split",
        str(split_path),
        "--motion-root",
        str(motion_root),
        "--artifact-mode",
        "pilot",
        "--selected-motion-key",
        selected[0],
        "--selected-motion-key",
        selected[1],
        "--output",
        str(output_path),
    ]

    assert build_lengths_main(arguments) == 0
    summary = json.loads(capsys.readouterr().out)
    inventory = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["inventory_sha256"] == inventory["inventory_sha256"]
    validate_reference_length_inventory(inventory, split_manifest=split)
    assert output_path.read_text(encoding="utf-8") == (
        json.dumps(
            inventory,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_lengths_main(arguments)
    assert build_lengths_main([*arguments, "--force"]) == 0
