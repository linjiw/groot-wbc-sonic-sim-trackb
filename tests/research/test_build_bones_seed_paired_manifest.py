from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts.research.build_bones_seed_paired_manifest import build_manifest


def _write_pair(robot_dir: Path, smpl_dir: Path, key: str) -> None:
    robot_dir.mkdir(parents=True, exist_ok=True)
    smpl_dir.mkdir(parents=True, exist_ok=True)
    frames = 2
    robot = {
        "root_trans_offset": np.zeros((frames, 3), dtype=np.float32),
        "pose_aa": np.zeros((frames, 30, 3), dtype=np.float32),
        "dof": np.zeros((frames, 29), dtype=np.float32),
        "root_rot": np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (frames, 1)),
        "smpl_joints": np.zeros((frames, 24, 3), dtype=np.float32),
        "fps": 30,
    }
    smpl = {
        "pose_aa": np.zeros((3, 72), dtype=np.float32),
        "transl": np.zeros((3, 3), dtype=np.float32),
        "smpl_joints": np.zeros((3, 24, 3), dtype=np.float32),
        "fps": 50.0,
        "original_pose_aa": np.zeros((frames, 72), dtype=np.float32),
        "original_fps": 30.0,
    }
    joblib.dump({key: robot}, robot_dir / f"{key}.pkl")
    joblib.dump(smpl, smpl_dir / f"{key}.pkl")


def _write_cohort(path: Path, keys: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "source": {"revision": "a" * 40},
                "selection": {"selection_sha256": "b" * 64},
                "motions": [
                    {
                        "motion_key": key,
                        "duration_source_frames": 8,
                        "category": "Baseline",
                        "package": "Locomotion",
                        "duration_bin": 0,
                        "is_mirror": False,
                    }
                    for key in keys
                ],
            }
        ),
        encoding="utf-8",
    )


def test_build_manifest_validates_release_timeline_and_hashes(tmp_path: Path) -> None:
    robot_dir = tmp_path / "robot_filtered"
    smpl_dir = tmp_path / "smpl_filtered"
    keys = ["motion_a", "motion_b"]
    for key in keys:
        _write_pair(robot_dir, smpl_dir, key)
    cohort = tmp_path / "cohort.json"
    _write_cohort(cohort, keys)

    manifest = build_manifest(cohort, robot_dir, smpl_dir)

    assert manifest["output"]["motion_keys"] == keys
    assert manifest["output"]["motion_count"] == 2
    assert len(manifest["output"]["paired_dataset_sha256"]) == 64
    assert all(variant["validation"]["paired_original_timeline"] for variant in manifest["output"]["variants"])


def test_build_manifest_rejects_nested_or_incomplete_inventory(tmp_path: Path) -> None:
    robot_dir = tmp_path / "robot_filtered"
    smpl_dir = tmp_path / "smpl_filtered"
    _write_pair(robot_dir, smpl_dir, "motion_a")
    (robot_dir / "nested").mkdir()
    joblib.dump({}, robot_dir / "nested" / "extra.pkl")
    cohort = tmp_path / "cohort.json"
    _write_cohort(cohort, ["motion_a"])

    with pytest.raises(ValueError, match="must be flat"):
        build_manifest(cohort, robot_dir, smpl_dir)
