#!/usr/bin/env python3
"""Validate and hash-bind a materialized official BONES-SEED SONIC cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

ROBOT_FIELDS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints")
SMPL_FIELDS = ("pose_aa", "transl", "smpl_joints", "original_pose_aa")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_hash(records: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for relative_path, file_hash in sorted(records):
        digest.update(relative_path.encode())
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = joblib.load(path)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a dictionary")
    return value


def _finite_array(value: Any, *, field: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != len(shape_tail) + 1 or tuple(array.shape[1:]) != shape_tail:
        raise ValueError(f"{field} has shape {array.shape}, expected (frames, {shape_tail})")
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values")
    return array


def _flat_inventory(root: Path, expected_keys: list[str], *, modality: str) -> None:
    direct = sorted(path.stem for path in root.glob("*.pkl"))
    recursive = sorted(path for path in root.rglob("*.pkl"))
    if len(recursive) != len(direct):
        raise ValueError(f"{modality} directory must be flat: {root}")
    if direct != expected_keys:
        missing = sorted(set(expected_keys) - set(direct))
        extras = sorted(set(direct) - set(expected_keys))
        raise ValueError(f"{modality} inventory mismatch: missing={missing}, extras={extras}")


def _validate_pair(
    key: str,
    robot_path: Path,
    smpl_path: Path,
    *,
    source_frames: int,
) -> dict[str, Any]:
    robot_container = _load_object(robot_path)
    if set(robot_container) != {key}:
        raise ValueError(f"{robot_path} must contain exactly key {key!r}")
    robot = robot_container[key]
    if not isinstance(robot, dict) or any(field not in robot for field in ROBOT_FIELDS):
        raise ValueError(f"{robot_path} is missing required robot fields")
    robot_frames = _finite_array(
        robot["root_trans_offset"], field=f"{key}.robot.root_trans_offset", shape_tail=(3,)
    ).shape[0]
    for field, shape_tail in (
        ("pose_aa", (30, 3)),
        ("dof", (29,)),
        ("root_rot", (4,)),
        ("smpl_joints", (24, 3)),
    ):
        array = _finite_array(robot[field], field=f"{key}.robot.{field}", shape_tail=shape_tail)
        if array.shape[0] != robot_frames:
            raise ValueError(f"{key}.robot.{field} frame count does not match root translation")
    robot_fps = float(robot.get("fps", 0))
    if robot_fps != 30.0:
        raise ValueError(f"{key} robot fps must be 30, got {robot_fps}")
    expected_robot_frames = math.ceil(source_frames / 4)
    if robot_frames != expected_robot_frames:
        raise ValueError(
            f"{key} robot frames {robot_frames} do not match 120->30 Hz metadata expectation "
            f"{expected_robot_frames}"
        )
    quaternion_norm_error = float(np.max(np.abs(np.linalg.norm(np.asarray(robot["root_rot"]), axis=1) - 1.0)))
    if quaternion_norm_error > 1e-4:
        raise ValueError(f"{key} robot root quaternion norm error is {quaternion_norm_error}")

    smpl = _load_object(smpl_path)
    if any(field not in smpl for field in SMPL_FIELDS):
        raise ValueError(f"{smpl_path} is missing required SMPL fields")
    smpl_frames = _finite_array(smpl["pose_aa"], field=f"{key}.smpl.pose_aa", shape_tail=(72,)).shape[0]
    for field, shape_tail in (("transl", (3,)), ("smpl_joints", (24, 3))):
        array = _finite_array(smpl[field], field=f"{key}.smpl.{field}", shape_tail=shape_tail)
        if array.shape[0] != smpl_frames:
            raise ValueError(f"{key}.smpl.{field} frame count does not match pose_aa")
    original_pose = _finite_array(smpl["original_pose_aa"], field=f"{key}.smpl.original_pose_aa", shape_tail=(72,))
    smpl_fps = float(smpl.get("fps", 0))
    original_fps = float(smpl.get("original_fps", 0))
    if smpl_fps != 50.0 or original_fps != 30.0:
        raise ValueError(f"{key} SMPL fps/original_fps must be 50/30, got {smpl_fps}/{original_fps}")
    if original_pose.shape[0] != robot_frames:
        raise ValueError(f"{key} SMPL original timeline does not match robot frames")
    duration_delta = abs((smpl_frames - 1) / smpl_fps - (robot_frames - 1) / robot_fps)
    if duration_delta > 1.0 / robot_fps:
        raise ValueError(f"{key} robot/SMPL runtime durations differ by {duration_delta:.6f}s")

    return {
        "robot": {
            "path": f"robot_filtered/{key}.pkl",
            "sha256": _sha256(robot_path),
            "frames": robot_frames,
            "fps": robot_fps,
            "required_fields": list(ROBOT_FIELDS),
        },
        "smpl": {
            "path": f"smpl_filtered/{key}.pkl",
            "sha256": _sha256(smpl_path),
            "frames": smpl_frames,
            "fps": smpl_fps,
            "original_frames": int(original_pose.shape[0]),
            "original_fps": original_fps,
            "required_fields": list(SMPL_FIELDS),
        },
        "validation": {
            "source_frames_120hz": source_frames,
            "paired_original_timeline": True,
            "duration_delta_seconds": duration_delta,
            "root_quaternion_max_norm_error": quaternion_norm_error,
        },
    }


def build_manifest(cohort_path: Path, robot_dir: Path, smpl_dir: Path) -> dict[str, Any]:
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    motions = cohort.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("cohort motions must be a non-empty list")
    by_key = {row["motion_key"]: row for row in motions}
    if len(by_key) != len(motions):
        raise ValueError("cohort contains duplicate motion keys")
    motion_keys = sorted(by_key)
    _flat_inventory(robot_dir, motion_keys, modality="robot")
    _flat_inventory(smpl_dir, motion_keys, modality="smpl")

    variants = []
    robot_records = []
    smpl_records = []
    for key in motion_keys:
        pair = _validate_pair(
            key,
            robot_dir / f"{key}.pkl",
            smpl_dir / f"{key}.pkl",
            source_frames=int(by_key[key]["duration_source_frames"]),
        )
        pair["motion_key"] = key
        pair["source"] = {
            field: by_key[key][field] for field in ("category", "package", "duration_bin", "is_mirror")
        }
        variants.append(pair)
        robot_records.append((pair["robot"]["path"], pair["robot"]["sha256"]))
        smpl_records.append((pair["smpl"]["path"], pair["smpl"]["sha256"]))

    return {
        "schema_version": 1,
        "kind": "official_bones_seed_paired_sonic_cohort",
        "generator": "scripts/research/build_bones_seed_paired_manifest.py",
        "source": {
            "cohort_manifest": str(cohort_path),
            "cohort_manifest_sha256": _sha256(cohort_path),
            "bones_seed_revision": cohort["source"]["revision"],
            "selection_sha256": cohort["selection"]["selection_sha256"],
        },
        "output": {
            "robot_dir": "robot_filtered",
            "smpl_dir": "smpl_filtered",
            "motion_count": len(motion_keys),
            "motion_keys": motion_keys,
            "robot_dataset_sha256": _dataset_hash(robot_records),
            "smpl_dataset_sha256": _dataset_hash(smpl_records),
            "paired_dataset_sha256": _dataset_hash(robot_records + smpl_records),
            "variants": variants,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--robot-dir", type=Path, required=True)
    parser.add_argument("--smpl-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.cohort.resolve(), args.robot_dir.resolve(), args.smpl_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "motion_count": manifest["output"]["motion_count"],
                "paired_dataset_sha256": manifest["output"]["paired_dataset_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
