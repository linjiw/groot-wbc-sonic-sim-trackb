#!/usr/bin/env python3
"""Validate a SONIC VLA LeRobot dataset before GR00T fine-tuning.

The checks are intentionally file-based so they can run in the data collection
environment without Isaac Lab. Parquet inspection uses pandas/pyarrow when
available; metadata checks still run without them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REQUIRED_MODALITY_TOP_LEVEL = {"state", "action", "video", "annotation"}
REQUIRED_ACTION_MODALITIES = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}
REQUIRED_DATA_COLUMNS = {
    "observation.state",
    "observation.eef_state",
    "observation.root_orientation",
    "observation.projected_gravity",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "teleop.smpl_pose",
    "task_index",
}


class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)

    def print(self) -> None:
        for message in self.info:
            print(f"[info] {message}")
        for message in self.warnings:
            print(f"[warn] {message}")
        for message in self.errors:
            print(f"[error] {message}")
        print(
            json.dumps(
                {
                    "ok": not self.errors,
                    "errors": len(self.errors),
                    "warnings": len(self.warnings),
                },
                indent=2,
            )
        )


def load_json(path: Path, report: ValidationReport) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        report.error(f"missing required file: {path}")
    except json.JSONDecodeError as exc:
        report.error(f"invalid JSON in {path}: {exc}")
    return None


def find_parquet_files(dataset: Path) -> list[Path]:
    return sorted((dataset / "data").glob("*.parquet"))


def flatten_numeric(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr.reshape(-1)


def check_vector_column(
    df: Any,
    column: str,
    expected_dim: int,
    report: ValidationReport,
    max_rows: int,
) -> None:
    if column not in df.columns:
        report.error(f"missing required parquet column: {column}")
        return

    sample = df[column].head(max_rows)
    bad_shape = 0
    bad_numeric = 0
    for value in sample:
        arr = flatten_numeric(value)
        if arr.shape[0] != expected_dim:
            bad_shape += 1
            continue
        if not np.all(np.isfinite(arr.astype(float))):
            bad_numeric += 1

    if bad_shape:
        report.error(f"{column} has {bad_shape} sampled rows with wrong dim; expected {expected_dim}")
    if bad_numeric:
        report.error(f"{column} has {bad_numeric} sampled rows with NaN/Inf")


def check_no_nan_column(df: Any, column: str, report: ValidationReport, max_rows: int) -> None:
    if column not in df.columns:
        report.error(f"missing required parquet column: {column}")
        return

    bad = 0
    for value in df[column].head(max_rows):
        arr = flatten_numeric(value)
        try:
            numeric = arr.astype(float)
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(numeric)):
            bad += 1
    if bad:
        report.error(f"{column} has {bad} sampled rows with NaN/Inf")


def check_smpl_not_stale(df: Any, report: ValidationReport, max_rows: int) -> None:
    column = "teleop.smpl_pose"
    if column not in df.columns:
        return

    stale = 0
    for value in df[column].head(max_rows):
        arr = flatten_numeric(value)
        if arr.shape[0] == 63 and np.allclose(arr.astype(float), 0.0):
            stale += 1
    if stale:
        report.warn(f"{column} has {stale} sampled all-zero rows; run process_dataset.py")


def check_timestamps(df: Any, report: ValidationReport) -> None:
    if "timestamp" not in df.columns:
        report.warn("timestamp column not found; skipped frame-spacing check")
        return

    timestamps = np.asarray(df["timestamp"], dtype=float)
    if timestamps.size < 3:
        return
    diffs = np.diff(timestamps)
    if np.any(diffs < -1e-6):
        report.error("timestamps are not monotonic")
    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        report.error("timestamp median frame delta is non-positive")
        return
    max_jitter = float(np.max(np.abs(diffs - median_dt)))
    report.note(f"timestamp median dt={median_dt:.6f}s max jitter={max_jitter:.6f}s")


def check_modality(dataset: Path, report: ValidationReport) -> None:
    modality = load_json(dataset / "meta" / "modality.json", report)
    if modality is None:
        return

    missing_top = REQUIRED_MODALITY_TOP_LEVEL - set(modality)
    if missing_top:
        report.error(f"meta/modality.json missing top-level keys: {sorted(missing_top)}")

    actions = modality.get("action", {})
    for name, (original_key, expected_dim) in REQUIRED_ACTION_MODALITIES.items():
        entry = actions.get(name)
        if entry is None:
            report.error(f"meta/modality.json action missing '{name}'")
            continue
        if entry.get("original_key") != original_key:
            report.error(
                f"action.{name} original_key={entry.get('original_key')!r}; expected {original_key!r}"
            )
        if entry.get("end", 0) - entry.get("start", 0) != expected_dim:
            report.error(f"action.{name} dimension mismatch; expected {expected_dim}")

    videos = modality.get("video", {})
    if "ego_view" not in videos:
        report.error("meta/modality.json video missing 'ego_view'")

    annotation = modality.get("annotation", {})
    task_entry = annotation.get("human.task_description")
    if task_entry is None:
        report.error("meta/modality.json annotation missing 'human.task_description'")
    elif task_entry.get("original_key") != "task_index":
        report.error(
            "annotation.human.task_description original_key="
            f"{task_entry.get('original_key')!r}; expected 'task_index'"
        )


def check_info_and_videos(dataset: Path, report: ValidationReport) -> None:
    info = load_json(dataset / "meta" / "info.json", report)
    if info is None:
        return

    discarded = info.get("discarded_episode_indices", [])
    if discarded:
        report.warn(f"dataset still records discarded episodes: {discarded}")

    fps = info.get("fps")
    if fps is not None:
        report.note(f"dataset fps={fps}")

    ego_dir = dataset / "videos" / "observation.images.ego_view"
    if not ego_dir.exists():
        report.error(f"missing ego-view video directory: {ego_dir}")
        return
    videos = sorted(ego_dir.glob("*.mp4"))
    if not videos:
        report.error(f"no ego-view mp4 files found under {ego_dir}")
    else:
        report.note(f"found {len(videos)} ego-view video file(s)")


def check_parquet(dataset: Path, report: ValidationReport, max_rows: int) -> None:
    parquet_files = find_parquet_files(dataset)
    if not parquet_files:
        report.error(f"no parquet files found under {dataset / 'data'}")
        return
    report.note(f"found {len(parquet_files)} parquet file(s)")

    try:
        import pandas as pd
    except ImportError:
        report.warn("pandas is not installed; skipped parquet content checks")
        return

    for path in parquet_files:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:  # noqa: BLE001 - report all parquet read failures.
            report.error(f"failed to read parquet {path}: {exc}")
            continue

        report.note(f"{path.name}: {len(df)} rows, {len(df.columns)} columns")
        missing = REQUIRED_DATA_COLUMNS - set(df.columns)
        if missing:
            report.error(f"{path.name} missing columns: {sorted(missing)}")

        check_vector_column(df, "action.motion_token", 64, report, max_rows)
        check_vector_column(df, "teleop.left_hand_joints", 7, report, max_rows)
        check_vector_column(df, "teleop.right_hand_joints", 7, report, max_rows)

        for column in REQUIRED_DATA_COLUMNS - {
            "action.motion_token",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
            "task_index",
        }:
            check_no_nan_column(df, column, report, max_rows)
        check_smpl_not_stale(df, report, max_rows)
        check_timestamps(df, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        help="Path to a LeRobot dataset directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Path to a LeRobot dataset directory. Alias for the positional dataset argument.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500,
        help="Maximum rows per parquet file to sample for vector/numeric checks",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        help="Optional path to write the validation summary as JSON",
    )
    args = parser.parse_args()

    dataset_arg = args.dataset_path or args.dataset
    if dataset_arg is None:
        parser.error("provide a dataset path as a positional argument or --dataset-path")

    dataset = dataset_arg.resolve()
    report = ValidationReport()
    if not dataset.exists():
        report.error(f"dataset path does not exist: {dataset}")
        report.print()
        return 1
    if not dataset.is_dir():
        report.error(f"dataset path is not a directory: {dataset}")
        report.print()
        return 1

    check_modality(dataset, report)
    check_info_and_videos(dataset, report)
    check_parquet(dataset, report, args.max_rows)
    report.print()
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        with args.json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": not report.errors,
                    "errors": report.errors,
                    "warnings": report.warnings,
                    "info": report.info,
                    "dataset": str(dataset),
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
    return 1 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
