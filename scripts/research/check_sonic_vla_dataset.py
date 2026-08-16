#!/usr/bin/env python3
"""Validate a SONIC VLA LeRobot dataset before GR00T fine-tuning.

The checks are intentionally file-based so they can run in the data collection
environment without Isaac Lab. Parquet inspection uses pandas/pyarrow when
available; metadata checks still run without them.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

REQUIRED_MODALITY_TOP_LEVEL = {"state", "action", "video", "annotation"}
REQUIRED_ACTION_MODALITIES = {
    "motion_token": ("action.motion_token", 64),
    "left_hand_joints": ("teleop.left_hand_joints", 7),
    "right_hand_joints": ("teleop.right_hand_joints", 7),
}
LIVE_VR_DATA_COLUMNS = {
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
SYNTHETIC_G1_DATA_COLUMNS = {
    "observation.state",
    "observation.projected_gravity",
    "action.motion_token",
    "teleop.left_hand_joints",
    "teleop.right_hand_joints",
    "reference.g1_qpos",
    "task_index",
}
SYNTHETIC_G1_FORBIDDEN_FEATURES = {
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.smpl_frame_index",
    "teleop.vr_3pt_position",
    "teleop.vr_3pt_orientation",
}
GROOT_SONIC_ACTION_HORIZON = 40
GROOT_FLOAT32_FEATURE_SHAPES = {
    "observation.state": (43,),
    "observation.projected_gravity": (3,),
    "action.motion_token": (64,),
    "teleop.left_hand_joints": (7,),
    "teleop.right_hand_joints": (7,),
    "reference.g1_qpos": (36,),
}
GROOT_FLOAT32_FEATURES = set(GROOT_FLOAT32_FEATURE_SHAPES)


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


def load_jsonl(path: Path, report: ValidationReport) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    report.error(f"invalid JSON in {path}:{line_number}: {exc}")
                    continue
                if not isinstance(value, dict):
                    report.error(f"expected JSON object in {path}:{line_number}")
                    continue
                rows.append(value)
    except FileNotFoundError:
        report.error(f"missing required file: {path}")
    return rows


def _parse_video_rate(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if np.isfinite(rate) and rate > 0 else None


def _parse_video_frame_count(stream: dict[str, Any]) -> int | None:
    for key in ("nb_read_frames", "nb_frames"):
        value = stream.get(key)
        try:
            frame_count = int(value)
        except (TypeError, ValueError):
            continue
        if frame_count >= 0:
            return frame_count
    return None


def probe_video(
    path: Path,
    report: ValidationReport,
    *,
    expected_frames: int | None = None,
    expected_fps: float | None = None,
) -> None:
    """Require a non-empty, decodable video stream for the strict loader gate."""
    if path.stat().st_size <= 0:
        report.error(f"empty video file: {path}")
        return

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        report.error("ffprobe is required for --groot-loader-ready video validation")
        return
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            (
                "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,"
                "nb_frames,nb_read_frames,duration"
            ),
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        report.error(f"video is not decodable: {path}: {detail}")
        return
    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        report.error(f"invalid ffprobe output for {path}: {exc}")
        return
    streams = probe.get("streams", [])
    if not streams:
        report.error(f"video has no decodable stream: {path}")
        return
    stream = streams[0]
    if stream.get("width") != 640 or stream.get("height") != 480:
        report.error(
            f"ego video resolution must be 640x480 for UNITREE_G1_SONIC; "
            f"got {stream.get('width')}x{stream.get('height')} in {path}"
        )
    if expected_frames is not None:
        frame_count = _parse_video_frame_count(stream)
        if frame_count is None:
            report.error(f"could not determine decoded frame count for {path}")
        elif frame_count != expected_frames:
            report.error(
                f"video/episode frame mismatch for {path}: "
                f"video={frame_count}, episode={expected_frames}"
            )
    if expected_fps is not None:
        fps = _parse_video_rate(stream.get("avg_frame_rate"))
        if fps is None:
            fps = _parse_video_rate(stream.get("r_frame_rate"))
        if fps is None:
            report.error(f"could not determine video FPS for {path}")
        elif not np.isclose(fps, expected_fps, atol=0.05, rtol=0.0):
            report.error(
                f"video/dataset FPS mismatch for {path}: video={fps:g}, "
                f"dataset={expected_fps:g}"
            )


def _parquet_row_array(value: Any) -> np.ndarray:
    """Return one parquet cell without flattening away its declared feature shape."""
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        array = np.asarray(array.reshape(-1)[0])
    return array


def _check_accepted_float32_feature(
    frame_subset: Any,
    *,
    path: Path,
    key: str,
    expected_shape: tuple[int, ...],
    report: ValidationReport,
) -> list[np.ndarray]:
    """Exhaustively validate one strict numeric feature for accepted rows."""
    if key not in frame_subset.columns:
        report.error(f"{path.name} missing strict training column: {key}")
        return []

    bad_shape = 0
    bad_dtype = 0
    nonnumeric = 0
    nonfinite = 0
    valid: list[np.ndarray] = []
    for value in frame_subset[key]:
        try:
            array = _parquet_row_array(value)
        except (TypeError, ValueError):
            bad_shape += 1
            bad_dtype += 1
            nonnumeric += 1
            continue
        shape_ok = array.shape == expected_shape
        dtype_ok = array.dtype == np.dtype("float32")
        try:
            numeric = np.issubdtype(array.dtype, np.number)
        except TypeError:
            numeric = False

        finite = False
        if numeric:
            try:
                finite = bool(np.all(np.isfinite(array)))
            except TypeError:
                numeric = False

        if not shape_ok:
            bad_shape += 1
        if not dtype_ok:
            bad_dtype += 1
        if not numeric:
            nonnumeric += 1
        elif not finite:
            nonfinite += 1
        if shape_ok and dtype_ok and numeric and finite:
            valid.append(array)

    prefix = f"{path.name} accepted rows in column {key}"
    if bad_shape:
        report.error(
            f"{prefix} contain {bad_shape} value(s) with wrong shape; expected {expected_shape}"
        )
    if bad_dtype:
        report.error(f"{prefix} contain {bad_dtype} value(s) not stored as float32")
    if nonnumeric:
        report.error(f"{prefix} contain {nonnumeric} nonnumeric value(s)")
    if nonfinite:
        report.error(f"{prefix} contain {nonfinite} value(s) with NaN/Inf")
    return valid


def check_groot_loader_ready(dataset: Path, report: ValidationReport, *, profile: str) -> None:
    """Check invariants required by the official UNITREE_G1_SONIC loader.

    This is a file-level compatibility gate. A real Isaac-GR00T loader smoke is
    still required before publishing a dataset or changing the pinned GR00T
    revision.
    """
    if profile != "synthetic_g1":
        report.error("--groot-loader-ready currently requires --profile synthetic_g1")

    info = load_json(dataset / "meta" / "info.json", report)
    if info is None:
        return

    for key in ("features", "data_path", "video_path", "chunks_size"):
        if key not in info:
            report.error(f"meta/info.json missing GR00T loader field: {key}")
    if not isinstance(info.get("features"), dict) or not info.get("features"):
        report.error("meta/info.json features must be a non-empty object")
    if not isinstance(info.get("data_path"), str) or "{episode_index" not in info.get(
        "data_path", ""
    ):
        report.error("meta/info.json data_path must contain an episode_index template")
    if not isinstance(info.get("video_path"), str) or "{video_key" not in info.get(
        "video_path", ""
    ):
        report.error("meta/info.json video_path must contain a video_key template")
    if (
        not isinstance(info.get("chunks_size"), int)
        or isinstance(info.get("chunks_size"), bool)
        or info.get("chunks_size", 0) < 1
    ):
        report.error("meta/info.json chunks_size must be a positive integer")

    features = info.get("features", {})
    for key, expected_shape in GROOT_FLOAT32_FEATURE_SHAPES.items():
        entry = features.get(key)
        if not isinstance(entry, dict):
            report.error(f"meta/info.json features missing {key}")
            continue
        if entry.get("dtype") != "float32":
            report.error(
                f"{key} must be stored as float32 for the synthetic GR00T contract; "
                f"got {entry.get('dtype')!r}"
            )
        declared_shape = entry.get("shape")
        if not isinstance(declared_shape, (list, tuple)) or tuple(declared_shape) != expected_shape:
            report.error(
                f"meta/info.json feature {key} has shape {declared_shape!r}; "
                f"expected {list(expected_shape)}"
            )

    stats = load_json(dataset / "meta" / "stats.json", report)
    if stats is not None:
        for key in GROOT_FLOAT32_FEATURES:
            if key not in stats:
                report.error(f"meta/stats.json missing training feature: {key}")

    episodes = load_jsonl(dataset / "meta" / "episodes.jsonl", report)
    discarded = set(info.get("discarded_episode_indices", []))
    accepted = [episode for episode in episodes if episode.get("episode_index") not in discarded]
    if not accepted:
        report.error("dataset contains no accepted episodes")
    for episode in accepted:
        episode_index = episode.get("episode_index")
        length = episode.get("length")
        if not isinstance(length, int) or isinstance(length, bool):
            report.error(f"episode {episode_index} has invalid length: {length!r}")
        elif length < GROOT_SONIC_ACTION_HORIZON:
            report.error(
                f"episode {episode_index} has {length} frames; UNITREE_G1_SONIC "
                f"requires at least {GROOT_SONIC_ACTION_HORIZON} for its action horizon"
            )

    total_episodes = info.get("total_episodes")
    if isinstance(total_episodes, int) and total_episodes != len(episodes):
        report.error(
            f"meta/info.json total_episodes={total_episodes} but episodes.jsonl has "
            f"{len(episodes)} entries"
        )
    declared_total_frames = info.get("total_frames")
    episode_total_frames = sum(
        episode["length"] for episode in episodes if isinstance(episode.get("length"), int)
    )
    if isinstance(declared_total_frames, int) and declared_total_frames != episode_total_frames:
        report.error(
            f"meta/info.json total_frames={declared_total_frames} but episodes.jsonl "
            f"sums to {episode_total_frames}"
        )

    parquet_files = find_parquet_files(dataset)
    if len(parquet_files) != len(episodes):
        report.error(
            f"expected one parquet per episode ({len(episodes)}); found {len(parquet_files)}"
        )

    try:
        import pandas as pd
    except ImportError:
        report.error("pandas/pyarrow are required for --groot-loader-ready")
    else:
        accepted_indices = {
            episode.get("episode_index") for episode in accepted if "episode_index" in episode
        }
        token_values: list[np.ndarray] = []
        observed_episode_lengths: dict[int, int] = {}
        for path in parquet_files:
            try:
                df = pd.read_parquet(path)
            except Exception as exc:  # noqa: BLE001
                report.error(f"failed to read parquet {path}: {exc}")
                continue
            if "episode_index" in df.columns:
                for episode_index, count in df["episode_index"].value_counts().items():
                    if not isinstance(episode_index, (int, np.integer)) or isinstance(
                        episode_index, (bool, np.bool_)
                    ):
                        report.error(
                            f"{path.name} contains noninteger episode_index {episode_index!r}"
                        )
                        continue
                    observed_episode_lengths[int(episode_index)] = observed_episode_lengths.get(
                        int(episode_index), 0
                    ) + int(count)
                frame_subset = df[df["episode_index"].isin(accepted_indices)]
            else:
                report.error(f"{path.name} missing episode_index; cannot identify accepted rows")
                frame_subset = df
            for key, expected_shape in GROOT_FLOAT32_FEATURE_SHAPES.items():
                valid = _check_accepted_float32_feature(
                    frame_subset,
                    path=path,
                    key=key,
                    expected_shape=expected_shape,
                    report=report,
                )
                if key == "action.motion_token":
                    token_values.extend(valid)

        for episode in accepted:
            episode_index = episode.get("episode_index")
            observed_length = observed_episode_lengths.get(episode_index, 0)
            if observed_length != episode.get("length"):
                report.error(
                    f"episode {episode_index} metadata length={episode.get('length')} but parquet "
                    f"contains {observed_length} frames"
                )
        if token_values:
            if not any(np.any(np.abs(token) > 1e-12) for token in token_values):
                report.error("accepted action.motion_token data is entirely zero")

    videos = sorted(
        path
        for path in (dataset / "videos").rglob("*.mp4")
        if "observation.images.ego_view" in path.parts
    )
    if len(videos) != len(episodes):
        report.error(f"expected one ego video per episode ({len(episodes)}); found {len(videos)}")
    episode_lengths = {
        episode.get("episode_index"): episode.get("length")
        for episode in episodes
        if isinstance(episode.get("episode_index"), int) and isinstance(episode.get("length"), int)
    }
    expected_fps = info.get("fps")
    if not isinstance(expected_fps, int | float) or isinstance(expected_fps, bool):
        expected_fps = None
    observed_video_indices: list[int] = []
    for path in videos:
        episode_suffix = path.stem.removeprefix("episode_")
        try:
            video_episode_index = int(episode_suffix)
        except ValueError:
            report.error(f"could not infer episode index from ego video filename: {path}")
            probe_video(path, report, expected_fps=expected_fps)
            continue
        observed_video_indices.append(video_episode_index)
        expected_frames = episode_lengths.get(video_episode_index)
        if expected_frames is None:
            report.error(f"ego video has no matching episode metadata: {path}")
        probe_video(
            path,
            report,
            expected_frames=expected_frames,
            expected_fps=expected_fps,
        )
    expected_video_indices = sorted(episode_lengths)
    if sorted(observed_video_indices) != expected_video_indices:
        report.error(
            "ego video episode indices do not match episodes.jsonl: "
            f"videos={sorted(observed_video_indices)}, episodes={expected_video_indices}"
        )

    if not report.errors:
        report.note(
            f"GR00T file gate passed for {len(accepted)} accepted episode(s), "
            f"action horizon={GROOT_SONIC_ACTION_HORIZON}"
        )


def find_parquet_files(dataset: Path) -> list[Path]:
    return sorted((dataset / "data").rglob("*.parquet"))


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
        report.error(
            f"{column} has {bad_shape} sampled rows with wrong dim; expected {expected_dim}"
        )
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


def check_modality(dataset: Path, report: ValidationReport, *, profile: str = "live_vr") -> None:
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

    if profile == "synthetic_g1":
        expected_states = {
            "left_leg",
            "right_leg",
            "waist",
            "left_arm",
            "right_arm",
            "left_hand",
            "right_hand",
            "projected_gravity",
        }
        missing_states = expected_states - set(modality.get("state", {}))
        if missing_states:
            report.error(
                "synthetic_g1 modality missing registered G1 SONIC states: "
                f"{sorted(missing_states)}"
            )
        unexpected_actions = set(actions) - set(REQUIRED_ACTION_MODALITIES)
        if unexpected_actions:
            report.error(
                "synthetic_g1 modality contains non-training actions: "
                f"{sorted(unexpected_actions)}"
            )

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


def check_info_and_videos(
    dataset: Path, report: ValidationReport, *, profile: str = "live_vr"
) -> None:
    info = load_json(dataset / "meta" / "info.json", report)
    if info is None:
        return

    discarded = info.get("discarded_episode_indices", [])
    if discarded:
        report.warn(f"dataset still records discarded episodes: {discarded}")

    fps = info.get("fps")
    if fps is not None:
        report.note(f"dataset fps={fps}")

    if profile == "synthetic_g1":
        declared_features = set(info.get("features", {}))
        forbidden = SYNTHETIC_G1_FORBIDDEN_FEATURES & declared_features
        if forbidden:
            report.error(
                "synthetic_g1 metadata declares unavailable human/VR features: "
                f"{sorted(forbidden)}"
            )

    videos_root = dataset / "videos"
    if not videos_root.exists():
        report.error(f"missing video directory: {videos_root}")
        return
    videos = sorted(
        path for path in videos_root.rglob("*.mp4") if "observation.images.ego_view" in path.parts
    )
    if not videos:
        report.error(f"no ego-view mp4 files found under {videos_root}")
    else:
        report.note(f"found {len(videos)} ego-view video file(s)")


def check_parquet(
    dataset: Path, report: ValidationReport, max_rows: int, *, profile: str = "live_vr"
) -> None:
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
        required_columns = (
            SYNTHETIC_G1_DATA_COLUMNS if profile == "synthetic_g1" else LIVE_VR_DATA_COLUMNS
        )
        missing = required_columns - set(df.columns)
        if missing:
            report.error(f"{path.name} missing columns: {sorted(missing)}")
        if profile == "synthetic_g1":
            forbidden_columns = SYNTHETIC_G1_FORBIDDEN_FEATURES & set(df.columns)
            if forbidden_columns:
                report.error(
                    f"{path.name} contains unavailable human/VR columns: "
                    f"{sorted(forbidden_columns)}"
                )

        check_vector_column(df, "action.motion_token", 64, report, max_rows)
        check_vector_column(df, "teleop.left_hand_joints", 7, report, max_rows)
        check_vector_column(df, "teleop.right_hand_joints", 7, report, max_rows)
        if profile == "synthetic_g1":
            check_vector_column(df, "reference.g1_qpos", 36, report, max_rows)

        for column in required_columns - {
            "action.motion_token",
            "teleop.left_hand_joints",
            "teleop.right_hand_joints",
            "reference.g1_qpos",
            "task_index",
        }:
            check_no_nan_column(df, column, report, max_rows)
        if profile == "live_vr":
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
        "--profile",
        choices=("live_vr", "synthetic_g1"),
        default="live_vr",
        help="Validation profile; live_vr preserves the existing teleoperation checks",
    )
    parser.add_argument(
        "--groot-loader-ready",
        action="store_true",
        help=(
            "Apply the strict UNITREE_G1_SONIC file gate: aggregate stats, float32 "
            "training features, decodable videos, and >=40-frame accepted episodes"
        ),
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
    report.note(f"validation profile={args.profile}")
    if not dataset.exists():
        report.error(f"dataset path does not exist: {dataset}")
        report.print()
        return 1
    if not dataset.is_dir():
        report.error(f"dataset path is not a directory: {dataset}")
        report.print()
        return 1

    check_modality(dataset, report, profile=args.profile)
    check_info_and_videos(dataset, report, profile=args.profile)
    check_parquet(dataset, report, args.max_rows, profile=args.profile)
    if args.groot_loader_ready:
        check_groot_loader_ready(dataset, report, profile=args.profile)
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
