from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.research.check_sonic_vla_dataset import (
    GROOT_FLOAT32_FEATURE_SHAPES,
    ValidationReport,
    check_groot_loader_ready,
    probe_video,
)
from scripts.research.create_tiny_sonic_vla_fixture import create_fixture


def _make_loader_ready_fixture(dataset: Path) -> tuple[Path, pd.DataFrame]:
    create_fixture(dataset, episodes=1, frames=40, fps=50, profile="synthetic_g1")
    parquet_path = dataset / "data" / "train-00000-of-00001.parquet"
    frame = pd.read_parquet(parquet_path)
    for key in GROOT_FLOAT32_FEATURE_SHAPES:
        frame[key] = [np.asarray(value, dtype=np.float32) for value in frame[key]]
    frame.to_parquet(parquet_path, index=False)

    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": (
                "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            ),
            "features": {
                key: {"dtype": "float32", "shape": list(shape)}
                for key, shape in GROOT_FLOAT32_FEATURE_SHAPES.items()
            },
        }
    )
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (dataset / "meta" / "stats.json").write_text(
        json.dumps({key: {} for key in GROOT_FLOAT32_FEATURE_SHAPES}),
        encoding="utf-8",
    )
    return parquet_path, frame


def test_strict_gate_rejects_schema_only_short_fixture(tmp_path: Path) -> None:
    dataset = tmp_path / "fixture"
    create_fixture(dataset, episodes=1, frames=8, fps=50, profile="synthetic_g1")
    report = ValidationReport()

    check_groot_loader_ready(dataset, report, profile="synthetic_g1")

    assert report.errors
    assert any("missing GR00T loader field: features" in error for error in report.errors)
    assert any("meta/stats.json" in error for error in report.errors)
    assert any("requires at least 40" in error for error in report.errors)
    assert any("empty video file" in error for error in report.errors)


def test_video_probe_rejects_frame_count_and_fps_mismatch(tmp_path: Path, monkeypatch) -> None:
    video_path = tmp_path / "episode_000000.mp4"
    video_path.write_bytes(b"not-empty")
    probe_output = {
        "streams": [
            {
                "width": 640,
                "height": 480,
                "avg_frame_rate": "25/1",
                "r_frame_rate": "25/1",
                "nb_frames": "174",
                "nb_read_frames": "174",
            }
        ]
    }
    monkeypatch.setattr(
        "scripts.research.check_sonic_vla_dataset.shutil.which",
        lambda command: "/usr/bin/ffprobe",
    )
    monkeypatch.setattr(
        "scripts.research.check_sonic_vla_dataset.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(probe_output),
            stderr="",
        ),
    )
    report = ValidationReport()

    probe_video(video_path, report, expected_frames=175, expected_fps=50.0)

    assert any("video=174, episode=175" in error for error in report.errors)
    assert any("video=25, dataset=50" in error for error in report.errors)


def test_strict_gate_requires_float32_reference_qpos_metadata(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "fixture"
    _make_loader_ready_fixture(dataset)
    info_path = dataset / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["reference.g1_qpos"]["dtype"] = "float64"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.research.check_sonic_vla_dataset.probe_video",
        lambda *args, **kwargs: None,
    )
    report = ValidationReport()

    check_groot_loader_ready(dataset, report, profile="synthetic_g1")

    assert any("reference.g1_qpos must be stored as float32" in error for error in report.errors)


def test_strict_gate_accepts_all_valid_float32_rows(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "fixture"
    _make_loader_ready_fixture(dataset)
    monkeypatch.setattr(
        "scripts.research.check_sonic_vla_dataset.probe_video",
        lambda *args, **kwargs: None,
    )
    report = ValidationReport()

    check_groot_loader_ready(dataset, report, profile="synthetic_g1")

    assert report.errors == []


def test_strict_gate_exhaustively_rejects_late_and_nonnumeric_rows(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = tmp_path / "fixture"
    parquet_path, frame = _make_loader_ready_fixture(dataset)
    frame.at[20, "action.motion_token"] = np.zeros(63, dtype=np.float32)
    frame.at[21, "reference.g1_qpos"] = np.zeros(36, dtype=np.float64)
    nonfinite_state = np.zeros(43, dtype=np.float32)
    nonfinite_state[0] = np.nan
    frame.at[22, "observation.state"] = nonfinite_state
    frame.at[23, "observation.projected_gravity"] = np.asarray(["invalid", "0", "-1"])
    monkeypatch.setattr(pd, "read_parquet", lambda path: frame.copy())
    monkeypatch.setattr(
        "scripts.research.check_sonic_vla_dataset.probe_video",
        lambda *args, **kwargs: None,
    )
    report = ValidationReport()

    check_groot_loader_ready(dataset, report, profile="synthetic_g1")

    assert any(
        parquet_path.name in error and "action.motion_token" in error and "wrong shape" in error
        for error in report.errors
    )
    assert any(
        "reference.g1_qpos" in error and "not stored as float32" in error for error in report.errors
    )
    assert any("observation.state" in error and "NaN/Inf" in error for error in report.errors)
    assert any(
        "observation.projected_gravity" in error and "nonnumeric" in error
        for error in report.errors
    )
