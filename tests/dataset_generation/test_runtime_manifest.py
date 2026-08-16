from __future__ import annotations

import json
from pathlib import Path

import pytest

from gear_sonic.dataset_generation.runtime_manifest import (
    build_runtime_capture_context,
    invalidate_runtime_success_manifest,
    read_runtime_success_manifest,
    write_runtime_success_manifest,
)


def test_capture_context_hashes_scene_and_motion_inputs(tmp_path: Path) -> None:
    scene_path = tmp_path / "factory_aisle.usda"
    motion_path = tmp_path / "motion.pkl"
    scene_path.write_bytes(b"scene")
    motion_path.write_bytes(b"motion")

    context = build_runtime_capture_context(
        terrain_type="scene_usd",
        scene_id=None,
        scene_usd_path=scene_path,
        task="walk through the aisle",
        motion_file=motion_path,
        camera_provenance="isaac_sim",
        camera_name="ego_camera",
        camera_track_root=False,
        render_ego=True,
        camera_attached_link="torso_link/head_link",
        camera_resolution=[480, 640],
        render_frame_skip=1,
        use_encoder="g1",
    )

    assert context["scene_id"] == "factory_aisle"
    assert context["scene"]["hash"].startswith("sha256:")
    assert context["motion"]["hash"].startswith("sha256:")
    assert context["camera"]["name"] == "ego_camera"


def test_runtime_success_manifest_round_trip(tmp_path: Path) -> None:
    trajectory_dir = tmp_path / "trajectory"
    rendering_dir = tmp_path / "rendering"
    trajectory_dir.mkdir()
    rendering_dir.mkdir()
    trajectory_path = trajectory_dir / "000000.trajectory.pkl"
    video_path = rendering_dir / "000000.mp4"
    trajectory_path.write_bytes(b"trajectory")
    video_path.write_bytes(b"video")
    path = write_runtime_success_manifest(
        tmp_path / "run" / "success.json",
        checkpoint="sonic_release/last.pt",
        exit_reason="max_render_steps",
        num_envs=1,
        policy_iterations=66,
        physics_steps=65,
        artifacts={
            "trajectory_dir": str(trajectory_dir),
            "rendering_dir": str(rendering_dir),
        },
    )

    payload = read_runtime_success_manifest(path)

    assert payload["status"] == "success"
    assert payload["physics_steps"] == 65
    assert payload["schema_version"] == 2
    assert payload["artifacts"]["trajectory_dir"] == str(trajectory_dir)
    assert payload["artifact_pairs"]["000000"]["trajectory_path"] == str(trajectory_path)
    assert payload["artifact_pairs"]["000000"]["video_path"] == str(video_path)
    assert payload["artifact_pairs"]["000000"]["trajectory_hash"].startswith("sha256:")
    assert payload["artifact_pairs"]["000000"]["video_hash"].startswith("sha256:")
    assert not list(path.parent.glob("*.tmp"))


def test_runtime_manifest_rejects_zero_step_marker(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sonic_isaac_eval_success",
                "status": "success",
                "physics_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one physics step"):
        read_runtime_success_manifest(path)


def test_runtime_manifest_rejects_impossible_counts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        write_runtime_success_manifest(
            tmp_path / "success.json",
            checkpoint="checkpoint.pt",
            exit_reason="bad",
            num_envs=1,
            policy_iterations=2,
            physics_steps=3,
            artifacts={},
        )


def test_invalidates_stale_success_before_new_run(tmp_path: Path) -> None:
    path = tmp_path / "success.json"
    path.write_text('{"status": "success"}\n', encoding="utf-8")

    invalidate_runtime_success_manifest(path)

    assert not path.exists()
    invalidate_runtime_success_manifest(path)
