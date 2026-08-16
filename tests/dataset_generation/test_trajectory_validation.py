from __future__ import annotations

import numpy as np

from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory


def _trajectory(frame_count: int = 3) -> dict:
    root_quat = np.zeros((frame_count, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    reference_qpos = np.zeros((frame_count, 36), dtype=np.float32)
    reference_qpos[:, 2] = 0.8
    reference_qpos[:, 3] = 1.0
    return {
        "schema_version": 2,
        "kind": "sonic_physics_trajectory",
        "total_frames": frame_count,
        "fps": 50.0,
        "num_joints": 29,
        "dof_pos": np.zeros((frame_count, 29), dtype=np.float32),
        "dof_vel": np.zeros((frame_count, 29), dtype=np.float32),
        "root_pos_w": np.zeros((frame_count, 3), dtype=np.float32),
        "root_quat_w": root_quat,
        "root_lin_vel_w": np.zeros((frame_count, 3), dtype=np.float32),
        "root_ang_vel_w": np.zeros((frame_count, 3), dtype=np.float32),
        "projected_gravity_b": np.tile([0.0, 0.0, -1.0], (frame_count, 1)),
        "applied_joint_action": np.zeros((frame_count, 29), dtype=np.float32),
        "action_motion_token": np.ones((frame_count, 64), dtype=np.float32),
        "reference_g1_qpos": reference_qpos,
        "motion_id": np.zeros(frame_count, dtype=np.int64),
        "motion_time_step": np.arange(frame_count, dtype=np.int64),
        "motion_time_s": np.arange(frame_count, dtype=np.float32) / 50.0,
        "tracking_metrics": {"error_joint_pos": np.zeros(frame_count, dtype=np.float32)},
    }


def test_accepts_complete_sonic_physics_trajectory() -> None:
    report = validate_sonic_trajectory(_trajectory())

    assert report.ok
    assert report.frame_count == 3


def test_rejects_zero_placeholder_latent() -> None:
    trajectory = _trajectory()
    trajectory["action_motion_token"][:] = 0.0

    report = validate_sonic_trajectory(trajectory)

    assert not report.ok
    assert any("entirely zero" in error for error in report.errors)


def test_rejects_bad_reference_shape_and_reset_boundary() -> None:
    trajectory = _trajectory()
    trajectory["reference_g1_qpos"] = np.zeros((3, 35), dtype=np.float32)
    trajectory["motion_time_s"] = np.array([0.0, 0.02, 0.0], dtype=np.float32)

    report = validate_sonic_trajectory(trajectory)

    assert not report.ok
    assert any("reference_g1_qpos shape mismatch" in error for error in report.errors)
    assert any("not monotonic" in error for error in report.errors)


def test_rejects_multiple_motion_ids_by_default() -> None:
    trajectory = _trajectory()
    trajectory["motion_id"] = np.array([1, 1, 2], dtype=np.int64)

    report = validate_sonic_trajectory(trajectory)

    assert not report.ok
    assert any("multiple motion IDs" in error for error in report.errors)
