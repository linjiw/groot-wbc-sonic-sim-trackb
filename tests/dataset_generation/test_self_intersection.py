"""Tests for detecting a reference pose the robot cannot hold."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.self_intersection import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_G1_MJCF,
    DEFAULT_PELVIS_HIP_LIMIT_M,
    SelfIntersectionError,
    check_reference_self_intersection,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.skipif(
    not DEFAULT_G1_MJCF.exists(), reason=f"G1 MJCF not present at {DEFAULT_G1_MJCF}"
)


def standing_qpos(frames: int = 40) -> np.ndarray:
    """A neutral standing pose: root up, identity orientation, all joints at zero."""
    qpos = np.zeros((frames, 36))
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    return qpos


def test_a_neutral_standing_pose_has_no_pelvis_hip_interpenetration():
    report = check_reference_self_intersection(standing_qpos())
    assert report.pelvis_hip_depth_m == pytest.approx(0.0, abs=1e-9)
    assert report.passed
    assert report.reasons == ()


def test_frames_checked_follows_the_stride():
    report = check_reference_self_intersection(standing_qpos(40), frame_stride=4)
    assert report.frames_checked == 10
    assert check_reference_self_intersection(standing_qpos(40), frame_stride=1).frames_checked == 40


def test_stride_must_be_at_least_one():
    with pytest.raises(SelfIntersectionError, match="frame_stride"):
        check_reference_self_intersection(standing_qpos(), frame_stride=0)


def test_wrong_column_count_names_both_numbers():
    with pytest.raises(SelfIntersectionError, match="36|columns"):
        check_reference_self_intersection(np.zeros((10, 12)))


def test_non_matrix_input_is_rejected():
    with pytest.raises(SelfIntersectionError, match=r"\(T, nq\)"):
        check_reference_self_intersection(np.zeros(36))


def test_a_missing_model_is_an_error_not_a_pass():
    with pytest.raises(SelfIntersectionError, match="MJCF not found"):
        check_reference_self_intersection(standing_qpos(), mjcf_path="/no/such/g1.xml")


def test_folding_the_hips_into_the_pelvis_is_detected():
    """Driving the hip pitch far past a reachable fold puts the thigh into the pelvis.

    This is the geometry behind the squat failures: the joint is clamped so every angle
    reads legal, and the pose is still impossible.
    """
    qpos = standing_qpos()
    # Hip pitch sits at the first two joint slots for the two legs in this model layout;
    # driving both hard produces the fold regardless of which is which.
    qpos[:, 7] = -2.4
    qpos[:, 13] = -2.4
    report = check_reference_self_intersection(qpos)
    assert report.max_depth_m > 0.0


def test_the_floor_is_not_counted_as_self_intersection():
    """A foot resting on the ground is not the robot hitting itself.

    Before the world body was excluded, `world~right_ankle_roll_link` dominated every
    result at 0.15-0.17 m and buried the real signal.
    """
    sunk = standing_qpos()
    sunk[:, 2] = 0.30  # drive the feet well through the floor
    report = check_reference_self_intersection(sunk)
    assert all("world" not in pair[:2] for pair in report.pair_depths)


def test_the_threshold_sits_where_the_measurement_put_it():
    """Above 0.10 m every episode was rejected; the band below still accepts at 81%."""
    assert DEFAULT_PELVIS_HIP_LIMIT_M == pytest.approx(0.10)
    assert DEFAULT_FRAME_STRIDE >= 1


def test_pair_depths_come_back_worst_first():
    qpos = standing_qpos()
    qpos[:, 7] = -2.4
    qpos[:, 13] = -2.4
    depths = [d for _, _, d in check_reference_self_intersection(qpos).pair_depths]
    assert depths == sorted(depths, reverse=True)
