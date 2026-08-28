"""Tests for screening untrackable reference motions before simulating them."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.motion_prefilter import (
    DEFAULT_SATURATION_LIMIT,
    PRESIM_SATURATION_LIMIT,
    RECORDED_SATURATION_LIMIT,
    PrefilterError,
    load_joint_limits,
    screen_reference_motion,
)

MJCF = """<mujoco>
  <default>
    <default class="knee"><joint range="-0.087267 2.8798"/></default>
  </default>
  <worldbody>
    <body>
      <joint name="hip_pitch" range="-2.5307 2.8798"/>
      <joint name="left_knee" class="knee"/>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_UNBOUNDED = """<mujoco>
  <worldbody>
    <body>
      <joint name="hip_pitch" range="-2.5307 2.8798"/>
      <joint name="unbounded"/>
    </body>
  </worldbody>
</mujoco>
"""

NAMES = ["a", "b"]
LIMITS = np.array([[-1.0, 1.0], [-1.0, 1.0]])


def qpos(dofs: np.ndarray, root_z: float = 0.75) -> np.ndarray:
    frames = len(dofs)
    block = np.zeros((frames, 7 + dofs.shape[1]))
    block[:, 2] = root_z
    block[:, 3] = 1.0
    block[:, 7:] = dofs
    return block


def test_class_default_ranges_are_resolved(tmp_path):
    """Only some G1 joints carry an explicit range; the rest inherit from a class.

    Reading only explicit ranges leaves a third of the robot unscreened while appearing
    to work.
    """
    path = tmp_path / "g1.xml"
    path.write_text(MJCF, encoding="utf-8")
    names, limits = load_joint_limits(path)
    assert names[:2] == ["hip_pitch", "left_knee"]
    assert limits[1] == pytest.approx([-0.087267, 2.8798])


def test_joint_without_range_or_class_is_an_error(tmp_path):
    """Skipping it would silently break alignment with the qpos DOF columns.

    Limits are matched to DOFs positionally, so dropping one joint shifts every joint
    after it onto the wrong range -- and the screen would still return a plausible number.
    """
    path = tmp_path / "unbounded.xml"
    path.write_text(MJCF_UNBOUNDED, encoding="utf-8")
    with pytest.raises(PrefilterError, match="no range and no class default"):
        load_joint_limits(path)


def test_mid_range_motion_has_no_saturation():
    dofs = np.zeros((50, 2))
    report = screen_reference_motion(qpos(dofs), NAMES, LIMITS)
    assert report.saturated_cell_fraction == 0.0
    assert report.saturated_frame_fraction == 0.0
    assert report.passed
    assert report.worst_joints == ()


def test_motion_pinned_at_a_limit_is_fully_saturated():
    dofs = np.full((50, 2), 1.0)
    report = screen_reference_motion(qpos(dofs), NAMES, LIMITS)
    assert report.saturated_cell_fraction == pytest.approx(1.0)
    assert not report.passed
    assert "joint_saturation" in report.reasons[0]


def test_saturation_counts_both_ends_of_the_range():
    low = np.full((10, 2), -1.0)
    high = np.full((10, 2), 1.0)
    assert screen_reference_motion(qpos(low), NAMES, LIMITS).saturated_cell_fraction == 1.0
    assert screen_reference_motion(qpos(high), NAMES, LIMITS).saturated_cell_fraction == 1.0


def test_cell_and_frame_fractions_differ_when_one_joint_saturates():
    dofs = np.zeros((100, 2))
    dofs[:, 0] = 1.0  # one of two joints pinned, every frame
    report = screen_reference_motion(qpos(dofs), NAMES, LIMITS)
    assert report.saturated_cell_fraction == pytest.approx(0.5)
    assert report.saturated_frame_fraction == pytest.approx(1.0)


def test_worst_joints_names_the_offender_first():
    dofs = np.zeros((100, 2))
    dofs[:80, 1] = 1.0
    dofs[:20, 0] = 1.0
    report = screen_reference_motion(qpos(dofs), NAMES, LIMITS)
    assert report.worst_joints[0][0] == "b"
    assert report.worst_joints[0][1] == pytest.approx(0.8)


def test_threshold_decides_pass_or_fail():
    dofs = np.zeros((100, 2))
    dofs[:10, 0] = 1.0  # 5% of cells
    assert screen_reference_motion(qpos(dofs), NAMES, LIMITS, saturation_limit=0.10).passed
    assert not screen_reference_motion(qpos(dofs), NAMES, LIMITS, saturation_limit=0.02).passed


def test_presim_threshold_sits_between_the_measured_pass_and_fail_values():
    """Source motion at 0.0071 peaked at 318 N (under the 343 N gate); 0.0116 hit 712 N."""
    assert 0.0071 < PRESIM_SATURATION_LIMIT < 0.0116


def test_recorded_threshold_is_a_different_scale_and_not_interchangeable():
    """The same crouch reads 0.0444 raw and 0.1391 as the recorded reference.

    Screening a raw CSV against the recorded-reference threshold would pass the very
    motion that produced 9686 N of self-contact.
    """
    assert RECORDED_SATURATION_LIMIT > PRESIM_SATURATION_LIMIT * 5
    assert DEFAULT_SATURATION_LIMIT == PRESIM_SATURATION_LIMIT
    raw_crouch = 0.0444
    assert raw_crouch > PRESIM_SATURATION_LIMIT  # correctly screened
    assert raw_crouch < RECORDED_SATURATION_LIMIT  # wrongly passed on the wrong scale


def test_crouch_is_reported_but_not_screened_on_height_alone():
    """Crouching is a behaviour the corpus wants; being low is not a reason to skip."""
    report = screen_reference_motion(qpos(np.zeros((50, 2)), root_z=0.45), NAMES, LIMITS)
    assert report.is_crouch
    assert report.passed


def test_malformed_qpos_is_rejected():
    with pytest.raises(PrefilterError, match=r"\(T, 7\+J\) qpos"):
        screen_reference_motion(np.zeros((10, 4)), NAMES, LIMITS)
    with pytest.raises(PrefilterError, match=r"\(T, 7\+J\) qpos"):
        screen_reference_motion(np.zeros(10), NAMES, LIMITS)


def test_degenerate_joint_range_is_rejected():
    bad = np.array([[1.0, 1.0], [-1.0, 1.0]])
    with pytest.raises(PrefilterError, match="range must be positive"):
        screen_reference_motion(qpos(np.zeros((10, 2))), NAMES, bad)


def test_extra_dofs_beyond_known_limits_are_ignored_not_fatal():
    dofs = np.zeros((10, 5))
    report = screen_reference_motion(qpos(dofs), NAMES, LIMITS)
    assert report.frames == 10
    assert report.saturated_cell_fraction == 0.0
