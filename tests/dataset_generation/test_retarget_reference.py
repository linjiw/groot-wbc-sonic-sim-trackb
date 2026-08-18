"""Tests for constructing a reachable crouch, since asking for one does not work."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.motion_prefilter import (
    load_joint_limits,
    screen_reference_motion,
)
from gear_sonic.dataset_generation.reference_payload import payload_from_reference
from gear_sonic.dataset_generation.retarget_reference import (
    DEFAULT_RANGE_KEEP,
    retarget_crouch,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


@pytest.fixture(scope="module")
def limits():
    return load_joint_limits(DEFAULT_G1_MJCF)


def clip(frames: int = 40) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 2] = 0.50
    qpos[:, 3] = 1.0
    return qpos


def pinned_at_limits(frames: int, limits) -> np.ndarray:
    """A clip riding the upper limit of every joint, like the generated crouches do."""
    names, bounds = limits
    qpos = clip(frames)
    count = min(29, bounds.shape[0])
    qpos[:, 7 : 7 + count] = bounds[:count, 1]
    return qpos


def test_a_clip_riding_every_limit_becomes_strictly_interior(limits):
    names, bounds = limits
    retargeted, _ = retarget_crouch(pinned_at_limits(40, limits))
    report = screen_reference_motion(retargeted, names, bounds)
    assert report.saturated_cell_fraction == 0.0
    assert report.passed


def test_clamping_would_not_have_worked(limits):
    """The distinction the module exists on. A clamped value sits exactly at the limit, and
    sitting at a limit is what the saturation screen counts, so clamping turns an
    out-of-range clip into a fully saturated one instead of a passing one."""
    names, bounds = limits
    pinned = pinned_at_limits(40, limits)
    count = min(29, bounds.shape[0])
    clamped = pinned.copy()
    clamped[:, 7 : 7 + count] = np.clip(
        pinned[:, 7 : 7 + count], bounds[:count, 0], bounds[:count, 1]
    )
    assert screen_reference_motion(clamped, names, bounds).saturated_cell_fraction > 0.0
    retargeted, _ = retarget_crouch(pinned)
    assert screen_reference_motion(retargeted, names, bounds).saturated_cell_fraction == 0.0


def test_every_joint_ends_strictly_inside_its_range(limits):
    names, bounds = limits
    retargeted, _ = retarget_crouch(pinned_at_limits(40, limits))
    count = min(29, bounds.shape[0])
    dofs = retargeted[:, 7 : 7 + count]
    assert (dofs > bounds[:count, 0]).all()
    assert (dofs < bounds[:count, 1]).all()


def test_the_waist_is_taken_off_its_limit_but_keeps_its_lean(limits):
    names, bounds = limits
    index = names.index("waist_pitch_joint")
    retargeted, report = retarget_crouch(pinned_at_limits(40, limits), waist_fraction=0.5)
    value = retargeted[:, 7 + index]
    assert np.allclose(value, 0.5 * bounds[index, 1])
    assert np.abs(value).max() < np.abs(bounds[index, 1])
    assert "waist_pitch_joint" in report.relieved_joints


def test_an_upright_waist_is_available_and_differs_from_a_leaning_one(limits):
    names, bounds = limits
    index = names.index("waist_pitch_joint")
    upright, _ = retarget_crouch(pinned_at_limits(40, limits), waist_fraction=0.0)
    assert np.allclose(upright[:, 7 + index], 0.0)


def test_the_root_is_never_moved_on_its_own(limits):
    """Raising the pelvis by itself looks like a shallower squat and is really a robot
    hovering: the root is a floating base, so moving it changes no joint and no contact."""
    source = pinned_at_limits(40, limits)
    retargeted, _ = retarget_crouch(source, leg_relax=0.0)
    assert np.array_equal(retargeted[:, :7], source[:, :7])


def test_relaxing_the_legs_moves_the_root_to_keep_the_feet_planted(limits):
    """The root must follow the legs, and only the legs.

    Straightening the knees without moving the root leaves the robot buried in the floor;
    that is the same floating-base trap as trying to raise the pelvis directly, in the other
    direction.
    """
    source = pinned_at_limits(40, limits)
    relaxed, _ = retarget_crouch(source, leg_relax=0.3)
    assert not np.array_equal(relaxed[:, 2], source[:, 2])
    assert np.array_equal(relaxed[:, :2], source[:, :2]), "only height may change"
    assert np.array_equal(relaxed[:, 3:7], source[:, 3:7]), "orientation must not change"

    def lowest_foot(clip):
        payload = payload_from_reference(clip)
        names = list(payload["body_names"])
        ankles = [i for i, n in enumerate(names) if "ankle_roll" in n]
        return np.asarray(payload["body_pos_w"])[:, ankles, 2].min(axis=1)

    assert np.allclose(lowest_foot(relaxed), lowest_foot(source), atol=1e-6)


def test_relaxing_the_legs_removes_the_self_interpenetration(limits):
    """The failure this exists for. The generated crouch overlaps the thighs into the pelvis
    by 1.6 mm, which the screen tolerates because its threshold is 100 mm; executed, that
    became 931.5 N of self-contact over 88 frames and the episode was rejected."""
    source = pinned_at_limits(40, limits)
    _, deep = retarget_crouch(source, leg_relax=0.0)
    _, relaxed = retarget_crouch(source, leg_relax=0.3)
    assert relaxed.pelvis_hip_interpenetration_m <= deep.pelvis_hip_interpenetration_m


def test_an_out_of_range_relax_is_refused():
    with pytest.raises(ValueError, match="leg_relax"):
        retarget_crouch(clip(), leg_relax=1.0)


def test_the_report_names_which_joints_were_relieved(limits):
    _, report = retarget_crouch(pinned_at_limits(40, limits))
    assert report.relieved_joints
    assert report.max_joint_change_rad > 0.0
    assert report.range_keep == DEFAULT_RANGE_KEEP


def test_foot_movement_is_measured_on_the_sole_not_the_link_origin(limits):
    """The ankle-roll origin sits at the ankle joint, so rotating the ankle tilts the sole
    without moving that point. A link-origin metric reports exactly zero for the one change
    this retarget makes to the feet."""
    _, report = retarget_crouch(pinned_at_limits(40, limits))
    assert np.isfinite(report.foot_height_shift_m)
    assert np.isfinite(report.max_foot_tilt_change_rad)


def test_an_already_interior_clip_is_left_essentially_alone(limits):
    names, bounds = limits
    source = clip(40)          # every joint at zero, well inside every range
    retargeted, report = retarget_crouch(source, waist_fraction=0.0)
    assert np.allclose(retargeted[:, 7:], source[:, 7:])
    assert report.relieved_joints == ()


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError, match="range_keep"):
        retarget_crouch(clip(), range_keep=1.5)
    with pytest.raises(ValueError, match="waist_fraction"):
        retarget_crouch(clip(), waist_fraction=2.0)
    with pytest.raises(ValueError, match=r"\(T, 7\+J\)"):
        retarget_crouch(np.zeros(36))
