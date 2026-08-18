"""Tests for bending one nominal motion locally, rather than pairing two separate clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.local_adaptation import (
    CROUCH_JOINTS,
    adaptation_profile,
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


def walk(frames: int = 90) -> np.ndarray:
    """A body walking along +x with a small leg swing."""
    qpos = np.zeros((frames, 36))
    qpos[:, 0] = np.linspace(0.0, 4.0, frames)
    qpos[:, 2] = 0.74
    qpos[:, 3] = 1.0
    phase = np.linspace(0.0, 6 * np.pi, frames)
    qpos[:, 7] = 0.25 * np.sin(phase)          # left hip pitch
    qpos[:, 10] = 0.30 + 0.20 * np.sin(phase)  # left knee
    qpos[:, 13] = 0.25 * np.sin(phase + np.pi)
    qpos[:, 16] = 0.30 + 0.20 * np.sin(phase + np.pi)
    return qpos


# ---- route progress and the profile -------------------------------------------------------

def test_progress_is_path_length_not_frame_index():
    """An obstacle sits at a place. Two motions of equal duration reach it at different
    frames, so a frame-indexed window would centre the adaptation somewhere else."""
    xy = np.zeros((100, 2))
    xy[:50, 0] = np.linspace(0.0, 0.5, 50)      # slow first half
    xy[50:, 0] = np.linspace(0.5, 4.0, 50)      # fast second half
    progress = route_progress(xy)
    assert progress[0] == pytest.approx(0.0)
    assert progress[-1] == pytest.approx(1.0)
    assert progress[50] < 0.25, "half the frames, far less than half the route"


def test_a_stationary_motion_still_yields_monotone_progress():
    progress = route_progress(np.zeros((20, 2)))
    assert np.all(np.diff(progress) >= 0)
    assert progress[-1] == pytest.approx(1.0)


def test_the_profile_peaks_at_the_station_and_vanishes_away_from_it():
    progress = np.linspace(0.0, 1.0, 200)
    alpha = adaptation_profile(progress, 0.5, window=0.2)
    assert alpha.max() == pytest.approx(1.0, abs=1e-6)
    assert alpha[np.argmin(np.abs(progress - 0.5))] == pytest.approx(1.0, abs=1e-6)
    assert alpha[0] == 0.0 and alpha[-1] == 0.0


def test_the_profile_has_no_step_a_tracker_would_fight():
    alpha = adaptation_profile(np.linspace(0.0, 1.0, 400), 0.5, window=0.2)
    assert np.abs(np.diff(alpha)).max() < 0.05


def test_the_adaptation_is_local_rather_than_whole_route():
    """A clip crouched from frame zero cannot show a decision made from what the robot sees:
    the obstacle is not visible when the crouch begins."""
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.15)
    assert report.active_fraction < 0.6
    assert report.active_fraction > 0.05


# ---- what must not change -----------------------------------------------------------------

def test_the_root_path_and_heading_are_untouched():
    """Both motions must walk the same line, or the scene separated two journeys rather than
    two behaviours."""
    nominal = walk()
    adapted, report = local_crouch(nominal, 0.55, target_drop_m=0.15)
    assert np.array_equal(adapted[:, :2], nominal[:, :2])
    assert np.array_equal(adapted[:, 3:7], nominal[:, 3:7])
    assert report.root_path_preserved


def test_duration_and_frame_count_are_preserved():
    nominal = walk(90)
    adapted, _ = local_crouch(nominal, 0.55, target_drop_m=0.15)
    assert adapted.shape == nominal.shape


def test_the_crouch_is_knee_driven_and_never_folds_the_waist():
    """The failure mode of every generated crouch, excluded by construction: the waist is not
    in the joint set the operator may touch."""
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.20)
    assert report.waist_change_rad == pytest.approx(0.0, abs=1e-9)
    assert not any("waist" in joint for joint in CROUCH_JOINTS)


def test_the_feet_stay_where_the_nominal_put_them():
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.15)
    assert report.foot_height_shift_m < 1e-3


def test_frames_outside_the_window_are_bit_identical_to_the_nominal():
    """Anything else would make the two motions differ where the obstacle is not."""
    nominal = walk()
    adapted, _ = local_crouch(nominal, 0.5, target_drop_m=0.15, window=0.15)
    alpha = adaptation_profile(route_progress(nominal[:, :2]), 0.5, window=0.15)
    quiet = alpha < 1e-6
    assert quiet.any()
    assert np.allclose(adapted[quiet, 7:], nominal[quiet, 7:], atol=1e-9)


# ---- what it achieves ----------------------------------------------------------------------

def test_a_deeper_target_produces_a_deeper_crouch_until_the_cap_binds():
    """Below the excursion cap the target is honoured; above it the clip is capped and says
    so, rather than reaching the target through a motion the robot cannot hold on to its
    route. Unbounded, this operator asked the knee for 1.629 rad -- 93 degrees beyond its
    gait -- and the result lost 0.71 m of forward progress in execution."""
    _, shallow = local_crouch(walk(), 0.55, target_drop_m=0.02, max_excursion=1.0)
    _, deep = local_crouch(walk(), 0.55, target_drop_m=0.10, max_excursion=1.0)
    assert deep.silhouette_drop_m > shallow.silhouette_drop_m


def test_a_target_beyond_the_cap_is_reported_rather_than_reached():
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.50, max_excursion=0.4)
    assert report.excursion_capped
    assert report.max_joint_change_rad <= 0.4 + 1e-6
    assert report.silhouette_drop_m < 0.50


def test_the_target_drop_is_hit_when_the_cap_allows_it():
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.05, max_excursion=1.2)
    assert not report.excursion_capped
    assert report.silhouette_drop_m == pytest.approx(0.05, abs=0.02)


def test_the_silhouette_is_a_capsule_surface_and_not_a_joint_centre():
    """What an overhead obstacle meets is the top of the collision geometry, which sits a
    capsule radius above the highest link origin. Reporting the joint centre would understate
    the robot's height and let a shelf be placed where it would actually be hit.

    The *change* under a knee-driven crouch is close to the root's, and should be: an upright
    torso lowers uniformly. It is the absolute height that a joint-centre proxy gets wrong.
    """
    from gear_sonic.dataset_generation.reference_payload import payload_from_reference

    nominal = walk()
    _, report = local_crouch(nominal, 0.55, target_drop_m=0.15)
    payload = payload_from_reference(nominal)
    highest_link = float(np.asarray(payload["body_pos_w"])[:, :, 2].max())
    assert report.nominal_silhouette_m > highest_link, (
        "the silhouette must clear the highest link origin by a capsule radius"
    )


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError, match="station_fraction"):
        local_crouch(walk(), 1.5)
    with pytest.raises(ValueError, match="target_drop_m"):
        local_crouch(walk(), 0.5, target_drop_m=-0.1)
    with pytest.raises(ValueError, match=r"\(T, 7\+J\)"):
        local_crouch(np.zeros(36), 0.5)


# ---- arm tuck ------------------------------------------------------------------------------

def swinging_arms(frames: int = 90) -> np.ndarray:
    """A walk whose arms swing out to the sides, so there is width to take in."""
    qpos = walk(frames)
    phase = np.linspace(0.0, 6 * np.pi, frames)
    from gear_sonic.dataset_generation.motion_prefilter import load_joint_limits
    names, _ = load_joint_limits(DEFAULT_G1_MJCF)
    for index, name in enumerate(names[:29]):
        if "shoulder_roll" in name:
            sign = 1.0 if name.startswith("left") else -1.0
            qpos[:, 7 + index] = sign * (0.35 + 0.15 * np.sin(phase))
    return qpos


def test_the_tuck_leaves_the_legs_and_the_root_alone():
    """The reason this operator is easier than the crouch: nothing about the support or
    contact schedule moves, so a tracker has only the arms to follow differently."""
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    nominal = swinging_arms()
    adapted, report = local_arm_tuck(nominal, 0.55, target_reduction_m=0.06)
    assert report.leg_change_rad == pytest.approx(0.0, abs=1e-12)
    assert np.array_equal(adapted[:, :7], nominal[:, :7])
    assert report.root_path_preserved


def test_the_tuck_narrows_the_robot_where_the_obstacle_is():
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    _, report = local_arm_tuck(swinging_arms(), 0.55, target_reduction_m=0.06)
    assert report.adapted_half_width_at_station_m < report.nominal_half_width_at_station_m


def test_the_tuck_never_makes_the_robot_wider_anywhere():
    """The failure of the first version, which blended toward the arm pose at the clip's own
    narrowest frame. That pose is narrow only alongside that frame's torso orientation, and
    transplanted elsewhere in the gait it made two real clips 20 to 40 mm wider.
    """
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    _, report = local_arm_tuck(swinging_arms(), 0.55, target_reduction_m=0.08)
    assert report.adapted_half_width_m <= report.nominal_half_width_m + 1e-6


def test_a_deeper_target_takes_more_width_in():
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    _, light = local_arm_tuck(swinging_arms(), 0.55, target_reduction_m=0.04)
    _, heavy = local_arm_tuck(swinging_arms(), 0.55, target_reduction_m=0.10)
    assert heavy.half_width_reduction_m > light.half_width_reduction_m


def test_the_tuck_is_local_like_the_crouch():
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    _, report = local_arm_tuck(swinging_arms(), 0.55, target_reduction_m=0.06)
    assert 0.05 < report.active_fraction < 0.6


def test_bad_tuck_arguments_are_refused():
    from gear_sonic.dataset_generation.local_adaptation import local_arm_tuck

    with pytest.raises(ValueError, match="station_fraction"):
        local_arm_tuck(walk(), 1.4)
    with pytest.raises(ValueError, match="target_reduction_m"):
        local_arm_tuck(walk(), 0.5, target_reduction_m=0.0)


#: Highest crouch excursion SONIC has been observed to hold (motion 005, the matched 2x2), and the
#: lowest observed to fail (x000, pure reference drift at 0.225 m/s with zero contact).
VERIFIED_CROUCH_KNEE_RAD = 0.994
REJECTED_CROUCH_KNEE_RAD = 1.000


def test_the_crouch_cap_sits_below_every_excursion_observed_to_fail():
    """The cap's whole job. It was once 1.00 -- exactly the value that failed -- because the clip
    that failed was pushed to the cap when its target drop was out of reach."""
    from gear_sonic.dataset_generation.local_adaptation import MAX_CROUCH_EXCURSION_RAD

    assert MAX_CROUCH_EXCURSION_RAD < REJECTED_CROUCH_KNEE_RAD


def test_the_two_operators_do_not_share_an_excursion_cap():
    """One number for both is the defect this splits apart.

    The tuck's bound comes from a 1.300 rad arm rotation that put the robot into a wall; the
    crouch's comes from a strength sweep whose trackability boundary sat between 1.05 and 1.31
    rad. Collapsing them again would re-forbid a verified family.
    """
    from gear_sonic.dataset_generation.local_adaptation import (
        MAX_CROUCH_EXCURSION_RAD,
        MAX_TUCK_EXCURSION_RAD,
    )

    assert MAX_TUCK_EXCURSION_RAD < MAX_CROUCH_EXCURSION_RAD


def test_the_crouch_leaves_the_waist_alone_unless_asked():
    """The guidance asks for a knee-driven crouch holding torso pitch near nominal, and the
    verified clip moves the waist by exactly 0.000 rad. Enabling the waist by default made the
    repository stop reproducing the clip physics had checked."""
    from gear_sonic.dataset_generation.local_adaptation import local_crouch

    clip = walk()
    _, report = local_crouch(clip, 0.55, target_drop_m=0.08)
    assert report.waist_change_rad == pytest.approx(0.0, abs=1e-9)

    _, spent = local_crouch(clip, 0.55, target_drop_m=0.08, waist_use_fraction=0.85)
    assert spent.waist_change_rad > 0.0
