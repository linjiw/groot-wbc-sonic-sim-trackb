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

def test_a_deeper_target_produces_a_deeper_crouch():
    _, shallow = local_crouch(walk(), 0.55, target_drop_m=0.10)
    _, deep = local_crouch(walk(), 0.55, target_drop_m=0.20)
    assert deep.silhouette_drop_m > shallow.silhouette_drop_m
    assert deep.adapted_silhouette_m < shallow.adapted_silhouette_m


def test_the_target_drop_is_hit_on_the_measured_silhouette():
    _, report = local_crouch(walk(), 0.55, target_drop_m=0.15)
    assert report.silhouette_drop_m == pytest.approx(0.15, abs=0.02)


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
