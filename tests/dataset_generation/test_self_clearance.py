"""Tests for measuring how close the robot comes to itself, including when it does not touch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.self_clearance import (
    clearance_change,
    self_clearance,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


def standing(frames: int = 6) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    return qpos


def test_clearance_is_reported_with_the_pair_responsible():
    report = self_clearance(standing(), frame_stride=1)
    assert report.frames_checked == 6
    assert report.tightest_pair[0] and report.tightest_pair[1]
    assert np.isfinite(report.min_clearance_m)


def test_the_pair_is_never_a_body_with_itself_or_its_own_parent():
    """Parent and child meet at a joint and touch there in every pose, so counting them
    would report a permanent zero that says nothing about the motion."""
    report = self_clearance(standing(), frame_stride=1)
    assert report.tightest_pair[0] != report.tightest_pair[1]


def test_a_malformed_clip_is_refused():
    with pytest.raises(ValueError, match=r"\(T, nq\)"):
        self_clearance(np.zeros(36))
    with pytest.raises(ValueError, match="columns"):
        self_clearance(np.zeros((4, 12)))


# ---- the comparison, which is the number worth using ---------------------------------------

def test_a_clip_compared_with_itself_has_changed_nothing():
    change = clearance_change(standing(), standing(), frame_stride=1)
    assert change.worst_change_m == pytest.approx(0.0, abs=1e-9)
    assert not change.is_tighter


def test_bringing_the_legs_up_registers_as_a_loss_of_clearance():
    """A crouch lifts the heel toward the pelvis, which is what the operator's depth costs."""
    nominal = standing()
    crouched = standing()
    crouched[:, 7] = 0.8      # left hip pitch
    crouched[:, 10] = 1.2     # left knee
    change = clearance_change(crouched, nominal, frame_stride=1)
    assert change.is_tighter
    assert change.adapted_clearance_m < change.nominal_clearance_m


def test_clips_of_different_length_cannot_be_compared_frame_by_frame():
    with pytest.raises(ValueError, match="same shape"):
        clearance_change(standing(6), standing(8))


def test_the_comparison_exists_because_the_absolute_minimum_saturates():
    """A plain accepted walk already reports about -0.08 m, because a wrist geom passes
    through a hip geom in ordinary arm swing. A global minimum is therefore pinned by whichever
    pair overlaps permanently and cannot distinguish a safe clip from a dangerous one; the
    change relative to a nominal that is known to track can.
    """
    nominal = standing()
    nominal[:, 7] = 0.1
    adapted = nominal.copy()
    adapted[:, 10] = 1.0
    absolute_nominal = self_clearance(nominal, frame_stride=1).min_clearance_m
    absolute_adapted = self_clearance(adapted, frame_stride=1).min_clearance_m
    change = clearance_change(adapted, nominal, frame_stride=1)
    # The change is attributable to a specific pair; the absolute minimum need not have moved.
    assert change.pair[0] and change.pair[1]
    assert np.isfinite(absolute_nominal) and np.isfinite(absolute_adapted)
