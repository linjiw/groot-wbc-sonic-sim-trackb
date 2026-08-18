"""Tests for whether the obstacle was visible before the robot began adapting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.perception_timing import (
    PerceptionTiming,
    field_of_view_rad,
    perception_timing,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


def walk(frames: int = 90) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 0] = np.linspace(0.0, 4.0, frames)
    qpos[:, 2] = 0.74
    qpos[:, 3] = 1.0
    return qpos


AHEAD = (2.0, -1.5, 1.10, 2.5, 1.5, 1.20)


def test_the_field_of_view_comes_from_the_recorded_intrinsics():
    """69 x 55 degrees, from a 1.88 mm focal length against a 2.60 x 1.95 mm aperture."""
    h, v = field_of_view_rad()
    assert np.degrees(h) == pytest.approx(69.4, abs=0.5)
    assert np.degrees(v) == pytest.approx(54.9, abs=0.5)
    assert h > v


def test_a_late_adaptation_can_be_a_response_to_what_was_seen():
    nominal = walk()
    adapted = nominal.copy()
    adapted[45:, 7] = 0.5                       # departs from frame 45
    timing = perception_timing(nominal, adapted, AHEAD)
    assert timing.t_adaptation_onset == 45
    assert timing.ordering_holds


def test_adapting_from_frame_zero_cannot_be_a_response():
    """The whole-route crouch, and the reason the operator is local. Nothing was visible
    yet, so such a pair can only support map-conditioned selection."""
    nominal = walk()
    adapted = nominal.copy()
    adapted[:, 7] = 0.5                         # departs immediately
    timing = perception_timing(nominal, adapted, AHEAD)
    assert timing.t_adaptation_onset == 0
    assert not timing.ordering_holds
    assert "map-conditioned only" in timing.verdict()


def test_an_obstacle_behind_the_robot_is_never_visible():
    nominal = walk()
    adapted = nominal.copy()
    adapted[45:, 7] = 0.5
    behind = (-4.0, -1.5, 1.10, -3.5, 1.5, 1.20)
    timing = perception_timing(nominal, adapted, behind)
    assert timing.t_first_visible is None
    assert not timing.ordering_holds
    assert "never enters the camera" in timing.verdict()


def test_an_obstacle_beside_the_robot_is_outside_the_horizontal_field_of_view():
    nominal = walk()
    adapted = nominal.copy()
    adapted[45:, 7] = 0.5
    beside = (1.9, 8.0, 1.10, 2.1, 9.0, 1.20)
    assert perception_timing(nominal, adapted, beside).t_first_visible is None


def test_an_identical_pair_has_no_adaptation_onset():
    nominal = walk()
    timing = perception_timing(nominal, nominal.copy(), AHEAD)
    assert timing.t_adaptation_onset is None
    assert "never departs" in timing.verdict()


def test_the_reaction_window_is_reported_in_seconds():
    nominal = walk()
    adapted = nominal.copy()
    adapted[60:, 7] = 0.5
    timing = perception_timing(nominal, adapted, AHEAD, fps=30.0)
    assert timing.reaction_window_s == pytest.approx(2.0, abs=0.05)


def test_clips_of_different_length_cannot_be_compared():
    with pytest.raises(ValueError, match="same shape"):
        perception_timing(walk(90), walk(80), AHEAD)


def test_the_ordering_is_the_whole_verdict():
    """Any missing moment means the question cannot be answered, not that it passed."""
    assert not PerceptionTiming(None, 10, 20, 30.0).ordering_holds
    assert not PerceptionTiming(0, None, 20, 30.0).ordering_holds
    assert not PerceptionTiming(0, 10, None, 30.0).ordering_holds
    assert PerceptionTiming(0, 10, 20, 30.0).ordering_holds
