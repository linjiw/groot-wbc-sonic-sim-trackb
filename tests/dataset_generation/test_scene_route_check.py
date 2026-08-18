"""A rollout must not be spent on an obstacle the robot never reaches.

Eight rollouts were spent finding this out the expensive way: two families' shelves sat in the
reference motion's frame while the rollout offset the motion by −2.0 m, so every cell came back
accepted and was read as evidence about the geometric window. It was evidence about nothing.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.dataset_generation.scene_route_check import (  # noqa: E402
    check_route_meets_obstacle,
)

#: A shelf spanning half a metre of x across a three-metre corridor.
FOOTPRINT = (1.98, -1.5, 2.48, 1.5)


def straight_walk(x0: float, x1: float, frames: int = 200) -> np.ndarray:
    path = np.zeros((frames, 2))
    path[:, 0] = np.linspace(x0, x1, frames)
    return path


def test_a_path_through_the_obstacle_passes():
    check = check_route_meets_obstacle(straight_walk(0.0, 4.0), FOOTPRINT)
    assert check.passes
    assert check.closest_approach_m == 0.0
    assert "enters" in check.explain()


def test_a_path_offset_out_of_the_frame_is_refused():
    """The exact failure: the same walk, shifted by the conversion's −2.0 m scene start."""
    check = check_route_meets_obstacle(straight_walk(-2.0, 2.0 - 2.0 + 2.0 - 2.0), FOOTPRINT)
    assert not check.passes
    assert check.closest_approach_m > 0.0
    assert "never enters" in check.explain()


def test_the_refusal_names_the_shift_that_would_fix_it():
    """A diagnosis is worth more than a rejection, and the offset is the tell: a value close to a
    conversion's --scene-start is a frame mismatch rather than a mis-chosen station."""
    check = check_route_meets_obstacle(straight_walk(-2.0, -0.05), FOOTPRINT)
    assert not check.passes
    # The path's closest point is its far end; the obstacle centre sits at x = 2.23.
    assert check.suggested_shift_x_m < -2.0
    assert "different frames" in check.explain()


def test_passing_beside_the_obstacle_is_also_refused():
    """Right x, wrong y. A corridor-spanning shelf makes this rare, but a one-sided lateral
    obstacle makes it the common case, and the check must not assume the obstacle spans the room."""
    path = straight_walk(0.0, 4.0)
    path[:, 1] = 3.0
    check = check_route_meets_obstacle(path, FOOTPRINT)
    assert not check.passes
    assert check.closest_approach_m > 1.0


def test_a_malformed_path_is_refused_rather_than_guessed():
    for bad in (np.zeros(10), np.zeros((10, 3))):
        try:
            check_route_meets_obstacle(bad, FOOTPRINT)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for shape {np.asarray(bad).shape}")
