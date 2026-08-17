"""Tests for building scene triplets where geometry decides which motion works."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.counterfactual_family import (
    GEOMETRY_REGIMES,
    CounterfactualError,
    ObstacleSpec,
    build_family,
    find_collision_boundary,
    swept_clearance_to_box,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule

BODY_NAMES = ["pelvis", "torso_link"]

#: A two-link stand-in for the G1's 29 capsules. The full model needs all 14 collision links
#: present in the recording; these tests are about the search, not the body.
TEST_CAPSULES = {
    "pelvis": (CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.0), radius=0.12),),
    "torso_link": (CollisionCapsule(start=(0.0, 0.0, 0.0), end=(0.0, 0.0, 0.15), radius=0.12),),
}


def straight_walk(frames: int = 60, torso_z: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """A body walking along +x with the torso held at a fixed height."""
    pos = np.zeros((frames, 2, 3))
    pos[:, 0, 0] = np.linspace(0.0, 3.0, frames)      # pelvis
    pos[:, 0, 2] = 0.75
    pos[:, 1, 0] = np.linspace(0.0, 3.0, frames)      # torso
    pos[:, 1, 2] = torso_z
    quat = np.zeros((frames, 2, 4))
    quat[:, :, 0] = 1.0
    return pos, quat


def shelf(height: float) -> ObstacleSpec:
    return ObstacleSpec(
        name="shelf",
        size=(0.4, 2.0, 0.1),
        base_center=(1.5, 0.0, height),
        axis=(0.0, 0.0, -1.0),
        regime="overhead",
    )


def test_a_high_obstacle_is_clear_of_the_swept_volume():
    pos, quat = straight_walk()
    clearance, _ = swept_clearance_to_box(
        pos, quat, BODY_NAMES, capsules=TEST_CAPSULES,
        box=shelf(3.0).box_at(0.0),
    )
    assert clearance > 0.5


def test_an_obstacle_through_the_body_reports_interference():
    pos, quat = straight_walk()
    clearance, frame = swept_clearance_to_box(
        pos, quat, BODY_NAMES, capsules=TEST_CAPSULES,
        box=shelf(1.0).box_at(0.0),
    )
    assert clearance < 0.0
    assert 0 <= frame < len(pos)


def test_the_reported_frame_is_where_the_obstacle_sits():
    """A caller should be able to point at the moment of contact, not just assert one."""
    pos, quat = straight_walk(frames=61)
    # The shelf sits at x = 1.5, which is the midpoint of a 0..3 m walk.
    _, frame = swept_clearance_to_box(pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=shelf(1.0).box_at(0.0))
    assert 24 <= frame <= 36


def test_the_boundary_search_finds_the_zero_crossing():
    pos, quat = straight_walk()
    spec = shelf(3.0)

    def clearance_at(parameter):
        return swept_clearance_to_box(pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(parameter))

    result = find_collision_boundary(clearance_at, 0.0, 1.9, tolerance_m=0.002)
    assert abs(result.clearance_m) < 0.02
    assert result.iterations > 0


def test_lowering_an_obstacle_past_the_body_clears_again():
    """The parameter sweep is not monotone, and the search must be bracketed accordingly.

    A shelf lowered far enough passes *below* the torso and the clearance turns positive
    again. The endpoint checks catch a caller who brackets across that second crossing.
    """
    pos, quat = straight_walk()
    spec = shelf(3.0)
    through, _ = swept_clearance_to_box(
        pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(1.8)
    )
    below, _ = swept_clearance_to_box(
        pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(2.5)
    )
    assert through < 0.0 < below


def test_a_search_that_never_interferes_is_an_error():
    """Silently assuming the bracket returns a confident answer about nothing."""
    pos, quat = straight_walk()
    spec = shelf(5.0)

    def clearance_at(parameter):
        return swept_clearance_to_box(pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(parameter))

    with pytest.raises(CounterfactualError, match="never interferes"):
        find_collision_boundary(clearance_at, 0.0, 0.2)


def test_a_search_already_interfering_at_the_clear_end_is_an_error():
    pos, quat = straight_walk()
    spec = shelf(1.0)

    def clearance_at(parameter):
        return swept_clearance_to_box(pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(parameter))

    with pytest.raises(CounterfactualError, match="already interferes"):
        find_collision_boundary(clearance_at, 0.0, 1.0)


def test_a_built_family_straddles_the_boundary():
    pos, quat = straight_walk()
    family = build_family("f0", pos, quat, BODY_NAMES, shelf(3.0), search_high=1.9, capsules=TEST_CAPSULES)
    assert family.separated
    assert family.easy_clearance_m > 0.0 > family.hard_clearance_m
    assert family.easy_parameter < family.boundary.parameter < family.hard_parameter


def test_a_margin_too_small_to_separate_is_refused():
    """Two easy scenes with different pixels are not a counterfactual."""
    pos, quat = straight_walk()
    with pytest.raises(CounterfactualError, match="straddle the boundary"):
        build_family(
            "f0", pos, quat, BODY_NAMES, shelf(3.0),
            search_high=1.9, margin_m=0.0, capsules=TEST_CAPSULES,
        )


def test_an_unknown_regime_is_refused():
    pos, quat = straight_walk()
    bad = ObstacleSpec("x", (0.4, 2.0, 0.1), (1.5, 0.0, 3.0), (0.0, 0.0, -1.0), "diagonal")
    with pytest.raises(CounterfactualError, match="unknown geometry regime"):
        build_family("f0", pos, quat, BODY_NAMES, bad, search_high=1.9, capsules=TEST_CAPSULES)


def test_the_regimes_are_the_four_the_plan_names():
    assert set(GEOMETRY_REGIMES) == {"overhead", "lateral", "floor", "compound"}


def test_a_lower_body_needs_a_lower_shelf_to_be_hit():
    """The asymmetry the whole design rests on: geometry that stops one motion and not
    another.

    Measured on real episodes: at a shelf underside of 1.25 m a brisk walk interferes by
    0.041 m while a duck-under clears by 0.041 m.
    """
    tall_pos, tall_quat = straight_walk(torso_z=1.10)
    ducked_pos, ducked_quat = straight_walk(torso_z=0.80)
    spec = shelf(3.0)

    def boundary_for(pos, quat):
        return find_collision_boundary(
            lambda p: swept_clearance_to_box(pos, quat, BODY_NAMES, capsules=TEST_CAPSULES, box=spec.box_at(p)),
            0.0, 1.9, tolerance_m=0.002,
        ).parameter

    # The ducked body needs the shelf brought further down before it interferes.
    assert boundary_for(ducked_pos, ducked_quat) > boundary_for(tall_pos, tall_quat)


def test_obstacle_box_moves_with_the_parameter():
    spec = shelf(2.0)
    high = spec.box_at(0.0)
    low = spec.box_at(0.5)
    assert low[2] == pytest.approx(high[2] - 0.5)
    assert low[0] == pytest.approx(high[0])
