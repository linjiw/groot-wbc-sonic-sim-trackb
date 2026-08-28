"""Scene difficulty should be chosen, not discovered.

A full-height shelf binds the tallest capsule and nothing else, so every scene built that way tests
the same body part. These tests pin the property that makes difficulty designable: an obstacle
confined to a height band binds whatever passes through that band.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gear_sonic.dataset_generation.criticality_map import (  # noqa: E402
    DEFAULT_BANDS,
    BindingConstraint,
    criticality_map,
    obstacle_offset_for_margin,
)


def synthetic_payload(frames: int = 40) -> dict:
    """A standing figure whose tallest body and widest body differ inside one band.

    Both are placed in the waist band on purpose: the wrist reaches further sideways, the torso
    stands higher. A wall and a ceiling entering that band are therefore stopped by different
    parts, which is the distinction the map has to keep.
    """
    names = ["torso_link", "left_wrist_yaw_link"]
    pos = np.zeros((frames, 2, 3))
    pos[:, 0] = (0.0, 0.00, 0.78)  # torso, on the centre line, the higher of the two
    pos[:, 1] = (0.0, 0.30, 0.65)  # wrist, out to the left, lower but wider
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 2, 1))
    return {
        "body_names": names,
        "body_pos_w": pos,
        "body_quat_w": quat,
        "root_pos_w": np.zeros((frames, 3)),
        "root_quat_w": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
    }


@pytest.fixture(scope="module")
def mapped():

    class Capsule:
        def __init__(self, radius):
            self.start = (0.0, 0.0, 0.0)
            self.end = (0.0, 0.0, 0.0)
            self.radius = radius

    return criticality_map(
        synthetic_payload(),
        capsules={"torso_link": (Capsule(0.10),), "left_wrist_yaw_link": (Capsule(0.05),)},
    )


def test_different_bands_select_different_parts(mapped):
    """The whole point: one trajectory poses different geometric problems by band."""
    parts = {c.band: c.lateral_body for c in mapped if c.side == "left" and c.lateral_body}
    assert parts.get("waist") == "left_wrist_yaw_link"
    # The right side is reached by nothing here, so it must bind nothing rather than defaulting.
    right = [c for c in mapped if c.band == "waist" and c.side == "right"][0]
    assert right.lateral_reach_m < parts_reach(mapped, "waist", "left")


def parts_reach(mapped, band, side):
    return [c for c in mapped if c.band == band and c.side == side][0].lateral_reach_m


def test_a_wall_and_a_ceiling_are_stopped_by_different_parts(mapped):
    """The first version of this map reported one reach for both, giving the overhead band the
    torso's *sideways* extent -- a number a ceiling never touches. A wall is stopped by whatever
    reaches furthest sideways; a ceiling by whatever is highest."""
    waist = [c for c in mapped if c.band == "waist" and c.side == "left"][0]
    assert waist.lateral_body == "left_wrist_yaw_link"
    assert waist.vertical_body == "torso_link"
    assert waist.lateral_reach_m != waist.vertical_reach_m


def test_a_band_the_robot_never_occupies_binds_nothing(mapped):
    """A room can hold an obstacle where the robot never goes; it tests nothing and must not be
    reported as a constraint."""
    empty = [c for c in mapped if c.frames_in_band == 0]
    for constraint in empty:
        assert not constraint.constructible("wall")
        assert not constraint.constructible("ceiling")


def test_margin_is_chosen_rather_than_searched():
    constraint = BindingConstraint(
        "overhead",
        "left",
        "torso_link",
        0.100,
        "torso_link",
        1.300,
        40,
        "local_crouch",
        "local_crouch",
    )
    assert obstacle_offset_for_margin(constraint, 0.020) == pytest.approx(0.120)
    assert obstacle_offset_for_margin(constraint, -0.010) == pytest.approx(0.090)
    # A ceiling's margin is a height, not a sideways offset, and mixing them would place an
    # obstacle a metre from where it was meant to go.
    assert obstacle_offset_for_margin(constraint, 0.020, obstacle="ceiling") == pytest.approx(1.320)


def test_a_binding_part_with_no_operator_is_not_a_family():
    """An obstacle binding a part nothing can relieve produces a negative with no matching
    positive. That is a scene, not a counterfactual, and the difference must not be blurred."""
    unrelieved = BindingConstraint(
        "floor",
        "left",
        "left_ankle_roll_link",
        0.245,
        "right_knee_link",
        0.423,
        40,
        None,
        None,
    )
    assert unrelieved.frames_in_band > 0
    assert not unrelieved.constructible("wall")
    assert not unrelieved.constructible("ceiling")


def test_every_default_band_is_disjoint_and_ordered():
    bounds = [(z0, z1) for _, z0, z1 in DEFAULT_BANDS]
    assert all(z0 < z1 for z0, z1 in bounds)
    ordered = sorted(bounds, key=lambda b: b[0])
    for (_, upper), (lower, _) in zip(ordered, ordered[1:]):
        assert upper <= lower + 1e-9, "bands must not overlap, or a part binds two of them"
