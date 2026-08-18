"""Scene difficulty should be chosen, not discovered.

A full-height shelf binds the tallest capsule and nothing else, so every scene built that way tests
the same body part. These tests pin the property that makes difficulty designable: an obstacle
confined to a height band binds whatever passes through that band.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    """A standing figure with two bodies at known heights and lateral offsets."""
    names = ["torso_link", "left_wrist_yaw_link"]
    pos = np.zeros((frames, 2, 3))
    pos[:, 0] = (0.0, 0.00, 1.20)   # torso, on the centre line, high
    pos[:, 1] = (0.0, 0.30, 0.70)   # wrist, out to the left, at waist height
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
    capsules = {
        "torso_link": ((np.zeros(3), np.zeros(3), 0.10),),
        "left_wrist_yaw_link": ((np.zeros(3), np.zeros(3), 0.05),),
    }

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
    parts = {c.band: c.body for c in mapped if c.side == "left" and c.body}
    assert parts.get("overhead") == "torso_link"
    assert parts.get("waist") == "left_wrist_yaw_link"


def test_a_band_the_robot_never_occupies_binds_nothing(mapped):
    """A room can hold an obstacle where the robot never goes; it tests nothing and must not be
    reported as a constraint."""
    empty = [c for c in mapped if c.frames_in_band == 0]
    for constraint in empty:
        assert not constraint.constructible


def test_margin_is_chosen_rather_than_searched():
    constraint = BindingConstraint("overhead", "left", "torso_link", 0.100, 40, "local_crouch")
    assert obstacle_offset_for_margin(constraint, 0.020) == pytest.approx(0.120)
    assert obstacle_offset_for_margin(constraint, -0.010) == pytest.approx(0.090)


def test_a_binding_part_with_no_operator_is_not_a_family():
    """An obstacle binding a part nothing can relieve produces a negative with no matching
    positive. That is a scene, not a counterfactual, and the difference must not be blurred."""
    unrelieved = BindingConstraint("floor", "left", "left_ankle_roll_link", 0.245, 40, None)
    assert unrelieved.frames_in_band > 0
    assert not unrelieved.constructible


def test_every_default_band_is_disjoint_and_ordered():
    bounds = [(z0, z1) for _, z0, z1 in DEFAULT_BANDS]
    assert all(z0 < z1 for z0, z1 in bounds)
    ordered = sorted(bounds, key=lambda b: b[0])
    for (_, upper), (lower, _) in zip(ordered, ordered[1:]):
        assert upper <= lower + 1e-9, "bands must not overlap, or a part binds two of them"
