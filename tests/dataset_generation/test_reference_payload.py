"""Tests for grading a generated reference without rolling it out."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.reference_payload import (
    payload_from_reference,
)
from gear_sonic.dataset_generation.self_intersection import (
    DEFAULT_G1_MJCF,
    SelfIntersectionError,
)

pytestmark = pytest.mark.skipif(
    not Path(DEFAULT_G1_MJCF).exists(), reason="G1 MJCF not present"
)


def standing(frames: int = 12) -> np.ndarray:
    qpos = np.zeros((frames, 36))
    qpos[:, 2] = 0.78          # root height
    qpos[:, 3] = 1.0           # identity quaternion, wxyz
    return qpos


def test_a_reference_becomes_a_payload_the_predicates_can_read():
    payload = payload_from_reference(standing())
    for key in ("body_pos_w", "body_quat_w", "body_names", "root_pos_w", "root_quat_w"):
        assert key in payload
    assert payload["body_pos_w"].shape == (12, len(payload["body_names"]), 3)
    assert payload["body_quat_w"].shape == (12, len(payload["body_names"]), 4)


def test_the_world_body_is_dropped_so_names_match_the_recorder():
    """The capsule model is keyed by link name, which only works if the sets agree."""
    payload = payload_from_reference(standing())
    assert "world" not in payload["body_names"]
    assert "torso_link" in payload["body_names"]
    assert "left_ankle_roll_link" in payload["body_names"]
    assert len(payload["body_names"]) == 30


def test_the_payload_declares_itself_a_reference():
    """Predicates calibrated on rollouts must be able to tell, because tracking lowers the
    foot by about 35 mm and an absolute threshold gives the opposite verdict otherwise."""
    assert payload_from_reference(standing())["kind"] == "reference"


def test_the_root_is_taken_from_the_clip_not_recomputed():
    qpos = standing()
    qpos[:, 0] = np.linspace(0.0, 2.0, len(qpos))
    payload = payload_from_reference(qpos)
    assert payload["root_pos_w"][-1, 0] == pytest.approx(2.0)
    assert payload["root_pos_w"][0, 0] == pytest.approx(0.0)


def test_forward_kinematics_actually_moves_the_bodies():
    """A payload of zeros would satisfy every shape assertion above and mean nothing."""
    low, high = standing(), standing()
    high[:, 2] = 1.20
    delta = (
        payload_from_reference(high)["body_pos_w"][:, :, 2]
        - payload_from_reference(low)["body_pos_w"][:, :, 2]
    )
    assert np.allclose(delta, 1.20 - 0.78, atol=1e-6)


def test_a_clip_with_the_wrong_column_count_is_refused():
    with pytest.raises(SelfIntersectionError, match="columns"):
        payload_from_reference(np.zeros((10, 20)))


def test_a_one_dimensional_clip_is_refused():
    with pytest.raises(SelfIntersectionError, match=r"\(T, nq\)"):
        payload_from_reference(np.zeros(36))
