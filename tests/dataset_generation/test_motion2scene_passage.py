import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage

BEAM = {"center_xy_m": [1.0, 0.0], "yaw_rad": 0.0, "length_m": 0.1}


def capture():
    root = np.c_[np.linspace(0, 2, 100), np.zeros(100), np.ones(100)]
    return {
        "root_pos_w": root,
        "body_pos_w": root[:, None].copy(),
        "projected_gravity_b": np.tile([0, 0, -1.0], (100, 1)),
        "motion_time_s": np.arange(100) / 50,
        "fps": 50,
    }, np.zeros((100, 2, 3))


def test_shared_world_and_beam_contact_change_passage():
    payload, forces = capture()
    assert score_passage(payload, forces, BEAM)["pass"]
    forces[50, 1, 2] = 2.0
    assert not score_passage(payload, forces, BEAM)["pass"]
    assert score_passage(payload, forces, BEAM)["observed_beam_contact"]
    forces[:] = 0
    shifted = {**BEAM, "center_xy_m": [3.0, 0.0]}
    assert score_passage(payload, forces, shifted)["incomplete_crossing"]


def test_reset_cannot_supply_later_success():
    payload, forces = capture()
    payload["motion_time_s"][40:] -= 0.8
    result = score_passage(payload, forces, BEAM)
    assert result["first_episode_frames"] == 40
    assert not result["pass"]


def test_trailing_body_and_stabilization_are_required():
    payload, forces = capture()
    trailing = payload["body_pos_w"].copy()
    trailing[..., 0] -= 1.0
    payload["body_pos_w"] = np.concatenate([payload["body_pos_w"], trailing], axis=1)
    assert not score_passage(payload, forces, BEAM)["pass"]
    payload, forces = capture()
    payload["projected_gravity_b"][50:, 2] = 0
    assert score_passage(payload, forces, BEAM)["stabilization_failed"]


def test_missing_or_nonfinite_contact_is_not_zero_contact():
    payload, forces = capture()
    with pytest.raises(ValueError):
        score_passage(payload, forces[:-1], BEAM)
    forces[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        score_passage(payload, forces, BEAM)
