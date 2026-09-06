import pickle

import numpy as np
import pytest
from test_trajectory_validation import _trajectory

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)


def test_reset_loader_keeps_short_failed_first_episode_and_tail(tmp_path):
    data = _trajectory(8)
    data["motion_time_s"] = np.array([0, 0.02, 0, 0.02, 0.04, 0.06, 0.08, 0])
    data["root_pos_w"][:, 0] = np.arange(8)
    path = tmp_path / "capture.pkl"
    path.write_bytes(pickle.dumps(data))
    loaded = load_reset_capture(path)
    assert loaded["total_frames"] == 8
    np.testing.assert_array_equal(loaded["root_pos_w"][:, 0], np.arange(8))
    np.testing.assert_array_equal(loaded["motion_time_s"], data["motion_time_s"])


def test_reset_does_not_excuse_nonfinite_actions(tmp_path):
    data = _trajectory(8)
    data["motion_time_s"][4:] -= 0.08
    data["applied_joint_action"][0, 0] = np.nan
    path = tmp_path / "capture.pkl"
    path.write_bytes(pickle.dumps(data))
    with pytest.raises(ValueError, match="invalid measurement"):
        load_reset_capture(path)
