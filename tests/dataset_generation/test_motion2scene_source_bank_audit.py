import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_source_bank_audit import (
    source_bank_errors,
)
from gear_sonic.dataset_generation.kimodo_motion_adapter import KIMODO_G1_JOINT_NAMES


def test_named_joint_order_and_wrong_source_detection():
    names = list(reversed(KIMODO_G1_JOINT_NAMES))
    entry = {
        "root_trans_offset": np.c_[np.arange(120) / 30, np.zeros((120, 2))],
        "dof": np.broadcast_to(np.arange(29) / 10, (120, 29)),
        "fps": 30,
    }
    root = np.c_[np.arange(199) / 50, np.zeros((199, 2))]
    bank = {
        "root_xyz": np.stack([root, root]),
        "joint_pos": np.broadcast_to(np.arange(29)[::-1] / 10, (2, 199, 29)).copy(),
        "fps": 50,
    }
    assert all(
        r["joint_scalar_interpolation_rad"] == 0
        for r in source_bank_errors(bank, [entry, entry], names)
    )
    bank["joint_pos"][1, :, 0] += 0.1
    assert (
        source_bank_errors(bank, [entry, entry], names)[1]["joint_scalar_interpolation_rad"] > 0.09
    )
    bank["root_xyz"][1, :, 0] += 0.2
    with pytest.raises(ValueError, match="source route"):
        source_bank_errors(bank, [entry, entry], names)
