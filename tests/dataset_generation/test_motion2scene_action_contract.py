import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_action_contract import (
    command_permission,
    paired_prefix,
    requested_skill,
)


def test_walk_is_commit_and_crouch_is_one_request():
    assert requested_skill(0, 0.2, 0.2, 0, False) == (0, True)
    assert requested_skill(1, 0.2, 0.2, 0, False) == (1, True)
    assert requested_skill(1, 0.2, 0.22, 0, True) == (0, True)
    assert requested_skill(1, 0.2, 0.22, 0, False) == (0, True)
    assert requested_skill(1, 0.2, 3.3, 1, True) == (0, True)


def test_guard_is_independent_of_scene_and_refuses_illegal_jumps():
    assert command_permission(1, 0, 0.3, 0.01, 0) == (True, [])
    assert not command_permission(1, 0, 0.42, 0, 0)[0]
    assert not command_permission(1, 0, 0.3, 0.051, 0)[0]
    assert not command_permission(1, 0, 0.3, 0, 0.011)[0]
    with pytest.raises(ValueError):
        requested_skill(1, 0.21, 0.21, 0, False)


def test_prefix_checks_actions_and_tokens_not_only_poses():
    keys = (
        "dof_pos",
        "dof_vel",
        "root_pos_w",
        "root_quat_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "applied_joint_action",
        "action_motion_token",
        "reference_g1_qpos",
    )
    a = {k: np.zeros((20, 3)) for k in keys}
    a["motion_time_s"] = np.arange(20) / 50
    b = {k: v.copy() for k, v in a.items()}
    assert paired_prefix(a, b, 0.2)["exact_match"]
    b["action_motion_token"][9, 0] = 0.01
    assert not paired_prefix(a, b, 0.2)["exact_match"]
    b["action_motion_token"][9, 0] = 0
    b["dof_pos"][15, 0] = 1
    assert paired_prefix(a, b, 0.2)["exact_match"]
