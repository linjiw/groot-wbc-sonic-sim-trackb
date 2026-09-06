import copy

from gear_sonic.dataset_generation.hallucination.motion2scene_comparison_contract import (
    pair_decisions,
    reserved_layout,
)


def test_reserved_neighborhood_includes_boundary_but_not_distant_coordinate():
    layouts = [{"station": 0.35, "underside_m": 1.27}]
    assert reserved_layout(0.36, 1.275, layouts)
    assert not reserved_layout(0.37, 1.27, layouts)
    assert not reserved_layout(0.35, 1.28, layouts)


def test_matching_precommand_state_does_not_require_matching_action():
    a = dict(
        phase_s=0.3,
        capture_frame=14,
        capture_elapsed_s=0.28,
        active_before=0,
        packet={},
        state={},
        features=[0.0] * 214,
        joint_names=[],
        root_pos_w=[0] * 3,
        root_quat_w=[1, 0, 0, 0],
        requested_action=0,
    )
    b = copy.deepcopy(a)
    b["requested_action"] = 1
    assert pair_decisions(a, b)["valid"]
    b["state"]["dof_pos"] = [1.0]
    assert not pair_decisions(a, b)["valid"]
    b = copy.deepcopy(a)
    b["phase_s"] = 0.2
    assert not pair_decisions(a, b)["valid"]
