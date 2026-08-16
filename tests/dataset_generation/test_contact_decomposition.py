"""Gates for separating robot self-contact from robot-environment contact.

The motivating measurement: on three M0 rollouts the raw non-foot contact maximum
was 33.8 N, 268.8 N and 309.2 N, and the *largest* occurred on a bare plane with
no obstacles at all. Every one of those contacts was a wrist-into-hip
action-reaction pair, and the unpaired (environment) residual was exactly zero.
A single summed per-body force therefore cannot gate navigation safety.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.contact_decomposition import (
    decompose_contact_forces,
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.trajectory_acceptance import (
    G1_FOOT_CONTACT_BODY_NAMES,
    LocomotionAcceptanceThresholds,
    evaluate_locomotion_trajectory,
)
from tests.dataset_generation.test_trajectory_acceptance import _acceptable_trajectory

BODIES = ("pelvis", "left_hip_roll_link", "left_wrist_yaw_link", "left_ankle_roll_link")


def _frames(count: int) -> np.ndarray:
    return np.zeros((count, len(BODIES), 3), dtype=np.float64)


# --------------------------------------------------------------------------
# Core separation
# --------------------------------------------------------------------------


def test_action_reaction_pair_is_self_contact_not_external():
    forces = _frames(3)
    forces[1, 1] = [-287.7852, 17.159288, -111.8089]
    forces[1, 2] = [287.7852, -17.159288, 111.8089]

    result = decompose_contact_forces(forces, BODIES)

    assert result.max_self_contact == pytest.approx(309.2183, abs=1e-3)
    assert result.max_external_contact == 0.0
    assert result.self_contact_pairs == (("left_hip_roll_link", "left_wrist_yaw_link"),)
    assert result.external_contact_bodies == ()
    assert result.self_contact_events == 1
    assert result.max_self_contact_frame == 1


def test_unpaired_force_is_external_contact():
    forces = _frames(2)
    forces[0, 0] = [50.0, 0.0, 0.0]  # pelvis into a wall, no reaction partner

    result = decompose_contact_forces(forces, BODIES)

    assert result.max_external_contact == pytest.approx(50.0)
    assert result.max_self_contact == 0.0
    assert result.external_contact_bodies == ("pelvis",)


def test_foot_ground_contact_is_excluded_from_external_total():
    forces = _frames(2)
    forces[0, 3] = [0.0, 0.0, 400.0]  # ankle carrying body weight on the floor

    result = decompose_contact_forces(forces, BODIES, foot_body_names=("left_ankle_roll_link",))

    assert result.max_external_contact == 0.0
    assert result.external_contact_bodies == ()


def test_foot_is_counted_as_external_when_not_declared_a_foot():
    forces = _frames(2)
    forces[0, 3] = [0.0, 0.0, 400.0]

    result = decompose_contact_forces(forces, BODIES, foot_body_names=())

    assert result.max_external_contact == pytest.approx(400.0)


def test_self_contact_and_external_contact_coexist_in_one_frame():
    forces = _frames(1)
    forces[0, 1] = [10.0, 0.0, 0.0]
    forces[0, 2] = [-10.0, 0.0, 0.0]
    forces[0, 0] = [0.0, 25.0, 0.0]

    result = decompose_contact_forces(forces, BODIES)

    assert result.max_self_contact == pytest.approx(10.0)
    assert result.max_external_contact == pytest.approx(25.0)
    assert result.external_contact_bodies == ("pelvis",)


def test_near_miss_pairing_is_treated_as_external_not_silently_matched():
    """A body touching scene and self at once must not hide the external part."""
    forces = _frames(1)
    forces[0, 1] = [10.0, 0.0, 0.0]
    forces[0, 2] = [-9.0, 0.0, 0.0]  # 1 N of the reaction is absorbed by the scene

    result = decompose_contact_forces(forces, BODIES, pair_atol=1e-4)

    assert result.self_contact_events == 0
    assert result.max_external_contact == pytest.approx(10.0)
    assert set(result.external_contact_bodies) == {"left_hip_roll_link", "left_wrist_yaw_link"}


def test_pair_tolerance_is_caller_controlled():
    forces = _frames(1)
    forces[0, 1] = [10.0, 0.0, 0.0]
    forces[0, 2] = [-9.0, 0.0, 0.0]

    loose = decompose_contact_forces(forces, BODIES, pair_atol=2.0)

    assert loose.self_contact_events == 1
    assert loose.max_external_contact == 0.0


def test_three_way_contact_leaves_a_residual():
    """Two bodies pair; a third unpaired body still surfaces as external."""
    forces = _frames(1)
    forces[0, 1] = [10.0, 0.0, 0.0]
    forces[0, 2] = [-10.0, 0.0, 0.0]
    forces[0, 3] = [7.0, 0.0, 0.0]

    result = decompose_contact_forces(forces, BODIES, foot_body_names=())

    assert result.max_self_contact == pytest.approx(10.0)
    assert result.max_external_contact == pytest.approx(7.0)


def test_zero_forces_produce_empty_report():
    result = decompose_contact_forces(_frames(5), BODIES)

    assert result.max_self_contact == 0.0
    assert result.max_external_contact == 0.0
    assert result.self_contact_events == 0
    assert result.to_dict()["self_contact_pairs"] == []


@pytest.mark.parametrize(
    "forces,names",
    [
        (np.zeros((2, 3)), BODIES),
        (np.zeros((2, 4, 2)), BODIES),
    ],
)
def test_malformed_force_arrays_are_rejected(forces, names):
    with pytest.raises(ValueError):
        decompose_contact_forces(forces, names)


def test_name_count_must_match_force_axis():
    with pytest.raises(ValueError, match="entries but forces have"):
        decompose_contact_forces(_frames(2), BODIES[:2])


def test_payload_wrapper_uses_declared_foot_names():
    payload = {
        "robot_contact_force_w": _frames(1),
        "contact_body_names": BODIES,
        "allowed_foot_contact_body_names": ("left_ankle_roll_link",),
    }
    payload["robot_contact_force_w"][0, 3] = [0.0, 0.0, 400.0]

    assert decompose_payload_contacts(payload).max_external_contact == 0.0


# --------------------------------------------------------------------------
# Integration with the acceptance gate
# --------------------------------------------------------------------------


def _with_contact(vectors: list[tuple[int, str, list[float]]]):
    trajectory = _acceptable_trajectory()
    names = list(trajectory["contact_body_names"])
    for frame, body, value in vectors:
        trajectory["robot_contact_force_w"][frame, names.index(body)] = value
    trajectory["robot_contact_force_norm_w"] = np.linalg.norm(
        trajectory["robot_contact_force_w"], axis=-1
    ).astype(np.float32)
    trajectory["max_nonfoot_contact_force_n"] = np.max(
        trajectory["robot_contact_force_norm_w"][
            :, [i for i, n in enumerate(names) if n not in G1_FOOT_CONTACT_BODY_NAMES]
        ],
        axis=1,
    ).astype(np.float32)
    return trajectory


def test_gait_self_contact_no_longer_rejects_an_otherwise_clean_episode():
    """The regression this whole module exists for."""
    trajectory = _with_contact(
        [
            (4, "left_hip_roll_link", [-287.7852, 17.159288, -111.8089]),
            (4, "left_wrist_yaw_link", [287.7852, -17.159288, 111.8089]),
        ]
    )

    report = evaluate_locomotion_trajectory(trajectory)

    assert report.accepted
    assert "disallowed_robot_contact" not in report.rejection_reasons
    decomposition = report.diagnostics["contact_decomposition"]
    assert decomposition["max_self_contact_force_n"] == pytest.approx(309.2183, abs=1e-3)
    assert decomposition["max_external_contact_force_n"] == 0.0
    # The undecomposed number is preserved so the change stays auditable.
    assert report.diagnostics["raw_max_nonfoot_contact_force_n"] == pytest.approx(
        309.2183, abs=1e-3
    )


def test_a_purely_upward_knee_load_is_support_not_collision():
    """Measured on a crouching reference: knee force was [0, 0, 104.9] N while the
    root was at 0.41 m. That is kneeling on the floor, not striking a rack."""
    trajectory = _with_contact([(4, "left_knee_link", [0.0, 0.0, 104.9])])

    report = evaluate_locomotion_trajectory(trajectory)

    assert "disallowed_robot_contact" not in report.rejection_reasons
    decomposition = report.diagnostics["contact_decomposition"]
    assert decomposition["max_lateral_contact_force_n"] == pytest.approx(0.0, abs=1e-9)
    assert decomposition["max_support_contact_force_n"] == pytest.approx(104.9, abs=1e-3)
    assert decomposition["external_contact_bodies"] == []


def test_support_load_beyond_the_provisional_ceiling_is_rejected():
    trajectory = _with_contact([(4, "left_knee_link", [0.0, 0.0, 1500.0])])
    report = evaluate_locomotion_trajectory(trajectory)
    assert not report.accepted
    assert "excessive_nonfoot_support_load" in report.rejection_reasons


def test_scene_contact_still_rejects():
    trajectory = _with_contact([(4, "pelvis", [20.0, 0.0, 0.0])])

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "disallowed_robot_contact" in report.rejection_reasons
    assert report.diagnostics["contact_decomposition"]["external_contact_bodies"] == ["pelvis"]


def test_self_contact_above_the_provisional_threshold_is_rejected():
    trajectory = _with_contact(
        [
            (4, "left_hip_roll_link", [-1000.0, 0.0, 0.0]),
            (4, "left_wrist_yaw_link", [1000.0, 0.0, 0.0]),
        ]
    )

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "excessive_self_contact" in report.rejection_reasons


def test_self_contact_threshold_is_configurable():
    trajectory = _with_contact(
        [
            (4, "left_hip_roll_link", [-50.0, 0.0, 0.0]),
            (4, "left_wrist_yaw_link", [50.0, 0.0, 0.0]),
        ]
    )

    strict = LocomotionAcceptanceThresholds(max_self_contact_force_n=10.0)
    assert "excessive_self_contact" in (
        evaluate_locomotion_trajectory(trajectory, strict).rejection_reasons
    )
    assert evaluate_locomotion_trajectory(trajectory).accepted


def test_negative_self_contact_threshold_is_rejected():
    with pytest.raises(ValueError, match="contact thresholds must be non-negative"):
        LocomotionAcceptanceThresholds(max_self_contact_force_n=-1.0)


# --------------------------------------------------------------------------
# Crouch vs fall
# --------------------------------------------------------------------------


def _with_heights(executed_z, reference_z):
    trajectory = _acceptable_trajectory()
    frames = len(trajectory["root_pos_w"])
    trajectory["root_pos_w"][:, 2] = np.asarray(executed_z, dtype=np.float32)
    trajectory["reference_g1_qpos"][:, 2] = np.asarray(reference_z, dtype=np.float32)
    assert frames == len(trajectory["reference_g1_qpos"])
    return trajectory


def test_a_tracked_crouch_is_not_a_fall():
    """Measured on 02_multi_text_ee_constraint: reference dips to 0.323 m and the
    robot follows it to within 4 cm while upright. An absolute 0.50 m floor called
    that a fall and rejected a legitimate behaviour."""
    frames = len(_acceptable_trajectory()["root_pos_w"])
    reference = np.full(frames, 0.78)
    reference[10:25] = 0.323
    executed = reference - 0.039

    report = evaluate_locomotion_trajectory(_with_heights(executed, reference))

    assert report.accepted
    assert "fall_root_height" not in report.rejection_reasons
    assert "root_sank_below_reference" not in report.rejection_reasons
    assert report.diagnostics["min_reference_root_height_m"] == pytest.approx(0.323, abs=1e-3)


def test_sinking_below_the_commanded_height_is_still_a_fall():
    """A robot that does not follow its commanded height fails at any absolute height."""
    frames = len(_acceptable_trajectory()["root_pos_w"])
    reference = np.full(frames, 0.78)
    executed = reference.copy()
    executed[12] = 0.78 - 0.40  # sinks 40 cm below command, still above 0.25 m

    report = evaluate_locomotion_trajectory(_with_heights(executed, reference))

    assert not report.accepted
    assert "root_sank_below_reference" in report.rejection_reasons


def test_absolute_collapse_is_still_a_fall_even_if_the_reference_collapses_too():
    """A reference that itself goes to the floor must not license a collapsed episode."""
    frames = len(_acceptable_trajectory()["root_pos_w"])
    reference = np.full(frames, 0.78)
    reference[15] = 0.10
    executed = reference.copy()

    report = evaluate_locomotion_trajectory(_with_heights(executed, reference))

    assert not report.accepted
    assert "fall_root_height" in report.rejection_reasons


def test_height_gates_are_configurable():
    frames = len(_acceptable_trajectory()["root_pos_w"])
    reference = np.full(frames, 0.78)
    executed = reference - 0.10
    trajectory = _with_heights(executed, reference)

    assert evaluate_locomotion_trajectory(trajectory).accepted
    strict = LocomotionAcceptanceThresholds(max_root_height_below_reference_m=0.05)
    assert "root_sank_below_reference" in (
        evaluate_locomotion_trajectory(trajectory, strict).rejection_reasons
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_root_height_m": 0.0},
        {"max_root_height_below_reference_m": 0.0},
        {"max_root_height_below_reference_m": -0.1},
    ],
)
def test_invalid_height_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError, match="fall thresholds must be positive"):
        LocomotionAcceptanceThresholds(**kwargs)


def _with_tilt(executed_rad, reference_rad):
    """Set roll on both executed and reference root quaternions (wxyz)."""
    trajectory = _acceptable_trajectory()
    frames = len(trajectory["root_quat_w"])

    def quat(angle):
        return np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0], dtype=np.float32)

    trajectory["root_quat_w"] = np.tile(quat(0.0), (frames, 1)).astype(np.float32)
    trajectory["root_quat_w"][frames // 2] = quat(executed_rad)
    trajectory["reference_g1_qpos"][:, 3:7] = np.tile(quat(0.0), (frames, 1))
    trajectory["reference_g1_qpos"][frames // 2, 3:7] = quat(reference_rad)
    return trajectory


def test_a_tracked_bow_is_not_a_fall():
    """03_full_body_keyframes executes 40.6 deg of tilt while its reference commands
    41.9 deg. An absolute 0.60 rad threshold rejected a well-tracked bow."""
    report = evaluate_locomotion_trajectory(_with_tilt(0.708, 0.732))

    assert report.accepted
    assert "fall_root_tilt" not in report.rejection_reasons
    assert "root_tilted_beyond_reference" not in report.rejection_reasons
    assert report.diagnostics["max_reference_root_tilt_rad"] == pytest.approx(0.732, abs=1e-3)


def test_tilting_further_than_commanded_is_still_a_fall():
    report = evaluate_locomotion_trajectory(_with_tilt(0.90, 0.0))

    assert not report.accepted
    assert "root_tilted_beyond_reference" in report.rejection_reasons


def test_near_horizontal_torso_is_a_fall_even_if_commanded():
    report = evaluate_locomotion_trajectory(_with_tilt(1.5, 1.5))

    assert not report.accepted
    assert "fall_root_tilt" in report.rejection_reasons
