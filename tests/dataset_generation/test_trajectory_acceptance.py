from __future__ import annotations

import numpy as np

from gear_sonic.dataset_generation.trajectory_acceptance import (
    G1_CONTACT_BODY_NAMES,
    G1_FOOT_CONTACT_BODY_NAMES,
    LocomotionAcceptanceThresholds,
    evaluate_locomotion_trajectory,
)
from tests.dataset_generation.test_trajectory_validation import _trajectory


def _acceptable_trajectory() -> dict:
    trajectory = _trajectory(frame_count=40)
    trajectory["root_pos_w"][:, 2] = 0.8
    trajectory["reference_g1_qpos"][:, 2] = 0.8
    trajectory["root_pos_w"][:, 0] = np.linspace(0.0, 1.0, 40)
    trajectory["reference_g1_qpos"][:, 0] = np.linspace(0.0, 1.0, 40)
    trajectory["contact_body_names"] = G1_CONTACT_BODY_NAMES
    contact_force_vectors = np.zeros((40, len(G1_CONTACT_BODY_NAMES), 3), dtype=np.float32)
    left_foot_index = G1_CONTACT_BODY_NAMES.index(G1_FOOT_CONTACT_BODY_NAMES[0])
    right_foot_index = G1_CONTACT_BODY_NAMES.index(G1_FOOT_CONTACT_BODY_NAMES[1])
    contact_force_vectors[:, left_foot_index, 2] = 50.0
    contact_force_vectors[:, right_foot_index, 2] = 50.0
    trajectory["robot_contact_force_w"] = contact_force_vectors
    trajectory["robot_contact_force_norm_w"] = np.linalg.norm(contact_force_vectors, axis=-1)
    trajectory["left_foot_ground_contact_force_w"] = np.tile([0.0, 0.0, 50.0], (40, 1)).astype(
        np.float32
    )
    trajectory["right_foot_ground_contact_force_w"] = np.tile([0.0, 0.0, 50.0], (40, 1)).astype(
        np.float32
    )
    trajectory["support_floor_prim_path"] = "/World/ground/terrain/Structure/Floor"
    trajectory["max_nonfoot_contact_force_n"] = np.zeros(40, dtype=np.float32)
    trajectory["left_foot_contact_force_n"] = np.full(40, 50.0, dtype=np.float32)
    trajectory["right_foot_contact_force_n"] = np.full(40, 50.0, dtype=np.float32)
    return trajectory


def test_accepts_supported_collision_free_tracking() -> None:
    report = evaluate_locomotion_trajectory(_acceptable_trajectory())

    assert report.accepted
    assert not report.rejection_reasons
    # Scene vs self contact, and absolute vs commanded-relative height, are each
    # separate gates; see contact_decomposition and the crouch-vs-fall tests.
    assert len(report.gates) == 14
    gate_names = {gate.name for gate in report.gates}
    assert {
        "nonfoot_scene_contact",
        "self_contact",
        "root_height",
        "root_height_tracking",
    } <= gate_names
    assert report.diagnostics["contact_decomposition"]["self_contact_events"] == 0
    assert report.diagnostics["raw_max_nonfoot_contact_force_n"] == 0.0


def test_reports_individual_route_fall_and_contact_failures() -> None:
    trajectory = _acceptable_trajectory()
    trajectory["root_pos_w"][-10:, 0] += 1.0
    trajectory["root_pos_w"][3, 2] = 0.2
    # A wall push is horizontal; a purely upward force is the floor, not a crash.
    trajectory["robot_contact_force_w"][4, 0, 0] = 20.0
    trajectory["robot_contact_force_norm_w"][4, 0] = 20.0

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "reference_endpoint_tracking_error" in report.rejection_reasons
    assert "reference_path_tracking_error" in report.rejection_reasons
    assert "fall_root_height" in report.rejection_reasons
    assert "disallowed_robot_contact" in report.rejection_reasons
    assert report.diagnostics["max_nonfoot_contact_frame"] == 4
    assert report.diagnostics["max_nonfoot_contact_bodies"] == ["pelvis"]


def test_requires_contact_evidence_instead_of_assuming_no_collision() -> None:
    trajectory = _acceptable_trajectory()
    for key in (
        "robot_contact_force_norm_w",
        "robot_contact_force_w",
        "contact_body_names",
        "left_foot_ground_contact_force_w",
        "right_foot_ground_contact_force_w",
        "support_floor_prim_path",
        "max_nonfoot_contact_force_n",
        "left_foot_contact_force_n",
        "right_foot_contact_force_n",
    ):
        trajectory.pop(key)

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert report.errors
    assert "missing physics contact evidence" in report.errors[0]


def test_recomputes_contact_summaries_from_per_body_evidence() -> None:
    trajectory = _acceptable_trajectory()
    trajectory["robot_contact_force_w"][4, 0, 0] = 500.0
    trajectory["max_nonfoot_contact_force_n"][:] = 0.0

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert any("does not match" in error for error in report.errors)


def test_recomputes_foot_support_from_per_body_evidence() -> None:
    trajectory = _acceptable_trajectory()
    trajectory["left_foot_ground_contact_force_w"][:] = 0.0
    trajectory["right_foot_ground_contact_force_w"][:] = 0.0
    trajectory["left_foot_contact_force_n"][:] = 50.0
    trajectory["right_foot_contact_force_n"][:] = 50.0

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "insufficient_foot_support" in report.rejection_reasons


def test_rejects_foot_wall_contact_as_non_ground_collision() -> None:
    trajectory = _acceptable_trajectory()
    left_foot_index = G1_CONTACT_BODY_NAMES.index(G1_FOOT_CONTACT_BODY_NAMES[0])
    trajectory["robot_contact_force_w"][7, left_foot_index, 0] = 30.0
    trajectory["robot_contact_force_norm_w"] = np.linalg.norm(
        trajectory["robot_contact_force_w"], axis=-1
    )

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "disallowed_foot_non_ground_contact" in report.rejection_reasons
    assert report.diagnostics["max_foot_non_ground_contact_frame"] == 7


def test_rejects_duplicate_or_incomplete_contact_body_names() -> None:
    trajectory = _acceptable_trajectory()
    names = list(G1_CONTACT_BODY_NAMES)
    names[-1] = names[-2]
    trajectory["contact_body_names"] = tuple(names)

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "contact_body_names must be unique" in report.errors
    assert any("must exactly cover" in error for error in report.errors)


def test_rejects_caller_widening_the_foot_contact_exemption() -> None:
    trajectory = _acceptable_trajectory()
    trajectory["allowed_foot_contact_body_names"] = (
        *G1_FOOT_CONTACT_BODY_NAMES,
        "pelvis",
    )

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert any("registered G1 feet" in error for error in report.errors)


def test_rejects_stationary_robot_on_short_reference_route() -> None:
    trajectory = _acceptable_trajectory()
    trajectory["reference_g1_qpos"][:, 0] = np.linspace(0.0, 0.25, 40)
    trajectory["root_pos_w"][:, 0] = 0.0

    report = evaluate_locomotion_trajectory(trajectory)

    assert not report.accepted
    assert "insufficient_executed_motion" in report.rejection_reasons


def test_thresholds_validate_supported_fraction() -> None:
    try:
        LocomotionAcceptanceThresholds(min_supported_fraction=1.1)
    except ValueError as exc:
        assert "between zero and one" in str(exc)
    else:
        raise AssertionError("invalid supported fraction was accepted")
