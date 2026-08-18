from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from gear_sonic.research.lace.reference_feasibility import (
    REFERENCE_FEASIBILITY_KIND,
    ReferenceFeasibilityThresholds,
    compute_reference_feasibility_proxy,
)
from gear_sonic.research.lace.schema import canonical_sha256


def _inputs() -> dict:
    fps = 10.0
    time = np.arange(6, dtype=np.float64) / fps
    joints = np.column_stack((0.5 * time, -0.25 * time))
    root = np.column_stack((time, np.zeros_like(time), np.ones_like(time)))
    rotations = np.column_stack((np.zeros_like(time), np.zeros_like(time), 0.2 * time))
    half_angle = 0.1 * time
    root_quaternion = np.column_stack(
        (
            np.zeros_like(time),
            np.zeros_like(time),
            np.sin(half_angle),
            np.cos(half_angle),
        )
    )
    feet = np.zeros((len(time), 2, 3), dtype=np.float64)
    feet[:, 0, :] = [0.0, 0.1, 0.02]
    feet[:, 1, :] = [0.0, -0.1, 0.02]
    return {
        "motion_key": "walk__A001",
        "joint_position": joints,
        "root_position": root,
        "root_quaternion_xyzw": root_quaternion,
        "source_root_quaternion_xyzw": root_quaternion,
        "source_root_rotation_vector": rotations,
        "reference_foot_position": feet,
        "reference_foot_contact": np.ones((len(time), 2), dtype=bool),
        "reference_fps": fps,
        "joint_lower_limits": [-1.0, -1.0],
        "joint_upper_limits": [1.0, 1.0],
        "soft_joint_lower_limits": [-0.9, -0.9],
        "soft_joint_upper_limits": [0.9, 0.9],
        "joint_velocity_limits": [2.0, 2.0],
        "joint_names": ["joint_a", "joint_b"],
        "robot_contract_sha256": "a" * 64,
        "source_file_sha256": "b" * 64,
        "source_schema_and_file_hash_verified": True,
        "release_filename_filter_pass": True,
    }


def test_reference_proxy_is_deterministic_reference_only_and_self_bound() -> None:
    inputs = _inputs()
    original = deepcopy(inputs)

    first = compute_reference_feasibility_proxy(**inputs)
    second = compute_reference_feasibility_proxy(**inputs)

    assert first == second
    assert np.array_equal(inputs["joint_position"], original["joint_position"])
    assert first["kind"] == REFERENCE_FEASIBILITY_KIND
    assert first["policy_rollout_used"] is False
    assert first["inverse_dynamics_used"] is False
    assert first["reference_constraint_screen_pass"] is True
    assert first["hard_corruption_exclusion"] is False
    assert first["diagnostics"]["root_speed_meters_per_second"]["max"] == pytest.approx(1.0)
    assert first["diagnostics"]["root_angular_speed_radians_per_second"]["max"] == pytest.approx(
        0.2
    )
    assert len(first["continuous_feature_names"]) == len(first["continuous_feature_vector"]) == 26
    assert first["scope"] == "kinematic proxy, not dynamic feasibility proof"
    assert (
        first["reference_contact_contract"]["independent_physical_contact_labels_available"]
        is False
    )
    assert first["reference_feasibility_sha256"] == canonical_sha256(
        first,
        digest_field="reference_feasibility_sha256",
    )


def test_hard_limit_and_velocity_flags_are_separate_from_continuous_features() -> None:
    inputs = _inputs()
    joints = np.asarray(inputs["joint_position"]).copy()
    joints[-1, 0] = 1.5
    inputs["joint_position"] = joints

    result = compute_reference_feasibility_proxy(**inputs)

    assert result["reference_constraint_flags"]["joint_hard_limit_violation"] is True
    assert result["reference_constraint_flags"]["joint_velocity_limit_violation"] is True
    assert result["reference_constraint_screen_pass"] is False
    assert result["diagnostics"]["joint_hard_excess_normalized"]["max"] == pytest.approx(0.25)
    assert result["diagnostics"]["joint_velocity_ratio"]["max"] > 1.0


def test_release_keyword_filter_is_only_an_explicit_eligibility_flag() -> None:
    inputs = _inputs()
    inputs["release_filename_filter_pass"] = False

    result = compute_reference_feasibility_proxy(**inputs)

    assert result["release_eligibility"]["pass"] is False
    assert result["reference_constraint_screen_pass"] is True
    assert result["scope"] == "kinematic proxy, not dynamic feasibility proof"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.update(reference_fps=0.0), "reference_fps"),
        (
            lambda values: values.update(joint_position=np.zeros((2, 2))),
            "at least three",
        ),
        (
            lambda values: values.update(root_position=np.zeros((6, 2))),
            "root_position must have shape",
        ),
        (
            lambda values: values.update(joint_velocity_limits=[2.0, 0.0]),
            "must be positive",
        ),
        (lambda values: values.update(joint_names=["joint_a"]), "must contain 2"),
        (
            lambda values: values.update(release_filename_filter_pass=1),
            "must be boolean",
        ),
        (
            lambda values: values.update(source_schema_and_file_hash_verified=1),
            "must be boolean",
        ),
    ],
)
def test_invalid_reference_contracts_fail_closed(mutation, message: str) -> None:
    inputs = _inputs()
    mutation(inputs)

    with pytest.raises(ValueError, match=message):
        compute_reference_feasibility_proxy(**inputs)


def test_threshold_contract_rejects_invalid_values() -> None:
    inputs = _inputs()

    with pytest.raises(ValueError, match="joint_limit_proximity_fraction"):
        compute_reference_feasibility_proxy(
            **inputs,
            thresholds=ReferenceFeasibilityThresholds(joint_limit_proximity_fraction=0.5),
        )
    with pytest.raises(ValueError, match="joint_velocity_ratio_limit"):
        compute_reference_feasibility_proxy(
            **inputs,
            thresholds=ReferenceFeasibilityThresholds(joint_velocity_ratio_limit=0.0),
        )


def test_quaternion_and_contact_corruption_are_separate_hard_exclusions() -> None:
    quaternion_inputs = _inputs()
    quaternions = np.asarray(quaternion_inputs["source_root_quaternion_xyzw"]).copy()
    quaternions[2] *= 1.1
    quaternion_inputs["source_root_quaternion_xyzw"] = quaternions
    quaternion_result = compute_reference_feasibility_proxy(**quaternion_inputs)

    assert quaternion_result["hard_corruption_exclusion"] is True
    assert quaternion_result["hard_corruption_checks"]["source_quaternion_norm_violation"] is True
    assert quaternion_result["reference_constraint_screen_pass"] is True

    contact_inputs = _inputs()
    contacts = np.asarray(contact_inputs["reference_foot_contact"]).copy()
    contacts[0, 0] = False
    contact_inputs["reference_foot_contact"] = contacts
    contact_result = compute_reference_feasibility_proxy(**contact_inputs)
    assert contact_result["hard_corruption_exclusion"] is True
    assert (
        contact_result["hard_corruption_checks"]["reference_contact_kinematic_rule_mismatch"]
        is True
    )


def test_source_integrity_and_source_temporal_discontinuity_fail_closed() -> None:
    integrity_inputs = _inputs()
    integrity_inputs["source_schema_and_file_hash_verified"] = False
    integrity_result = compute_reference_feasibility_proxy(**integrity_inputs)
    assert integrity_result["hard_corruption_exclusion"] is True

    discontinuity_inputs = _inputs()
    source_rotations = np.asarray(discontinuity_inputs["source_root_rotation_vector"]).copy()
    source_rotations[3:, 2] += np.pi
    discontinuity_inputs["source_root_rotation_vector"] = source_rotations
    source_quaternions = np.zeros((len(source_rotations), 4), dtype=np.float64)
    source_quaternions[:, 2] = np.sin(0.5 * source_rotations[:, 2])
    source_quaternions[:, 3] = np.cos(0.5 * source_rotations[:, 2])
    discontinuity_inputs["source_root_quaternion_xyzw"] = source_quaternions
    discontinuity_result = compute_reference_feasibility_proxy(**discontinuity_inputs)
    assert discontinuity_result["hard_corruption_exclusion"] is True
    assert (
        discontinuity_result["hard_corruption_checks"]["root_orientation_step_discontinuity"]
        is True
    )
