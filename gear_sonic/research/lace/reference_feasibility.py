"""Reference-only feasibility proxies for LACE representation controls.

These descriptors deliberately avoid policy rollouts and inverse dynamics. They
screen corrupt or kinematically implausible G1 references and provide continuous
RQ1 covariates. They are a kinematic proxy, not a dynamic feasibility proof.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gear_sonic.research.lace.schema import canonical_sha256

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

REFERENCE_FEASIBILITY_KIND = "lace_reference_feasibility_proxy"
REFERENCE_FEASIBILITY_SCHEMA_VERSION = 2
REFERENCE_FEASIBILITY_SCOPE = "kinematic proxy, not dynamic feasibility proof"
SONIC_CONTACT_RULE_ID = "sonic_motion_lib_foot_detect_v1"


@dataclass(frozen=True)
class ReferenceFeasibilityThresholds:
    """Outcome-independent G1 reference screening thresholds."""

    joint_limit_tolerance_radians: float = 1e-4
    joint_velocity_ratio_limit: float = 1.0
    joint_limit_proximity_fraction: float = 0.05
    quaternion_norm_tolerance: float = 1e-3
    root_rotation_representation_tolerance_radians: float = 1e-4
    root_orientation_step_limit_radians: float = math.pi / 2.0
    contact_height_threshold_meters: float = 0.05
    contact_displacement_squared_threshold_m2: float = 0.0005
    contact_boundary_relative_margin: float = 0.10

    def validate(self) -> None:
        nonnegative = (
            "joint_limit_tolerance_radians",
            "quaternion_norm_tolerance",
            "root_rotation_representation_tolerance_radians",
        )
        positive = (
            "joint_velocity_ratio_limit",
            "root_orientation_step_limit_radians",
            "contact_height_threshold_meters",
            "contact_displacement_squared_threshold_m2",
        )
        for name in nonnegative:
            _finite_scalar(getattr(self, name), name, nonnegative=True)
        for name in positive:
            _positive_scalar(getattr(self, name), name)
        for name in (
            "joint_limit_proximity_fraction",
            "contact_boundary_relative_margin",
        ):
            value = _positive_scalar(getattr(self, name), name)
            if value >= 0.5:
                raise ValueError(f"{name} must be in (0, 0.5)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    _require(value == value.lower(), f"{name} must be a lowercase SHA-256")
    return value


def _finite_scalar(value: Any, name: str, *, nonnegative: bool = False) -> float:
    _require(
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value)),
        f"{name} must be a finite scalar",
    )
    result = float(value)
    if nonnegative:
        _require(result >= 0.0, f"{name} must be nonnegative")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    _require(result > 0.0, f"{name} must be a positive finite scalar")
    return result


def _array(values: ArrayLike, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(values)
    _require(raw.ndim == ndim and all(size > 0 for size in raw.shape), f"{name} shape is invalid")
    _require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must be real numeric",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(np.all(np.isfinite(result)), f"{name} must contain only finite values")
    return result


def _binary_array(values: ArrayLike, name: str, ndim: int) -> BoolArray:
    raw = np.asarray(values)
    _require(raw.ndim == ndim and all(size > 0 for size in raw.shape), f"{name} shape is invalid")
    if np.issubdtype(raw.dtype, np.bool_):
        return np.asarray(raw, dtype=np.bool_)
    _require(
        np.issubdtype(raw.dtype, np.number) and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must be boolean or binary numeric",
    )
    numeric = np.asarray(raw, dtype=np.float64)
    _require(
        np.all(np.isfinite(numeric)) and np.all((numeric == 0.0) | (numeric == 1.0)),
        f"{name} must contain only binary values",
    )
    return np.asarray(numeric, dtype=np.bool_)


def _axis_angle_to_quaternion_xyzw(rotation_vectors: FloatArray) -> FloatArray:
    angles = np.linalg.norm(rotation_vectors, axis=1)
    half_angles = 0.5 * angles
    scale = np.empty_like(angles)
    nonzero = angles > 1e-12
    scale[nonzero] = np.sin(half_angles[nonzero]) / angles[nonzero]
    scale[~nonzero] = 0.5
    quaternions = np.column_stack(
        (
            rotation_vectors * scale[:, None],
            np.cos(half_angles),
        )
    )
    return _normalize_quaternions(quaternions, "axis-angle quaternions")


def _normalize_quaternions(quaternions: FloatArray, name: str) -> FloatArray:
    norms = np.linalg.norm(quaternions, axis=1)
    _require(np.all(norms > 1e-12), f"{name} contain a zero-norm quaternion")
    return quaternions / norms[:, None]


def _quaternion_conjugate_xyzw(quaternions: FloatArray) -> FloatArray:
    result = quaternions.copy()
    result[:, :3] *= -1.0
    return result


def _quaternion_multiply_xyzw(left: FloatArray, right: FloatArray) -> FloatArray:
    left_xyz, left_w = left[:, :3], left[:, 3:4]
    right_xyz, right_w = right[:, :3], right[:, 3:4]
    xyz = left_w * right_xyz + right_w * left_xyz + np.cross(left_xyz, right_xyz)
    scalar = left_w * right_w - np.sum(left_xyz * right_xyz, axis=1, keepdims=True)
    return np.column_stack((xyz, scalar))


def _relative_rotation_vectors_world(quaternions: FloatArray) -> FloatArray:
    relative = _quaternion_multiply_xyzw(
        quaternions[1:],
        _quaternion_conjugate_xyzw(quaternions[:-1]),
    )
    relative = _normalize_quaternions(relative, "relative root rotations")
    negative_scalar = relative[:, 3] < 0.0
    relative[negative_scalar] *= -1.0
    vector_norm = np.linalg.norm(relative[:, :3], axis=1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[:, 3], 0.0, 1.0))
    scale = np.zeros_like(angle)
    nonzero = vector_norm > 1e-12
    scale[nonzero] = angle[nonzero] / vector_norm[nonzero]
    return relative[:, :3] * scale[:, None]


def _quaternion_geodesic_angle(left: FloatArray, right: FloatArray) -> FloatArray:
    cosine_half_angle = np.clip(np.abs(np.sum(left * right, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(cosine_half_angle)


def _summary(values: FloatArray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    _require(flat.size > 0 and np.all(np.isfinite(flat)), "summary input must be finite")
    return {
        "mean": float(np.mean(flat)),
        "rms": float(np.sqrt(np.mean(np.square(flat)))),
        "q90": float(np.quantile(flat, 0.90)),
        "max": float(np.max(flat)),
    }


def _summary_or_zero(values: FloatArray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return _summary(flat) if flat.size else {name: 0.0 for name in ("mean", "rms", "q90", "max")}


def _limit_diagnostics(
    joints: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    proximity_fraction: float,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, float]:
    joint_range = upper - lower
    below_limit = np.maximum(lower[None, :] - joints, 0.0)
    above_limit = np.maximum(joints - upper[None, :], 0.0)
    excess = np.maximum(below_limit, above_limit)
    normalized_excess = excess / joint_range[None, :]
    normalized_position = (joints - lower[None, :]) / joint_range[None, :]
    proximity = np.maximum(
        np.clip(
            (proximity_fraction - normalized_position) / proximity_fraction,
            0.0,
            1.0,
        ),
        np.clip(
            (normalized_position - (1.0 - proximity_fraction)) / proximity_fraction,
            0.0,
            1.0,
        ),
    )
    violation_fraction = float(np.mean(excess > tolerance))
    return proximity, normalized_excess, violation_fraction


def _sonic_foot_contact(
    foot_position: FloatArray,
    *,
    height_threshold: float,
    displacement_squared_threshold: float,
) -> tuple[BoolArray, FloatArray]:
    displacement_squared = np.sum(np.square(np.diff(foot_position, axis=0)), axis=2)
    displacement_squared = np.concatenate(
        (displacement_squared, displacement_squared[[-1]]),
        axis=0,
    )
    contact = (displacement_squared < displacement_squared_threshold) & (
        foot_position[:, :, 2] < height_threshold
    )
    return contact, displacement_squared


def compute_reference_feasibility_proxy(
    *,
    motion_key: str,
    joint_position: ArrayLike,
    root_position: ArrayLike,
    root_quaternion_xyzw: ArrayLike,
    source_root_quaternion_xyzw: ArrayLike,
    source_root_rotation_vector: ArrayLike,
    reference_foot_position: ArrayLike,
    reference_foot_contact: ArrayLike,
    reference_fps: Real,
    joint_lower_limits: ArrayLike,
    joint_upper_limits: ArrayLike,
    soft_joint_lower_limits: ArrayLike,
    soft_joint_upper_limits: ArrayLike,
    joint_velocity_limits: ArrayLike,
    joint_names: Sequence[str],
    robot_contract_sha256: str,
    source_file_sha256: str,
    source_schema_and_file_hash_verified: bool,
    release_filename_filter_pass: bool,
    thresholds: ReferenceFeasibilityThresholds | None = None,
) -> dict[str, Any]:
    """Compute a frozen reference-only descriptor and separate integrity flags."""

    _require(isinstance(motion_key, str) and motion_key, "motion_key must be non-empty")
    _require(
        isinstance(release_filename_filter_pass, bool),
        "release_filename_filter_pass must be boolean",
    )
    _require(
        isinstance(source_schema_and_file_hash_verified, bool),
        "source_schema_and_file_hash_verified must be boolean",
    )
    fps = _positive_scalar(reference_fps, "reference_fps")
    robot_digest = _sha256(robot_contract_sha256, "robot_contract_sha256")
    source_digest = _sha256(source_file_sha256, "source_file_sha256")
    config = thresholds if thresholds is not None else ReferenceFeasibilityThresholds()
    _require(
        isinstance(config, ReferenceFeasibilityThresholds),
        "thresholds must be ReferenceFeasibilityThresholds",
    )
    config.validate()

    joints = _array(joint_position, "joint_position", 2)
    time_steps, joint_count = joints.shape
    _require(time_steps >= 3, "reference feasibility requires at least three frames")
    root = _array(root_position, "root_position", 2)
    root_quaternion_raw = _array(root_quaternion_xyzw, "root_quaternion_xyzw", 2)
    feet = _array(reference_foot_position, "reference_foot_position", 3)
    contacts = _binary_array(reference_foot_contact, "reference_foot_contact", 2)
    _require(root.shape == (time_steps, 3), f"root_position must have shape [{time_steps}, 3]")
    _require(
        root_quaternion_raw.shape == (time_steps, 4),
        f"root_quaternion_xyzw must have shape [{time_steps}, 4]",
    )
    _require(
        feet.shape == (time_steps, 2, 3),
        f"reference_foot_position must have shape [{time_steps}, 2, 3]",
    )
    _require(
        contacts.shape == (time_steps, 2),
        f"reference_foot_contact must have shape [{time_steps}, 2]",
    )

    source_quaternion_raw = _array(
        source_root_quaternion_xyzw,
        "source_root_quaternion_xyzw",
        2,
    )
    source_rotation_vector = _array(
        source_root_rotation_vector,
        "source_root_rotation_vector",
        2,
    )
    _require(
        source_quaternion_raw.shape[1:] == (4,),
        "source_root_quaternion_xyzw must have shape [source_frames, 4]",
    )
    _require(
        source_rotation_vector.shape == (source_quaternion_raw.shape[0], 3),
        "source_root_rotation_vector must match source quaternion frames",
    )
    _require(
        source_quaternion_raw.shape[0] >= 2,
        "source root rotations require at least two frames",
    )

    lower = _array(joint_lower_limits, "joint_lower_limits", 1)
    upper = _array(joint_upper_limits, "joint_upper_limits", 1)
    soft_lower = _array(soft_joint_lower_limits, "soft_joint_lower_limits", 1)
    soft_upper = _array(soft_joint_upper_limits, "soft_joint_upper_limits", 1)
    velocity_limits = _array(joint_velocity_limits, "joint_velocity_limits", 1)
    for values, name in (
        (lower, "joint_lower_limits"),
        (upper, "joint_upper_limits"),
        (soft_lower, "soft_joint_lower_limits"),
        (soft_upper, "soft_joint_upper_limits"),
        (velocity_limits, "joint_velocity_limits"),
    ):
        _require(values.shape == (joint_count,), f"{name} must have shape [{joint_count}]")
    _require(np.all(upper > lower), "joint upper limits must exceed lower limits")
    _require(np.all(soft_upper > soft_lower), "soft joint upper limits must exceed lower limits")
    _require(
        np.all(soft_lower >= lower) and np.all(soft_upper <= upper),
        "soft joint limits must be contained within hard limits",
    )
    _require(np.all(velocity_limits > 0.0), "joint velocity limits must be positive")
    _require(
        isinstance(joint_names, Sequence) and not isinstance(joint_names, (str, bytes)),
        "joint_names must be an ordered sequence",
    )
    ordered_joint_names = list(joint_names)
    _require(
        len(ordered_joint_names) == joint_count
        and all(isinstance(name, str) and name for name in ordered_joint_names)
        and len(set(ordered_joint_names)) == joint_count,
        f"joint_names must contain {joint_count} unique non-empty names",
    )

    joint_velocity = np.diff(joints, axis=0) * fps
    joint_acceleration = np.diff(joint_velocity, axis=0) * fps
    velocity_ratio = np.abs(joint_velocity) / velocity_limits[None, :]
    hard_proximity, hard_excess, hard_violation_fraction = _limit_diagnostics(
        joints,
        lower,
        upper,
        proximity_fraction=float(config.joint_limit_proximity_fraction),
        tolerance=float(config.joint_limit_tolerance_radians),
    )
    soft_proximity, soft_excess, soft_violation_fraction = _limit_diagnostics(
        joints,
        soft_lower,
        soft_upper,
        proximity_fraction=float(config.joint_limit_proximity_fraction),
        tolerance=float(config.joint_limit_tolerance_radians),
    )

    root_velocity = np.diff(root, axis=0) * fps
    root_acceleration = np.diff(root_velocity, axis=0) * fps
    root_speed = np.linalg.norm(root_velocity, axis=1)
    root_acceleration_magnitude = np.linalg.norm(root_acceleration, axis=1)

    root_quaternions = _normalize_quaternions(root_quaternion_raw, "root quaternions")
    root_rotation_steps = _relative_rotation_vectors_world(root_quaternions)
    angular_velocity = root_rotation_steps * fps
    angular_speed = np.linalg.norm(angular_velocity, axis=1)
    angular_acceleration = np.diff(angular_velocity, axis=0) * fps
    angular_acceleration_magnitude = np.linalg.norm(angular_acceleration, axis=1)

    source_quaternion_norm = np.linalg.norm(source_quaternion_raw, axis=1)
    _require(
        np.all(source_quaternion_norm > 1e-12),
        "source_root_quaternion_xyzw contains a zero-norm quaternion",
    )
    source_quaternions = source_quaternion_raw / source_quaternion_norm[:, None]
    source_axis_angle_quaternions = _axis_angle_to_quaternion_xyzw(source_rotation_vector)
    source_root_rotation_steps = _relative_rotation_vectors_world(source_quaternions)
    representation_disagreement = _quaternion_geodesic_angle(
        source_quaternions,
        source_axis_angle_quaternions,
    )
    quaternion_norm_deviation = np.abs(source_quaternion_norm - 1.0)

    expected_contact, foot_displacement_squared = _sonic_foot_contact(
        feet,
        height_threshold=float(config.contact_height_threshold_meters),
        displacement_squared_threshold=float(config.contact_displacement_squared_threshold_m2),
    )
    contact_mismatch = contacts ^ expected_contact
    foot_speed = np.sqrt(foot_displacement_squared) * fps
    contact_speed = foot_speed[contacts]
    contact_height = feet[:, :, 2][contacts]
    contact_transitions = np.count_nonzero(contacts[1:] != contacts[:-1])
    duration_seconds = (time_steps - 1) / fps
    transitions_per_foot_second = float(contact_transitions / (2.0 * duration_seconds))
    contact_count = np.sum(contacts, axis=1)
    height_relative_distance = np.abs(
        feet[:, :, 2] - float(config.contact_height_threshold_meters)
    ) / float(config.contact_height_threshold_meters)
    displacement_relative_distance = np.abs(
        foot_displacement_squared - float(config.contact_displacement_squared_threshold_m2)
    ) / float(config.contact_displacement_squared_threshold_m2)
    contact_boundary_fraction = float(
        np.mean(
            np.minimum(height_relative_distance, displacement_relative_distance)
            < float(config.contact_boundary_relative_margin)
        )
    )

    velocity_summary = _summary(velocity_ratio)
    acceleration_summary = _summary(np.abs(joint_acceleration))
    hard_proximity_summary = _summary(hard_proximity)
    soft_proximity_summary = _summary(soft_proximity)
    hard_excess_summary = _summary(hard_excess)
    soft_excess_summary = _summary(soft_excess)
    root_speed_summary = _summary(root_speed)
    root_acceleration_summary = _summary(root_acceleration_magnitude)
    angular_speed_summary = _summary(angular_speed)
    angular_acceleration_summary = _summary(angular_acceleration_magnitude)
    contact_speed_summary = _summary_or_zero(contact_speed)
    contact_height_summary = _summary_or_zero(contact_height)
    foot_clearance_summary = _summary(feet[:, :, 2])

    hard_joint_limit_violation = bool(
        np.any(hard_excess > float(config.joint_limit_tolerance_radians))
    )
    soft_joint_limit_violation = bool(
        np.any(soft_excess > float(config.joint_limit_tolerance_radians))
    )
    joint_velocity_violation = bool(
        np.any(velocity_ratio > float(config.joint_velocity_ratio_limit))
    )
    corruption_checks = {
        "source_schema_and_file_hash_verified": source_schema_and_file_hash_verified,
        "source_quaternion_norm_violation": bool(
            np.any(quaternion_norm_deviation > float(config.quaternion_norm_tolerance))
        ),
        "root_rotation_representation_mismatch": bool(
            np.any(
                representation_disagreement
                > float(config.root_rotation_representation_tolerance_radians)
            )
        ),
        "root_orientation_step_discontinuity": bool(
            np.any(
                np.linalg.norm(source_root_rotation_steps, axis=1)
                > float(config.root_orientation_step_limit_radians)
            )
            or np.any(
                np.linalg.norm(root_rotation_steps, axis=1)
                > float(config.root_orientation_step_limit_radians)
            )
        ),
        "reference_contact_kinematic_rule_mismatch": bool(np.any(contact_mismatch)),
    }
    hard_corruption_exclusion = (
        any(
            value
            for name, value in corruption_checks.items()
            if name != "source_schema_and_file_hash_verified"
        )
        or not corruption_checks["source_schema_and_file_hash_verified"]
    )

    feature_names = [
        "log_duration_seconds",
        "joint_velocity_ratio_q90",
        "joint_velocity_ratio_max",
        "joint_acceleration_abs_rms",
        "joint_acceleration_abs_q90",
        "joint_hard_limit_proximity_mean",
        "joint_hard_limit_proximity_q90",
        "joint_soft_limit_proximity_mean",
        "joint_soft_limit_proximity_q90",
        "joint_hard_excess_fraction",
        "joint_hard_excess_normalized_max",
        "joint_soft_excess_fraction",
        "joint_soft_excess_normalized_max",
        "root_speed_q90",
        "root_speed_max",
        "root_acceleration_q90",
        "root_angular_speed_q90",
        "root_angular_speed_max",
        "root_angular_acceleration_q90",
        "foot_no_support_fraction",
        "foot_double_support_fraction",
        "foot_contact_transitions_per_foot_second",
        "foot_contact_speed_q90",
        "foot_contact_height_q90",
        "foot_clearance_q90",
        "foot_contact_boundary_fraction",
    ]
    feature_vector = [
        math.log(max(duration_seconds, np.finfo(np.float64).tiny)),
        velocity_summary["q90"],
        velocity_summary["max"],
        acceleration_summary["rms"],
        acceleration_summary["q90"],
        hard_proximity_summary["mean"],
        hard_proximity_summary["q90"],
        soft_proximity_summary["mean"],
        soft_proximity_summary["q90"],
        hard_violation_fraction,
        hard_excess_summary["max"],
        soft_violation_fraction,
        soft_excess_summary["max"],
        root_speed_summary["q90"],
        root_speed_summary["max"],
        root_acceleration_summary["q90"],
        angular_speed_summary["q90"],
        angular_speed_summary["max"],
        angular_acceleration_summary["q90"],
        float(np.mean(contact_count == 0)),
        float(np.mean(contact_count == 2)),
        transitions_per_foot_second,
        contact_speed_summary["q90"],
        contact_height_summary["q90"],
        foot_clearance_summary["q90"],
        contact_boundary_fraction,
    ]
    _require(all(math.isfinite(value) for value in feature_vector), "feature vector is non-finite")

    result: dict[str, Any] = {
        "kind": REFERENCE_FEASIBILITY_KIND,
        "schema_version": REFERENCE_FEASIBILITY_SCHEMA_VERSION,
        "motion_key": motion_key,
        "source_file_sha256": source_digest,
        "robot_contract_sha256": robot_digest,
        "reference_fps": fps,
        "reference_num_frames": time_steps,
        "source_file_num_frames": int(source_quaternion_raw.shape[0]),
        "duration_seconds_exclusive_endpoint": float(duration_seconds),
        "joint_names": ordered_joint_names,
        "thresholds": asdict(config),
        "release_eligibility": {
            "rule": "official_release_filename_filter",
            "pass": release_filename_filter_pass,
        },
        "hard_corruption_checks": corruption_checks,
        "hard_corruption_exclusion": bool(hard_corruption_exclusion),
        "reference_constraint_flags": {
            "joint_hard_limit_violation": hard_joint_limit_violation,
            "joint_soft_limit_violation": soft_joint_limit_violation,
            "joint_velocity_limit_violation": joint_velocity_violation,
        },
        "reference_constraint_screen_pass": bool(
            not hard_joint_limit_violation and not joint_velocity_violation
        ),
        "continuous_feature_names": feature_names,
        "continuous_feature_vector": [float(value) for value in feature_vector],
        "diagnostics": {
            "joint_velocity_ratio": velocity_summary,
            "joint_acceleration_abs_radians_per_second_squared": acceleration_summary,
            "joint_hard_limit_proximity": hard_proximity_summary,
            "joint_soft_limit_proximity": soft_proximity_summary,
            "joint_hard_excess_normalized": hard_excess_summary,
            "joint_soft_excess_normalized": soft_excess_summary,
            "source_quaternion_norm_deviation": _summary(quaternion_norm_deviation),
            "root_rotation_representation_disagreement_radians": _summary(
                representation_disagreement
            ),
            "root_orientation_step_radians": _summary(np.linalg.norm(root_rotation_steps, axis=1)),
            "source_root_orientation_step_radians": _summary(
                np.linalg.norm(source_root_rotation_steps, axis=1)
            ),
            "root_speed_meters_per_second": root_speed_summary,
            "root_acceleration_meters_per_second_squared": root_acceleration_summary,
            "root_angular_speed_radians_per_second": angular_speed_summary,
            "root_angular_acceleration_radians_per_second_squared": (angular_acceleration_summary),
            "foot_contact_speed_meters_per_second": contact_speed_summary,
            "foot_contact_height_meters": contact_height_summary,
            "foot_clearance_meters": foot_clearance_summary,
            "reference_contact_kinematic_mismatch_fraction": float(np.mean(contact_mismatch)),
        },
        "reference_contact_contract": {
            "rule_id": SONIC_CONTACT_RULE_ID,
            "runtime_source": "gear_sonic.utils.motion_lib.motion_lib_base:MotionLibBase.foot_detect",
            "contact_labels_are_kinematically_derived": True,
            "independent_physical_contact_labels_available": False,
            "consistency_check_scope": (
                "reproduces_the_frozen_SONIC_kinematic_contact_rule_not_physical_contact"
            ),
        },
        "scope": REFERENCE_FEASIBILITY_SCOPE,
        "inverse_dynamics_used": False,
        "policy_rollout_used": False,
        "dynamic_feasibility_proof": False,
    }
    result["reference_feasibility_sha256"] = canonical_sha256(result)
    return result
