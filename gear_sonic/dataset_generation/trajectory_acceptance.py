"""Reason-coded pilot acceptance gates for SONIC locomotion trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np

from gear_sonic.dataset_generation.contact_decomposition import decompose_contact_forces
from gear_sonic.dataset_generation.trajectory_validation import validate_sonic_trajectory

G1_CONTACT_BODY_NAMES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)
G1_FOOT_CONTACT_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
)


@dataclass(frozen=True)
class LocomotionAcceptanceThresholds:
    """Provisional M0 thresholds to calibrate on a reviewed 100-episode pilot."""

    min_frames: int = 40
    min_reference_displacement_m: float = 0.25
    min_executed_progress_ratio: float = 0.50
    endpoint_error_m: float = 0.35
    path_error_p95_m: float = 0.25
    # Absolute collapse floor, NOT "standing height". A deliberate crouch is a
    # legitimate behaviour: 02_multi_text_ee_constraint tracks a reference whose
    # pelvis dips to 0.323 m while staying upright (max tilt 0.245 rad) and within
    # 4 cm of its command. An absolute 0.50 m threshold called that a fall. Below
    # 0.25 m the pelvis is at roughly knee height, which no commanded G1 stance
    # uses. PROVISIONAL: a physical anchor, not a calibration.
    min_root_height_m: float = 0.25
    # A fall is the root sinking below what it was *commanded*, at any absolute
    # height. Observed executed-minus-reference minima were -0.039 and -0.040 m on
    # accepted episodes; 0.15 m is well outside that while still catching a real
    # sink. PROVISIONAL.
    max_root_height_below_reference_m: float = 0.15
    # Absolute limit only. 03_full_body_keyframes tracks a reference that *commands*
    # a 41.9 deg bow; the robot followed it to within 7 deg while upright at 0.64 m,
    # and an absolute 0.60 rad threshold called that a fall. Beyond ~80 deg the torso
    # is near-horizontal, which no commanded G1 locomotion stance uses. PROVISIONAL.
    max_abs_roll_pitch_rad: float = 1.40
    # Leaning further than commanded is a fall at any absolute angle. Observed excess
    # on accepted episodes was 0.124-0.139 rad. PROVISIONAL.
    max_roll_pitch_above_reference_rad: float = 0.35
    # Robot-to-environment contact on a non-foot body. This is the real navigation
    # safety gate and stays strict.
    max_nonfoot_contact_force_n: float = 1.0
    # Upward load carried by a non-foot body. A crouch that puts a knee on the floor
    # is legitimate (measured 105-549 N, purely vertical, while kneeling), so this is
    # a behaviour descriptor rather than a collision. 686 N is 2x nominal G1 body
    # weight, above which the load is impulsive enough to be worth rejecting.
    # PROVISIONAL.
    max_nonfoot_support_force_n: float = 686.0
    # Robot-to-self contact (e.g. wrist swinging into hip). PROVISIONAL and NOT yet
    # calibrated: measured peaks on three M0 rollouts were 33.8 N, 268.8 N and
    # 309.2 N, the largest of which occurred on a bare plane with no obstacles, so
    # this is a gait property rather than a scene failure. The default is anchored
    # to nominal G1 body weight (~35 kg, ~343 N) as a physical scale rather than
    # fitted to those three samples: a self-impact exceeding body weight is
    # unambiguously abnormal. Calibrate against the reviewed pilot before freezing.
    max_self_contact_force_n: float = 343.0
    max_foot_non_ground_contact_force_n: float = 1.0
    foot_support_force_n: float = 10.0
    min_supported_fraction: float = 0.90

    def __post_init__(self) -> None:
        if not isinstance(self.min_frames, int) or isinstance(self.min_frames, bool):
            raise ValueError("min_frames must be an integer")
        if self.min_frames < 1:
            raise ValueError("min_frames must be positive")
        for name, value in asdict(self).items():
            if name == "min_frames":
                continue
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_reference_displacement_m <= 0:
            raise ValueError("min_reference_displacement_m must be positive")
        if not 0.0 < self.min_executed_progress_ratio <= 1.0:
            raise ValueError("min_executed_progress_ratio must be in (0, 1]")
        if self.endpoint_error_m <= 0 or self.path_error_p95_m <= 0:
            raise ValueError("route thresholds must be positive")
        if (
            self.min_root_height_m <= 0
            or self.max_root_height_below_reference_m <= 0
            or self.max_abs_roll_pitch_rad <= 0
            or self.max_roll_pitch_above_reference_rad <= 0
        ):
            raise ValueError("fall thresholds must be positive")
        if (
            self.max_nonfoot_contact_force_n < 0
            or self.max_nonfoot_support_force_n < 0
            or self.max_self_contact_force_n < 0
            or self.max_foot_non_ground_contact_force_n < 0
            or self.foot_support_force_n < 0
        ):
            raise ValueError("contact thresholds must be non-negative")
        if not 0.0 <= self.min_supported_fraction <= 1.0:
            raise ValueError("min_supported_fraction must be between zero and one")


@dataclass(frozen=True)
class AcceptanceGate:
    name: str
    passed: bool
    value: float | None
    threshold: float | None
    comparison: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocomotionAcceptanceReport:
    gates: tuple[AcceptanceGate, ...]
    errors: tuple[str, ...]
    thresholds: LocomotionAcceptanceThresholds
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return not self.errors and all(gate.passed for gate in self.gates)

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        return tuple(gate.reason_code for gate in self.gates if not gate.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "errors": list(self.errors),
            "rejection_reasons": list(self.rejection_reasons),
            "thresholds": asdict(self.thresholds),
            "gates": [gate.to_dict() for gate in self.gates],
            "diagnostics": dict(self.diagnostics),
        }


def _wxyz_roll_pitch(quaternions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


def _validated_contact_evidence(
    payload: Mapping[str, Any],
) -> tuple[
    np.ndarray | None,
    tuple[str, ...] | None,
    np.ndarray | None,
    np.ndarray | None,
    tuple[str, ...],
]:
    """Return pair-resolved contact vectors or fail closed."""
    errors: list[str] = []
    required = (
        "robot_contact_force_w",
        "robot_contact_force_norm_w",
        "contact_body_names",
        "left_foot_ground_contact_force_w",
        "right_foot_ground_contact_force_w",
        "support_floor_prim_path",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing physics contact evidence: {key}")
    if errors:
        return None, None, None, None, tuple(errors)

    contact_force_vectors = np.asarray(payload["robot_contact_force_w"], dtype=np.float64)
    declared_contact_norms = np.asarray(payload["robot_contact_force_norm_w"], dtype=np.float64)
    left_ground_force = np.asarray(payload["left_foot_ground_contact_force_w"], dtype=np.float64)
    right_ground_force = np.asarray(payload["right_foot_ground_contact_force_w"], dtype=np.float64)
    raw_names = payload["contact_body_names"]
    if not isinstance(raw_names, (list, tuple)):
        return None, None, None, None, ("contact_body_names must be a list or tuple",)
    contact_body_names = tuple(raw_names)
    names_are_valid = all(isinstance(name, str) and name for name in contact_body_names)
    if not names_are_valid:
        errors.append("contact_body_names must contain only non-empty strings")
    else:
        if len(set(contact_body_names)) != len(contact_body_names):
            errors.append("contact_body_names must be unique")

        expected_names = set(G1_CONTACT_BODY_NAMES)
        actual_names = set(contact_body_names)
        if actual_names != expected_names or len(contact_body_names) != len(G1_CONTACT_BODY_NAMES):
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            errors.append(
                "contact_body_names must exactly cover the registered G1 bodies; "
                f"missing={missing}, unexpected={unexpected}"
            )

    frame_count = int(payload["total_frames"])
    expected_vector_shape = (frame_count, len(contact_body_names), 3)
    expected_norm_shape = (frame_count, len(contact_body_names))
    if contact_force_vectors.shape != expected_vector_shape:
        errors.append(
            "robot_contact_force_w shape mismatch: "
            f"expected {expected_vector_shape}, got {contact_force_vectors.shape}"
        )
    if declared_contact_norms.shape != expected_norm_shape:
        errors.append(
            "robot_contact_force_norm_w shape mismatch: "
            f"expected {expected_norm_shape}, got {declared_contact_norms.shape}"
        )
    elif np.any(declared_contact_norms < 0.0):
        errors.append("robot_contact_force_norm_w contains negative force magnitudes")
    if (
        contact_force_vectors.shape == expected_vector_shape
        and declared_contact_norms.shape == expected_norm_shape
    ):
        recomputed_norms = np.linalg.norm(contact_force_vectors, axis=-1)
        if not np.allclose(declared_contact_norms, recomputed_norms, atol=1e-4, rtol=1e-5):
            errors.append("robot_contact_force_norm_w does not match robot_contact_force_w")

    expected_ground_shape = (frame_count, 3)
    for key, values in (
        ("left_foot_ground_contact_force_w", left_ground_force),
        ("right_foot_ground_contact_force_w", right_ground_force),
    ):
        if values.shape != expected_ground_shape:
            errors.append(
                f"{key} shape mismatch: expected {expected_ground_shape}, got {values.shape}"
            )
    if payload.get("support_floor_prim_path") != "/World/ground/terrain/Structure/Floor":
        errors.append("support_floor_prim_path must identify the registered dataset floor")

    raw_foot_names = payload.get("allowed_foot_contact_body_names", G1_FOOT_CONTACT_BODY_NAMES)
    if not isinstance(raw_foot_names, (list, tuple)):
        errors.append("allowed_foot_contact_body_names must be a list or tuple")
    elif tuple(raw_foot_names) != G1_FOOT_CONTACT_BODY_NAMES:
        errors.append("allowed_foot_contact_body_names must exactly match the registered G1 feet")

    if errors:
        return None, None, None, None, tuple(errors)
    return contact_force_vectors, contact_body_names, left_ground_force, right_ground_force, ()


def cross_track_error(executed_xy: np.ndarray, reference_xy: np.ndarray) -> np.ndarray:
    """Distance from each executed position to the reference *path*, not to its position in time.

    ``‖executed(t) − reference(t)‖`` is one number covering two unrelated failures: leaving the
    route, and being behind on it. A crouched robot commits only the second -- its stride shortens,
    so it arrives at each place later -- and the combined number then charges it for a deviation it
    never made. Measured against the polyline, the nominal walk is 5% phase and a deep crouch is
    53%, and every crouch measured stays inside the route-adherence threshold it was failing.

    Schedule adherence is still worth knowing and is still reported; it is simply a different fact
    from route adherence, and one gate cannot answer both.
    """
    starts, ends = reference_xy[:-1], reference_xy[1:]
    spans = ends - starts
    lengths = np.einsum("ij,ij->i", spans, spans)
    lengths[lengths < 1e-12] = 1e-12
    out = np.empty(len(executed_xy), dtype=np.float64)
    for index, point in enumerate(executed_xy):
        t = np.clip(np.einsum("ij,ij->i", point - starts, spans) / lengths, 0.0, 1.0)
        closest = starts + t[:, None] * spans
        out[index] = float(np.min(np.linalg.norm(point - closest, axis=1)))
    return out


def evaluate_locomotion_trajectory(
    payload: Mapping[str, Any],
    thresholds: LocomotionAcceptanceThresholds | None = None,
) -> LocomotionAcceptanceReport:
    """Evaluate one raw rollout without collapsing failures into one Boolean."""
    thresholds = thresholds or LocomotionAcceptanceThresholds()
    raw_report = validate_sonic_trajectory(payload)
    if not raw_report.ok:
        return LocomotionAcceptanceReport((), raw_report.errors, thresholds)

    (
        contact_force_vectors,
        contact_body_names,
        left_ground_force,
        right_ground_force,
        contact_errors,
    ) = _validated_contact_evidence(payload)
    if contact_errors:
        return LocomotionAcceptanceReport((), contact_errors, thresholds)
    assert contact_force_vectors is not None
    assert contact_body_names is not None
    assert left_ground_force is not None
    assert right_ground_force is not None

    root_pos = np.asarray(payload["root_pos_w"], dtype=np.float64)
    root_quat = np.asarray(payload["root_quat_w"], dtype=np.float64)
    reference_root = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, :3]
    path_errors = np.linalg.norm(root_pos[:, :2] - reference_root[:, :2], axis=1)
    reference_delta = reference_root[-1, :2] - reference_root[0, :2]
    reference_displacement = float(np.linalg.norm(reference_delta))
    actual_delta = root_pos[-1, :2] - root_pos[0, :2]
    if reference_displacement > np.finfo(np.float64).eps:
        executed_progress = float(np.dot(actual_delta, reference_delta / reference_displacement))
        executed_progress_ratio = executed_progress / reference_displacement
    else:
        executed_progress_ratio = 0.0
    endpoint_error = float(path_errors[-1])
    # Route adherence gates; schedule adherence is reported beside it. See cross_track_error.
    schedule_error_p95 = float(np.quantile(path_errors, 0.95))
    path_error_p95 = float(
        np.quantile(cross_track_error(root_pos[:, :2], reference_root[:, :2]), 0.95)
    )
    min_root_height = float(np.min(root_pos[:, 2]))
    # Sinking below the commanded pelvis height is a fall whatever the absolute
    # value; a low absolute height is only a fall when nothing commanded it.
    reference_height = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, 2]
    height_below_reference = float(np.max(reference_height - root_pos[:, 2]))
    min_height_deficit_frame = int(np.argmax(reference_height - root_pos[:, 2]))
    roll, pitch = _wxyz_roll_pitch(root_quat)
    executed_tilt = np.maximum(np.abs(roll), np.abs(pitch))
    max_abs_roll_pitch = float(np.max(executed_tilt))
    # Tilting further than commanded is a fall at any absolute angle; a large
    # absolute tilt is only a fall when nothing commanded it.
    reference_roll, reference_pitch = _wxyz_roll_pitch(
        np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, 3:7]
    )
    reference_tilt = np.maximum(np.abs(reference_roll), np.abs(reference_pitch))
    tilt_above_reference = float(np.max(executed_tilt - reference_tilt))
    left_foot_index = contact_body_names.index(G1_FOOT_CONTACT_BODY_NAMES[0])
    right_foot_index = contact_body_names.index(G1_FOOT_CONTACT_BODY_NAMES[1])
    nonfoot_indices = [
        index
        for index, body_name in enumerate(contact_body_names)
        if body_name not in G1_FOOT_CONTACT_BODY_NAMES
    ]
    contact_force_norm = np.linalg.norm(contact_force_vectors, axis=-1)

    # The sensor reports one summed net force per body, so the raw non-foot maximum
    # cannot tell "wrist hit own hip" from "hip hit a wall". Split it by Newton's
    # third law before gating; see contact_decomposition for the method and limits.
    decomposition = decompose_contact_forces(
        contact_force_vectors,
        contact_body_names,
        foot_body_names=G1_FOOT_CONTACT_BODY_NAMES,
    )
    # Collision is a horizontal push *or* a downward one. A purely upward reaction is the floor
    # supporting a knee or hand during a crouch, which is behaviour, not a crash -- but excluding
    # all vertical force to exclude that also excluded every overhead collision. A crouch jammed
    # under a shelf was pushed down on torso_link at 1017.4 N with a horizontal component of
    # exactly 0.0, and this gate stayed silent; the episode was caught only by the drift that
    # followed, so a milder jam would have been accepted. Sign separates the two cases.
    if decomposition.max_overhead_contact > decomposition.max_lateral_contact:
        max_nonfoot_frame = decomposition.max_overhead_contact_frame
        max_nonfoot_force = decomposition.max_overhead_contact
    else:
        max_nonfoot_frame = decomposition.max_lateral_contact_frame
        max_nonfoot_force = decomposition.max_lateral_contact
    max_support_force = decomposition.max_support_contact
    max_self_contact_force = decomposition.max_self_contact
    max_self_contact_frame = decomposition.max_self_contact_frame

    # Retained for reporting and for comparison against pre-decomposition records.
    raw_nonfoot_by_frame = np.max(contact_force_norm[:, nonfoot_indices], axis=1)
    raw_max_nonfoot_force = float(raw_nonfoot_by_frame.max()) if raw_nonfoot_by_frame.size else 0.0
    left_foot_non_ground_force = np.linalg.norm(
        contact_force_vectors[:, left_foot_index] - left_ground_force,
        axis=-1,
    )
    right_foot_non_ground_force = np.linalg.norm(
        contact_force_vectors[:, right_foot_index] - right_ground_force,
        axis=-1,
    )
    foot_non_ground_force_by_frame = np.maximum(
        left_foot_non_ground_force,
        right_foot_non_ground_force,
    )
    max_foot_non_ground_frame = int(np.argmax(foot_non_ground_force_by_frame))
    max_foot_non_ground_force = float(foot_non_ground_force_by_frame[max_foot_non_ground_frame])
    left_foot_support_force = np.maximum(left_ground_force[:, 2], 0.0)
    right_foot_support_force = np.maximum(right_ground_force[:, 2], 0.0)
    supported_fraction = float(
        np.mean(
            np.maximum(left_foot_support_force, right_foot_support_force)
            >= thresholds.foot_support_force_n
        )
    )

    diagnostics: dict[str, Any] = {
        "contact_decomposition": decomposition.to_dict(),
        "raw_max_nonfoot_contact_force_n": raw_max_nonfoot_force,
        # Reported beside the gates. Whether a reference gate *binds* is not this function's call:
        # gate_policy.py decides that per episode, because an episode whose room was built around
        # the executed corridor is not being asked to hug its reference. Demoting here instead
        # would silently disarm the endpoint test for scene-first episodes, where it is the label.
        "schedule_error_p95_m": schedule_error_p95,
        "endpoint_error_m": endpoint_error,
        "phase_share_of_schedule_error": (
            float(1.0 - path_error_p95 / schedule_error_p95) if schedule_error_p95 > 1e-9 else 0.0
        ),
    }
    if max_nonfoot_force > 0.0:
        diagnostics.update(
            {
                "max_nonfoot_contact_frame": max_nonfoot_frame,
                "max_nonfoot_contact_time_s": max_nonfoot_frame / float(payload["fps"]),
            }
        )
        frame_forces = contact_force_norm[max_nonfoot_frame]
        peak_force = max(float(frame_forces[index]) for index in nonfoot_indices)
        diagnostics["max_nonfoot_contact_bodies"] = [
            contact_body_names[index]
            for index in nonfoot_indices
            if np.isclose(frame_forces[index], peak_force, atol=1e-5, rtol=1e-5)
        ]
    if max_self_contact_force > 0.0:
        diagnostics["max_self_contact_time_s"] = max_self_contact_frame / float(payload["fps"])
    diagnostics["min_reference_root_height_m"] = float(reference_height.min())
    diagnostics["max_reference_root_tilt_rad"] = float(reference_tilt.max())
    diagnostics["max_root_tilt_above_reference_rad"] = tilt_above_reference
    diagnostics["max_root_height_below_reference_m"] = height_below_reference
    diagnostics["max_root_height_deficit_frame"] = min_height_deficit_frame
    if max_foot_non_ground_force > 0.0:
        diagnostics["max_foot_non_ground_contact_frame"] = max_foot_non_ground_frame
        diagnostics["max_foot_non_ground_contact_time_s"] = max_foot_non_ground_frame / float(
            payload["fps"]
        )

    gates = (
        AcceptanceGate(
            "episode_length",
            int(payload["total_frames"]) >= thresholds.min_frames,
            float(payload["total_frames"]),
            float(thresholds.min_frames),
            ">=",
            "episode_too_short",
        ),
        AcceptanceGate(
            "reference_displacement",
            reference_displacement >= thresholds.min_reference_displacement_m,
            reference_displacement,
            thresholds.min_reference_displacement_m,
            ">=",
            "not_locomotion",
        ),
        AcceptanceGate(
            "executed_progress_ratio",
            executed_progress_ratio >= thresholds.min_executed_progress_ratio,
            executed_progress_ratio,
            thresholds.min_executed_progress_ratio,
            ">=",
            "insufficient_executed_motion",
        ),
        AcceptanceGate(
            "endpoint_error",
            endpoint_error <= thresholds.endpoint_error_m,
            endpoint_error,
            thresholds.endpoint_error_m,
            "<=",
            "reference_endpoint_tracking_error",
        ),
        AcceptanceGate(
            "path_error_p95",
            path_error_p95 <= thresholds.path_error_p95_m,
            path_error_p95,
            thresholds.path_error_p95_m,
            "<=",
            "reference_path_tracking_error",
        ),
        AcceptanceGate(
            "root_height",
            min_root_height >= thresholds.min_root_height_m,
            min_root_height,
            thresholds.min_root_height_m,
            ">=",
            "fall_root_height",
        ),
        AcceptanceGate(
            "root_height_tracking",
            height_below_reference <= thresholds.max_root_height_below_reference_m,
            height_below_reference,
            thresholds.max_root_height_below_reference_m,
            "<=",
            "root_sank_below_reference",
        ),
        AcceptanceGate(
            "root_tilt",
            max_abs_roll_pitch <= thresholds.max_abs_roll_pitch_rad,
            max_abs_roll_pitch,
            thresholds.max_abs_roll_pitch_rad,
            "<=",
            "fall_root_tilt",
        ),
        AcceptanceGate(
            "root_tilt_tracking",
            tilt_above_reference <= thresholds.max_roll_pitch_above_reference_rad,
            tilt_above_reference,
            thresholds.max_roll_pitch_above_reference_rad,
            "<=",
            "root_tilted_beyond_reference",
        ),
        AcceptanceGate(
            "nonfoot_scene_contact",
            max_nonfoot_force <= thresholds.max_nonfoot_contact_force_n,
            max_nonfoot_force,
            thresholds.max_nonfoot_contact_force_n,
            "<=",
            "disallowed_robot_contact",
        ),
        AcceptanceGate(
            "nonfoot_support_contact",
            max_support_force <= thresholds.max_nonfoot_support_force_n,
            max_support_force,
            thresholds.max_nonfoot_support_force_n,
            "<=",
            "excessive_nonfoot_support_load",
        ),
        AcceptanceGate(
            "self_contact",
            max_self_contact_force <= thresholds.max_self_contact_force_n,
            max_self_contact_force,
            thresholds.max_self_contact_force_n,
            "<=",
            "excessive_self_contact",
        ),
        AcceptanceGate(
            "foot_non_ground_contact",
            max_foot_non_ground_force <= thresholds.max_foot_non_ground_contact_force_n,
            max_foot_non_ground_force,
            thresholds.max_foot_non_ground_contact_force_n,
            "<=",
            "disallowed_foot_non_ground_contact",
        ),
        AcceptanceGate(
            "foot_support",
            supported_fraction >= thresholds.min_supported_fraction,
            supported_fraction,
            thresholds.min_supported_fraction,
            ">=",
            "insufficient_foot_support",
        ),
    )
    return LocomotionAcceptanceReport(gates, (), thresholds, diagnostics)
