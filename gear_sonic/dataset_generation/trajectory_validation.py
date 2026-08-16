"""Validation for raw SONIC-in-physics trajectory artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class TrajectoryValidationReport:
    """Machine-readable result from :func:`validate_sonic_trajectory`."""

    errors: tuple[str, ...]
    notes: tuple[str, ...]
    frame_count: int | None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "frame_count": self.frame_count,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


_REQUIRED_FRAME_FIELDS: dict[str, tuple[int, ...]] = {
    "root_pos_w": (3,),
    "root_quat_w": (4,),
    "root_lin_vel_w": (3,),
    "root_ang_vel_w": (3,),
    "projected_gravity_b": (3,),
    "action_motion_token": (64,),
    "reference_g1_qpos": (36,),
}


def _as_finite_frame_array(
    payload: Mapping[str, Any],
    key: str,
    *,
    frame_count: int | None,
    trailing_shape: tuple[int, ...] | None,
    errors: list[str],
) -> np.ndarray | None:
    if key not in payload:
        errors.append(f"missing required field: {key}")
        return None
    value = np.asarray(payload[key])
    if value.ndim < 1:
        errors.append(f"{key} must have a frame dimension; got shape {value.shape}")
        return None
    if frame_count is not None and value.shape[0] != frame_count:
        errors.append(f"{key} frame count mismatch: expected {frame_count}, got {value.shape[0]}")
    if trailing_shape is not None and value.shape[1:] != trailing_shape:
        errors.append(
            f"{key} shape mismatch: expected ({frame_count}, {', '.join(map(str, trailing_shape))}), "
            f"got {value.shape}"
        )
    try:
        finite = np.isfinite(value)
    except TypeError:
        errors.append(f"{key} must be numeric; got dtype {value.dtype}")
        return None
    if not finite.all():
        errors.append(f"{key} contains NaN or infinity")
    return value


def _check_unit_quaternions(
    quaternions: np.ndarray | None,
    *,
    key: str,
    errors: list[str],
    atol: float,
) -> None:
    if quaternions is None or quaternions.ndim != 2 or quaternions.shape[-1] != 4:
        return
    norms = np.linalg.norm(quaternions, axis=-1)
    if not np.allclose(norms, 1.0, atol=atol, rtol=0.0):
        max_error = float(np.max(np.abs(norms - 1.0)))
        errors.append(f"{key} contains non-unit quaternions (max norm error {max_error:.3g})")


def validate_sonic_trajectory(
    payload: Mapping[str, Any],
    *,
    require_single_motion: bool = True,
    quaternion_atol: float = 1e-3,
) -> TrajectoryValidationReport:
    """Validate a recorder payload before it can enter dataset export.

    This gate intentionally requires the actual 64D SONIC decoder input and a
    36D G1 reference qpos. A state-only replay trajectory is therefore valid for
    its historic use but not for synthetic VLA data generation.
    """

    errors: list[str] = []
    notes: list[str] = []
    if not isinstance(payload, Mapping):
        return TrajectoryValidationReport(
            errors=("trajectory payload must be a mapping",), notes=(), frame_count=None
        )

    if payload.get("kind") != "sonic_physics_trajectory":
        errors.append("kind must be 'sonic_physics_trajectory'")
    if payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")

    raw_frame_count = payload.get("total_frames")
    frame_count: int | None = None
    if isinstance(raw_frame_count, int) and not isinstance(raw_frame_count, bool):
        frame_count = raw_frame_count
        if frame_count < 1:
            errors.append("total_frames must be positive")
    else:
        errors.append("total_frames must be an integer")

    raw_fps = payload.get("fps")
    if not isinstance(raw_fps, int | float) or isinstance(raw_fps, bool):
        errors.append("fps must be numeric")
    elif not math.isfinite(float(raw_fps)) or float(raw_fps) <= 0.0:
        errors.append("fps must be positive and finite")

    num_joints = payload.get("num_joints")
    if not isinstance(num_joints, int) or isinstance(num_joints, bool) or num_joints < 1:
        errors.append("num_joints must be a positive integer")
        joint_shape = None
    else:
        joint_shape = (num_joints,)

    dof_pos = _as_finite_frame_array(
        payload,
        "dof_pos",
        frame_count=frame_count,
        trailing_shape=joint_shape,
        errors=errors,
    )
    dof_vel = _as_finite_frame_array(
        payload,
        "dof_vel",
        frame_count=frame_count,
        trailing_shape=joint_shape,
        errors=errors,
    )
    if dof_pos is not None and dof_vel is not None and dof_pos.shape != dof_vel.shape:
        errors.append(f"dof_pos/dof_vel shape mismatch: {dof_pos.shape} vs {dof_vel.shape}")

    arrays: dict[str, np.ndarray | None] = {}
    for key, trailing_shape in _REQUIRED_FRAME_FIELDS.items():
        arrays[key] = _as_finite_frame_array(
            payload,
            key,
            frame_count=frame_count,
            trailing_shape=trailing_shape,
            errors=errors,
        )

    _check_unit_quaternions(
        arrays["root_quat_w"], key="root_quat_w", errors=errors, atol=quaternion_atol
    )
    reference_qpos = arrays["reference_g1_qpos"]
    if reference_qpos is not None and reference_qpos.ndim == 2 and reference_qpos.shape[-1] == 36:
        _check_unit_quaternions(
            reference_qpos[:, 3:7],
            key="reference_g1_qpos[:, 3:7]",
            errors=errors,
            atol=quaternion_atol,
        )

    motion_token = arrays["action_motion_token"]
    if motion_token is not None and not np.any(np.abs(motion_token) > 1e-12):
        errors.append("action_motion_token is entirely zero; decoder-input latent was not captured")

    if "applied_joint_action" in payload:
        _as_finite_frame_array(
            payload,
            "applied_joint_action",
            frame_count=frame_count,
            trailing_shape=joint_shape,
            errors=errors,
        )

    contact_force_norm = None
    if "robot_contact_force_norm_w" in payload:
        contact_force_norm = _as_finite_frame_array(
            payload,
            "robot_contact_force_norm_w",
            frame_count=frame_count,
            trailing_shape=None,
            errors=errors,
        )
        body_names = payload.get("contact_body_names")
        if not isinstance(body_names, (list, tuple)) or not body_names:
            errors.append("contact_body_names must accompany robot_contact_force_norm_w")
        elif (
            contact_force_norm is not None
            and contact_force_norm.ndim == 2
            and contact_force_norm.shape[1] != len(body_names)
        ):
            errors.append(
                "contact_body_names count does not match robot_contact_force_norm_w body axis"
            )
        for key in (
            "max_nonfoot_contact_force_n",
            "left_foot_contact_force_n",
            "right_foot_contact_force_n",
        ):
            _as_finite_frame_array(
                payload,
                key,
                frame_count=frame_count,
                trailing_shape=(),
                errors=errors,
            )
        if "robot_contact_force_w" in payload:
            _as_finite_frame_array(
                payload,
                "robot_contact_force_w",
                frame_count=frame_count,
                trailing_shape=(
                    (len(body_names), 3) if isinstance(body_names, (list, tuple)) else None
                ),
                errors=errors,
            )
        for key in (
            "left_foot_ground_contact_force_w",
            "right_foot_ground_contact_force_w",
        ):
            if key in payload:
                _as_finite_frame_array(
                    payload,
                    key,
                    frame_count=frame_count,
                    trailing_shape=(3,),
                    errors=errors,
                )

    motion_ids = None
    if "motion_id" in payload:
        motion_ids = _as_finite_frame_array(
            payload,
            "motion_id",
            frame_count=frame_count,
            trailing_shape=(),
            errors=errors,
        )
    else:
        errors.append("missing required field: motion_id")

    motion_times = None
    if "motion_time_step" in payload:
        motion_time_steps = _as_finite_frame_array(
            payload,
            "motion_time_step",
            frame_count=frame_count,
            trailing_shape=(),
            errors=errors,
        )
        if motion_time_steps is not None and not np.issubdtype(motion_time_steps.dtype, np.integer):
            errors.append("motion_time_step must use an integer dtype")
    else:
        errors.append("missing required field: motion_time_step")

    if "motion_time_s" in payload:
        motion_times = _as_finite_frame_array(
            payload,
            "motion_time_s",
            frame_count=frame_count,
            trailing_shape=(),
            errors=errors,
        )
    else:
        errors.append("missing required field: motion_time_s")

    if require_single_motion and motion_ids is not None and np.unique(motion_ids).size != 1:
        errors.append("trajectory spans multiple motion IDs and must be split into episodes")
    if motion_times is not None and motion_times.size > 1 and np.any(np.diff(motion_times) < 0):
        errors.append("motion_time_s is not monotonic and must be split at reset boundaries")

    tracking_metrics = payload.get("tracking_metrics")
    if not isinstance(tracking_metrics, Mapping) or not tracking_metrics:
        errors.append("tracking_metrics must be a non-empty mapping")
    else:
        for name, values in tracking_metrics.items():
            _as_finite_frame_array(
                tracking_metrics,
                str(name),
                frame_count=frame_count,
                trailing_shape=None,
                errors=errors,
            )

    if frame_count is not None:
        notes.append(f"validated {frame_count} physics frames")
    if motion_token is not None and motion_token.ndim == 2:
        notes.append(f"captured motion token shape {motion_token.shape}")
    if reference_qpos is not None and reference_qpos.ndim == 2:
        notes.append(f"captured G1 reference qpos shape {reference_qpos.shape}")
    if contact_force_norm is not None and contact_force_norm.ndim == 2:
        notes.append(f"captured robot contact forces for {contact_force_norm.shape[1]} bodies")

    return TrajectoryValidationReport(
        errors=tuple(errors), notes=tuple(notes), frame_count=frame_count
    )
