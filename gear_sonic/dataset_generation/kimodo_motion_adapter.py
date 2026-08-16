"""Convert Kimodo G1 MuJoCo qpos output into SONIC motion-library entries."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    BONES_CSV_JOINT_NAMES,
    MJ_TO_IL,
    convert_sequence,
)

KIMODO_G1_QPOS_DIM = 36
KIMODO_G1_ROOT_POS_SLICE = slice(0, 3)
KIMODO_G1_ROOT_QUAT_SLICE = slice(3, 7)
KIMODO_G1_DOF_SLICE = slice(7, 36)
KIMODO_G1_JOINT_NAMES = tuple(name.removesuffix("_dof") for name in BONES_CSV_JOINT_NAMES)
# Selection indices that turn an Isaac Lab-ordered 29D vector into the
# MuJoCo/Kimodo order above. The legacy converter calls the same mapping
# ``MJ_TO_IL`` because each output MuJoCo index selects one Isaac Lab index.
G1_ISAACLAB_TO_MUJOCO_DOF = tuple(int(index) for index in MJ_TO_IL)
G1_MUJOCO_TO_ISAACLAB_DOF = tuple(int(index) for index in np.argsort(G1_ISAACLAB_TO_MUJOCO_DOF))
G1_ISAACLAB_JOINT_NAMES = tuple(KIMODO_G1_JOINT_NAMES[index] for index in G1_MUJOCO_TO_ISAACLAB_DOF)


class KimodoQposError(ValueError):
    """Raised when Kimodo qpos data violates the G1 interchange contract."""


def _as_qpos_array(qpos: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(qpos, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != KIMODO_G1_QPOS_DIM:
        raise KimodoQposError(
            f"Kimodo G1 qpos must have shape [T, {KIMODO_G1_QPOS_DIM}], got {array.shape}"
        )
    return array


def validate_kimodo_qpos(
    qpos: np.ndarray | Sequence[Sequence[float]],
    *,
    min_frames: int = 2,
    quaternion_norm_tolerance: float = 5e-3,
) -> np.ndarray:
    """Validate and return a float64 view of Kimodo G1 qpos data."""
    array = _as_qpos_array(qpos)
    if array.shape[0] < min_frames:
        raise KimodoQposError(
            f"Kimodo G1 motion must have at least {min_frames} frames, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise KimodoQposError("Kimodo G1 qpos contains NaN or infinity")

    quaternions = array[:, KIMODO_G1_ROOT_QUAT_SLICE]
    norms = np.linalg.norm(quaternions, axis=1)
    if np.any(norms < 1e-8):
        raise KimodoQposError("Kimodo G1 qpos contains a zero root quaternion")
    max_error = float(np.max(np.abs(norms - 1.0)))
    if max_error > quaternion_norm_tolerance:
        raise KimodoQposError(
            "Kimodo G1 root quaternion is not unit length: "
            f"maximum norm error {max_error:.6g} exceeds {quaternion_norm_tolerance:.6g}"
        )
    return array


def load_kimodo_qpos_csv(path: str | Path) -> np.ndarray:
    """Load the headerless 36-column CSV emitted by Kimodo's G1 exporter."""
    try:
        qpos = np.loadtxt(path, delimiter=",", dtype=np.float64)
    except ValueError as exc:
        raise KimodoQposError(f"failed to parse Kimodo qpos CSV {path}: {exc}") from exc
    return validate_kimodo_qpos(qpos)


def _normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise KimodoQposError("cannot normalize a zero root quaternion")
    return quaternions / norms


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for broadcast-compatible WXYZ quaternions."""
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def transform_qpos_to_scene(
    qpos: np.ndarray | Sequence[Sequence[float]],
    *,
    scene_start_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    scene_yaw: float = 0.0,
    canonicalize_horizontal_origin: bool = True,
) -> np.ndarray:
    """Place a Z-up, +X-forward Kimodo qpos clip in a scene-local frame.

    The source pelvis height is preserved. ``scene_start_xyz[2]`` is the support-surface
    height to add, not the desired pelvis height.
    """
    array = validate_kimodo_qpos(qpos).copy()
    start = np.asarray(scene_start_xyz, dtype=np.float64)
    if start.shape != (3,) or not np.all(np.isfinite(start)):
        raise KimodoQposError("scene_start_xyz must contain three finite values")
    if not math.isfinite(float(scene_yaw)):
        raise KimodoQposError("scene_yaw must be finite")

    positions = array[:, KIMODO_G1_ROOT_POS_SLICE]
    if canonicalize_horizontal_origin:
        positions[:, :2] -= positions[0, :2].copy()
    cosine, sine = math.cos(scene_yaw), math.sin(scene_yaw)
    yaw_rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    array[:, KIMODO_G1_ROOT_POS_SLICE] = positions @ yaw_rotation.T + start

    yaw_quaternion = np.array(
        [math.cos(scene_yaw / 2.0), 0.0, 0.0, math.sin(scene_yaw / 2.0)],
        dtype=np.float64,
    )
    quaternions = _normalize_quaternions_wxyz(array[:, KIMODO_G1_ROOT_QUAT_SLICE])
    transformed_quaternions = _quat_multiply_wxyz(yaw_quaternion, quaternions)
    array[:, KIMODO_G1_ROOT_QUAT_SLICE] = _normalize_quaternions_wxyz(transformed_quaternions)
    return array


def qpos_to_sonic_motion_entry(
    qpos: np.ndarray | Sequence[Sequence[float]],
    *,
    source_fps: float = 30.0,
    scene_start_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    scene_yaw: float = 0.0,
    canonicalize_horizontal_origin: bool = True,
) -> dict:
    """Convert Kimodo G1 qpos to the motion-library entry consumed by SONIC.

    The reused legacy motion-library converter adds an all-zero ``smpl_joints`` field for
    compatibility with that internal loader. It is not valid SMPL supervision and must not be
    copied into the synthetic G1 LeRobot export.
    """
    if not math.isfinite(float(source_fps)) or source_fps <= 0:
        raise KimodoQposError("source_fps must be a positive finite number")
    if not float(source_fps).is_integer():
        raise KimodoQposError(
            "the current SONIC motion-lib entry stores integer fps; "
            f"got non-integral source_fps={source_fps}"
        )

    transformed = transform_qpos_to_scene(
        qpos,
        scene_start_xyz=scene_start_xyz,
        scene_yaw=scene_yaw,
        canonicalize_horizontal_origin=canonicalize_horizontal_origin,
    )
    frame_count = transformed.shape[0]
    body_pos_w = np.zeros((frame_count, 1, 3), dtype=np.float32)
    body_pos_w[:, 0, :] = transformed[:, KIMODO_G1_ROOT_POS_SLICE]
    body_quat_w = np.zeros((frame_count, 1, 4), dtype=np.float32)
    body_quat_w[:, 0, :] = transformed[:, KIMODO_G1_ROOT_QUAT_SLICE]

    sequence = {
        "joint_pos": transformed[:, KIMODO_G1_DOF_SLICE].astype(np.float32),
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "joint_order": "mj",
    }
    return convert_sequence(sequence, fps=int(source_fps))


def save_sonic_motion_file(
    path: str | Path,
    *,
    motion_key: str,
    motion_entry: dict,
    compress: int = 3,
) -> None:
    """Write one SONIC motion-library mapping using the repository's joblib format."""
    if not isinstance(motion_key, str) or not motion_key.strip():
        raise ValueError("motion_key must be a non-empty string")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({motion_key: motion_entry}, output, compress=compress)
