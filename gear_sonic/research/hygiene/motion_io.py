"""Reader/writer/validator for SONIC motion-library clips, used by the hygiene screen.

WHY this module exists
----------------------
The dynamic-feasibility screen and the repair operator both need to (a) read a SONIC
``.pkl`` clip without dragging in Isaac Lab or the training stack, (b) put it on the
*same* timeline the policy actually sees, and (c) write it back without silently
changing dtype or breaking the redundancy the format carries.  Doing that inline in
each caller is how format drift starts, so it lives here once.

Format (verified on the 4950-clip Bones-SEED bank)
--------------------------------------------------
One joblib ``.pkl`` per clip.  Top level is a dict with exactly one key -- the motion
key, which equals the file stem.  Its value is a dict with:

===================  ==============  ==================================================
field                shape / dtype   meaning
===================  ==============  ==================================================
root_trans_offset    (T, 3) f32      pelvis position in world frame [m]
pose_aa              (T, 30, 3) f32  axis-angle; row 0 = root orientation,
                                     row i+1 = ``DOF_AXIS[i] * dof[:, i]``
dof                  (T, 29) f32     joint angles, MuJoCo (MJCF actuator) order [rad]
root_rot             (T, 4) f32      root quaternion, XYZW (scipy convention)
smpl_joints          (T, 24, 3) f32  SMPL joint positions in world frame [m]
fps                  int             30 on the shipped bank
===================  ==============  ==================================================

``pose_aa`` and ``dof`` are redundant by construction; :func:`validate_motion` checks
that redundancy to ``1e-6`` because a repair or resample that updates one and forgets
the other produces a clip that trains differently than it screens.

Resampling
----------
SONIC loads at 30 Hz and resamples to ``target_fps: 50`` *at load time*, so any
feasibility statement about "what the policy tracks" has to be made on the 50 Hz
timeline.  :func:`resample_to` reproduces the pinned rule rather than inventing one --
see :func:`resample_to` for the source references and the one documented numerical
deviation (float32 timeline arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import DOF_AXIS

MOTION_KEYS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints", "fps")

NUM_DOF = 29
NUM_POSE_ROWS = 30  # root orientation + 29 single-axis joints
NUM_SMPL_JOINTS = 24

SONIC_TARGET_FPS = 50

#: ``pose_aa[:, i + 1] == DOF_AXIS[i] * dof[:, i]`` must hold to this absolute tolerance.
POSE_DOF_REDUNDANCY_TOL = 1e-6
#: ``root_rot`` must be a unit quaternion to this absolute tolerance.
QUATERNION_NORM_TOL = 1e-3
#: ``root_rot`` must agree with ``pose_aa[:, 0]`` (up to global sign) to this tolerance.
ROOT_ROT_CONSISTENCY_TOL = 1e-4

_ARRAY_FIELDS: dict[str, tuple[int, ...]] = {
    "root_trans_offset": (3,),
    "pose_aa": (NUM_POSE_ROWS, 3),
    "dof": (NUM_DOF,),
    "root_rot": (4,),
    "smpl_joints": (NUM_SMPL_JOINTS, 3),
}


@dataclass(frozen=True)
class Motion:
    """One SONIC motion-library clip, held in memory in its on-disk layout."""

    key: str
    root_trans_offset: np.ndarray
    pose_aa: np.ndarray
    dof: np.ndarray
    root_rot: np.ndarray
    smpl_joints: np.ndarray
    fps: int

    @property
    def num_frames(self) -> int:
        """Number of frames ``T`` (taken from ``root_trans_offset``)."""
        return int(self.root_trans_offset.shape[0])

    @property
    def duration_s(self) -> float:
        """Clip duration on its own timeline, ``T / fps`` seconds."""
        return float(self.num_frames) / float(self.fps)

    def replace(self, **kw: Any) -> "Motion":
        """Return a copy with the given fields replaced (dataclass semantics)."""
        return dataclass_replace(self, **kw)

    def to_payload(self) -> dict[str, dict[str, Any]]:
        """Return the exact nested dict that :mod:`joblib` should dump for this clip."""
        return {
            self.key: {
                "root_trans_offset": self.root_trans_offset,
                "pose_aa": self.pose_aa,
                "dof": self.dof,
                "root_rot": self.root_rot,
                "smpl_joints": self.smpl_joints,
                "fps": int(self.fps),
            }
        }


def load_motion(path: str | Path) -> Motion:
    """Load one clip.  Raises ``ValueError`` if the container is not the SONIC layout.

    The motion key is taken from the payload (not from the filename) so that a clip
    whose key and stem disagree is still readable; :func:`validate_motion` is where
    structural opinions live, not here.
    """
    path = Path(path)
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a dict at the top level, got {type(payload).__name__}")
    if len(payload) != 1:
        raise ValueError(f"{path}: expected exactly one motion key, got {sorted(payload)!r}")
    key = next(iter(payload))
    body = payload[key]
    if not isinstance(body, dict):
        raise ValueError(f"{path}: motion {key!r} is not a dict")
    missing = [name for name in MOTION_KEYS if name not in body]
    if missing:
        raise ValueError(f"{path}: motion {key!r} is missing fields {missing}")
    return Motion(
        key=str(key),
        root_trans_offset=np.asarray(body["root_trans_offset"]),
        pose_aa=np.asarray(body["pose_aa"]),
        dof=np.asarray(body["dof"]),
        root_rot=np.asarray(body["root_rot"]),
        smpl_joints=np.asarray(body["smpl_joints"]),
        fps=int(body["fps"]),
    )


def save_motion(motion: Motion, path: str | Path) -> None:
    """Write ``motion`` to ``path`` atomically, preserving the on-disk float32 dtypes.

    The dump goes to a temporary file in the destination directory and is then
    ``os.replace``-d into position, so a crashed or killed writer never leaves a
    half-written clip that a later resumable run would happily skip.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = motion.to_payload()
    body = payload[motion.key]
    for name in _ARRAY_FIELDS:
        body[name] = np.ascontiguousarray(body[name], dtype=np.float32)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def validate_motion(motion: Motion) -> list[str]:
    """Return a list of human-readable problems; empty means the clip is clean.

    Checks, in order: key/fps sanity, per-field shape, float32 dtype, finiteness,
    ``root_rot`` unit norm, the ``pose_aa``/``dof`` redundancy, and agreement between
    ``root_rot`` and ``pose_aa[:, 0]``.  Later checks are skipped when an earlier one
    already made them meaningless (e.g. a wrong shape suppresses the redundancy test)
    so the caller gets the root cause instead of a cascade.
    """
    problems: list[str] = []
    if not isinstance(motion.key, str) or not motion.key:
        problems.append("key must be a non-empty string")
    if not isinstance(motion.fps, (int, np.integer)) or isinstance(motion.fps, bool):
        problems.append(f"fps must be an int, got {type(motion.fps).__name__}")
    elif int(motion.fps) <= 0:
        problems.append(f"fps must be positive, got {motion.fps}")

    num_frames = None
    shapes_ok = True
    for name, trailing in _ARRAY_FIELDS.items():
        array = getattr(motion, name)
        if not isinstance(array, np.ndarray):
            problems.append(f"{name} must be a numpy array, got {type(array).__name__}")
            shapes_ok = False
            continue
        if array.ndim != 1 + len(trailing) or array.shape[1:] != trailing:
            problems.append(f"{name} must have shape (T, {', '.join(map(str, trailing))}), got {array.shape}")
            shapes_ok = False
            continue
        if num_frames is None:
            num_frames = int(array.shape[0])
        elif int(array.shape[0]) != num_frames:
            problems.append(f"{name} has {array.shape[0]} frames, expected {num_frames}")
            shapes_ok = False
        if array.dtype != np.float32:
            problems.append(f"{name} must be float32, got {array.dtype}")
        if array.size and not np.isfinite(array).all():
            problems.append(f"{name} contains non-finite values")
            shapes_ok = False
    if num_frames is not None and num_frames < 2:
        problems.append(f"clip must have at least 2 frames, got {num_frames}")
    if not shapes_ok:
        return problems

    norms = np.linalg.norm(motion.root_rot.astype(np.float64), axis=-1)
    worst_norm = float(np.max(np.abs(norms - 1.0))) if norms.size else 0.0
    if worst_norm > QUATERNION_NORM_TOL:
        problems.append(f"root_rot is not unit norm (max |‖q‖ - 1| = {worst_norm:.3e})")

    expected_pose = DOF_AXIS.astype(np.float64)[None, :, :] * motion.dof.astype(np.float64)[:, :, None]
    redundancy_error = float(np.max(np.abs(motion.pose_aa[:, 1:, :].astype(np.float64) - expected_pose)))
    if redundancy_error > POSE_DOF_REDUNDANCY_TOL:
        frame = int(np.argmax(np.abs(motion.pose_aa[:, 1:, :].astype(np.float64) - expected_pose).max(axis=(1, 2))))
        problems.append(
            "pose_aa[:, 1:] != DOF_AXIS * dof "
            f"(max abs error {redundancy_error:.3e} > {POSE_DOF_REDUNDANCY_TOL:.0e}, worst at frame {frame})"
        )

    root_quat_xyzw = axis_angle_to_quaternion_xyzw(motion.pose_aa[:, 0, :].astype(np.float64))
    sign = np.sign(np.sum(root_quat_xyzw * motion.root_rot.astype(np.float64), axis=-1))
    sign[sign == 0.0] = 1.0
    root_error = float(np.max(np.abs(root_quat_xyzw * sign[:, None] - motion.root_rot.astype(np.float64))))
    if root_error > ROOT_ROT_CONSISTENCY_TOL:
        problems.append(
            f"root_rot disagrees with pose_aa[:, 0] (max abs error {root_error:.3e} "
            f"> {ROOT_ROT_CONSISTENCY_TOL:.0e})"
        )
    return problems


def motion_sha256(motion: Motion) -> str:
    """Content hash of a clip: stable across re-dumps, unlike the compressed ``.pkl``.

    joblib compression is not byte-reproducible, so hashing the container file does not
    identify the *content* that was screened.  This hashes the key, fps and the raw
    little-endian float32 bytes of every array instead, in the fixed ``MOTION_KEYS``
    order, which is what a reviewer actually wants pinned in a screen record.
    """
    digest = hashlib.sha256()
    digest.update(motion.key.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(str(int(motion.fps)).encode("ascii"))
    for name in _ARRAY_FIELDS:
        array = np.ascontiguousarray(getattr(motion, name), dtype="<f4")
        digest.update(b"\x00")
        digest.update(name.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def resample_to(motion: Motion, target_fps: int = SONIC_TARGET_FPS) -> Motion:
    """Put a clip on SONIC's load-time timeline (default 50 Hz).

    Reproduces the pinned rule rather than inventing one.  Sources:

    * ``gear_sonic/utils/motion_lib/torch_humanoid_batch.py``
      (``Humanoid_Batch.interploate_pose`` / ``_compute_frame_blend`` / ``fk_batch``)
      -- the code that actually runs at load time, and
    * ``gear_sonic/research/lace/reference_feasibility_manifest.py``
      (``_runtime_reference``) -- LACE's frozen restatement of the same rule.

    The rule, verbatim:

    1. ``duration = (T_src - 1) / fps_src``
    2. ``times = arange(0, duration, 1 / target_fps)`` in **float32**, endpoint
       **exclusive** -- so the last source frame is generally *not* sampled.
    3. ``phase = times / duration``; ``coordinates = phase * (T_src - 1)``;
       ``index_0 = floor(coordinates)``; ``index_1 = min(index_0 + 1, T_src - 1)``;
       ``blend = coordinates - index_0``.
    4. ``pose_aa`` is converted to quaternions and **slerped**; ``root_trans_offset`` is
       **lerped**.  ``dof`` is then recovered as ``sum(pose_aa[:, 1:], axis=-1)``, which
       is exact because every G1 joint axis is a signed unit basis vector.

    Two documented deviations from the torch original, both deliberate:

    * The timeline is built with ``numpy.arange(..., dtype=float32)`` instead of
      ``torch.arange(..., dtype=torch.float32)``.  The **frame count is identical** for
      every ``T_src`` in ``[2, 4000)`` at 30 -> 50 Hz (verified), but individual sample
      times differ by up to ~3e-8 s because the two libraries accumulate float32
      differently.  That propagates to ``blend`` at the ~1e-6 level, i.e. below the
      float32 resolution of the angles themselves.
    * The slerp is evaluated in float64 and rounded once at the end -- see :func:`_slerp`.
      Measured against the torch implementation on real bank clips the two agree to
      <= 1.1e-3 rad (0.06 deg) per joint, the difference being float32 cancellation in
      SONIC's ``sqrt(1 - cos^2)``.
    * ``smpl_joints`` is lerped.  SONIC does not resample it at load (it never reads it),
      so there is no upstream rule to copy; lerp keeps the field on the same timeline as
      everything else instead of leaving a silently stale array behind.

    ``fps == target_fps`` is a no-op that returns the same object.
    """
    target_fps = int(target_fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    source_fps = int(motion.fps)
    if source_fps == target_fps:
        return motion
    source_frames = motion.num_frames
    if source_frames < 2:
        raise ValueError(f"{motion.key}: cannot resample a clip with {source_frames} frame(s)")

    duration = (source_frames - 1) * 1.0 / source_fps
    times = np.arange(0.0, duration, 1.0 / target_fps, dtype=np.float32)
    expected = int(math.ceil(duration / (1.0 / target_fps)))
    if times.shape[0] != expected:
        raise ValueError(
            f"{motion.key}: float32 timeline has {times.shape[0]} samples, expected {expected}"
        )
    if times.shape[0] < 2:
        raise ValueError(
            f"{motion.key}: resampling {source_frames} frames from {source_fps} to {target_fps} Hz "
            f"yields {times.shape[0]} frame(s)"
        )

    phase = times / np.float32(duration)
    coordinates = phase * np.float32(source_frames - 1)
    index_0 = np.floor(coordinates).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, source_frames - 1)
    blend = (coordinates - index_0).astype(np.float32)

    pose_quat = axis_angle_to_quaternion_wxyz(motion.pose_aa.astype(np.float32))
    resampled_quat = _slerp(pose_quat[index_0], pose_quat[index_1], blend[:, None, None])
    pose_aa = quaternion_wxyz_to_axis_angle(resampled_quat).astype(np.float32)
    dof = pose_aa[:, 1:, :].sum(axis=-1).astype(np.float32)
    root_rot = resampled_quat[:, 0][:, [1, 2, 3, 0]].astype(np.float32)
    root_trans = _lerp(
        motion.root_trans_offset.astype(np.float32)[index_0],
        motion.root_trans_offset.astype(np.float32)[index_1],
        blend[:, None],
    ).astype(np.float32)
    smpl_joints = _lerp(
        motion.smpl_joints.astype(np.float32)[index_0],
        motion.smpl_joints.astype(np.float32)[index_1],
        blend[:, None, None],
    ).astype(np.float32)
    return motion.replace(
        root_trans_offset=root_trans,
        pose_aa=pose_aa,
        dof=dof,
        root_rot=root_rot,
        smpl_joints=smpl_joints,
        fps=target_fps,
    )


def axis_angle_to_quaternion_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle -> WXYZ quaternion, matching ``gear_sonic.isaac_utils.rotations``."""
    axis_angle = np.asarray(axis_angle)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)
    half = angles * 0.5
    small = np.abs(angles) < 1e-6
    scale = np.where(small, 0.5 - (angles * angles) / 48.0, np.sin(half) / np.where(small, 1.0, angles))
    return np.concatenate([np.cos(half), axis_angle * scale], axis=-1).astype(axis_angle.dtype)


def axis_angle_to_quaternion_xyzw(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle -> XYZW quaternion (the ``root_rot`` convention on disk)."""
    quat = axis_angle_to_quaternion_wxyz(axis_angle)
    return quat[..., [1, 2, 3, 0]]


def quaternion_wxyz_to_axis_angle(quaternion: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """WXYZ quaternion -> axis-angle, matching ``quaternion_to_angle_axis`` (Ceres rule)."""
    quaternion = np.asarray(quaternion)
    cos_theta = quaternion[..., 0]
    vec = quaternion[..., 1:]
    sin_squared = np.sum(vec * vec, axis=-1)
    sin_theta = np.sqrt(np.maximum(sin_squared, eps))
    two_theta = 2.0 * np.where(
        cos_theta < 0.0,
        np.arctan2(-sin_theta, -cos_theta),
        np.arctan2(sin_theta, cos_theta),
    )
    k = np.where(sin_squared > 0.0, two_theta / np.maximum(sin_theta, eps), 2.0)
    return (vec * k[..., None]).astype(quaternion.dtype)


def _lerp(a: np.ndarray, b: np.ndarray, blend: np.ndarray) -> np.ndarray:
    return a * (1.0 - blend) + b * blend


def _slerp(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Restatement of ``gear_sonic.isaac_utils.rotations.slerp`` (WXYZ), evaluated in float64.

    Same formula, same short-circuit thresholds (``|sin| < 0.001`` -> quaternion lerp,
    ``|cos| >= 1`` -> ``q0``).  The one deliberate change is the working precision:
    SONIC evaluates ``sqrt(1 - cos^2)`` in float32, and for the near-parallel quaternions
    of adjacent 30 Hz frames (``cos`` within ~1e-5 of 1) that subtraction cancels to a few
    significant bits.  Evaluating in float64 and rounding once at the end keeps the two
    within ~1e-3 rad (0.06 deg) of each other on real clips, with ours the more accurate;
    that is far below the angular resolution anything downstream of here resolves.
    """
    dtype = q0.dtype
    q0 = q0.astype(np.float64)
    q1 = q1.astype(np.float64)
    blend = blend.astype(np.float64)
    cos_half_theta = np.sum(q0 * q1, axis=-1)
    q1 = np.where((cos_half_theta < 0.0)[..., None], -q1, q1)
    cos_half_theta = np.clip(np.abs(cos_half_theta)[..., None], -1.0, 1.0)
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(np.maximum(1.0 - cos_half_theta * cos_half_theta, 0.0))
    safe_sin = np.where(np.abs(sin_half_theta) < 1e-12, 1.0, sin_half_theta)
    ratio_a = np.sin((1.0 - blend) * half_theta) / safe_sin
    ratio_b = np.sin(blend * half_theta) / safe_sin
    result = ratio_a * q0 + ratio_b * q1
    result = np.where(np.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, result)
    result = np.where(cos_half_theta >= 1.0, q0, result)
    return result.astype(dtype)
