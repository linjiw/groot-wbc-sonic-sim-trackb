"""Read, validate, resample, and re-write SONIC G1 motion-library clips.

WHY this module exists: the hygiene package has to touch the *same bytes* SONIC
trains on.  Every other tool in the package (the dynamic screen, the repair
operator, the sampler diagnostics) needs one loader that (a) knows the on-disk
contract exactly, (b) never silently changes dtype or the redundant
``pose_aa``/``dof`` encoding, and (c) reproduces SONIC's load-time 30 -> 50 Hz
resample rather than inventing a resample of its own.  A screen run on a
timeline that differs from the training timeline measures the wrong motion.

On-disk contract (verified against the 4950-clip Bones-SEED bank):
    one joblib ``.pkl`` per clip, top-level dict with exactly one key equal to
    the motion key (and to the file stem).  The value is a dict with

        root_trans_offset : (T, 3)     float32   pelvis position, world frame
        pose_aa           : (T, 30, 3) float32   axis-angle; row 0 is the root
                                                 orientation, rows 1..29 are
                                                 ``DOF_AXIS[i] * dof[:, i]``
        dof               : (T, 29)    float32   MuJoCo/MJCF actuator order
        root_rot          : (T, 4)     float32   quaternion, XYZW (scipy)
        smpl_joints       : (T, 24, 3) float32
        fps               : int                  30 for the release bank

``pose_aa`` and ``dof`` are redundant by construction; :func:`validate_motion`
checks that redundancy because a repair operator that edits one and forgets the
other produces a clip that behaves differently in MuJoCo (which reads ``dof``)
than in SONIC's ``Humanoid_Batch`` FK (which reads ``pose_aa``).

Resampling: SONIC's motion library resamples every clip to ``target_fps: 50`` at
load time.  The pinned rule lives in
``gear_sonic/research/lace/reference_feasibility_manifest.py::_runtime_reference``
and is reproduced here verbatim (float32 ``torch.arange`` with an *exclusive*
endpoint, quaternion ``slerp`` for rotations, ``lerp`` for translation, joint
angles recovered by summing the interpolated rotation vectors).  We call the
same ``gear_sonic.isaac_utils.rotations`` helpers LACE calls, so the two agree
by construction instead of by coincidence.  Torch is imported lazily so that
merely importing this module stays cheap for multiprocessing workers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace as _dataclass_replace
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    DOF_AXIS,
    NUM_BODIES,
    NUM_DOF,
)

__all__ = [
    "DOF_AXIS",
    "MOTION_KEYS",
    "Motion",
    "load_motion",
    "motion_sha256",
    "resample_to",
    "save_motion",
    "validate_motion",
]

MOTION_KEYS = ("root_trans_offset", "pose_aa", "dof", "root_rot", "smpl_joints", "fps")

#: SONIC's configured training rate (``target_fps`` in the universal-token configs).
DEFAULT_TARGET_FPS = 50

#: Tolerance for the ``pose_aa[:, 1:] == DOF_AXIS * dof`` identity.  The bank is
#: written in float32 from exactly this product, so the observed error is 0.0;
#: 1e-6 leaves room for a repair operator that rebuilds one side in float64.
POSE_AA_DOF_ATOL = 1e-6

#: Tolerance for ``|root_rot| == 1``.  float32 quaternions on the release bank
#: land within ~6e-8 of unit norm.
QUAT_NORM_ATOL = 1e-5

_EXPECTED_SHAPES = {
    "root_trans_offset": (3,),
    "pose_aa": (NUM_BODIES, 3),
    "dof": (NUM_DOF,),
    "root_rot": (4,),
    "smpl_joints": (24, 3),
}

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Motion:
    """One SONIC motion-library clip, exactly as stored on disk.

    The arrays are held as given (float32 on the release bank); nothing in this
    class upcasts, renormalizes, or reorders.  Use :meth:`replace` to derive an
    edited clip and :meth:`to_payload` to get a dict ready for ``joblib.dump``.
    """

    key: str
    root_trans_offset: FloatArray
    pose_aa: FloatArray
    dof: FloatArray
    root_rot: FloatArray
    smpl_joints: FloatArray
    fps: int

    @property
    def num_frames(self) -> int:
        """Frame count taken from ``dof`` (all fields share this axis)."""
        return int(self.dof.shape[0])

    @property
    def duration_s(self) -> float:
        """Clip duration in seconds at the clip's own frame rate."""
        return float(self.num_frames) / float(self.fps)

    def replace(self, **kw: Any) -> "Motion":
        """Return a copy with the named fields replaced."""
        return _dataclass_replace(self, **kw)

    def to_payload(self) -> dict[str, dict[str, Any]]:
        """Return ``{key: {field: array, ..., "fps": int}}`` ready for ``joblib.dump``."""
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
    """Load one clip ``.pkl``.

    Raises ``ValueError`` when the file is not a single-key motion-library dict
    or is missing any field of :data:`MOTION_KEYS`.  Structural problems raise;
    *numeric* problems are reported by :func:`validate_motion` instead, so that a
    caller can screen a suspect clip rather than being unable to open it.
    """
    path = Path(path)
    payload = joblib.load(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a dict at top level, got {type(payload).__name__}")
    if len(payload) != 1:
        raise ValueError(f"{path}: expected exactly one motion key, found {len(payload)}")
    key = next(iter(payload))
    record = payload[key]
    if not isinstance(record, dict):
        raise ValueError(f"{path}: motion {key!r} is not a dict")
    missing = [name for name in MOTION_KEYS if name not in record]
    if missing:
        raise ValueError(f"{path}: motion {key!r} is missing fields {missing}")
    return Motion(
        key=str(key),
        root_trans_offset=np.asarray(record["root_trans_offset"]),
        pose_aa=np.asarray(record["pose_aa"]),
        dof=np.asarray(record["dof"]),
        root_rot=np.asarray(record["root_rot"]),
        smpl_joints=np.asarray(record["smpl_joints"]),
        fps=int(record["fps"]),
    )


def save_motion(motion: Motion, path: str | Path) -> None:
    """Write ``motion`` to ``path`` atomically, preserving dtypes.

    The temporary file is created in the destination directory so that
    ``os.replace`` is a same-filesystem rename: a reader either sees the old file
    or the complete new one, never a truncated pickle.  A repair sweep over 4950
    clips that is interrupted must not leave half-written motions behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    os.close(handle)
    tmp_path = Path(tmp_name)
    try:
        joblib.dump(motion.to_payload(), tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def validate_motion(motion: Motion) -> list[str]:
    """Return a list of human-readable problems; empty when the clip is clean.

    Checks, in order: field shapes and the shared frame axis, dtype, finiteness,
    ``fps``, quaternion unit norm, agreement between ``root_rot`` and
    ``pose_aa[:, 0]``, and the ``pose_aa[:, 1:] == DOF_AXIS * dof`` redundancy.
    """
    problems: list[str] = []

    if not isinstance(motion.key, str) or not motion.key:
        problems.append("key is empty or not a string")
    if not isinstance(motion.fps, (int, np.integer)) or int(motion.fps) <= 0:
        problems.append(f"fps must be a positive integer, got {motion.fps!r}")

    arrays = {
        "root_trans_offset": motion.root_trans_offset,
        "pose_aa": motion.pose_aa,
        "dof": motion.dof,
        "root_rot": motion.root_rot,
        "smpl_joints": motion.smpl_joints,
    }
    frame_counts: dict[str, int] = {}
    for name, array in arrays.items():
        if not isinstance(array, np.ndarray):
            problems.append(f"{name} is not a numpy array (got {type(array).__name__})")
            continue
        expected = _EXPECTED_SHAPES[name]
        if array.ndim != len(expected) + 1 or tuple(array.shape[1:]) != expected:
            problems.append(f"{name} has shape {array.shape}, expected (T, {', '.join(map(str, expected))})")
            continue
        frame_counts[name] = int(array.shape[0])
        if array.dtype != np.float32:
            problems.append(f"{name} has dtype {array.dtype}, expected float32")
        if not np.isfinite(array).all():
            problems.append(f"{name} contains {int((~np.isfinite(array)).sum())} non-finite values")

    if len(set(frame_counts.values())) > 1:
        problems.append(f"frame axes disagree: {frame_counts}")
    if frame_counts and min(frame_counts.values()) < 2:
        problems.append(f"clip has fewer than 2 frames: {frame_counts}")

    if problems:
        # Shape/dtype damage makes the numeric checks below meaningless.
        return problems

    norms = np.linalg.norm(motion.root_rot.astype(np.float64), axis=-1)
    worst_norm = float(np.max(np.abs(norms - 1.0)))
    if worst_norm > QUAT_NORM_ATOL:
        problems.append(
            f"root_rot is not unit norm: max |‖q‖-1| = {worst_norm:.3e} > {QUAT_NORM_ATOL:.1e}"
        )

    expected_pose = DOF_AXIS.astype(np.float64)[None, :, :] * motion.dof.astype(np.float64)[:, :, None]
    residual = np.abs(motion.pose_aa.astype(np.float64)[:, 1:, :] - expected_pose)
    worst = float(residual.max())
    if worst > POSE_AA_DOF_ATOL:
        frame, joint, axis = np.unravel_index(int(np.argmax(residual)), residual.shape)
        problems.append(
            "pose_aa/dof redundancy broken: max |pose_aa[:, 1:] - DOF_AXIS * dof| = "
            f"{worst:.3e} > {POSE_AA_DOF_ATOL:.1e} at frame {frame}, joint {joint}, axis {axis} "
            f"({int((residual > POSE_AA_DOF_ATOL).any(axis=(1, 2)).sum())} of {residual.shape[0]} frames affected)"
        )

    root_quat_xyzw = _axis_angle_to_quat_xyzw(motion.pose_aa.astype(np.float64)[:, 0, :])
    # A quaternion and its negation are the same rotation; compare on |dot|.
    alignment = np.abs(np.sum(root_quat_xyzw * motion.root_rot.astype(np.float64), axis=-1))
    worst_alignment = float(np.min(alignment))
    if worst_alignment < 1.0 - 1e-4:
        problems.append(
            "root_rot disagrees with pose_aa[:, 0]: min |dot| = "
            f"{worst_alignment:.6f} (frame {int(np.argmin(alignment))})"
        )

    return problems


def motion_sha256(motion: Motion) -> str:
    """Content digest of a clip: stable across files, sensitive to any edit.

    We hash the arrays rather than the ``.pkl`` because joblib pickles are not
    byte-reproducible, and because a repaired clip held in memory must be
    identifiable before it is ever written.
    """
    digest = hashlib.sha256()
    digest.update(motion.key.encode("utf-8"))
    digest.update(b"|")
    for name in MOTION_KEYS:
        if name == "fps":
            digest.update(str(int(motion.fps)).encode("ascii"))
            continue
        array = np.ascontiguousarray(getattr(motion, name))
        digest.update(f"{name}:{array.dtype.str}:{array.shape}|".encode("utf-8"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def resample_to(motion: Motion, target_fps: int = DEFAULT_TARGET_FPS) -> Motion:
    """Resample a clip the way SONIC's motion library does at load time.

    The rule, pinned by ``lace/reference_feasibility_manifest.py`` and originally
    implemented in ``gear_sonic/utils/motion_lib/torch_humanoid_batch.py``:

        duration = (T_src - 1) / source_fps
        times    = torch.arange(0, duration, 1 / target_fps, dtype=float32)   # exclusive end
        phase    = times / duration
        coords   = phase * (T_src - 1)
        i0, i1   = floor(coords), min(i0 + 1, T_src - 1);  blend = coords - i0
        pose     = slerp(quat(pose_aa)[i0], quat(pose_aa)[i1], blend)
        trans    = lerp(root_trans_offset[i0], root_trans_offset[i1], blend)
        dof      = sum(axis_angle(pose)[:, 1:], axis=-1)
        root_rot = quat_xyzw(pose[:, 0])

    When ``source_fps == target_fps`` SONIC skips interpolation entirely and the
    clip is returned unchanged.

    ``smpl_joints`` is *not* part of SONIC's runtime interpolation (the motion
    library never reads it).  We lerp it on the same index/blend so the returned
    clip stays internally consistent; treat those values as a convenience, not as
    a claim about SONIC's runtime.
    """
    if int(target_fps) <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    source_frames = motion.num_frames
    if int(motion.fps) == int(target_fps):
        return motion
    if source_frames < 2:
        raise ValueError(f"{motion.key}: cannot resample a {source_frames}-frame clip")

    import torch  # noqa: PLC0415 - lazy: keeps module import cheap for worker processes

    from gear_sonic.isaac_utils.rotations import (  # noqa: PLC0415
        axis_angle_to_quaternion,
        matrix_to_quaternion,
        quaternion_to_matrix,
        slerp,
    )
    from gear_sonic.trl.utils.torch_transform import quaternion_to_angle_axis  # noqa: PLC0415

    pose = torch.from_numpy(np.ascontiguousarray(motion.pose_aa, dtype=np.float32))
    translation = torch.from_numpy(np.ascontiguousarray(motion.root_trans_offset, dtype=np.float32))
    with torch.no_grad():
        pose_quaternion = axis_angle_to_quaternion(pose)
        duration = (source_frames - 1) * 1.0 / float(motion.fps)
        times = torch.arange(0, duration, 1.0 / target_fps, dtype=torch.float32, device=torch.device("cpu"))
        phase = times / duration
        coordinates = phase * (source_frames - 1)
        index_0 = torch.floor(coordinates).to(dtype=torch.long)
        index_1 = torch.minimum(index_0 + 1, torch.tensor(source_frames - 1, dtype=torch.long))
        blend = coordinates - index_0
        out_quaternion = slerp(pose_quaternion[index_0], pose_quaternion[index_1], blend[:, None, None])
        out_translation = translation[index_0] * (1.0 - blend[:, None]) + translation[index_1] * blend[:, None]
        out_pose_aa = quaternion_to_angle_axis(out_quaternion)
        out_dof = out_pose_aa[:, 1:, :].sum(dim=-1)
        out_root_rot_wxyz = matrix_to_quaternion(quaternion_to_matrix(out_quaternion)[:, 0])
        out_root_rot = out_root_rot_wxyz[:, [1, 2, 3, 0]]

    index_0_np = index_0.numpy()
    index_1_np = index_1.numpy()
    blend_np = blend.numpy()[:, None, None]
    smpl = motion.smpl_joints.astype(np.float32)
    out_smpl = smpl[index_0_np] * (1.0 - blend_np) + smpl[index_1_np] * blend_np

    return motion.replace(
        root_trans_offset=np.ascontiguousarray(out_translation.numpy(), dtype=np.float32),
        pose_aa=np.ascontiguousarray(out_pose_aa.numpy(), dtype=np.float32),
        dof=np.ascontiguousarray(out_dof.numpy(), dtype=np.float32),
        root_rot=np.ascontiguousarray(out_root_rot.numpy(), dtype=np.float32),
        smpl_joints=np.ascontiguousarray(out_smpl, dtype=np.float32),
        fps=int(target_fps),
    )


def motion_from_dof(
    key: str,
    *,
    dof: np.ndarray,
    root_trans_offset: np.ndarray,
    root_aa: np.ndarray | None = None,
    fps: int = DEFAULT_TARGET_FPS,
) -> Motion:
    """Build a clip from joint angles, rebuilding ``pose_aa``/``root_rot`` consistently.

    Exists so that tests and the repair operator can synthesize or rewrite a clip
    without independently re-deriving the redundant encoding (and drifting from
    it).  ``root_aa`` defaults to the identity rotation.
    """
    dof = np.ascontiguousarray(dof, dtype=np.float32)
    root_trans_offset = np.ascontiguousarray(root_trans_offset, dtype=np.float32)
    frames = int(dof.shape[0])
    if dof.shape != (frames, NUM_DOF):
        raise ValueError(f"dof must be (T, {NUM_DOF}), got {dof.shape}")
    if root_trans_offset.shape != (frames, 3):
        raise ValueError(f"root_trans_offset must be ({frames}, 3), got {root_trans_offset.shape}")
    if root_aa is None:
        root_aa = np.zeros((frames, 3), dtype=np.float32)
    root_aa = np.ascontiguousarray(root_aa, dtype=np.float32)

    pose_aa = np.zeros((frames, NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 0, :] = root_aa
    pose_aa[:, 1:, :] = (DOF_AXIS[None, :, :] * dof[:, :, None]).astype(np.float32)
    root_rot = _axis_angle_to_quat_xyzw(root_aa.astype(np.float64)).astype(np.float32)
    return Motion(
        key=key,
        root_trans_offset=root_trans_offset,
        pose_aa=pose_aa,
        dof=dof,
        root_rot=root_rot,
        smpl_joints=np.zeros((frames, 24, 3), dtype=np.float32),
        fps=int(fps),
    )


def _axis_angle_to_quat_xyzw(axis_angle: NDArray[np.float64]) -> NDArray[np.float64]:
    """Axis-angle -> XYZW quaternion, matching ``scipy.spatial.transform`` output."""
    angle = np.linalg.norm(axis_angle, axis=-1)
    half = 0.5 * angle
    small = angle < 1e-8
    scale = np.where(small, 0.5 - angle**2 / 48.0, np.sin(half) / np.where(small, 1.0, angle))
    return np.concatenate([axis_angle * scale[..., None], np.cos(half)[..., None]], axis=-1)
