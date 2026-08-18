"""Turn a generated reference clip into something the predicates and envelope can read.

Grading a behaviour has meant rolling it out: convert the motion, spawn Isaac, track it,
record the trajectory, then measure. That is minutes of contended GPU per clip, and it is the
wrong tool for the question *did the generator produce the behaviour I asked for*, which is
answerable from the reference alone. Three of the taxonomy's body modes turn out not to
produce their behaviour at all, and each was discovered only after its motions had been
rolled out.

Forward kinematics closes that loop. The reference is a 36-column MuJoCo qpos clip and the
G1 MJCF's body names are the same strings the recorder writes -- ``torso_link``,
``left_ankle_roll_link`` -- so a payload assembled from ``mj_forward`` satisfies the same
contract a recorded trajectory does, and every predicate and envelope measurement works on it
unchanged. Testing a rephrased prompt costs a second instead of a rollout.

**What this cannot tell you.** A reference is what the generator asked for; an executed
trajectory is what the controller managed. They differ, and the difference is exactly what
the acceptance gate exists to catch. So this is a screen for behavioural *intent* and never a
substitute for physics: a reference that ducks may still be untrackable, and only a rollout
knows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .self_intersection import DEFAULT_G1_MJCF, SelfIntersectionError, _load_model


def payload_from_reference(
    reference_qpos: np.ndarray,
    *,
    fps: float = 30.0,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
) -> dict:
    """Assemble a recorder-shaped payload from a reference clip by forward kinematics.

    ``reference_qpos`` is the 36-column layout the corpus uses: 3 root translation, 4 root
    quaternion (wxyz), 29 joint DOFs.
    """
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise SelfIntersectionError(f"expected (T, nq) reference qpos, got {qpos.shape}")

    mujoco, model = _load_model(mjcf_path)
    if qpos.shape[1] != model.nq:
        raise SelfIntersectionError(
            f"reference has {qpos.shape[1]} columns but the model expects {model.nq}"
        )
    data = mujoco.MjData(model)

    # Body 0 is ``world``. Dropping it leaves the same 30 links the recorder writes, in the
    # same order, which is what lets the capsule model be keyed by name without a mapping.
    names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(1, model.nbody)
    )

    frames = len(qpos)
    positions = np.zeros((frames, len(names), 3))
    quaternions = np.zeros((frames, len(names), 4))
    for index in range(frames):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        positions[index] = data.xpos[1:]
        quaternions[index] = data.xquat[1:]

    return {
        "body_pos_w": positions,
        "body_quat_w": quaternions,
        "body_names": names,
        "root_pos_w": qpos[:, :3].copy(),
        "root_quat_w": qpos[:, 3:7].copy(),
        "fps": float(fps),
        "total_frames": frames,
        "quat_format": "wxyz",
        "kind": "reference",
    }


def payload_from_csv(
    path: str | Path, *, fps: float = 30.0, mjcf_path: str | Path = DEFAULT_G1_MJCF
) -> dict:
    """Convenience wrapper over a generated Kimodo CSV."""
    return payload_from_reference(
        np.loadtxt(Path(path), delimiter=","), fps=fps, mjcf_path=mjcf_path
    )
