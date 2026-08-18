"""Make a generated crouch reachable, since asking for a reachable one does not work.

Four prompt phrasings were tried, including one written specifically to forbid the failure
("keeping the back straight and never folding forward at the waist"). Every phrasing that
lowers the body pins ``waist_pitch_joint`` at its limit on 100% of frames; the one phrasing
that stays inside the limits produces a motion 8 mm *taller* than a plain walk. The generator's
crouch is a waist fold, and forbidding the fold removes the crouch rather than changing how it
is made. So the crouch has to be constructed.

Two edits do it, and both are small:

**Take the waist off its limit.** The clip holds +0.521 rad against a +0.520 limit for its
entire duration. Setting it to a fraction of the limit keeps the forward lean -- which is
doing real work, since leaning shifts the centre of mass over the feet in a deep squat -- while
leaving the joint headroom.

**Give every joint a margin.** Clamping is not enough and is worth being precise about: a
clamped value sits *exactly at* the limit, and sitting at a limit is what the saturation
screen counts. Clamping converts an out-of-range clip into a fully saturated one. Shrinking
each joint's range toward its centre by a few percent puts it strictly inside instead.

What this does **not** do is re-solve the pose. Root translation is a floating base, so raising
the pelvis moves the whole robot without changing a single joint -- it looks like a shallower
squat and is really a robot hovering. Anything that needs the legs re-solved needs an
optimiser, and this is not one.

**A retargeted reference is not a verified motion.** It is reachable, which is what the screen
measures. Whether SONIC can track it is a rollout's question, and the foot-contact diagnostics
here exist so that an obviously untrackable result can be rejected before spending one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .motion_prefilter import load_joint_limits
from .self_intersection import DEFAULT_G1_MJCF

#: Fraction of each joint's half-range to keep. 0.97 was enough to take the measured crouch
#: clips from 0.0098 saturation to exactly zero, against a screen threshold of 0.009.
DEFAULT_RANGE_KEEP = 0.97

#: Waist pitch as a fraction of its limit. Zero stands the torso fully upright and costs
#: 50 mm of silhouette; half keeps the lean that helps a deep squat balance while leaving the
#: joint 50% headroom.
DEFAULT_WAIST_FRACTION = 0.5


@dataclass(frozen=True)
class RetargetReport:
    """What the retarget changed, and what it cost."""

    frames: int
    #: Joints that were sitting at a limit before, and are not after.
    relieved_joints: tuple[str, ...]
    #: Largest change applied to any joint, in radians.
    max_joint_change_rad: float
    #: Mean vertical movement of the lowest point of either foot's collision geometry, in
    #: metres. Measured on the capsule rather than the link origin: the ankle-roll origin sits
    #: at the ankle joint, so rotating the ankle tilts the sole without moving that point at
    #: all, and a link-origin metric reports exactly zero for the one change this retarget
    #: actually makes to the feet.
    foot_height_shift_m: float
    #: Largest change in either foot's pitch, in radians. A retarget that relieves the ankle
    #: necessarily tilts the sole, and this is how much.
    max_foot_tilt_change_rad: float
    waist_fraction: float
    range_keep: float


def retarget_crouch(
    reference_qpos: np.ndarray,
    *,
    waist_fraction: float = DEFAULT_WAIST_FRACTION,
    range_keep: float = DEFAULT_RANGE_KEEP,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    epsilon: float = 1e-3,
) -> tuple[np.ndarray, RetargetReport]:
    """Return a reachable version of a crouch clip, and a report of what changed."""
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError(f"expected (T, 7+J) reference qpos, got {qpos.shape}")
    if not 0.0 <= range_keep <= 1.0:
        raise ValueError(f"range_keep must be in [0, 1]; got {range_keep}")
    if not -1.0 <= waist_fraction <= 1.0:
        raise ValueError(f"waist_fraction must be in [-1, 1]; got {waist_fraction}")

    names, limits = load_joint_limits(mjcf_path)
    out = qpos.copy()
    dofs = out[:, 7:]
    count = min(dofs.shape[1], limits.shape[0])
    lower, upper = limits[:count, 0], limits[:count, 1]

    before = dofs[:, :count].copy()
    at_limit_before = (
        (np.abs(before - lower) < epsilon) | (np.abs(before - upper) < epsilon)
        | (before < lower) | (before > upper)
    )

    if "waist_pitch_joint" in names:
        index = names.index("waist_pitch_joint")
        if index < count:
            sign = np.sign(np.median(before[:, index])) or 1.0
            edge = upper[index] if sign > 0 else lower[index]
            out[:, 7 + index] = waist_fraction * edge

    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * range_keep
    out[:, 7 : 7 + count] = np.clip(out[:, 7 : 7 + count], centre - half, centre + half)

    after = out[:, 7 : 7 + count]
    at_limit_after = (np.abs(after - lower) < epsilon) | (np.abs(after - upper) < epsilon)
    relieved = tuple(
        names[i] for i in range(count)
        if at_limit_before[:, i].any() and not at_limit_after[:, i].any()
    )

    # Foot movement, as a cheap proxy for whether contacts survived. Computed from the
    # ankle-roll link so it needs forward kinematics; a large shift is a reason to look at
    # the clip rather than roll it out.
    from .reference_payload import payload_from_reference

    from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    def sole_and_tilt(clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        payload = payload_from_reference(clip, mjcf_path=mjcf_path)
        body_names = list(payload["body_names"])
        starts, ends, radii, owners = body_capsules_world(
            np.asarray(payload["body_pos_w"], dtype=np.float64),
            np.asarray(payload["body_quat_w"], dtype=np.float64),
            body_names, capsules=G1_COLLISION_CAPSULES,
        )
        columns = [i for i, owner in enumerate(owners) if "ankle" in owner]
        if not columns:
            raise ValueError("no ankle capsules in the collision model")
        lows = np.minimum(starts[:, columns, 2], ends[:, columns, 2]) - radii[None, columns]
        # Sole pitch, from the capsule's own axis: the foot capsule runs along the foot.
        axis = ends[:, columns, :] - starts[:, columns, :]
        length = np.linalg.norm(axis, axis=-1) + 1e-9
        tilt = np.arcsin(np.clip(axis[..., 2] / length, -1.0, 1.0))
        return lows.min(axis=1), tilt

    try:
        low_after, tilt_after = sole_and_tilt(out)
        low_before, tilt_before = sole_and_tilt(qpos)
        shift = float(np.mean(np.abs(low_after - low_before)))
        tilt_change = float(np.abs(tilt_after - tilt_before).max())
    except (ValueError, OSError, KeyError):
        shift = float("nan")
        tilt_change = float("nan")

    return out, RetargetReport(
        frames=len(qpos),
        relieved_joints=relieved,
        max_joint_change_rad=float(np.abs(after - before).max()),
        foot_height_shift_m=shift,
        max_foot_tilt_change_rad=tilt_change,
        waist_fraction=waist_fraction,
        range_keep=range_keep,
    )
