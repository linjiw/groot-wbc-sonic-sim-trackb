"""Bend one nominal motion locally, instead of pairing two independently generated ones.

Every family so far pairs two separately generated clips: a walk and a crouch that were never
the same motion. A reviewer can reasonably ask whether the shelf separated the *behaviours* or
merely two different journeys that happened to be labelled differently, and the honest answer
is that the construction cannot tell them apart.

This operator removes the question. It starts from a nominal motion that has already been
tracked and accepted, and changes only the degrees of freedom needed to clear one obstacle,
over only the stretch of route where that obstacle is. Everything else is held:

* root XY and yaw, so both motions walk the same line
* duration and frame count, so gait phase stays aligned
* start and goal
* waist pitch, so the lowering is knee-driven rather than a fold
* stance-foot height, so contacts survive

The adapted motion is therefore the nominal motion plus a local deviation, and the two differ
by exactly the adaptation under test.

**Local, not whole-route.** A clip crouched from frame zero cannot demonstrate a decision made
from what the robot sees, because the obstacle is not visible when the crouch begins -- such a
pair can only support map-conditioned selection. Centring the adaptation on the obstacle
station in route-progress coordinates makes the onset late enough to be a response.

**Clearance is measured on collision-capsule surfaces**, never on a joint position or the root
height. The G1's own half-width runs from 0.273 m with arms tucked to 0.664 m at peak arm
swing, so a joint-centre proxy is wrong in both directions depending on the pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .motion_prefilter import load_joint_limits
from .self_intersection import DEFAULT_G1_MJCF

#: Joints the crouch is allowed to use. The waist is deliberately absent: folding at the waist
#: is how the generator produced its crouches, and it is the one motion the G1 cannot hold.
CROUCH_JOINTS = ("hip_pitch", "knee", "ankle_pitch")

#: Fraction of each joint's half-range the adapted motion may occupy. Leaving headroom is not
#: cosmetic -- a reference that rides a limit is what the saturation screen rejects, and a
#: reference that sits exactly at one is what clamping produces.
DEFAULT_RANGE_KEEP = 0.94

#: Half-width of the adaptation window, in route progress. 0.18 means the robot is fully
#: adapted across about a third of its route, with ramps either side.
DEFAULT_WINDOW = 0.18

#: Fraction of the window spent ramping in and out. Abrupt onsets are not trackable.
DEFAULT_RAMP = 0.45


@dataclass(frozen=True)
class LocalCrouchReport:
    """What the operator achieved, and what it left alone."""

    frames: int
    station_fraction: float
    #: Metres the collision-capsule silhouette peak came down by, at its lowest.
    silhouette_drop_m: float
    nominal_silhouette_m: float
    adapted_silhouette_m: float
    #: Largest change in waist pitch, in radians. Should be ~0: the crouch is knee-driven.
    waist_change_rad: float
    #: Largest movement of the lower foot's sole, in metres. Should be ~0: contacts are held.
    foot_height_shift_m: float
    #: Fraction of the clip where the adaptation is active at all.
    active_fraction: float
    root_path_preserved: bool
    scale_applied: float


def route_progress(root_xy: np.ndarray) -> np.ndarray:
    """Cumulative path length, normalised to [0, 1].

    Progress along the route rather than frame index: two motions of the same duration can
    reach the obstacle at different frames, and an obstacle sits at a place, not a time.
    """
    steps = np.linalg.norm(np.diff(root_xy, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(steps)])
    total = cumulative[-1]
    return cumulative / total if total > 1e-9 else np.linspace(0.0, 1.0, len(root_xy))


def adaptation_profile(
    progress: np.ndarray,
    station: float,
    *,
    window: float = DEFAULT_WINDOW,
    ramp: float = DEFAULT_RAMP,
) -> np.ndarray:
    """A smooth 0 -> 1 -> 0 profile centred on ``station`` in route progress."""
    distance = np.abs(progress - station)
    inner = window * (1.0 - ramp)
    alpha = np.clip((window - distance) / max(window - inner, 1e-9), 0.0, 1.0)
    # Smoothstep, so onset and recovery have no velocity discontinuity for a tracker to fight.
    return alpha * alpha * (3.0 - 2.0 * alpha)


def _silhouette(qpos: np.ndarray, mjcf_path) -> np.ndarray:
    """Per-frame top of the collision geometry, in world z."""
    from .reference_payload import payload_from_reference
    from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    payload = payload_from_reference(qpos, mjcf_path=mjcf_path)
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]), capsules=G1_COLLISION_CAPSULES,
    )
    return (np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]).max(axis=1)


def _sole_height(qpos: np.ndarray, mjcf_path) -> np.ndarray:
    """Per-frame lowest point of either foot's collision geometry."""
    from .reference_payload import payload_from_reference
    from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    payload = payload_from_reference(qpos, mjcf_path=mjcf_path)
    starts, ends, radii, owners = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]), capsules=G1_COLLISION_CAPSULES,
    )
    columns = [i for i, owner in enumerate(owners) if "ankle" in owner]
    lows = np.minimum(starts[:, columns, 2], ends[:, columns, 2]) - radii[None, columns]
    return lows.min(axis=1)


def local_crouch(
    nominal_qpos: np.ndarray,
    station_fraction: float,
    *,
    target_drop_m: float = 0.15,
    window: float = DEFAULT_WINDOW,
    ramp: float = DEFAULT_RAMP,
    range_keep: float = DEFAULT_RANGE_KEEP,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    max_scale: float = 3.0,
) -> tuple[np.ndarray, LocalCrouchReport]:
    """Crouch a nominal motion locally, around one station, by about ``target_drop_m``.

    Returns the adapted clip and a report. The scale needed to reach the target drop is found
    by bisection on the measured capsule silhouette rather than assumed from joint angles,
    because the relationship between leg flexion and how low the *body* actually gets depends
    on the pose and is not worth modelling.
    """
    qpos = np.asarray(nominal_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError(f"expected (T, 7+J) reference qpos, got {qpos.shape}")
    if not 0.0 <= station_fraction <= 1.0:
        raise ValueError(f"station_fraction must be in [0, 1]; got {station_fraction}")
    if target_drop_m <= 0.0:
        raise ValueError(f"target_drop_m must be positive; got {target_drop_m}")

    names, limits = load_joint_limits(mjcf_path)
    count = min(qpos.shape[1] - 7, limits.shape[0])
    legs = [
        i for i, name in enumerate(names[:count])
        if any(key in name for key in CROUCH_JOINTS)
    ]
    if not legs:
        raise ValueError("no crouch joints found in the model")

    alpha = adaptation_profile(
        route_progress(qpos[:, :2]), station_fraction, window=window, ramp=ramp
    )
    nominal_soles = _sole_height(qpos, mjcf_path)
    nominal_silhouette = _silhouette(qpos, mjcf_path)
    lower, upper = limits[:count, 0], limits[:count, 1]
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * range_keep

    def build(scale: float) -> np.ndarray:
        out = qpos.copy()
        # Flex the legs in proportion to the local profile, so the crouch is confined to the
        # stretch of route the obstacle occupies and the rest of the clip is untouched.
        factor = 1.0 + alpha[:, None] * scale
        out[:, 7 + np.asarray(legs)] = qpos[:, 7 + np.asarray(legs)] * factor
        out[:, 7 : 7 + count] = np.clip(out[:, 7 : 7 + count], centre - half, centre + half)
        # The root follows the legs, per frame, so the feet stay where the nominal put them.
        out[:, 2] += nominal_soles - _sole_height(out, mjcf_path)
        return out

    # Bisect on the achieved silhouette drop.
    low, high = 0.0, max_scale
    for _ in range(12):
        middle = 0.5 * (low + high)
        drop = float(
            (nominal_silhouette - _silhouette(build(middle), mjcf_path)).max()
        )
        if drop < target_drop_m:
            low = middle
        else:
            high = middle
    adapted = build(0.5 * (low + high))

    adapted_silhouette = _silhouette(adapted, mjcf_path)
    waist_index = names.index("waist_pitch_joint") if "waist_pitch_joint" in names else None
    waist_change = (
        float(np.abs(adapted[:, 7 + waist_index] - qpos[:, 7 + waist_index]).max())
        if waist_index is not None and waist_index < count else 0.0
    )

    return adapted, LocalCrouchReport(
        frames=len(qpos),
        station_fraction=station_fraction,
        silhouette_drop_m=float((nominal_silhouette - adapted_silhouette).max()),
        nominal_silhouette_m=float(nominal_silhouette.min()),
        adapted_silhouette_m=float(adapted_silhouette.min()),
        waist_change_rad=waist_change,
        foot_height_shift_m=float(
            np.abs(_sole_height(adapted, mjcf_path) - nominal_soles).max()
        ),
        active_fraction=float((alpha > 0.01).mean()),
        root_path_preserved=bool(
            np.array_equal(adapted[:, :2], qpos[:, :2])
            and np.array_equal(adapted[:, 3:7], qpos[:, 3:7])
        ),
        scale_applied=float(0.5 * (low + high)),
    )


#: Joints the arm tuck may use. Upper body only: the legs, the root and the gait are not
#: touched at all, which is why this operator is easier than the crouch and why its output is
#: far more likely to remain trackable -- nothing about the support or contact schedule moves.
TUCK_JOINTS = ("shoulder", "elbow", "wrist")


@dataclass(frozen=True)
class LocalTuckReport:
    """What the arm tuck achieved, and what it left alone."""

    frames: int
    station_fraction: float
    #: Metres the collision-capsule half-width came in by, at its narrowest.
    half_width_reduction_m: float
    #: Widest the robot gets **inside the adaptation window**, before and after. These, not
    #: the whole-clip maxima, are what a gap placed at the station would test: the operator is
    #: local, so the clip's overall widest frame is usually outside the window and barely
    #: moves. Reporting the whole-clip figure made a working tuck look like it did nothing.
    nominal_half_width_at_station_m: float
    adapted_half_width_at_station_m: float
    #: Whole-clip maxima, kept so a caller can see the tuck did not widen the robot elsewhere.
    nominal_half_width_m: float
    adapted_half_width_m: float
    #: Largest change to any leg joint, in radians. Should be exactly 0.
    leg_change_rad: float
    active_fraction: float
    root_path_preserved: bool
    scale_applied: float


def _half_width(qpos: np.ndarray, mjcf_path) -> np.ndarray:
    """Per-frame half-width across the direction of travel, on capsule surfaces.

    Across the heading rather than the world y axis, and on the capsule rather than a link
    origin: the G1's measured half-width runs 0.273 m with arms tucked to 0.664 m at peak arm
    swing, so both choices change the answer by more than any tuck would.
    """
    from .reference_payload import payload_from_reference
    from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    payload = payload_from_reference(qpos, mjcf_path=mjcf_path)
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    quat = np.asarray(payload["root_quat_w"], dtype=np.float64)
    w, x, y, z = (quat[:, i] for i in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lateral = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)

    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]), capsules=G1_COLLISION_CAPSULES,
    )
    centres = 0.5 * (starts + ends)
    offsets = centres[:, :, :2] - root[:, None, :2]
    return (np.abs(np.einsum("tcd,td->tc", offsets, lateral)) + radii[None, :]).max(axis=1)


def local_arm_tuck(
    nominal_qpos: np.ndarray,
    station_fraction: float,
    *,
    target_reduction_m: float = 0.08,
    window: float = DEFAULT_WINDOW,
    ramp: float = DEFAULT_RAMP,
    range_keep: float = DEFAULT_RANGE_KEEP,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
) -> tuple[np.ndarray, LocalTuckReport]:
    """Draw the arms in toward the torso locally, narrowing the silhouette.

    The lateral counterpart to :func:`local_crouch`, and a deliberately easier operator: the
    root, the legs and the contact schedule are untouched, so the only thing a tracker has to
    follow differently is the arms.

    Asking a generator for this produced motions 64 mm *wider* than a plain walk, which is why
    it is constructed here instead. Narrowing is done by scaling the arm joints toward the
    posture they hold at their narrowest, rather than toward zero, since zero is a T-pose in
    some conventions and would widen the robot.
    """
    qpos = np.asarray(nominal_qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] < 8:
        raise ValueError(f"expected (T, 7+J) reference qpos, got {qpos.shape}")
    if not 0.0 <= station_fraction <= 1.0:
        raise ValueError(f"station_fraction must be in [0, 1]; got {station_fraction}")
    if target_reduction_m <= 0.0:
        raise ValueError(f"target_reduction_m must be positive; got {target_reduction_m}")

    names, limits = load_joint_limits(mjcf_path)
    count = min(qpos.shape[1] - 7, limits.shape[0])
    arms = [
        i for i, name in enumerate(names[:count])
        if any(key in name for key in TUCK_JOINTS)
    ]
    legs = [
        i for i, name in enumerate(names[:count])
        if any(key in name for key in CROUCH_JOINTS)
    ]
    if not arms:
        raise ValueError("no arm joints found in the model")

    alpha = adaptation_profile(
        route_progress(qpos[:, :2]), station_fraction, window=window, ramp=ramp
    )
    nominal_width = _half_width(qpos, mjcf_path)
    lower, upper = limits[:count, 0], limits[:count, 1]
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * range_keep

    # Which way each arm joint has to move to narrow the robot is not knowable from the
    # joint's name, and guessing it wrongly is how the first version of this operator made
    # two clips *wider* than the walk they came from. It blended toward the arm pose at the
    # clip's own narrowest frame, which is narrow only in combination with that frame's torso
    # orientation; transplanted elsewhere in the gait it is not.
    #
    # So the direction is measured. Each arm joint is nudged both ways and the sign that
    # reduces the mean half-width over the active window is kept.
    active = alpha > 0.05
    if not active.any():
        active = np.ones(len(qpos), dtype=bool)
    baseline = float(nominal_width[active].mean())
    probe_step = 0.15
    direction = np.zeros(len(arms))

    # Mirrored joints are probed together. Half-width is a maximum over capsules, and in a
    # symmetric arm pose both wrists attain it at once -- moving one alone cannot lower a
    # maximum that two capsules share, so a per-joint probe sees a flat objective and gives
    # up. Real clips swing out of phase and hide this; a symmetric one exposes it.
    groups: dict[str, list[int]] = {}
    for position, joint in enumerate(arms):
        stem = names[joint].removeprefix("left_").removeprefix("right_")
        groups.setdefault(stem, []).append(position)

    for members in groups.values():
        best_delta, best_signs = 0.0, [0.0] * len(members)
        # A mirrored pair narrows when the two sides move oppositely; a midline joint when it
        # moves either way. Both hypotheses are tried and the better kept.
        candidates = [[1.0] * len(members), [-1.0] * len(members)]
        if len(members) == 2:
            candidates += [[1.0, -1.0], [-1.0, 1.0]]
        for signs in candidates:
            trial = qpos.copy()
            for sign, position in zip(signs, members):
                joint = arms[position]
                trial[:, 7 + joint] = np.clip(
                    trial[:, 7 + joint] + sign * probe_step, lower[joint], upper[joint]
                )
            reduction = baseline - float(_half_width(trial, mjcf_path)[active].mean())
            if reduction > best_delta:
                best_delta, best_signs = reduction, list(signs)
        for sign, position in zip(best_signs, members):
            direction[position] = sign

    def build(scale: float) -> np.ndarray:
        out = qpos.copy()
        step = alpha[:, None] * scale * direction[None, :]
        out[:, 7 + np.asarray(arms)] = qpos[:, 7 + np.asarray(arms)] + step
        out[:, 7 : 7 + count] = np.clip(out[:, 7 : 7 + count], centre - half, centre + half)
        return out

    low, high = 0.0, 1.5
    for _ in range(10):
        middle = 0.5 * (low + high)
        reduction = float((nominal_width - _half_width(build(middle), mjcf_path)).max())
        if reduction < target_reduction_m:
            low = middle
        else:
            high = middle
    adapted = build(0.5 * (low + high))
    adapted_width = _half_width(adapted, mjcf_path)

    return adapted, LocalTuckReport(
        frames=len(qpos),
        station_fraction=station_fraction,
        half_width_reduction_m=float((nominal_width - adapted_width).max()),
        nominal_half_width_at_station_m=float(nominal_width[active].max()),
        adapted_half_width_at_station_m=float(adapted_width[active].max()),
        nominal_half_width_m=float(nominal_width.max()),
        adapted_half_width_m=float(adapted_width.max()),
        leg_change_rad=float(
            np.abs(adapted[:, 7 + np.asarray(legs)] - qpos[:, 7 + np.asarray(legs)]).max()
        ) if legs else 0.0,
        active_fraction=float((alpha > 0.01).mean()),
        root_path_preserved=bool(
            np.array_equal(adapted[:, :3], qpos[:, :3])
            and np.array_equal(adapted[:, 3:7], qpos[:, 3:7])
        ),
        scale_applied=float(0.5 * (low + high)),
    )
