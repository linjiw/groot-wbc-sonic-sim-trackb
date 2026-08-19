"""Join two clips where the gait already agrees, because a robot never starts a crouch at frame 0.

Every clip in the corpus is played from its first frame to its last. That is enough to ask whether
a behaviour clears an obstacle, and it is not enough to ask anything a closed loop asks: when to
begin adapting, from which point in the stride, when to recover, and how to get from the first
obstacle to the second without the commanded reference jumping. A clip that crouches from frame
zero cannot even support a claim about a decision, because the obstacle was not visible when the
crouch began — the perception timing gate exists to catch exactly that.

Nothing in the repository represents gait phase, so this module builds it first and then uses it.

**Phase is read off the feet, not off a clock.** Two clips of the same nominal duration are at
different points of the stride at the same frame index, and after `deployable_retiming` has slowed
one of them they are not even the same length. The estimator here takes the signed difference
between the two soles' heights and its rate of change, which together trace a limit cycle over one
stride, and reports the angle around that cycle. It needs no contact sensor and no annotation.

**A transition is chosen, not assumed.** Given a route position where the switch must happen — set
by where the obstacle is, not by preference — the planner searches the frames either side of it in
*both* clips and scores every pairing by four things a tracker actually feels: how far apart the
gait phases are, how far the joints jump, how fast they jump, and whether the robot has both feet
down when it happens. Double support is preferred because a switch during flight changes the
reference for a leg that is mid-swing with nothing to push against.

**The target clip is re-anchored, never teleported.** Its root is rotated about z and translated so
that the frame being switched into sits exactly where the robot already is. This is the same
operation the controller performs on reset, applied without the reset: heading-only rotation, so
the reference is not tipped, and the height left alone, so a crouched clip stays crouched.

**A measured limit of the phase estimator, recorded rather than tuned away.** On the retimed
overhead crouch it reports 7 wraps against the nominal walk's 4 over the same route. The crouch
compresses the sole-height difference the estimator reads, so the loop shrinks toward the noise and
picks up spurious crossings. This does not reach the planner's decisions -- it compares phases
locally, near one station, and found gaps of 0.04 rad at both seams of a real walk-crouch-walk
composition -- but the *count* of strides in a crouched clip should not be trusted, and any future
use that integrates phase over a whole clip needs a contact-based estimator instead of this one.

What this module does not do is decide *whether* to switch. That is the selector's job, and the
cost this returns is one of its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .deployable_retiming import route_deviation_m, slerp

# One definition of route progress, not three. `local_adaptation` owns it because that is where an
# obstacle station is first expressed in it, and a second copy here would be the same mistake the
# window measurement already made once by living in a script.
from .local_adaptation import route_progress  # noqa: F401  (re-exported for callers of this module)

#: Frames of cross-fade across the join. Long enough that the tracker sees a ramp rather than a
#: step, short enough that the blended stretch is not a pose neither clip ever held. At 30 fps this
#: is 0.2 s, about a fifth of a stride.
DEFAULT_BLEND_FRAMES = 6

#: How far either side of the requested route position the planner may move the switch, in units of
#: normalised route progress. The switch has to happen near where the obstacle is; this says how
#: much freedom it has to find a good point in the stride while staying there.
DEFAULT_SEARCH_WINDOW = 0.08

#: Weights on the four costs. Velocity is weighted above position because a reference that steps in
#: position is a bounded error the tracker closes, while one that steps in velocity asks for an
#: acceleration no actuator has. Support is a bonus rather than a cost, so a clip with no clean
#: double support still yields a transition instead of failing.
DEFAULT_WEIGHTS = {"phase": 1.0, "pose": 1.0, "velocity": 2.0, "support": 0.5}

#: Metres below which a sole counts as down. The soles of a standing G1 sit within a centimetre of
#: the floor and a swing foot clears it by several, so this separates them with room to spare.
SOLE_DOWN_M = 0.03


@dataclass(frozen=True)
class TransitionPoint:
    """Where two clips may be joined, and what the join costs."""

    source_frame: int
    target_frame: int
    #: Normalised route progress at which the switch happens, measured on the source clip.
    route_progress: float
    #: Angular distance between the two gait phases, in radians, wrapped to [0, pi].
    phase_gap_rad: float
    #: Largest single-joint difference across the join, in radians.
    pose_jump_rad: float
    #: Largest single-joint difference of the per-frame joint velocity, in radians per frame.
    velocity_jump_rad_per_frame: float
    #: 1.0 when both feet are down on both sides of the join, 0.0 when neither is.
    support: float
    cost: float


def sole_heights(qpos: np.ndarray, mjcf_path) -> np.ndarray:
    """Per-frame lowest point of each foot's collision geometry, as (T, 2) — left then right.

    Capsule surfaces rather than joint centres, for the reason `local_adaptation` gives: the
    distance from an ankle joint to the bottom of the foot is not a constant of the pose.
    """
    from .reference_payload import payload_from_reference
    from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

    payload = payload_from_reference(np.asarray(qpos, dtype=np.float64), mjcf_path=mjcf_path)
    starts, ends, radii, owners = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )
    lows = np.minimum(starts[:, :, 2], ends[:, :, 2]) - radii[None, :]
    sides = []
    for side in ("left", "right"):
        columns = [i for i, owner in enumerate(owners) if "ankle" in owner and side in owner]
        if not columns:
            raise ValueError(f"no {side} ankle capsule in the collision model")
        sides.append(lows[:, columns].min(axis=1))
    return np.stack(sides, axis=1)


def gait_phase(soles: np.ndarray) -> np.ndarray:
    """The angle around the stride's limit cycle, per frame, in [-pi, pi].

    The signed height difference between the feet is zero twice a stride and extremal in between,
    so on its own it cannot tell stepping left from stepping right. Paired with its own rate of
    change it traces a loop, and the angle around that loop is monotone through the stride and
    comparable between two clips that never shared a frame index.

    Both axes are scaled by their own spread before the angle is taken. Without that the loop is a
    thin ellipse -- the difference is centimetres, its derivative is millimetres per frame -- and the
    angle would spend almost all of the stride near two values.

    The angle **advances** with time, and zero is the moment the feet are level and separating. The
    sign matters: a phase that ran backwards would still be a valid cycle coordinate and would still
    match two clips correctly, but every caller reading it as "how far through the stride" would be
    reading it inverted.
    """
    heights = np.asarray(soles, dtype=np.float64)
    if heights.ndim != 2 or heights.shape[1] != 2:
        raise ValueError(f"expected (T, 2) sole heights, got {heights.shape}")
    if len(heights) < 3:
        raise ValueError("a stride cannot be read from fewer than three frames")

    difference = heights[:, 0] - heights[:, 1]
    difference = difference - difference.mean()
    rate = np.gradient(difference)
    scale_d = float(difference.std())
    scale_r = float(rate.std())
    if scale_d < 1e-9 or scale_r < 1e-9:
        # A clip with no stride — standing, or both feet moving together. There is no phase to
        # report, and returning zeros says so without pretending the loop exists.
        return np.zeros(len(heights))
    return np.arctan2(difference / scale_d, rate / scale_r)


def phase_gap(a: float, b: float) -> float:
    """Angular distance between two phases, wrapped to [0, pi]."""
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def support_profile(soles: np.ndarray, *, down_m: float = SOLE_DOWN_M) -> np.ndarray:
    """Per-frame fraction of feet that are down, in {0, 0.5, 1}."""
    heights = np.asarray(soles, dtype=np.float64)
    floor = float(heights.min())
    return (heights - floor < down_m).mean(axis=1)


def _joint_velocity(qpos: np.ndarray) -> np.ndarray:
    """Per-frame joint velocity of a reference, in radians per frame."""
    return np.gradient(np.asarray(qpos, dtype=np.float64)[:, 7:], axis=0)


def plan_transition(
    source_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    at_progress: float,
    source_soles: np.ndarray,
    target_soles: np.ndarray,
    window: float = DEFAULT_SEARCH_WINDOW,
    weights: dict[str, float] | None = None,
) -> TransitionPoint:
    """Find where to leave ``source_qpos`` and join ``target_qpos``, near one route position.

    ``at_progress`` is where the switch must happen, in normalised route progress — set by the
    obstacle, not chosen for convenience. The planner may move within ``window`` of it to find a
    point in the stride where the two clips agree.

    Sole heights are passed in rather than computed here because they cost a forward-kinematics
    pass over the whole clip and a caller planning several transitions on one pair of clips should
    pay for that once.
    """
    source = np.asarray(source_qpos, dtype=np.float64)
    target = np.asarray(target_qpos, dtype=np.float64)
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"the two clips must describe the same body: {source.shape[1]} against "
            f"{target.shape[1]} columns"
        )
    if not 0.0 <= at_progress <= 1.0:
        raise ValueError(f"at_progress must lie in [0, 1]; got {at_progress}")
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    source_progress = route_progress(source[:, :2])
    target_progress = route_progress(target[:, :2])
    source_phase = gait_phase(source_soles)
    target_phase = gait_phase(target_soles)
    source_support = support_profile(source_soles)
    target_support = support_profile(target_soles)
    source_velocity = _joint_velocity(source)
    target_velocity = _joint_velocity(target)

    # Only frames near the required route position are candidates, on both sides. A join that is
    # kinematically perfect but 2 m from the obstacle is not a transition, it is a different route.
    sources = np.flatnonzero(np.abs(source_progress - at_progress) <= window)
    targets = np.flatnonzero(np.abs(target_progress - at_progress) <= window)
    # The join needs a frame after it in the target and before it in the source, so the ends are
    # not candidates: there is nothing to blend into or out of.
    sources = sources[(sources > 0) & (sources < len(source) - 1)]
    targets = targets[(targets > 0) & (targets < len(target) - 1)]
    if not len(sources) or not len(targets):
        raise ValueError(
            f"neither clip has usable frames within {window} of route progress {at_progress}; "
            "the switch cannot be placed where the obstacle is"
        )

    best: TransitionPoint | None = None
    for i in sources:
        for j in targets:
            gap = phase_gap(float(source_phase[i]), float(target_phase[j]))
            pose = float(np.abs(target[j, 7:] - source[i, 7:]).max())
            velocity = float(np.abs(target_velocity[j] - source_velocity[i]).max())
            support = 0.5 * (float(source_support[i]) + float(target_support[j]))
            cost = (
                w["phase"] * gap / np.pi
                + w["pose"] * pose
                + w["velocity"] * velocity
                - w["support"] * support
            )
            if best is None or cost < best.cost:
                best = TransitionPoint(
                    source_frame=int(i),
                    target_frame=int(j),
                    route_progress=float(source_progress[i]),
                    phase_gap_rad=gap,
                    pose_jump_rad=pose,
                    velocity_jump_rad_per_frame=velocity,
                    support=support,
                    cost=float(cost),
                )
    assert best is not None
    return best


def anchor_to(
    target_qpos: np.ndarray, target_frame: int, root_pos: np.ndarray, root_quat: np.ndarray
) -> np.ndarray:
    """Rotate and translate a clip so ``target_frame`` sits at a given root pose.

    Heading-only rotation and planar translation. Rolling or pitching the clip to match would tip a
    reference the robot is expected to hold upright, and translating in z would lift a crouch out of
    its own crouch; both are changes to the behaviour rather than to where it happens.
    """
    clip = np.asarray(target_qpos, dtype=np.float64).copy()
    pos = np.asarray(root_pos, dtype=np.float64)
    quat = np.asarray(root_quat, dtype=np.float64)

    yaw_from = _heading(clip[target_frame, 3:7])
    yaw_to = _heading(quat)
    delta = yaw_to - yaw_from
    cos, sin = np.cos(delta), np.sin(delta)

    origin = clip[target_frame, :2].copy()
    offset = clip[:, :2] - origin
    clip[:, 0] = pos[0] + cos * offset[:, 0] - sin * offset[:, 1]
    clip[:, 1] = pos[1] + sin * offset[:, 0] + cos * offset[:, 1]

    spin = np.array([np.cos(0.5 * delta), 0.0, 0.0, np.sin(0.5 * delta)])
    clip[:, 3:7] = _quat_mul(np.repeat(spin[None, :], len(clip), axis=0), clip[:, 3:7])
    return clip


def _heading(quat: np.ndarray) -> float:
    """Yaw of a (w, x, y, z) quaternion, in radians."""
    w, x, y, z = (float(v) for v in quat)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=1,
    )


def stitch(
    source_qpos: np.ndarray,
    target_qpos: np.ndarray,
    point: TransitionPoint,
    *,
    blend_frames: int = DEFAULT_BLEND_FRAMES,
) -> np.ndarray:
    """Concatenate two clips at a planned point, with a cross-fade and the target re-anchored.

    The result is one clip: the source up to the join, then ``blend_frames`` where the two are
    mixed, then the target. The mixing weight is a smoothstep so the blend has no velocity step of
    its own at either end — a linear fade would trade one discontinuity for two smaller ones.
    """
    source = np.asarray(source_qpos, dtype=np.float64)
    target = np.asarray(target_qpos, dtype=np.float64)
    if blend_frames < 0:
        raise ValueError(f"blend_frames must not be negative; got {blend_frames}")

    anchored = anchor_to(
        target, point.target_frame, source[point.source_frame, :3], source[point.source_frame, 3:7]
    )
    head = source[: point.source_frame]
    tail = anchored[point.target_frame :]
    span = min(blend_frames, len(tail), len(source) - point.source_frame)
    if span <= 0:
        return np.concatenate([head, tail], axis=0)

    alpha = np.linspace(0.0, 1.0, span + 2)[1:-1]
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    over_source = source[point.source_frame : point.source_frame + span]
    over_target = tail[:span]

    blended = np.empty_like(over_target)
    linear = np.r_[0:3, 7 : source.shape[1]]
    blended[:, linear] = (1.0 - alpha)[:, None] * over_source[:, linear] + alpha[
        :, None
    ] * over_target[:, linear]
    blended[:, 3:7] = slerp(over_source[:, 3:7], over_target[:, 3:7], alpha)
    return np.concatenate([head, blended, tail[span:]], axis=0)


def join_discontinuity(clip: np.ndarray, frame: int) -> tuple[float, float]:
    """Largest joint position and velocity step across one frame of a finished clip.

    Read back off the stitched result rather than predicted from the plan, so a blend that failed
    to smooth the join is visible instead of assumed away.
    """
    q = np.asarray(clip, dtype=np.float64)
    if not 1 <= frame < len(q) - 1:
        raise ValueError(f"frame {frame} has no neighbours in a clip of {len(q)}")
    position = float(np.abs(q[frame, 7:] - q[frame - 1, 7:]).max())
    velocity = float(
        np.abs((q[frame + 1, 7:] - q[frame, 7:]) - (q[frame, 7:] - q[frame - 1, 7:])).max()
    )
    return position, velocity


def holds_route(
    source_qpos: np.ndarray, stitched: np.ndarray, *, tolerance_m: float = 0.05
) -> bool:
    """Whether the stitched clip still walks the source's line through the room.

    A transition that veers is a different route, and a scene built against the source no longer
    describes it. The tolerance is looser than `deployable_retiming`'s because the target clip
    genuinely leaves the source's path after the join — this asks whether it leaves it *near* the
    join, which is where a re-anchoring error would show.
    """
    joined = np.asarray(stitched, dtype=np.float64)
    source = np.asarray(source_qpos, dtype=np.float64)
    reach = min(len(joined), len(source))
    return route_deviation_m(source[:, :2], joined[:reach, :2]) <= tolerance_m
