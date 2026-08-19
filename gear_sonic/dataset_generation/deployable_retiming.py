"""Ask a crouched robot for a crouched pace, so the adapted clip can pay its own transport cost.

`local_adaptation` holds duration and the forward schedule fixed on purpose: a matched pair whose
root XY, yaw, frame count and gait phase are identical differs by exactly the adaptation under
test, and that is what makes the counterfactual attributable. The price is that the adapted
reference is internally inconsistent. It asks for a crouch *and* for undiminished progress, and a
crouched G1 cannot walk as fast as an upright one. On the first completed overhead family the
adapted cell cleared its ceiling at 49 N and was still rejected -- `reference_endpoint_tracking_error`,
0.524 m against a 0.35 m budget -- while the nominal it was bent from already spent 0.257 m of
that budget, leaving roughly 90 mm for the adaptation to consume. Endpoint lag then rises
monotonically with crouch depth: 0.257, 0.313, 0.399, 0.509 m at increasing squat. Every crouch
deeper than about 5 cm trips the gate, which bounds the overhead band structurally rather than
incidentally, because a ceiling can only be relieved by lowering the robot.

The response here is *not* to change the matched pair and *not* to move the gate. It is to keep two
artifacts with different jobs:

* **causal reference** -- what `local_adaptation` already emits. Same root path, same duration,
  same gait phase, minimum edit. It answers *why the behaviour must change* and it is what the
  paper's attribution rests on. Nothing in this module touches it.
* **deployable skill** -- the same adaptation, retimed. It answers *how the robot executes the
  changed behaviour*, and it is what a `motion_lib` entry, a fine-tuning set and a real G1 want.

What retiming holds:

* the route **as a curve** -- the same polyline through the same room, so an obstacle placed
  against the nominal is still placed against this clip, and route progress (arclength, normalised)
  is unchanged, so the obstacle's station is unchanged
* the start and the goal, exactly
* every joint angle, at the route position where the operator put it
* the frame rate, because a SONIC motion-lib entry stores integer fps

What retiming releases:

* frame-by-frame root XY identity with the nominal, and therefore duration. The clip gets longer.

The slowdown is a **time warp**, not a root edit: the root pose and the joint angles are resampled
together against a warped clock, so the legs still swing the distance the root travels and the
reference feet do not skate. Slowing the root alone while leaving the gait where it was would
produce exactly the inconsistency this module exists to remove.

**How much to slow down is measured, not modelled.** The shortfall accrues where the adaptation is
active, in proportion to how active it is, so if the executed clip arrived ``d`` metres short of
where the unadapted one arrived, and the adaptation was active over an alpha-weighted arclength
``W``, then commanding that stretch at ``1 - d/W`` of nominal pace restores the endpoint. That is
the same correction `local_adaptation.delivery_corrected_target` applies to the *geometric* target,
applied to the *transport* one. No relationship between knee angle and walking speed is assumed,
because four attempts to infer trackability from a clip rather than measure it have already been
wrong (see docs/prediction_register.md, P1-P4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Multiplier on the measured shortfall, mirroring `local_adaptation.DELIVERY_SAFETY` and for the
#: same reason. Overshooting costs clip duration, which is free; undershooting costs a rollout.
RETIME_SAFETY = 1.2

#: Slowest the window may be commanded, as a fraction of nominal pace. Below this the clip is not a
#: retimed walk any more, it is a different behaviour -- one nobody has asked the controller for and
#: whose stance durations leave the gait it was trained on. A request that hits this floor is
#: reported as floored rather than silently satisfied.
MIN_RETIME_RATIO = 0.35

#: Two clips whose largest joint departure is below this are the same clip, and there is no
#: adaptation to slow down for.
DEPARTURE_FLOOR_RAD = 1e-9

#: Metres of deviation from the nominal polyline that still counts as the same route. The resampled
#: root lands on chords of the original path rather than on the path itself, so the error is the
#: chord sag over one frame -- millimetres at walking pace -- and anything larger means the warp
#: went wrong, not that interpolation was coarse.
ROUTE_TOLERANCE_M = 0.01


@dataclass(frozen=True)
class RetimeReport:
    """What the retiming spent, and what it kept."""

    source_frames: int
    frames: int
    #: Output frames per source frame. Always >= 1: retiming only ever slows down.
    duration_scale: float
    #: Pace inside the fully-active window, as a fraction of nominal, as requested.
    commanded_ratio: float
    #: Pace actually delivered after the clip was rounded to a whole number of frames. Never
    #: faster than ``commanded_ratio``, because the rounding is taken in the slow direction.
    realised_ratio: float
    #: The shortfall the ratio was derived from, or NaN when a ratio was given directly.
    excess_lag_m: float
    #: Arclength over which the adaptation is active, weighted by how active it is. This is the
    #: distance the shortfall had to accrue over, so it is the distance the correction divides by.
    weighted_window_m: float
    #: Largest distance from a retimed root position to the nominal route, in metres.
    route_deviation_m: float
    start_shift_m: float
    goal_shift_m: float
    #: True when the requested slowdown hit ``MIN_RETIME_RATIO`` and the clip is therefore less
    #: slowed than the measurement asked for. Such a clip is expected to still arrive short.
    floored: bool
    #: Fraction of the source clip where the adaptation is active at all.
    active_fraction: float

    @property
    def holds_route(self) -> bool:
        """Whether an obstacle placed against the nominal is still placed against this clip."""
        return self.route_deviation_m <= ROUTE_TOLERANCE_M

    @property
    def added_seconds(self) -> float:
        """Extra wall-clock the deployable clip costs, in source-clip frame units per second."""
        return float(self.frames - self.source_frames)


def departure_profile(nominal_qpos: np.ndarray, adapted_qpos: np.ndarray) -> np.ndarray:
    """Per-frame adaptation activity, normalised to [0, 1], read off the two clips.

    Taken from the clips rather than from the station and window the operator was called with, for
    the reason `local_adaptation.active_frames` gives: a profile recomputed from parameters drifts
    away from the clip whenever a search, a clamp or an excursion cap changed what was actually
    applied. The largest joint departure at each frame is what the adaptation is *doing* there.
    """
    nominal = np.asarray(nominal_qpos, dtype=np.float64)
    adapted = np.asarray(adapted_qpos, dtype=np.float64)
    if nominal.ndim != 2 or adapted.ndim != 2:
        raise ValueError("expected (T, 7+J) reference qpos for both clips")
    if nominal.shape != adapted.shape:
        raise ValueError(
            "a deployable skill is retimed from its own causal reference, so the two clips must "
            f"have the same shape; got {nominal.shape} and {adapted.shape}"
        )
    departure = np.abs(adapted[:, 7:] - nominal[:, 7:]).max(axis=1)
    peak = float(departure.max())
    if peak <= DEPARTURE_FLOOR_RAD:
        raise ValueError("the two clips are identical, so there is no adaptation to retime")
    return departure / peak


def weighted_route_length(root_xy: np.ndarray, weights: np.ndarray) -> float:
    """Arclength weighted by ``weights``, in metres.

    The distance a shortfall had to accrue over. Unweighted arclength would divide the shortfall by
    the whole route, including the stretches the robot walked upright, and so under-correct.
    """
    xy = np.asarray(root_xy, dtype=np.float64)
    alpha = np.asarray(weights, dtype=np.float64)
    if len(xy) != len(alpha):
        raise ValueError(f"weights must match the clip: {len(alpha)} against {len(xy)} frames")
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    midpoints = 0.5 * (alpha[:-1] + alpha[1:])
    return float((steps * midpoints).sum())


def retime_ratio(
    excess_lag_m: float,
    weighted_window_m: float,
    *,
    safety: float = RETIME_SAFETY,
    floor: float = MIN_RETIME_RATIO,
) -> tuple[float, bool]:
    """The pace that pays back a measured shortfall, and whether the floor stopped it.

    ``excess_lag_m`` is the adapted clip's endpoint error **minus the nominal's**, not the raw
    endpoint error. The nominal's own lag is the controller's ordinary tracking behaviour on this
    route; it is inside the gate, it is not what the adaptation caused, and slowing the window to
    remove it would be correcting the nominal through the adapted clip.
    """
    if weighted_window_m <= 0.0:
        raise ValueError("the adaptation is active over no distance, so no pace can be derived")
    if not np.isfinite(excess_lag_m):
        raise ValueError(f"excess_lag_m must be finite; got {excess_lag_m}")
    if excess_lag_m <= 0.0:
        # The adaptation tracked at least as well as the nominal -- the arm tuck does exactly this,
        # passing through at 88-105% against the crouch's 46-67%. Nothing to pay back.
        return 1.0, False
    ratio = 1.0 - safety * excess_lag_m / weighted_window_m
    if ratio < floor:
        return floor, True
    return float(min(ratio, 1.0)), False


def slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-arc interpolation between unit quaternions, in MuJoCo's (w, x, y, z) order.

    Componentwise interpolation of a quaternion is wrong twice over: it leaves the unit sphere, and
    across a sign flip it takes the long way round, which on a turning route would swing the
    reference heading through most of a circle inside one frame.
    """
    dot = np.sum(q0 * q1, axis=1)
    q1 = np.where(dot[:, None] < 0.0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    # Nearly parallel: the slerp weights go 0/0, and a straight lerp is accurate to well under the
    # numerical noise in the source clip.
    parallel = sin_theta < 1e-8
    w0 = np.where(parallel, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(parallel, 1.0, sin_theta))
    w1 = np.where(parallel, t, np.sin(t * theta) / np.where(parallel, 1.0, sin_theta))
    out = w0[:, None] * q0 + w1[:, None] * q1
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norm > 1e-12, norm, 1.0)


def resample_clip(qpos: np.ndarray, source_index: np.ndarray) -> np.ndarray:
    """Sample a clip at fractional source frames, root and joints together.

    Root position and joint angles are interpolated linearly; the root quaternion is slerped.
    Resampling both against one clock is what keeps the swing consistent with the travel: warping
    the root alone would ask the reference feet to slide.
    """
    clip = np.asarray(qpos, dtype=np.float64)
    tau = np.asarray(source_index, dtype=np.float64)
    frames = len(clip)
    if clip.ndim != 2 or clip.shape[1] < 8:
        raise ValueError(f"expected (T, 7+J) reference qpos, got {clip.shape}")
    if frames < 2:
        raise ValueError("a clip of fewer than two frames cannot be retimed")
    if tau.min() < -1e-9 or tau.max() > frames - 1 + 1e-9:
        raise ValueError("source_index leaves the clip")

    tau = np.clip(tau, 0.0, frames - 1)
    low = np.clip(np.floor(tau).astype(int), 0, frames - 2)
    frac = tau - low
    high = low + 1

    out = np.empty((len(tau), clip.shape[1]), dtype=np.float64)
    linear = np.r_[0:3, 7 : clip.shape[1]]
    out[:, linear] = (1.0 - frac)[:, None] * clip[low][:, linear] + frac[:, None] * clip[high][
        :, linear
    ]
    out[:, 3:7] = slerp(clip[low][:, 3:7], clip[high][:, 3:7], frac)
    return out


def time_map(rate: np.ndarray) -> np.ndarray:
    """Fractional source frames to emit, one per output frame, for a per-frame pace.

    ``rate[i]`` is the fraction of nominal pace commanded while passing source frame ``i``, so
    traversing one source frame costs ``1 / rate`` output frames. The cumulative cost is the clock;
    inverting it gives the source position at each tick of that clock.

    The output length is rounded **up** to a whole number of frames and the samples spread evenly
    over the exact warped duration. That keeps three things at once: an integer frame rate, which
    the motion-lib entry requires; the exact first and last source frame, so start and goal are
    reached rather than approached; and a rounding error that is always in the slow direction,
    where it is harmless, rather than the fast one, where it would give back part of the correction.
    """
    r = np.asarray(rate, dtype=np.float64)
    if r.ndim != 1 or len(r) < 2:
        raise ValueError("rate must be one value per source frame, for at least two frames")
    if not np.all(np.isfinite(r)) or r.min() <= 0.0 or r.max() > 1.0 + 1e-12:
        raise ValueError("rate must lie in (0, 1]; retiming only ever slows a clip down")

    cost = 1.0 / r
    elapsed = np.concatenate([[0.0], np.cumsum(0.5 * (cost[:-1] + cost[1:]))])
    total = float(elapsed[-1])
    frames = int(np.ceil(total - 1e-9)) + 1
    ticks = np.linspace(0.0, total, frames)
    return np.interp(ticks, elapsed, np.arange(len(r), dtype=np.float64))


def deployable_clip(
    nominal_qpos: np.ndarray,
    adapted_qpos: np.ndarray,
    *,
    excess_lag_m: float | None = None,
    ratio: float | None = None,
    safety: float = RETIME_SAFETY,
    floor: float = MIN_RETIME_RATIO,
) -> tuple[np.ndarray, RetimeReport]:
    """Retime a causal adapted clip into a deployable one.

    Give either ``excess_lag_m`` -- the adapted clip's measured endpoint error minus the nominal's,
    from a rollout that has actually run -- or ``ratio`` directly, which is what a sweep does when
    it is looking for the boundary rather than correcting a known shortfall.

    Returns the retimed clip and a report. The report is not decoration: ``holds_route`` is the
    precondition for reusing the family's scene, and ``floored`` says the clip is expected to still
    arrive short, so a caller that ignores it will read a rejection as a fact about the operator
    when it is a fact about the request.
    """
    if (excess_lag_m is None) == (ratio is None):
        raise ValueError("give exactly one of excess_lag_m or ratio")

    nominal = np.asarray(nominal_qpos, dtype=np.float64)
    adapted = np.asarray(adapted_qpos, dtype=np.float64)
    alpha = departure_profile(nominal, adapted)
    # The adapted clip's own route, not the nominal's. For a matched pair they are the same line
    # to within the precision the clips were written at, and taking it from the clip being retimed
    # keeps the correction independent of which copy of the nominal a caller happened to load.
    window_m = weighted_route_length(adapted[:, :2], alpha)

    if ratio is None:
        chosen, floored = retime_ratio(float(excess_lag_m), window_m, safety=safety, floor=floor)
        lag = float(excess_lag_m)
    else:
        if not 0.0 < float(ratio) <= 1.0:
            raise ValueError(f"ratio must lie in (0, 1]; got {ratio}")
        chosen, floored = float(ratio), False
        lag = float("nan")

    # Slow in proportion to how active the adaptation is, so the pace change and the pose change
    # ramp together and the clip has no moment where the robot is upright but crawling.
    rate = 1.0 - (1.0 - chosen) * alpha
    tau = time_map(rate)
    retimed = resample_clip(adapted, tau)

    source_frames = len(adapted)
    frames = len(retimed)
    # linspace spread the ticks over the exact warped duration, so the realised pace is the
    # requested one diluted by however much the ceiling added.
    exact = float(np.sum(0.5 * (1.0 / rate[:-1] + 1.0 / rate[1:])))
    realised = chosen * exact / max(frames - 1, 1)

    return retimed, RetimeReport(
        source_frames=source_frames,
        frames=frames,
        duration_scale=float((frames - 1) / max(source_frames - 1, 1)),
        commanded_ratio=float(chosen),
        realised_ratio=float(realised),
        excess_lag_m=lag,
        weighted_window_m=float(window_m),
        route_deviation_m=route_deviation_m(adapted[:, :2], retimed[:, :2]),
        start_shift_m=float(np.linalg.norm(retimed[0, :3] - adapted[0, :3])),
        goal_shift_m=float(np.linalg.norm(retimed[-1, :3] - adapted[-1, :3])),
        floored=bool(floored),
        active_fraction=float((alpha > 0.01).mean()),
    )


def route_deviation_m(reference_xy: np.ndarray, retimed_xy: np.ndarray) -> float:
    """Largest distance from a retimed root position to the reference polyline, in metres.

    The question a scene asks of a retimed clip is not whether the two root traces line up frame by
    frame -- they deliberately do not -- but whether the robot still walks *the same line through
    the room*. Distance to the polyline answers that and is blind to where along it each frame
    fell, which is exactly the difference retiming introduces.
    """
    path = np.asarray(reference_xy, dtype=np.float64)
    points = np.asarray(retimed_xy, dtype=np.float64)
    starts, ends = path[:-1], path[1:]
    segment = ends - starts
    length_sq = np.maximum((segment * segment).sum(axis=1), 1e-12)
    offset = points[:, None, :] - starts[None, :, :]
    t = np.clip((offset * segment[None, :, :]).sum(axis=2) / length_sq[None, :], 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * segment[None, :, :]
    return float(np.linalg.norm(points[:, None, :] - closest, axis=2).min(axis=1).max())
