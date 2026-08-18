"""Ask whether the robot did the behaviour, not merely whether it survived doing something.

Acceptance answers "did the robot track its reference safely". That is necessary and it is
not the question a benchmark asks. An episode labelled ``side_step`` that turns and walks
forward is safe, well-tracked, collision-free, and mislabelled -- and no gate in the corpus
notices.

The review page already contains the specification. Under each behaviour card sits a
sentence written for a human reviewer: *confirm the robot actually stops*, *watch the
trailing foot*, *the reach should happen after the walk*, *check it is not just turning*.
Those are predicates. This module is them in code, so the episodes nobody will watch are
still checked.

Validity is therefore four separate questions, and collapsing them into one flag loses the
distinction that matters:

``physics_valid``      no disallowed contact, the robot stayed up and supported
``tracking_valid``     the controller followed what it was given, stably
``semantic_valid``     **the executed motion is the behaviour its label claims**
``scene_task_valid``   the episode achieved its goal in this scene

Only the third is here. Each predicate returns a verdict with the measurement behind it, so
a failure says *why* rather than merely that. Thresholds are stated as constants with the
reasoning attached; several are provisional and say so, because no reviewed sample has yet
calibrated them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Sequence

import numpy as np

#: Speed below which the root counts as stopped, in m/s. A standing G1 still shows small
#: root motion from balance, so this is not zero.
STOP_SPEED_MPS = 0.12

#: How long a stop must last to be a pause rather than a hesitation, in seconds.
MIN_PAUSE_S = 0.30

#: Heading change that counts as a turn, in radians. Below this a "turn in place" is drift.
MIN_TURN_RAD = math.radians(60)

#: Root translation a turn-in-place may accumulate before it is a turn *and a walk*.
MAX_TURN_TRANSLATION_M = 0.60

#: Fraction of total displacement that must be lateral for a side-step to be a side-step
#: rather than a turn-and-walk wearing its label.
MIN_LATERAL_FRACTION = 0.60

#: Heading a side-step may swing through while still "facing forward", in radians.
MAX_SIDESTEP_HEADING_RAD = math.radians(45)

#: Drop in torso height that counts as ducking, relative to the episode's own standing
#: height. Relative rather than absolute because the G1's standing height varies with gait.
MIN_DUCK_DROP_M = 0.08

#: Swing apex a foot must reach above its own stance height to count as stepping *over*
#: something rather than walking past it.
#:
#: Calibrated against a control, which is the check that matters: over accepted episodes the
#: trailing foot's apex is 0.125-0.153 m for a plain walk (n=14, median 0.136) and
#: 0.131-0.145 m for episodes labelled step_over (n=7, median 0.138). Mann-Whitney
#: p = 0.632 -- the labelled step-overs are statistically indistinguishable from walking.
#: The threshold sits above the walking population's maximum, so a motion that clears it is
#: doing something walking does not.
MIN_STEP_APEX_M = 0.18

#: The same criterion for a *reference* clip rather than an executed one. Two thresholds are
#: needed because tracking systematically lowers the foot: across the corpus the trailing
#: foot's apex drops 0.034 m from reference to execution for a plain walk and 0.039 m for a
#: clip labelled step_over -- the controller flattens everything, not step-overs selectively.
#: A single absolute number therefore cannot serve both domains, and 0.18 m happens to sit
#: between them, so it passes nearly every reference and fails nearly every rollout. That
#: looked like the generator producing a step-over that the controller then lost. It is not:
#: measured on references, step_over apex is 0.193-0.226 m against a plain walk's
#: 0.206-0.222 m, Mann-Whitney p = 0.693. The behaviour is absent in both domains.
#:
#: This threshold sits just above the observed reference walking range, so a clip has to
#: clear a walk's swing to count.
REFERENCE_STEP_APEX_M = 0.24

#: Metres the half-width must fall below the episode's own walking width for a narrowing to
#: count. Relative, not absolute, for the reason stand_to_walk taught: the G1's half-width
#: runs 0.273 m tucked to 0.664 m at peak arm swing, so a single absolute threshold would
#: pass wide motions that never narrow and fail narrow ones that do.
MIN_WIDTH_REDUCTION_M = 0.06

#: Provisional. No reviewed sample has calibrated these yet; they are first estimates chosen
#: to be permissive, so a false reject is less likely than a false accept.
PROVISIONAL = frozenset({"walk_to_stop", "walk_and_reach", "carry"})


class PredicateError(ValueError):
    """Raised when a predicate cannot be evaluated from the episode given."""


@dataclass(frozen=True)
class PredicateResult:
    """Whether the episode is the behaviour it claims, and the measurement behind it."""

    behaviour: str
    satisfied: bool
    reason: str
    measurements: dict[str, float] = field(default_factory=dict)
    provisional: bool = False


def _root_speed(payload: dict) -> tuple[np.ndarray, float]:
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    fps = float(payload.get("fps", 50.0))
    if len(root) < 3:
        raise PredicateError(f"need at least 3 frames, got {len(root)}")
    return np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1) * fps, fps


def _heading(payload: dict) -> np.ndarray:
    """Body heading per frame, from the root quaternion's yaw."""
    quat = np.asarray(payload["root_quat_w"], dtype=np.float64)
    w, x, y, z = quat.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _body_height(payload: dict, name: str = "torso_link") -> np.ndarray:
    names = list(payload["body_names"])
    if name not in names:
        raise PredicateError(f"{name!r} not among the recorded bodies")
    return np.asarray(payload["body_pos_w"], dtype=np.float64)[:, names.index(name), 2]


def _half_width_series(payload: dict) -> np.ndarray:
    """Half-width per frame, measured across the direction of travel.

    Across the heading rather than the world y axis: a motion walking along +y is not two
    metres wide, and using the world frame would report exactly that. Bodies are taken from
    the recorded link origins rather than the collision capsules, which keeps this a cheap
    check that any recording with body positions can answer.
    """
    if "body_pos_w" not in payload or payload["body_pos_w"] is None:
        raise PredicateError("no per-body positions recorded, so width cannot be measured")
    bodies = np.asarray(payload["body_pos_w"], dtype=np.float64)
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    yaw = _heading(payload)
    lateral_axis = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    offsets = bodies[:, :, :2] - root[:, None, :2]
    return np.abs(np.einsum("tbd,td->tb", offsets, lateral_axis)).max(axis=1)


def check_pause(payload: dict) -> PredicateResult:
    """The pause is the point: confirm the robot actually stops, then resumes."""
    speed, fps = _root_speed(payload)
    stopped = speed < STOP_SPEED_MPS
    # Longest run of stopped frames.
    longest = current = 0
    end = 0
    for index, is_stopped in enumerate(stopped):
        current = current + 1 if is_stopped else 0
        if current > longest:
            longest, end = current, index
    duration = longest / fps
    resumed = bool(end + 1 < len(speed) and speed[end + 1 :].max() > STOP_SPEED_MPS * 2)
    satisfied = duration >= MIN_PAUSE_S and resumed
    return PredicateResult(
        behaviour="pause",
        satisfied=satisfied,
        reason=(
            "stopped and resumed" if satisfied
            else f"longest stop {duration:.2f} s (need {MIN_PAUSE_S:.2f})"
            if duration < MIN_PAUSE_S else "stopped but never resumed"
        ),
        measurements={"pause_s": duration, "min_speed_mps": float(speed.min())},
    )


def check_turn_in_place(payload: dict) -> PredicateResult:
    """Check the feet pivot rather than the root sliding across the floor."""
    heading = np.unwrap(_heading(payload))
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    turned = float(abs(heading[-1] - heading[0]))
    # Displacement during the turning portion only: a turn followed by a walk is fine.
    turn_end = int(np.argmax(np.abs(heading - heading[0]) > MIN_TURN_RAD * 0.9))
    turn_end = turn_end if turn_end > 0 else len(root) - 1
    translation = float(np.linalg.norm(root[turn_end, :2] - root[0, :2]))
    satisfied = turned >= MIN_TURN_RAD and translation <= MAX_TURN_TRANSLATION_M
    return PredicateResult(
        behaviour="turn_in_place",
        satisfied=satisfied,
        reason=(
            "turned in place" if satisfied
            else f"heading change {turned:.2f} rad (need {MIN_TURN_RAD:.2f})"
            if turned < MIN_TURN_RAD
            else f"translated {translation:.2f} m while turning (max {MAX_TURN_TRANSLATION_M})"
        ),
        measurements={"heading_change_rad": turned, "turn_translation_m": translation},
    )


def check_side_step(payload: dict) -> PredicateResult:
    """Lateral travel must dominate, and the body must keep facing forward.

    Without the heading check a turn-and-walk satisfies "moved sideways relative to the
    start" perfectly well, which is exactly the disguise the review card warns about.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    heading = np.unwrap(_heading(payload))
    start_heading = float(heading[0])
    displacement = root[-1, :2] - root[0, :2]
    forward = np.array([math.cos(start_heading), math.sin(start_heading)])
    lateral = np.array([-forward[1], forward[0]])
    total = float(np.linalg.norm(displacement))
    if total < 1e-6:
        raise PredicateError("the robot did not move; a side-step cannot be assessed")
    lateral_fraction = float(abs(displacement @ lateral) / total)
    heading_swing = float(np.abs(heading - start_heading).max())
    satisfied = (
        lateral_fraction >= MIN_LATERAL_FRACTION
        and heading_swing <= MAX_SIDESTEP_HEADING_RAD
    )
    return PredicateResult(
        behaviour="side_step",
        satisfied=satisfied,
        reason=(
            "stepped sideways while facing forward" if satisfied
            else f"only {lateral_fraction:.0%} of travel was lateral "
                 f"(need {MIN_LATERAL_FRACTION:.0%})"
            if lateral_fraction < MIN_LATERAL_FRACTION
            else f"heading swung {heading_swing:.2f} rad -- this is a turn, not a side-step"
        ),
        measurements={
            "lateral_fraction": lateral_fraction,
            "heading_swing_rad": heading_swing,
        },
    )


def check_duck_under(payload: dict, *, shelf_z: float | None = None) -> PredicateResult:
    """The torso must actually drop, and come back up afterwards.

    When the shelf height is known the clearance is checked against it; without one the
    predicate can only confirm that a duck happened, not that it was deep enough for any
    particular obstacle -- and it says which of the two it did.
    """
    torso = _body_height(payload)
    standing = float(np.percentile(torso, 90))
    lowest = float(torso.min())
    drop = standing - lowest
    lowest_frame = int(np.argmin(torso))
    recovered = bool(
        lowest_frame + 1 < len(torso)
        and torso[lowest_frame + 1 :].max() >= standing - MIN_DUCK_DROP_M * 0.5
    )
    satisfied = drop >= MIN_DUCK_DROP_M and recovered
    measurements = {"duck_drop_m": drop, "lowest_torso_m": lowest, "standing_torso_m": standing}
    reason = (
        "ducked and recovered" if satisfied
        else f"torso dropped only {drop:.3f} m (need {MIN_DUCK_DROP_M})"
        if drop < MIN_DUCK_DROP_M else "ducked but never came back up"
    )
    if shelf_z is not None:
        measurements["shelf_clearance_m"] = shelf_z - lowest
        if satisfied and lowest >= shelf_z:
            satisfied = False
            reason = f"ducked, but the torso never got below the shelf at {shelf_z:.3f} m"
    return PredicateResult("duck_under", satisfied, reason, measurements)


def check_walk_to_stop(payload: dict) -> PredicateResult:
    """Decelerating into a hold, without overshooting and drifting on."""
    speed, fps = _root_speed(payload)
    hold_frames = max(int(0.4 * fps), 3)
    if len(speed) <= hold_frames:
        raise PredicateError("episode too short to contain a stop")
    final = float(speed[-hold_frames:].max())
    peak = float(speed.max())
    satisfied = final < STOP_SPEED_MPS and peak > STOP_SPEED_MPS * 3
    return PredicateResult(
        behaviour="walk_to_stop",
        satisfied=satisfied,
        reason=(
            "walked, then held still" if satisfied
            else f"still moving at {final:.3f} m/s at the end"
            if final >= STOP_SPEED_MPS else "never walked fast enough to be stopping"
        ),
        measurements={"final_speed_mps": final, "peak_speed_mps": peak},
        provisional=True,
    )


#: A start-from-rest is recognised by how slow the opening is *relative to the episode's own
#: peak*, not by an absolute speed. Measured over five accepted starts, the opening 0.3 s
#: peaks at 0.13-0.16 m/s while the episode goes on to reach 1.25-1.79 m/s -- so an absolute
#: threshold of 0.12 m/s false-rejected every one of them while the behaviour was plainly
#: present. The ratio separates them by an order of magnitude.
MAX_START_SPEED_FRACTION = 0.25


def check_stand_to_walk(payload: dict) -> PredicateResult:
    """Starting from rest: the first step is where a tracker usually slips.

    Judged on the opening speed as a fraction of the episode's own peak, because an
    absolute threshold cannot separate "still" from "walking slowly" across motions whose
    peak speed varies threefold.
    """
    speed, fps = _root_speed(payload)
    lead = max(int(0.3 * fps), 3)
    if len(speed) <= lead:
        raise PredicateError("episode too short to contain a start")
    initial = float(speed[:lead].max())
    later = float(speed[lead:].max())
    if later <= 0.0:
        raise PredicateError("the robot never moved; a start cannot be assessed")
    fraction = initial / later
    satisfied = fraction < MAX_START_SPEED_FRACTION and later > STOP_SPEED_MPS * 3
    return PredicateResult(
        behaviour="stand_to_walk",
        satisfied=satisfied,
        reason=(
            "started from rest" if satisfied
            else f"opened at {fraction:.0%} of its peak speed "
                 f"(need under {MAX_START_SPEED_FRACTION:.0%})"
            if fraction >= MAX_START_SPEED_FRACTION else "never got going"
        ),
        measurements={
            "initial_speed_mps": initial,
            "peak_speed_mps": later,
            "opening_fraction": fraction,
        },
    )


def _foot_heights(payload: dict) -> dict[str, np.ndarray]:
    names = list(payload["body_names"])
    feet = {n: i for i, n in enumerate(names) if "ankle_roll" in n}
    if len(feet) < 2:
        raise PredicateError(f"expected two ankle_roll links, found {sorted(feet)}")
    bodies = np.asarray(payload["body_pos_w"], dtype=np.float64)
    return {name: bodies[:, index, 2] for name, index in feet.items()}


def _step_apex_threshold(payload: dict) -> float:
    """Pick the apex threshold for the domain this payload came from.

    A reference clip carries ``kind == "reference"``; a recorded trajectory does not. Getting
    this wrong does not produce a small error, it produces the opposite verdict.
    """
    return (
        REFERENCE_STEP_APEX_M if payload.get("kind") == "reference" else MIN_STEP_APEX_M
    )


def check_step_over(
    payload: dict, *, obstacle_x: float | None = None, obstacle_top_z: float | None = None
) -> PredicateResult:
    """Both feet must clear, and the trailing one is the one that fails.

    The review card is explicit about this: *watch the trailing foot, not the leading one*.
    A motion that lifts the leading foot high and drags the trailing one is the characteristic
    failure, and checking only the maximum over both feet would pass it, because the leading
    foot's apex is the maximum.

    With the obstacle's position the check becomes a real clearance test at the crossing
    frame. Without it, the predicate can only confirm that both feet swung high, and says
    which of the two it did.
    """
    feet = _foot_heights(payload)
    stance = {name: float(np.percentile(height, 10)) for name, height in feet.items()}
    apex = {name: float(height.max()) - stance[name] for name, height in feet.items()}
    lowest_apex_foot = min(apex, key=apex.get)
    lowest_apex = apex[lowest_apex_foot]

    threshold = _step_apex_threshold(payload)
    measurements = {f"apex_{name}_m": value for name, value in apex.items()}
    measurements["apex_threshold_m"] = threshold
    satisfied = lowest_apex >= threshold
    reason = (
        "both feet cleared" if satisfied
        else f"{lowest_apex_foot} only rose {lowest_apex:.3f} m (need {threshold})"
    )

    if obstacle_x is not None and obstacle_top_z is not None:
        bodies = np.asarray(payload["body_pos_w"], dtype=np.float64)
        names = list(payload["body_names"])
        for name in feet:
            xs = bodies[:, names.index(name), 0]
            crossings = np.flatnonzero(np.diff(np.sign(xs - obstacle_x)))
            if crossings.size == 0:
                satisfied = False
                reason = f"{name} never crossed the obstacle at x={obstacle_x:.2f}"
                break
            frame = int(crossings[0])
            clearance = float(feet[name][frame] - obstacle_top_z)
            measurements[f"clearance_{name}_m"] = clearance
            if clearance < 0.0:
                satisfied = False
                reason = f"{name} passed {abs(clearance):.3f} m below the obstacle top"
                break
        else:
            if satisfied:
                reason = "both feet cleared the obstacle"

    return PredicateResult("step_over", satisfied, reason, measurements)


#: Behaviours with a predicate, and the callable that checks them. A behaviour absent here
#: has no semantic check yet, which is reported rather than silently passed.
def check_narrow_pass(payload: dict, *, gap_half_width_m: float | None = None) -> PredicateResult:
    """The silhouette must actually narrow, and open out again afterwards.

    **The two modes answer different questions and must not be swapped.** Without
    ``gap_half_width_m`` this asks whether a narrowing *happened* -- a semantic check on the
    label. With one, it asks whether the robot got *inside a specific gap* -- a geometric
    check on utility. A motion can pass the first and fail the second, and ``side_step`` is
    exactly that motion: across the corpus its within-episode width reduction is far larger
    than a plain walk's (median 0.189 m against 0.089 m, p = 0.007), and yet its absolute
    narrowest half-width is *larger* (median 0.277 m against 0.227 m). It narrows sharply
    from a wider stance and never gets below where a walk already sits.

    So side-stepping is a real behaviour that is useless to the lateral geometry regime, and
    a gap that stops a walk stops a side-step too. ``arm_tuck`` and ``shoulder_turn`` exist
    to ask for the absolute width the regime needs.

    Width is taken across the direction of travel, so a motion walking along +y is not
    reported as two metres wide.
    """
    widths = _half_width_series(payload)
    walking = float(np.percentile(widths, 90))
    narrowest = float(widths.min())
    reduction = walking - narrowest
    narrowest_frame = int(np.argmin(widths))
    recovered = bool(
        narrowest_frame + 1 < len(widths)
        and widths[narrowest_frame + 1:].max() >= walking - MIN_WIDTH_REDUCTION_M * 0.5
    )

    measurements = {
        "width_reduction_m": reduction,
        "narrowest_half_width_m": narrowest,
        "walking_half_width_m": walking,
    }
    if gap_half_width_m is not None:
        measurements["gap_clearance_m"] = gap_half_width_m - narrowest
        satisfied = narrowest < gap_half_width_m and recovered
        reason = (
            "narrowed enough for the gap and opened out again" if satisfied
            else f"narrowest half-width {narrowest:.3f} m never got inside the "
                 f"{gap_half_width_m:.3f} m gap"
            if narrowest >= gap_half_width_m else "fitted the gap but never opened out again"
        )
    else:
        satisfied = reduction >= MIN_WIDTH_REDUCTION_M and recovered
        reason = (
            "narrowed and opened out again" if satisfied
            else f"half-width fell only {reduction:.3f} m (need {MIN_WIDTH_REDUCTION_M})"
            if reduction < MIN_WIDTH_REDUCTION_M else "narrowed but never opened out again"
        )

    return PredicateResult(
        behaviour="narrow_pass",
        satisfied=satisfied,
        reason=reason,
        measurements=measurements,
        provisional=True,
    )


PREDICATES: dict[str, Callable[[dict], PredicateResult]] = {
    "walk_pause": check_pause,
    # "walk_look" deliberately has no entry. It was wired to check_pause, which reported
    # every one of its four accepted episodes as mislabelled -- but "pauses and looks
    # around" is a head/torso yaw excursion while the root keeps traversing, and root speed
    # cannot see it. Reusing the wrong predicate manufactured a finding. A real one needs
    # head orientation relative to the root, which is recorded and not yet implemented.
    "turn_in_place": check_turn_in_place,
    "side_step": check_side_step,
    "duck_under": check_duck_under,
    "walk_to_stop": check_walk_to_stop,
    "stand_to_walk": check_stand_to_walk,
    "step_over": check_step_over,
    "arm_tuck": check_narrow_pass,
    "shoulder_turn": check_narrow_pass,
}


def check_behaviour(behaviour: str, payload: dict, **kwargs) -> PredicateResult | None:
    """Run the predicate for a behaviour, or return None when none exists yet.

    None is deliberately not a pass. A caller reporting semantic validity must distinguish
    "checked and correct" from "nobody has written the check", which is the same distinction
    the accepted/unevaluable split makes elsewhere.
    """
    predicate = PREDICATES.get(behaviour)
    if predicate is None:
        return None
    result = predicate(payload, **kwargs)
    if behaviour in PROVISIONAL and not result.provisional:
        result = PredicateResult(
            result.behaviour, result.satisfied, result.reason, result.measurements, True
        )
    return result


def coverage(behaviours: Sequence[str]) -> dict[str, bool]:
    """Which behaviours have a semantic check and which are still on the honour system."""
    return {behaviour: behaviour in PREDICATES for behaviour in behaviours}
