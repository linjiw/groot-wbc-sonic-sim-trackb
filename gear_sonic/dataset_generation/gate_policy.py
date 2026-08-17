"""Gate an episode according to what its label actually is.

The acceptance gates grew up when every episode was a Kimodo reference tracked as closely
as possible, so "did the robot stay near the reference" was a reasonable proxy for "is this
episode any good". With a varied behaviour library that proxy fails in a measurable way:
acceptance became a function of episode *length* rather than of behaviour. Grading the same
episodes at longer horizons gives 83% at 1.2 s falling to 64% at 4.8 s; duration correlates
with rejection at r = -0.47; and reference-path error caused 40 of 54 rejections. Scaling
under that gate would bake two biases into the corpus — over-representation of short clips
and of early-episode dynamics, and behaviour families penalised for prompt length rather
than difficulty.

The fix follows from asking what the label is, which depends on how the episode was made:

* **scene-around-motion.** The room is built around the corridor the robot *executed*, so
  the scene explains the path the robot actually took. The reference was a means of
  producing behaviour, not the label. Path-versus-reference error is a tracking-quality
  diagnostic here, not a verdict.
* **scene-first.** The route was planned through fixed geometry and *is* the label, so
  reaching the planned goal is the thing being tested. Corridor-hugging p95 over a whole
  episode is still the wrong instrument; goal-region success plus stability is right.

Both directions keep every safety and validity gate — contact, support, falls, and whether
the episode contains locomotion at all. Nothing about this relaxes what the corpus promises
about collisions.

Measured on 164 evaluable episodes: dropping the path and endpoint gates for
scene-around-motion promotes **38 episodes**, taking acceptance from 62% to 85%. Every one
of those 38 has **0.0000 N** of lateral scene contact, foot support at or above 0.996, root
height at or above 0.544 m and tilt at or below 0.574 rad. They drifted from a reference and
hit nothing. The 24 that remain rejected fail for self-contact, scene contact, foot contact,
absent motion, or not being locomotion — none of which is about the reference.

Among the promoted are the winding motions (`05_root_path`, `06_root_waypoints`), which are
the most navigationally interesting episodes in the corpus and were being discarded for
missing a reference by centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Gates that measure agreement with the Kimodo reference rather than the episode's safety
#: or validity. Whether these bind depends on the generation direction.
REFERENCE_GATES = frozenset(
    {"reference_path_tracking_error", "reference_endpoint_tracking_error"}
)

#: Divergence of the reference-tracking error over the episode's second half, in m/s. The
#: second half is used because the first contains a settling transient that says nothing
#: about stability. Measured: episodes that are otherwise clean reach at most 0.139 m/s,
#: while episodes rejected for safety reasons reach 0.638 m/s.
#:
#: Set as a **backstop, not a discriminator**. On the current corpus it rejects nothing that
#: the safety gates do not already catch; it exists for the failure mode where a robot walks
#: away stably in the wrong direction and every other gate stays happy.
DEFAULT_MAX_DRIFT_RATE_MPS = 0.15

#: How close the executed endpoint must land to a planned goal, for scene-first episodes
#: where the planned route is the label. Provisional -- no scene-first episode has been
#: rolled out yet, so there is no measurement behind this number and it says so.
DEFAULT_GOAL_RADIUS_M = 0.75


class GatePolicyError(ValueError):
    """Raised when an episode cannot be graded under the requested policy."""


@dataclass(frozen=True)
class GatePolicy:
    """Which gates bind, for episodes made a particular way."""

    name: str
    #: True when the planned/commanded route is the episode's label, so departing from it
    #: is a failure rather than a diagnostic.
    reference_is_label: bool
    max_drift_rate_mps: float = DEFAULT_MAX_DRIFT_RATE_MPS
    goal_radius_m: float = DEFAULT_GOAL_RADIUS_M
    #: Gates demoted to diagnostics under this policy.
    demoted_gates: frozenset[str] = field(default_factory=frozenset)


#: The room was built around the executed corridor, so the executed trajectory is the label.
SCENE_AROUND_MOTION = GatePolicy(
    name="scene_around_motion",
    reference_is_label=False,
    demoted_gates=REFERENCE_GATES,
)

#: The route was planned through fixed geometry, so reaching the planned goal is the test.
#: Path-hugging is still not, which is why only the endpoint gate stays.
SCENE_FIRST = GatePolicy(
    name="scene_first",
    reference_is_label=True,
    demoted_gates=frozenset({"reference_path_tracking_error"}),
)

POLICIES = {policy.name: policy for policy in (SCENE_AROUND_MOTION, SCENE_FIRST)}


@dataclass(frozen=True)
class PolicyOutcome:
    """The verdict under a policy, with what was demoted rather than dropped."""

    policy: str
    accepted: bool
    rejection_reasons: tuple[str, ...]
    #: Gate failures that did not count under this policy, kept so provenance records that
    #: the episode departed from its reference even though that was not disqualifying.
    diagnostics: dict[str, float | bool | str] = field(default_factory=dict)
    demoted_failures: tuple[str, ...] = ()


def drift_rate_mps(
    executed_xy: np.ndarray, reference_xy: np.ndarray, fps: float
) -> float:
    """Divergence of tracking error over the episode's second half, in m/s.

    Chosen over the whole-episode slope after comparing candidates on 164 episodes: the
    whole-episode slope is nearly length-independent (r = +0.04 with duration, against
    +0.29 for absolute p95) but separates poorly, because an episode that fails early with a
    large constant offset has a small slope. Restricting to the second half skips the
    settling transient and gives the largest median separation of any candidate tried --
    0.0029 m/s for accepted episodes against 0.0574 m/s for those rejected on safety.
    """
    executed = np.asarray(executed_xy, dtype=np.float64).reshape(-1, 2)
    reference = np.asarray(reference_xy, dtype=np.float64).reshape(-1, 2)
    frames = min(len(executed), len(reference))
    if frames < 20:
        raise GatePolicyError(f"need at least 20 frames to fit a drift rate, got {frames}")
    if not np.isfinite(fps) or fps <= 0:
        raise GatePolicyError(f"fps must be positive and finite, got {fps}")

    error = np.linalg.norm(executed[:frames] - reference[:frames], axis=1)
    seconds = np.arange(frames) / float(fps)
    half = frames // 2
    slope, _ = np.polyfit(seconds[half:], error[half:], 1)
    return float(slope)


def endpoint_error_m(executed_xy: np.ndarray, goal_xy: np.ndarray) -> float:
    """Distance from where the robot stopped to where it was supposed to."""
    executed = np.asarray(executed_xy, dtype=np.float64).reshape(-1, 2)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    if executed.size == 0:
        raise GatePolicyError("executed path is empty")
    return float(np.linalg.norm(executed[-1] - goal))


def apply_policy(
    base_reasons: tuple[str, ...] | list[str],
    payload: dict,
    policy: GatePolicy = SCENE_AROUND_MOTION,
    *,
    planned_goal_xy: np.ndarray | None = None,
) -> PolicyOutcome:
    """Re-grade a base acceptance result under a generation-direction policy.

    ``base_reasons`` are the gate names that failed in the underlying evaluation. Gates the
    policy demotes are removed from the verdict and recorded as diagnostics; the policy's
    own gates are then applied on top.
    """
    reasons = [r for r in base_reasons if r not in policy.demoted_gates]
    demoted = tuple(r for r in base_reasons if r in policy.demoted_gates)
    diagnostics: dict[str, float | bool | str] = {"policy": policy.name}

    executed = np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, :2]
    fps = float(payload.get("fps", 50.0))

    try:
        rate = drift_rate_mps(executed, reference, fps)
        diagnostics["drift_rate_mps"] = rate
        if rate > policy.max_drift_rate_mps:
            reasons.append("unstable_reference_drift")
    except GatePolicyError as error:
        # Too short to fit a rate. That is not a pass: an episode that cannot be assessed
        # for stability should not be certified as stable.
        diagnostics["drift_rate_error"] = str(error)
        reasons.append("drift_rate_unmeasurable")

    if policy.reference_is_label:
        if planned_goal_xy is None:
            raise GatePolicyError(
                f"policy {policy.name!r} grades against a planned goal, which was not given"
            )
        error_m = endpoint_error_m(executed, planned_goal_xy)
        diagnostics["goal_error_m"] = error_m
        if error_m > policy.goal_radius_m:
            reasons.append("planned_goal_not_reached")

    return PolicyOutcome(
        policy=policy.name,
        accepted=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        diagnostics=diagnostics,
        demoted_failures=demoted,
    )
