"""Decide whether a generated reference deserves a rollout, before spending one.

Every check here already existed and was run separately, late, or not at all. The cost of
that shows up twice in the corpus. `step_over` clips were generated, screened, converted,
rolled out, rendered and graded fifteen times before anyone asked whether the reference
contained a step-over -- it never did, in any of them. `crouch_walk` clips were the most
functionally useful motions ever generated, 0.395 m lower than a walk, and every one was
discarded by a saturation check whose verdict came down to a single joint nobody looked at.

So the checks are composed into one gate, applied to the reference, in about a second:

============================  ==========================================================
``reference_semantic_valid``  does the clip contain the behaviour its prompt named
``embodiment_feasible``       can the G1 reach these configurations at all
``self_collision_free``       does the body pass through itself
``functionally_separated``    is it far enough from the nominal motion to build a family
============================  ==========================================================

These are four different facts and a single accepted/rejected flag hides which one failed --
the same conflation the accepted/unevaluable split exists to prevent, one stage earlier.
A clip can be semantically perfect and physically unreachable (`crouch_walk`), or perfectly
reachable and semantically empty (`step_over`), and the two demand opposite responses:
rephrase the prompt, or retarget the motion.

**This gate is not a physics verdict.** A reference that passes may still be untrackable, and
only a rollout knows. What it guarantees is that the rollout is worth running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .behaviour_predicates import PredicateError, check_behaviour
from .motion_envelope import EnvelopeSignature, compute_envelope
from .motion_prefilter import (
    PRESIM_SATURATION_LIMIT,
    PrefilterError,
    load_joint_limits,
    screen_reference_motion,
)
from .reference_payload import payload_from_reference
from .self_intersection import (
    DEFAULT_G1_MJCF,
    SelfIntersectionError,
    check_reference_self_intersection,
)

#: Metres of separation, along the regime's discriminating axis, below which a pair cannot
#: support a counterfactual family. Matched to the mining threshold so the gate and the miner
#: agree about what "separated" means.
MIN_FUNCTIONAL_SEPARATION_M = 0.05

#: Columns in the reference layout: 3 root translation, 4 root quaternion, 29 joint DOFs.
EXPECTED_QPOS_COLUMNS = 36


@dataclass(frozen=True)
class ReferenceVerdict:
    """Four independent facts about one reference clip, plus what to do about them."""

    motion_id: str
    body_mode: str
    reference_semantic_valid: bool | None
    embodiment_feasible: bool
    self_collision_free: bool
    functionally_separated: bool | None
    signature: EnvelopeSignature | None = None
    #: Fraction of (frame, joint) cells sitting at a limit. The single number that killed
    #: every crouch clip, and the joint responsible is in ``worst_joint``.
    saturated_cell_fraction: float = 0.0
    worst_joint: str = ""
    worst_joint_fraction: float = 0.0
    separation_m: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def worth_a_rollout(self) -> bool:
        """Every hard requirement met. An unchecked semantic is not a failure, but an
        unchecked *separation* is not either -- callers screening a pair must look at
        ``functionally_separated`` themselves."""
        return (
            self.embodiment_feasible
            and self.self_collision_free
            and self.reference_semantic_valid is not False
        )

    @property
    def diagnosis(self) -> str:
        """Which stage failed, in the words that say what to do next."""
        if not self.embodiment_feasible:
            if not self.worst_joint:
                return f"could not be assessed: {'; '.join(self.notes) or 'unknown'}"
            return (
                f"unreachable on the G1: {self.worst_joint} at a limit on "
                f"{self.worst_joint_fraction:.0%} of frames -- retarget, do not rephrase"
            )
        if not self.self_collision_free:
            return "the body passes through itself -- retarget"
        if self.reference_semantic_valid is False:
            return "the clip does not contain the behaviour its prompt named -- rephrase"
        if self.functionally_separated is False:
            return (
                f"only {self.separation_m:.3f} m from the nominal motion -- usable, but it "
                "cannot anchor a family"
            )
        if self.reference_semantic_valid is None:
            return "worth a rollout; no predicate exists for this behaviour"
        return "worth a rollout"


def screen_reference(
    reference_qpos: np.ndarray,
    motion_id: str,
    body_mode: str,
    *,
    nominal: EnvelopeSignature | None = None,
    regime: str | None = None,
    fps: float = 30.0,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    saturation_limit: float = PRESIM_SATURATION_LIMIT,
) -> ReferenceVerdict:
    """Run every cheap check on one reference clip.

    ``nominal`` and ``regime`` are optional: supply them to also ask whether this clip is far
    enough from a nominal motion to anchor a counterfactual family in that regime.
    """
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    notes: list[str] = []

    # A gate must fail closed. A clip with the wrong column count passes the saturation
    # check -- it finds whatever joints are present and none of them are at a limit -- while
    # every other check errors out, and the verdict came back "feasible". Shape is therefore
    # established before anything is asked about the contents.
    if qpos.ndim != 2 or qpos.shape[1] != EXPECTED_QPOS_COLUMNS:
        return ReferenceVerdict(
            motion_id=motion_id, body_mode=body_mode, reference_semantic_valid=None,
            embodiment_feasible=False, self_collision_free=False,
            functionally_separated=None,
            notes=(
                f"expected (T, {EXPECTED_QPOS_COLUMNS}) reference qpos, got {qpos.shape}",
            ),
        )

    # --- embodiment: can the robot reach these configurations at all ---------------------
    try:
        joint_names, joint_limits = load_joint_limits(mjcf_path)
        prefilter = screen_reference_motion(
            qpos, joint_names, joint_limits, saturation_limit=saturation_limit
        )
        feasible = prefilter.passed
        saturated = prefilter.saturated_cell_fraction
        worst, worst_fraction = (
            prefilter.worst_joints[0] if prefilter.worst_joints else ("", 0.0)
        )
        notes.extend(prefilter.reasons)
    except (PrefilterError, OSError, ValueError) as error:
        return ReferenceVerdict(
            motion_id=motion_id, body_mode=body_mode, reference_semantic_valid=None,
            embodiment_feasible=False, self_collision_free=False,
            functionally_separated=None, notes=(f"prefilter failed: {error}",),
        )

    # --- self-collision ------------------------------------------------------------------
    try:
        intersection = check_reference_self_intersection(qpos, mjcf_path=mjcf_path)
        collision_free = intersection.passed
        notes.extend(intersection.reasons)
    except (SelfIntersectionError, OSError, ValueError) as error:
        collision_free = False
        notes.append(f"self-intersection check failed: {error}")

    # --- semantics and envelope, both from one forward-kinematics pass --------------------
    semantic: bool | None = None
    signature: EnvelopeSignature | None = None
    try:
        payload = payload_from_reference(qpos, fps=fps, mjcf_path=mjcf_path)
        signature = compute_envelope(payload, motion_id, body_mode)
        result = check_behaviour(body_mode, payload)
        if result is not None:
            semantic = bool(result.satisfied)
            notes.append(result.reason)
    except (PredicateError, SelfIntersectionError, KeyError, ValueError) as error:
        notes.append(f"could not grade the reference: {error}")

    # --- functional separation from the nominal motion ------------------------------------
    separated: bool | None = None
    separation: float | None = None
    if nominal is not None and regime is not None and signature is not None:
        from .motion_envelope import REGIME_FIELD

        if regime not in REGIME_FIELD:
            raise ValueError(f"unknown regime {regime!r}")
        field_name, direction = REGIME_FIELD[regime]
        separation = float(
            direction * (getattr(signature, field_name) - getattr(nominal, field_name))
        )
        separated = bool(
            np.isfinite(separation) and separation >= MIN_FUNCTIONAL_SEPARATION_M
        )

    return ReferenceVerdict(
        motion_id=motion_id,
        body_mode=body_mode,
        reference_semantic_valid=semantic,
        embodiment_feasible=feasible,
        self_collision_free=collision_free,
        functionally_separated=separated,
        signature=signature,
        saturated_cell_fraction=saturated,
        worst_joint=worst,
        worst_joint_fraction=worst_fraction,
        separation_m=separation,
        notes=tuple(notes),
    )
