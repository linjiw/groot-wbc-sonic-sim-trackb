"""What a whole-body adaptation costs, so "minimal adaptation" can be measured.

The counterfactual family is more than a matched negative. In the easy scene *both* motions
succeed, and yet they are not equally good: crouching through an open room spends joint
travel, pelvis height and time for nothing. In the hard scene only the crouch survives. So
the scene does not merely forbid a motion, it **reverses the ranking** of the two, and the
learning problem that follows is not "will this collide" but

    m* = argmin_m  C(m)     subject to    P(success | S, m) >= tau

Everything on the right of that is measured elsewhere. This module is the left: a cost that
says how far a motion departs from ordinary walking, so that a policy which crouches in an
empty corridor can be scored as wrong even though nothing hits it.

**This is a preference sanity check, not a reportable quantity.** The ratio it produces --
1.41x for the crouch over the walk on the measured family -- depends on the weights, on how
each component is normalised, on whether duration and tracking error are included, and on
which motion is taken as nominal. None of those are frozen, so the number does not belong in
an abstract or a headline result. What it is good for is confirming the *sign*: where both
motions succeed, the walk should come out cheaper, and it does.

A safer decision rule while the weights are unfrozen is lexicographic: require task success
first, and among successful motions prefer the smallest deviation from nominal. There is also
a circularity to avoid -- once an adaptation operator minimises a cost, that same cost cannot
then be used to argue the operator produces minimal adaptation.

**The weights are a stated convention, not a measurement.** No reviewed sample has calibrated
them, and the ranking they produce should be checked against the one thing that is not a
convention: whichever motion a person would call the obvious choice in an open room. Every
component is reported alongside the total so a reader can reweight without rerunning
anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

#: Weights over the normalised components. Deliberately blunt: the pelvis term dominates
#: because lowering the body is the most visible and most expensive adaptation the G1 makes,
#: and duration is included because a slower traverse of the same route is a real cost to a
#: robot that has somewhere to be.
DEFAULT_WEIGHTS = {
    "pelvis_lowering": 3.0,
    "joint_travel": 1.0,
    "postural_deviation": 2.0,
    "duration": 0.5,
}

#: Pelvis height of the G1 walking normally, in metres, measured over the corpus's plain
#: walks. Lowering is scored against *this*, not against each motion's own upright height.
#:
#: Scoring against the motion's own height looks more robust and is wrong in the one case
#: that matters most: a clip crouched for its entire duration never rises, so its own 90th
#: percentile *is* the crouch, and the most expensive adaptation available scores zero. A
#: sustained crouch is the adaptation this whole line of work is built on, so the reference
#: has to come from outside the clip. Pass ``reference_pelvis_m`` for a different robot.
NOMINAL_PELVIS_HEIGHT_M = 0.75


@dataclass(frozen=True)
class AdaptationCost:
    """How far one motion departs from ordinary walking, component by component."""

    motion_id: str
    #: Mean drop of the pelvis below a normal walking height, in metres. Measured against an
    #: external reference, so a clip crouched throughout scores its full depth.
    pelvis_lowering_m: float
    #: The clip's own upright pelvis height, reported so the reference can be sanity-checked.
    pelvis_height_m: float
    #: Summed absolute joint motion per second, in radians -- how much the body works.
    joint_travel_rad_per_s: float
    #: RMS joint deviation from the motion's own median pose, in radians. A walk returns to
    #: the same posture every stride; a crouch holds a different one.
    postural_deviation_rad: float
    duration_s: float
    path_length_m: float
    total: float

    def components(self) -> dict[str, float]:
        return {
            "pelvis_lowering_m": self.pelvis_lowering_m,
            "pelvis_height_m": self.pelvis_height_m,
            "joint_travel_rad_per_s": self.joint_travel_rad_per_s,
            "postural_deviation_rad": self.postural_deviation_rad,
            "duration_s": self.duration_s,
        }


def adaptation_cost(
    payload: Mapping,
    motion_id: str = "",
    *,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
    reference_pose: np.ndarray | None = None,
    reference_pelvis_m: float = NOMINAL_PELVIS_HEIGHT_M,
) -> AdaptationCost:
    """Cost of one executed or reference motion.

    ``reference_pose`` is an optional joint vector standing for "ordinary walking"; without
    it the postural term is measured against the motion's own median pose, which still
    separates a sustained crouch from a walk because a walk returns to its median every
    stride and a crouch does not.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    fps = float(payload.get("fps", 50.0)) or 50.0
    frames = len(root)
    duration = frames / fps

    # Pelvis lowering against an external walking height. See NOMINAL_PELVIS_HEIGHT_M: a
    # clip that never stands up has no upright height of its own to be measured against.
    height = root[:, 2]
    lowering = float(np.mean(np.clip(reference_pelvis_m - height, 0.0, None)))

    dofs = payload.get("dof_pos")
    if dofs is None:
        dofs = np.asarray(payload.get("reference_g1_qpos"))[:, 7:]
    dofs = np.asarray(dofs, dtype=np.float64)

    travel = float(np.abs(np.diff(dofs, axis=0)).sum() / duration) if frames > 1 else 0.0
    baseline = (
        np.asarray(reference_pose, dtype=np.float64)
        if reference_pose is not None else np.median(dofs, axis=0)
    )
    deviation = float(np.sqrt(np.mean((dofs - baseline) ** 2)))

    steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    path_length = float(steps.sum())

    # Normalise each component to roughly unit scale before weighting, so the weights mean
    # what they look like they mean.
    total = (
        weights.get("pelvis_lowering", 0.0) * (lowering / NOMINAL_PELVIS_HEIGHT_M)
        + weights.get("joint_travel", 0.0) * (travel / 100.0)
        + weights.get("postural_deviation", 0.0) * deviation
        + weights.get("duration", 0.0) * (duration / 10.0)
    )

    return AdaptationCost(
        motion_id=motion_id,
        pelvis_lowering_m=lowering,
        pelvis_height_m=float(np.percentile(height, 90)),
        joint_travel_rad_per_s=travel,
        postural_deviation_rad=deviation,
        duration_s=duration,
        path_length_m=path_length,
        total=float(total),
    )


def prefers(
    cheaper: AdaptationCost, dearer: AdaptationCost, *, margin: float = 0.0
) -> bool:
    """Whether the first motion is preferred, by more than ``margin``.

    Separate from a bare ``<`` because a preference that survives no margin at all is not a
    preference worth reporting -- two motions whose costs differ in the fourth decimal are
    interchangeable, and calling one of them optimal overstates what was measured.
    """
    return cheaper.total < dearer.total - margin
