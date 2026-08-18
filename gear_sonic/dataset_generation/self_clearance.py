"""How close the robot comes to touching itself, including when it does not.

The existing self-intersection check answers a sign question: are any two bodies overlapping,
and by how much. It cannot answer the question that keeps mattering, which is *how much room
is left* when they are not overlapping -- MuJoCo reports contacts, and a pair that is 3 mm
apart generates no contact at all, so it looks identical to a pair half a metre apart.

That blind spot has now cost two experiments. A generated crouch overlapped thigh into pelvis
by 1.6 mm, passed a screen whose threshold is 100 mm, and produced 931.5 N of self-contact
once tracked. Relaxing it to *exactly zero* penetration was still rejected, because zero
penetration is not clearance: a few millimetres of tracking error is enough either way.

It is the third form of one mistake in this project -- a joint riding its limit, a clamp
placing values exactly at a limit, and a body pair sitting exactly at the boundary of
intersecting. Each time the check that passed was a sign test where a margin was needed. This
measures the margin.

Distances come from **MuJoCo's own collision geometry**, via ``mj_geomDistance``, and not from
the swept-volume capsules. That distinction has already been learned once here: the capsules
in ``G1_COLLISION_CAPSULES`` are deliberately conservative *outer* approximations built for
sweeping a volume along a path, and they interpenetrate each other permanently. Measured with
them, a plain walk, a shallow crouch and a deep crouch all report an identical -0.0478 m,
which is not a property of any of those motions.

Geoms on the same body are skipped, as are parent-child pairs, which meet at a joint and
touch there by construction.

**Read the absolute minimum with care; prefer the comparison.** A plain accepted walk already
reports -0.0817 m, because a wrist geom passes through a hip geom in ordinary arm swing and
the model tolerates it. So a global minimum is saturated by whichever pair overlaps
permanently, and it says almost nothing about whether a particular clip is dangerously close
to itself. What does say something is ``clearance_change``: how much worse a clip is than the
nominal motion it was derived from, per pair. For a matched adaptation operator that is
exactly the right question, because the nominal is known to track.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .self_intersection import DEFAULT_G1_MJCF, _load_model

#: Metres of clearance below which a reference is too close to itself to survive tracking.
#: Provisional: derived from one rejected pair, not from a calibrated sample. The crouch that
#: failed sat at 0.000 m and the walk it came from clears far more; where between those the
#: real threshold lies needs a handful of rollouts at known clearances to say.
PROVISIONAL_MIN_CLEARANCE_M = 0.02


@dataclass(frozen=True)
class ClearanceReport:
    """The tightest the robot gets to itself over a clip."""

    frames_checked: int
    #: Smallest capsule-surface gap between any non-adjacent pair, in metres. Negative means
    #: the capsules overlap.
    min_clearance_m: float
    #: The pair responsible, and the frame it happens on.
    tightest_pair: tuple[str, str]
    tightest_frame: int
    #: Per-frame minimum, for callers that want to see when the clip is tight.
    per_frame: np.ndarray

    @property
    def has_margin(self) -> bool:
        return self.min_clearance_m >= PROVISIONAL_MIN_CLEARANCE_M


def self_clearance(
    reference_qpos: np.ndarray,
    *,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    frame_stride: int = 2,
    cutoff: float = 0.5,
) -> ClearanceReport:
    """Smallest gap between the robot's own collision geoms over a clip."""
    qpos = np.asarray(reference_qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"expected (T, nq) reference qpos, got {qpos.shape}")

    mujoco, model = _load_model(mjcf_path)
    if qpos.shape[1] != model.nq:
        raise ValueError(
            f"reference has {qpos.shape[1]} columns but the model expects {model.nq}"
        )
    data = mujoco.MjData(model)

    def name_of(geom: int) -> str:
        body = model.geom_bodyid[geom]
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"

    # Pairs worth asking about: different bodies, neither of them the world, and not adjacent
    # across a joint.
    pairs = []
    for first in range(model.ngeom):
        for second in range(first + 1, model.ngeom):
            a, b = model.geom_bodyid[first], model.geom_bodyid[second]
            if a == b or a == 0 or b == 0:
                continue
            if model.body_parentid[a] == b or model.body_parentid[b] == a:
                continue
            pairs.append((first, second))

    per_frame = []
    best = np.inf
    tightest = ("", "")
    tightest_frame = 0
    scratch = np.zeros(6)
    for index in range(0, len(qpos), frame_stride):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        frame_best = np.inf
        for first, second in pairs:
            distance = mujoco.mj_geomDistance(model, data, first, second, cutoff, scratch)
            if distance < frame_best:
                frame_best = distance
                if distance < best:
                    best = distance
                    tightest = (name_of(first), name_of(second))
                    tightest_frame = index
        per_frame.append(frame_best)

    return ClearanceReport(
        frames_checked=len(per_frame),
        min_clearance_m=float(best),
        tightest_pair=tightest,
        tightest_frame=tightest_frame,
        per_frame=np.asarray(per_frame),
    )


@dataclass(frozen=True)
class ClearanceChange:
    """How much closer to itself an adapted clip gets than the nominal it came from."""

    #: Metres the tightest pair lost relative to the nominal. Negative means the adapted clip
    #: is closer to touching itself than the motion it was derived from.
    worst_change_m: float
    pair: tuple[str, str]
    frame: int
    nominal_clearance_m: float
    adapted_clearance_m: float

    @property
    def is_tighter(self) -> bool:
        return self.worst_change_m < 0.0


def clearance_change(
    adapted_qpos: np.ndarray,
    nominal_qpos: np.ndarray,
    *,
    mjcf_path: str | Path = DEFAULT_G1_MJCF,
    frame_stride: int = 2,
    cutoff: float = 0.5,
) -> ClearanceChange:
    """Per-pair clearance of an adapted clip against the nominal it was derived from.

    The absolute minimum is useless on its own -- an ordinary walk overlaps a wrist geom into
    a hip geom -- so what is reported is the pair that *deteriorated* most. A matched operator
    starts from a motion known to track, and the question worth asking is whether it brought
    any pair closer together than that motion already had them.
    """
    adapted = np.asarray(adapted_qpos, dtype=np.float64)
    nominal = np.asarray(nominal_qpos, dtype=np.float64)
    if adapted.shape != nominal.shape:
        raise ValueError(
            f"clips must have the same shape to compare per-frame; "
            f"got {adapted.shape} and {nominal.shape}"
        )

    mujoco, model = _load_model(mjcf_path)
    data = mujoco.MjData(model)

    pairs = []
    for first in range(model.ngeom):
        for second in range(first + 1, model.ngeom):
            a, b = model.geom_bodyid[first], model.geom_bodyid[second]
            if a == b or a == 0 or b == 0:
                continue
            if model.body_parentid[a] == b or model.body_parentid[b] == a:
                continue
            pairs.append((first, second))

    def name_of(geom: int) -> str:
        body = model.geom_bodyid[geom]
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or f"body{body}"

    scratch = np.zeros(6)

    def distances(clip: np.ndarray) -> np.ndarray:
        out = np.empty((len(range(0, len(clip), frame_stride)), len(pairs)))
        for row, index in enumerate(range(0, len(clip), frame_stride)):
            data.qpos[:] = clip[index]
            mujoco.mj_forward(model, data)
            for column, (first, second) in enumerate(pairs):
                out[row, column] = mujoco.mj_geomDistance(
                    model, data, first, second, cutoff, scratch
                )
        return out

    before, after = distances(nominal), distances(adapted)
    delta = after - before
    row, column = np.unravel_index(int(np.argmin(delta)), delta.shape)
    return ClearanceChange(
        worst_change_m=float(delta[row, column]),
        pair=(name_of(pairs[column][0]), name_of(pairs[column][1])),
        frame=int(row * frame_stride),
        nominal_clearance_m=float(before[row, column]),
        adapted_clearance_m=float(after[row, column]),
    )
