"""Which body part an obstacle binds, and how hard it can be made, from an executed trajectory.

Scene difficulty has been chosen by binary search: lower a shelf until something touches. That
finds *a* boundary but says nothing about *what* is being tested, and it can only ever bind the
tallest part of the robot -- a full-height shelf on a walk binds ``torso_link`` on every frame and
nothing else.

An obstacle occupying a *band* of height binds whatever passes through that band. On one plain walk
the bands select five different parts: torso overhead, elbow at chest height, wrist at waist, hip at
knee height, knee at floor level. So the same nominal motion can pose five distinct geometric
problems, each testing a different part of the body, without generating a single new clip.

That turns scene difficulty into two design parameters -- *which part* is the binding constraint,
and *how much margin* it is left -- instead of one discovered number.

**A binding part is not automatically a counterfactual.** A family needs an adaptation that relieves
the part the obstacle binds. Two exist: the crouch relieves the torso, the arm tuck relieves the
wrists and elbows. Hip and knee obstacles are constructible and currently have no operator to answer
them, so they would produce negatives with no matching positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

#: Height bands, in metres, named for the part of a room they correspond to. Chosen so each band
#: selects a different binding part on a nominal walk; they are not anatomical boundaries.
DEFAULT_BANDS = (
    ("overhead", 1.15, 1.60),
    ("chest", 0.85, 1.15),
    ("waist", 0.55, 0.85),
    ("knee", 0.25, 0.55),
    ("floor", 0.00, 0.25),
)

#: Which operator, if any, relieves a binding part. An obstacle binding a part with no operator
#: produces a negative that nothing can answer, which is a scene, not a family.
RELIEVED_BY = {
    "torso_link": "local_crouch",
    "left_wrist_yaw_link": "local_arm_tuck",
    "right_wrist_yaw_link": "local_arm_tuck",
    "left_elbow_link": "local_arm_tuck",
    "right_elbow_link": "local_arm_tuck",
}


@dataclass(frozen=True)
class BindingConstraint:
    """The part an obstacle in one band and side would meet first, and by how much."""

    band: str
    side: str
    body: str
    #: Metres the binding capsule reaches toward that side, measured from the root and including
    #: the capsule radius. An obstacle placed nearer than this intersects the robot.
    reach_m: float
    #: Frames in which any capsule occupies the band at all. A band the robot never enters cannot
    #: host an obstacle that tests anything.
    frames_in_band: int
    relieved_by: str | None

    @property
    def constructible(self) -> bool:
        """Whether a *family* can be built here, not merely a scene."""
        return self.frames_in_band > 0 and self.relieved_by is not None


def _lateral_frame(root_quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (root_quat[:, i] for i in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)


def criticality_map(
    payload: Mapping,
    *,
    bands: Sequence[tuple[str, float, float]] = DEFAULT_BANDS,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> list[BindingConstraint]:
    """For each height band and side, which body part an obstacle would meet first.

    Measured on the *executed* trajectory rather than the reference, because the reference is what
    was asked for and the execution is what the room will actually contain.
    """
    starts, ends, radii, names = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=capsules,
    )
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    lateral = _lateral_frame(np.asarray(payload["root_quat_w"], dtype=np.float64))

    centres = 0.5 * (starts + ends)
    offset = np.einsum("tcd,td->tc", centres[:, :, :2] - root[:, None, :2], lateral)
    high = np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]
    low = np.minimum(starts[:, :, 2], ends[:, :, 2]) - radii[None, :]

    out: list[BindingConstraint] = []
    for name, z0, z1 in bands:
        in_band = (high > z0) & (low < z1)
        for side, sign in (("left", 1.0), ("right", -1.0)):
            # Reach toward one side, radius included. Capsules outside the band cannot bind an
            # obstacle confined to it.
            reach = np.where(in_band, sign * offset + radii[None, :], -np.inf)
            best = reach.max(axis=1)
            finite = np.isfinite(best)
            if not finite.any():
                out.append(BindingConstraint(name, side, "", 0.0, 0, None))
                continue
            index = int(reach[finite].max(axis=1).argmax())
            frame = int(np.flatnonzero(finite)[index])
            body = names[int(reach[frame].argmax())]
            out.append(
                BindingConstraint(
                    band=name,
                    side=side,
                    body=body,
                    reach_m=float(best[finite].max()),
                    frames_in_band=int(finite.sum()),
                    relieved_by=RELIEVED_BY.get(body),
                )
            )
    return out


def obstacle_offset_for_margin(constraint: BindingConstraint, margin_m: float) -> float:
    """Where to place an obstacle's inner face to leave ``margin_m`` of clearance.

    Positive margin clears the robot; negative intersects it by that much. This is the whole point
    of the map: difficulty becomes a number chosen in advance rather than one discovered by
    lowering an obstacle until something touches.
    """
    return constraint.reach_m + margin_m
