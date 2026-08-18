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
    """What an obstacle in one band would meet first, for each of the two ways one can approach.

    A band admits two different obstacles and they are stopped by different parts of the robot.
    A **wall** -- a rack, a cabinet edge, a door frame -- approaches from the side and is stopped by
    whatever reaches furthest sideways within the band. A **ceiling** -- a shelf, a beam, a low
    lintel -- descends from above and is stopped by the highest point in the band, which is usually
    a different capsule entirely.

    Reporting one number for both was an error in the first version of this map: it gave the
    overhead band a reach of 0.097 m, the torso's sideways extent, which a shelf from above never
    touches. The overhead constraint is the peak height, 1.301 m on the same trajectory.
    """

    band: str
    side: str
    #: Part stopped by a wall approaching from ``side``, and how far it reaches that way from the
    #: root, radius included.
    lateral_body: str
    lateral_reach_m: float
    #: Part stopped by a ceiling descending into the band, and its height above the floor.
    vertical_body: str
    vertical_reach_m: float
    #: Frames in which any capsule occupies the band at all. A band the robot never enters cannot
    #: host an obstacle that tests anything.
    frames_in_band: int
    lateral_relieved_by: str | None
    vertical_relieved_by: str | None

    def constructible(self, obstacle: str) -> bool:
        """Whether a *family* can be built here, not merely a scene.

        ``obstacle`` is ``"wall"`` or ``"ceiling"``; they have different answers because they bind
        different parts and only some parts have an operator that relieves them.
        """
        if self.frames_in_band <= 0:
            return False
        if obstacle == "wall":
            return self.lateral_relieved_by is not None
        if obstacle == "ceiling":
            return self.vertical_relieved_by is not None
        raise ValueError(f"obstacle must be 'wall' or 'ceiling'; got {obstacle!r}")


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
        occupied = int(in_band.any(axis=1).sum())

        # A ceiling descending into the band is stopped by the highest point inside it, which is
        # usually a different capsule from the one a wall would meet.
        ceiling = np.where(in_band, high, -np.inf)
        if np.isfinite(ceiling).any():
            flat = int(np.argmax(np.where(np.isfinite(ceiling), ceiling, -np.inf)))
            top_body = names[flat % ceiling.shape[1]]
            top_reach = float(ceiling[np.isfinite(ceiling)].max())
        else:
            top_body, top_reach = "", 0.0

        for side, sign in (("left", 1.0), ("right", -1.0)):
            # A wall approaching from one side is stopped by whatever reaches furthest that way.
            wall = np.where(in_band, sign * offset + radii[None, :], -np.inf)
            if not np.isfinite(wall).any():
                out.append(
                    BindingConstraint(
                        name, side, "", 0.0, top_body, top_reach, occupied, None, None
                    )
                )
                continue
            flat = int(np.argmax(np.where(np.isfinite(wall), wall, -np.inf)))
            side_body = names[flat % wall.shape[1]]
            side_reach = float(wall[np.isfinite(wall)].max())
            out.append(
                BindingConstraint(
                    band=name,
                    side=side,
                    lateral_body=side_body,
                    lateral_reach_m=side_reach,
                    vertical_body=top_body,
                    vertical_reach_m=top_reach,
                    frames_in_band=occupied,
                    lateral_relieved_by=RELIEVED_BY.get(side_body),
                    vertical_relieved_by=RELIEVED_BY.get(top_body),
                )
            )
    return out


def obstacle_offset_for_margin(
    constraint: BindingConstraint, margin_m: float, *, obstacle: str = "wall"
) -> float:
    """Where to put an obstacle's near face to leave ``margin_m`` of clearance.

    For a wall this is a lateral offset from the root; for a ceiling it is a height above the floor.
    Positive margin clears the robot, negative intersects it by that much. This is the point of the
    map: difficulty becomes a number chosen in advance rather than one found by lowering an obstacle
    until something touches.
    """
    if obstacle == "wall":
        return constraint.lateral_reach_m + margin_m
    if obstacle == "ceiling":
        return constraint.vertical_reach_m + margin_m
    raise ValueError(f"obstacle must be 'wall' or 'ceiling'; got {obstacle!r}")
