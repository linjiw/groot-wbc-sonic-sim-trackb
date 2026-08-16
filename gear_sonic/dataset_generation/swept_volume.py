"""World-frame swept volume of the G1, for clearance that reflects the actual body.

Why this replaces a root cylinder
---------------------------------

Clearance was previously a single 0.45 m cylinder about the **root**. Measured on a
recorded rollout, that is wrong in both directions at once:

* too permissive -- the wrist reaches **0.514 m** from the root, outside the assumed
  cylinder, on 22 of 249 frames. Arms were simply not represented.
* too conservative -- 99.4% of (frame, body) samples sit well inside 0.45 m, so
  corridors were sized for an envelope the robot rarely fills.

Modelling each link's real collision geometry fixes both, and is what makes narrow
passages safe to generate rather than merely optimistic.

Geometry source
---------------

The 29 primitive collision capsules of the 14 collision-bearing G1 links, transcribed
from the Kimodo G1 MJCF (`g1skel34/xml/g1.xml`). The repository's own
`gear_sonic_deploy/g1/g1_29dof.xml` defines collisions as meshes, which carry no usable
radii; the Kimodo MJCF was already verified against it -- same 29 actuated joints, same
order, axes, limits, parent links and link transforms -- so its primitives describe the
same robot. The numbers are transcribed here rather than read at runtime so this module
does not depend on an external checkout.

A useful consistency check: exactly these 14 links carry collision geometry, and exactly
the links in this table were ever observed to register contact force in recorded
rollouts. The other 16 articulation bodies reported identically zero force because they
have no collision shape at all.

Limits
------

* Hands beyond `*_wrist_yaw_link` are not modelled; the G1 collision set stops at the
  wrist, so a dexterous hand extends past this envelope.
* Capsules are placed from recorded body pose. A trajectory recorded before
  `body_quat_w` existed cannot use this model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "G1_COLLISION_CAPSULES",
    "swept_half_width_per_frame",
    "CollisionCapsule",
    "SweptVolumeReport",
    "body_capsules_world",
    "swept_volume_clearance",
]


@dataclass(frozen=True)
class CollisionCapsule:
    """A capsule in a link's local frame. A sphere is a capsule with start == end."""

    start: tuple[float, float, float]
    end: tuple[float, float, float]
    radius: float


#: Link name -> collision capsules, in that link's local frame.
G1_COLLISION_CAPSULES: dict[str, tuple[CollisionCapsule, ...]] = {
    "pelvis": (
        CollisionCapsule((0.0000, 0.0000, -0.0800), (0.0000, 0.0000, -0.0800), 0.0700),
    ),
    "left_hip_roll_link": (
        CollisionCapsule((0.0200, 0.0000, 0.0000), (0.0200, 0.0000, -0.2000), 0.0500),
    ),
    "left_knee_link": (
        CollisionCapsule((0.0200, 0.0000, 0.0000), (0.0200, 0.0000, -0.2500), 0.0400),
    ),
    "left_ankle_roll_link": (
        CollisionCapsule((0.1000, -0.0260, -0.0250), (0.0500, -0.0270, -0.0250), 0.0100),
        CollisionCapsule((-0.0440, -0.0180, -0.0250), (0.1230, -0.0180, -0.0250), 0.0100),
        CollisionCapsule((-0.0520, -0.0100, -0.0250), (0.1300, -0.0100, -0.0250), 0.0100),
        CollisionCapsule((-0.0540, 0.0000, -0.0250), (0.1320, 0.0000, -0.0250), 0.0100),
        CollisionCapsule((-0.0520, 0.0100, -0.0250), (0.1300, 0.0100, -0.0250), 0.0100),
        CollisionCapsule((-0.0440, 0.0180, -0.0250), (0.1230, 0.0180, -0.0250), 0.0100),
        CollisionCapsule((0.1000, 0.0260, -0.0250), (0.0500, 0.0260, -0.0250), 0.0100),
    ),
    "right_hip_roll_link": (
        CollisionCapsule((0.0200, 0.0000, 0.0000), (0.0200, 0.0000, -0.2000), 0.0500),
    ),
    "right_knee_link": (
        CollisionCapsule((0.0200, 0.0000, 0.0000), (0.0200, 0.0000, -0.2500), 0.0400),
    ),
    "right_ankle_roll_link": (
        CollisionCapsule((0.1000, -0.0260, -0.0250), (0.0500, -0.0260, -0.0250), 0.0100),
        CollisionCapsule((-0.0440, -0.0180, -0.0250), (0.1230, -0.0180, -0.0250), 0.0080),
        CollisionCapsule((-0.0520, -0.0100, -0.0250), (0.1300, -0.0100, -0.0250), 0.0100),
        CollisionCapsule((-0.0540, 0.0000, -0.0250), (0.1320, 0.0000, -0.0250), 0.0100),
        CollisionCapsule((-0.0520, 0.0100, -0.0250), (0.1300, 0.0100, -0.0250), 0.0100),
        CollisionCapsule((-0.0440, 0.0180, -0.0250), (0.1230, 0.0180, -0.0250), 0.0080),
        CollisionCapsule((0.1000, 0.0260, -0.0250), (0.0500, 0.0260, -0.0250), 0.0100),
    ),
    "torso_link": (
        CollisionCapsule((0.0050, -0.0320, 0.2200), (0.0050, 0.0320, 0.2200), 0.0730),
        CollisionCapsule((0.0050, -0.0280, 0.1300), (0.0050, 0.0280, 0.1300), 0.0700),
        CollisionCapsule((0.0050, -0.0200, 0.0600), (0.0050, 0.0200, 0.0600), 0.0650),
        CollisionCapsule((0.0100, 0.0000, 0.4100), (0.0100, 0.0000, 0.4200), 0.0680),
    ),
    "left_shoulder_yaw_link": (
        CollisionCapsule((0.0000, 0.0000, -0.0800), (0.0000, 0.0000, 0.0500), 0.0350),
    ),
    "left_elbow_link": (
        CollisionCapsule((-0.0100, 0.0000, -0.0100), (0.1200, 0.0000, -0.0100), 0.0350),
    ),
    "left_wrist_yaw_link": (
        CollisionCapsule((0.0500, 0.0000, 0.0000), (0.1000, 0.0000, 0.0000), 0.0500),
    ),
    "right_shoulder_yaw_link": (
        CollisionCapsule((0.0000, 0.0000, -0.0800), (0.0000, 0.0000, 0.0500), 0.0350),
    ),
    "right_elbow_link": (
        CollisionCapsule((-0.0100, 0.0000, -0.0100), (0.1200, 0.0000, -0.0100), 0.0350),
    ),
    "right_wrist_yaw_link": (
        CollisionCapsule((0.0500, 0.0000, 0.0000), (0.1000, 0.0000, 0.0000), 0.0500),
    ),
}


def _quat_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3) rotation matrices."""
    w, x, y, z = np.moveaxis(np.asarray(quaternions, dtype=np.float64), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def body_capsules_world(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    *,
    capsules: Mapping[str, Sequence[CollisionCapsule]] = G1_COLLISION_CAPSULES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Place every collision capsule in world frame for every frame.

    Args:
        body_pos: ``(T, B, 3)`` link origins.
        body_quat: ``(T, B, 4)`` link orientations, wxyz.
        body_names: ``B`` link names, matching axis 1 of both arrays.

    Returns:
        ``(starts, ends, radii, owners)`` where starts/ends are ``(T, C, 3)``, radii is
        ``(C,)`` and owners names the link each capsule belongs to.
    """
    positions = np.asarray(body_pos, dtype=np.float64)
    quaternions = np.asarray(body_quat, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"body_pos must be (T, B, 3); got {positions.shape}")
    if quaternions.shape[:2] != positions.shape[:2] or quaternions.shape[-1] != 4:
        raise ValueError(f"body_quat must be (T, B, 4); got {quaternions.shape}")
    names = tuple(body_names)
    if len(names) != positions.shape[1]:
        raise ValueError(f"{len(names)} names but {positions.shape[1]} bodies")

    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in capsules if name not in index]
    if missing:
        raise ValueError(f"collision links absent from the recorded bodies: {sorted(missing)}")

    starts, ends, radii, owners = [], [], [], []
    for link, link_capsules in capsules.items():
        column = index[link]
        rotation = _quat_to_matrix(quaternions[:, column])  # (T, 3, 3)
        origin = positions[:, column]  # (T, 3)
        for capsule in link_capsules:
            local_start = np.asarray(capsule.start, dtype=np.float64)
            local_end = np.asarray(capsule.end, dtype=np.float64)
            starts.append(np.einsum("tij,j->ti", rotation, local_start) + origin)
            ends.append(np.einsum("tij,j->ti", rotation, local_end) + origin)
            radii.append(capsule.radius)
            owners.append(link)
    return (
        np.stack(starts, axis=1),
        np.stack(ends, axis=1),
        np.asarray(radii, dtype=np.float64),
        tuple(owners),
    )


def _segment_to_box_distance(
    start: np.ndarray, end: np.ndarray, box: tuple[float, float, float, float, float, float]
) -> np.ndarray:
    """Distance from each segment to an axis-aligned box, sampled along the segment.

    Sampling rather than solving exactly: the closest point between a segment and a box
    has no short closed form, and the samples are dense enough that the residual error is
    far below the clearance margins in use. Sampling can only ever *over*-estimate the
    distance between samples, so the number is padded conservatively by the sample step.
    """
    samples = 9
    ts = np.linspace(0.0, 1.0, samples).reshape(1, samples, 1)
    points = start[:, None, :] * (1 - ts) + end[:, None, :] * ts  # (N, S, 3)
    lower = np.asarray(box[:3], dtype=np.float64)
    upper = np.asarray(box[3:], dtype=np.float64)
    delta = np.maximum(np.maximum(lower - points, 0.0), points - upper)
    return np.linalg.norm(delta, axis=-1).min(axis=1)


@dataclass(frozen=True)
class SweptVolumeReport:
    """Minimum clearance between the robot's swept volume and a set of boxes."""

    min_clearance_m: float
    min_frame: int
    min_link: str | None
    min_obstacle: str | None
    per_frame_clearance: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_clearance_m": self.min_clearance_m,
            "min_frame": self.min_frame,
            "min_link": self.min_link,
            "min_obstacle": self.min_obstacle,
        }


def swept_volume_clearance(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    obstacles: Sequence[tuple[str, tuple[float, float, float, float, float, float]]],
    *,
    capsules: Mapping[str, Sequence[CollisionCapsule]] = G1_COLLISION_CAPSULES,
) -> SweptVolumeReport:
    """Minimum surface distance from the robot's collision volume to any obstacle box.

    Args:
        obstacles: ``(name, (min_x, min_y, min_z, max_x, max_y, max_z))`` in the same
            frame as ``body_pos``.

    Returns:
        A :class:`SweptVolumeReport`. Negative values mean interpenetration.
    """
    starts, ends, radii, owners = body_capsules_world(
        body_pos, body_quat, body_names, capsules=capsules
    )
    frames, count, _ = starts.shape
    best = np.full(frames, np.inf)
    best_link: str | None = None
    best_obstacle: str | None = None
    overall = np.inf
    for name, box in obstacles:
        flat_start = starts.reshape(-1, 3)
        flat_end = ends.reshape(-1, 3)
        distance = _segment_to_box_distance(flat_start, flat_end, box).reshape(frames, count)
        distance = distance - radii[None, :]
        per_frame = distance.min(axis=1)
        best = np.minimum(best, per_frame)
        candidate = float(distance.min())
        if candidate < overall:
            overall = candidate
            index = np.unravel_index(int(distance.argmin()), distance.shape)
            best_link = owners[index[1]]
            best_obstacle = name
    return SweptVolumeReport(
        min_clearance_m=float(overall),
        min_frame=int(np.argmin(best)),
        min_link=best_link,
        min_obstacle=best_obstacle,
        per_frame_clearance=best,
    )


def swept_half_width_per_frame(
    body_pos: np.ndarray,
    body_quat: np.ndarray,
    body_names: Sequence[str],
    reference_xy: np.ndarray | None = None,
    *,
    capsules: Mapping[str, Sequence[CollisionCapsule]] = G1_COLLISION_CAPSULES,
) -> np.ndarray:
    """Horizontal half-width of the robot's collision volume, per frame.

    This is the quantity a corridor actually has to accommodate at each point along
    the path, and it varies a lot: measured on a walking rollout it ranges from
    0.273 m with the arms tucked to 0.664 m at peak arm swing, around a median of
    0.377 m. Sizing a corridor by the worst case (0.664 m) is *more* conservative
    than the old uniform 0.45 m root radius; sizing it per frame is what allows
    genuinely tight passages where the robot is actually slim.

    Args:
        reference_xy: ``(T, 2)`` path to measure from; defaults to the pelvis.

    Returns:
        ``(T,)`` half-width in metres, including capsule radii.
    """
    starts, ends, radii, _ = body_capsules_world(
        body_pos, body_quat, body_names, capsules=capsules
    )
    if reference_xy is None:
        index = {name: i for i, name in enumerate(body_names)}
        reference_xy = np.asarray(body_pos, dtype=np.float64)[:, index["pelvis"], :2]
    reference = np.asarray(reference_xy, dtype=np.float64)
    if reference.shape != (starts.shape[0], 2):
        raise ValueError(f"reference_xy must be ({starts.shape[0]}, 2); got {reference.shape}")
    points = np.concatenate([starts, ends], axis=1)
    padded = np.concatenate([radii, radii])
    distance = np.linalg.norm(points[:, :, :2] - reference[:, None, :], axis=-1) + padded[None, :]
    return distance.max(axis=1)
