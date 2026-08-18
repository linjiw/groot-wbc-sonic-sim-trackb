"""Refuse to spend a rollout on a scene the robot never reaches.

Two families were built, eight cells rolled out, and every cell came back accepted -- read at the
time as evidence that the geometric window over-states the real one by more than 40 mm. It did not.
The obstacle had been placed in the *reference* motion's coordinate frame while the rollout offset
the motion by −2.0 m at conversion, so the robot walked past two metres clear of the shelf. The
nominal "cleared" an obstacle it never approached.

This is the same class of error already on record from an earlier family, where a shelf was
rendered half a metre from the position the boundary search had optimised. The guard written then
compared clearances; it did not check that the path enters the obstacle's footprint at all, which
is the cheaper and more basic question.

So: before any rollout, confirm the path the robot will actually follow passes through the
obstacle's footprint, in the frame physics will use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RouteCheck:
    """Whether a path meets an obstacle, and by how much if not."""

    frames_inside: int
    #: Metres from the path's closest approach to the obstacle footprint. Zero when it enters.
    closest_approach_m: float
    #: Signed x offset that would bring the path's closest point onto the footprint centre. A value
    #: near a conversion's ``--scene-start`` offset is the signature of a frame mismatch.
    suggested_shift_x_m: float

    @property
    def passes(self) -> bool:
        return self.frames_inside > 0

    def explain(self) -> str:
        if self.passes:
            return f"path enters the obstacle footprint on {self.frames_inside} frames"
        return (
            f"path never enters the obstacle footprint; closest approach "
            f"{self.closest_approach_m:.2f} m, and shifting the obstacle by "
            f"{self.suggested_shift_x_m:+.2f} m in x would place it on the path. An offset close "
            f"to a conversion --scene-start value means the scene and the motion are in different "
            f"frames."
        )


def check_route_meets_obstacle(
    path_xy: np.ndarray,
    footprint: tuple[float, float, float, float],
) -> RouteCheck:
    """Does ``path_xy`` pass through ``footprint`` = ``(x_min, y_min, x_max, y_max)``?

    ``path_xy`` must be in the same frame physics will place the robot in -- an executed root path,
    or a reference path with the conversion's scene-start offset already applied. Passing a raw
    reference path against a scene built for an offset rollout is exactly the mistake this exists
    to catch, and the function cannot detect it for you.
    """
    path = np.asarray(path_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(f"expected (T, 2) path, got {path.shape}")
    x_min, y_min, x_max, y_max = footprint

    inside = (
        (path[:, 0] >= x_min)
        & (path[:, 0] <= x_max)
        & (path[:, 1] >= y_min)
        & (path[:, 1] <= y_max)
    )
    dx = np.maximum(np.maximum(x_min - path[:, 0], path[:, 0] - x_max), 0.0)
    dy = np.maximum(np.maximum(y_min - path[:, 1], path[:, 1] - y_max), 0.0)
    distance = np.hypot(dx, dy)
    nearest = int(np.argmin(distance))
    return RouteCheck(
        frames_inside=int(inside.sum()),
        closest_approach_m=float(distance.min()),
        suggested_shift_x_m=float(path[nearest, 0] - 0.5 * (x_min + x_max)),
    )


def capsule_box_clearance(
    starts: np.ndarray,
    ends: np.ndarray,
    radii: np.ndarray,
    box: tuple[float, float, float, float, float, float],
) -> tuple[float, int, int]:
    """Smallest gap between any collision capsule and an axis-aligned box, over a whole clip.

    ``check_route_meets_obstacle`` asks whether the *root path* enters an obstacle's footprint,
    which is the right question for a ceiling the robot walks under and the wrong one for a wall it
    walks past: the root never enters a wall's footprint, so that check would reject every lateral
    scene ever built.

    This asks the question that actually decides the outcome for either shape -- how close does the
    body come to the solid -- and returns ``(clearance, frame, capsule)``. Negative clearance means
    the capsule and the box overlap by that much.
    """
    x0, y0, z0, x1, y1, z1 = box
    lower = np.array([x0, y0, z0], dtype=np.float64)
    upper = np.array([x1, y1, z1], dtype=np.float64)

    # Sample along each capsule's axis. The segment-to-box distance has no short closed form, and a
    # sampled minimum is exact enough at the millimetre scale these scenes are placed to.
    samples = np.linspace(0.0, 1.0, 9)[None, None, :, None]
    points = starts[:, :, None, :] * (1.0 - samples) + ends[:, :, None, :] * samples
    clamped = np.clip(points, lower, upper)
    distance = np.linalg.norm(points - clamped, axis=-1) - radii[None, :, None]

    flat = int(np.argmin(distance))
    frame, capsule, _ = np.unravel_index(flat, distance.shape)
    return float(distance.min()), int(frame), int(capsule)
