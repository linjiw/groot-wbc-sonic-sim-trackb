"""Place a Kimodo reference motion where the scene can actually contain it.

Why this exists
---------------

The 2026-08-15 batch accepted 2 of 16 rollouts. The dominant cause was not the
controller and not the gates: every reference was dropped at a scene's route
start with a fixed yaw, so a motion whose root path curves 2.5 m laterally walked
into a rack no matter how well SONIC tracked it. Recorded scene-contact forces
reached 1850 N.

This module chooses the placement instead of assuming it. Given a motion's
canonical root path and a scene's obstacle footprints, it searches yaw and
translation for placements whose whole swept path keeps a required clearance and
stays inside the walkable rectangle.

Agreement with the adapter is the point
---------------------------------------

The transform searched here is exactly the one
``kimodo_motion_adapter.transform_qpos_to_scene`` applies: canonicalise the path
so its first XY sits at the origin, rotate by ``scene_yaw`` about +Z, then add
``scene_start_xyz``. A placement that this planner accepts is therefore the same
placement the converter will produce, and the obstacle model is imported from
``scene_asset_preflight`` so the planner cannot drift from the gate that judges
the result.

Tracking margin
---------------

Planning against the *reference* path underestimates risk, because the executed
root deviates from it. Measured on accepted M0 episodes, p95 path error was
0.22-0.23 m and endpoint error 0.20-0.22 m. The default margin is therefore
0.30 m on top of the scene's declared body-clearance radius: enough to cover the
observed deviation with headroom, and stated as a measured quantity rather than
a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from gear_sonic.dataset_generation.scene_asset_preflight import (
    SceneObstacleMap,
    load_scene_obstacle_map,
)

__all__ = [
    "DEFAULT_TRACKING_MARGIN_M",
    "Placement",
    "canonical_path_xy",
    "plan_placements",
    "transform_path",
]

#: Metres of extra clearance beyond the scene's declared body radius, covering the
#: measured gap between the planned reference path and the executed root path.
DEFAULT_TRACKING_MARGIN_M = 0.30


@dataclass(frozen=True)
class Placement:
    """One accepted (translation, yaw) placement of a motion in a scene."""

    scene_id: str
    start_xy: tuple[float, float]
    yaw_rad: float
    clearance_m: float
    nearest_obstacle: str | None
    path_length_m: float
    net_displacement_m: float
    end_xy: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "start_xy": list(self.start_xy),
            "yaw_rad": self.yaw_rad,
            "yaw_deg": math.degrees(self.yaw_rad),
            "clearance_m": self.clearance_m,
            "nearest_obstacle": self.nearest_obstacle,
            "path_length_m": self.path_length_m,
            "net_displacement_m": self.net_displacement_m,
            "end_xy": list(self.end_xy),
        }


def canonical_path_xy(qpos: np.ndarray) -> np.ndarray:
    """Root XY of a Kimodo qpos clip, translated so the first sample is the origin.

    Mirrors ``transform_qpos_to_scene(..., canonicalize_horizontal_origin=True)``.
    """
    array = np.asarray(qpos, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"expected a (T, >=2) qpos array; got {array.shape}")
    xy = array[:, :2].copy()
    xy -= xy[0]
    return xy


def transform_path(path_xy: np.ndarray, start_xy: Sequence[float], yaw_rad: float) -> np.ndarray:
    """Rotate a canonical path about +Z by ``yaw_rad`` then translate to ``start_xy``."""
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return np.asarray(path_xy, dtype=np.float64) @ rotation.T + np.asarray(
        start_xy, dtype=np.float64
    )


def _decimate(path_xy: np.ndarray, max_points: int) -> np.ndarray:
    """Reduce a dense path to at most ``max_points`` while keeping both endpoints.

    Clearance is evaluated per segment, so dropping interior samples only makes
    each segment longer -- it never lets the path cut a corner it did not take.
    """
    if len(path_xy) <= max_points:
        return path_xy
    indices = np.unique(
        np.concatenate(
            [np.linspace(0, len(path_xy) - 1, max_points).astype(int), [len(path_xy) - 1]]
        )
    )
    return path_xy[indices]


def _spread_select(candidates: list[Placement], count: int) -> list[Placement]:
    """Greedy farthest-point selection so returned placements are not clustered.

    Distance mixes start position with heading (yaw scaled to metres by one metre
    per radian) so two placements that differ only in facing still count as
    distinct.
    """
    if len(candidates) <= count:
        return candidates
    ordered = sorted(candidates, key=lambda placement: -placement.clearance_m)
    chosen = [ordered[0]]
    while len(chosen) < count:
        best = None
        best_distance = -1.0
        for candidate in ordered:
            if candidate in chosen:
                continue
            distance = min(
                math.dist(candidate.start_xy, picked.start_xy)
                + abs(_wrap_angle(candidate.yaw_rad - picked.yaw_rad))
                for picked in chosen
            )
            if distance > best_distance:
                best_distance = distance
                best = candidate
        if best is None:
            break
        chosen.append(best)
    return chosen


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def plan_placements(
    path_xy: np.ndarray,
    scene: SceneObstacleMap | str,
    *,
    max_results: int = 8,
    tracking_margin_m: float = DEFAULT_TRACKING_MARGIN_M,
    yaw_steps: int = 24,
    translation_step_m: float = 0.25,
    max_path_points: int = 96,
    package_dir: str | None = None,
) -> list[Placement]:
    """Search yaw and translation for placements the scene can contain.

    Args:
        path_xy: ``(T, 2)`` canonical root path, first point at the origin.
        scene: A loaded :class:`SceneObstacleMap`, or a scene id to load.
        max_results: Maximum placements to return, spread apart rather than
            clustered around the single best-clearance pose.
        tracking_margin_m: Extra clearance on top of the scene's declared body
            radius, covering executed-vs-reference deviation.
        yaw_steps: Number of headings sampled over a full turn.
        translation_step_m: Grid spacing for candidate start positions.
        max_path_points: Path decimation cap, to bound the search cost.

    Returns:
        Accepted placements, best clearance first.
    """
    obstacle_map = (
        load_scene_obstacle_map(scene, package_dir)
        if isinstance(scene, str) and package_dir is not None
        else load_scene_obstacle_map(scene)
        if isinstance(scene, str)
        else scene
    )
    canonical = _decimate(np.asarray(path_xy, dtype=np.float64), max_path_points)
    if len(canonical) < 2:
        raise ValueError("path must contain at least two points")

    required = obstacle_map.route_clearance_radius_m + tracking_margin_m
    min_x, min_y = obstacle_map.walkable_min_xy
    max_x, max_y = obstacle_map.walkable_max_xy

    # Full path length is reported from the undecimated path so the metric matches
    # what a reader would compute from the motion itself.
    full = np.asarray(path_xy, dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(full, axis=0), axis=1).sum())

    accepted: list[Placement] = []
    xs = np.arange(min_x, max_x + 1e-9, translation_step_m)
    ys = np.arange(min_y, max_y + 1e-9, translation_step_m)
    for yaw_index in range(yaw_steps):
        yaw = 2.0 * math.pi * yaw_index / yaw_steps
        rotated = transform_path(canonical, (0.0, 0.0), yaw)
        # Reject translations whose rotated extent cannot fit before doing any
        # per-segment distance work.
        extent_min = rotated.min(axis=0)
        extent_max = rotated.max(axis=0)
        for start_x in xs:
            if start_x + extent_min[0] < min_x + required or start_x + extent_max[0] > max_x - required:
                continue
            for start_y in ys:
                if (
                    start_y + extent_min[1] < min_y + required
                    or start_y + extent_max[1] > max_y - required
                ):
                    continue
                placed = rotated + np.array([start_x, start_y])
                points = [(float(x), float(y)) for x, y in placed]
                clearance, nearest = obstacle_map.clearance_to_obstacles(points)
                if clearance < required:
                    continue
                accepted.append(
                    Placement(
                        scene_id=obstacle_map.scene_id,
                        start_xy=(float(start_x), float(start_y)),
                        yaw_rad=float(yaw),
                        clearance_m=float(clearance),
                        nearest_obstacle=nearest,
                        path_length_m=path_length,
                        net_displacement_m=float(np.linalg.norm(full[-1] - full[0])),
                        end_xy=(float(placed[-1][0]), float(placed[-1][1])),
                    )
                )

    selected = _spread_select(accepted, max_results)
    return sorted(selected, key=lambda placement: -placement.clearance_m)


def summarize_placements(placements: Iterable[Placement]) -> dict[str, Any]:
    """Aggregate statistics for a planning report."""
    items = list(placements)
    if not items:
        return {"count": 0}
    clearances = [placement.clearance_m for placement in items]
    return {
        "count": len(items),
        "min_clearance_m": min(clearances),
        "max_clearance_m": max(clearances),
        "distinct_yaw_deg": sorted({round(math.degrees(p.yaw_rad), 1) for p in items}),
    }
