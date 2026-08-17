"""Sample walkable routes through an existing scene, to generate motion *from* geometry.

Every episode in the corpus so far was made scene-around-motion: the room is built around
a path the robot already walked. That guarantees traversability but destroys the causal
link the dataset is supposed to teach -- the geometry is a function of the action, so the
action cannot be a response to the geometry. A policy can fit such data while ignoring
vision entirely.

This module inverts it. The scene is fixed, a route is planned through whatever gaps the
furniture leaves, and that route becomes a root-trajectory constraint on generation. The
obstacles now determine the path, which is the direction of causation a visual navigation
dataset needs.

Two things make the planning honest rather than decorative:

* **Clearance is a hard constraint, not a cost.** A cell is walkable only if the robot's
  swept half-width plus a margin fits. Routes are planned on that eroded free space, so a
  returned route has the clearance it claims rather than merely preferring it.
* **The route is time-parameterised.** Kimodo constrains the root per frame, so a path is
  not a constraint until it has a speed. Routes are resampled to a constant-speed
  trajectory at the generation frame rate, and a route that cannot be walked within the
  model's 300-frame ceiling is rejected instead of silently truncated.

Coordinates: scene/Isaac frame is z-up with the route in (x, y). Kimodo is y-up with the
ground plane in (x, z), and ``MujocoQposConverter`` maps mujoco (x, y, z) = kimodo
(z, x, y). So a scene route point (X, Y) is the Kimodo constraint point (x=Y, z=X). That
permutation is cyclic, so handedness is preserved and no axis is mirrored.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable, Sequence

import numpy as np

#: Widest swept half-width measured over a walking rollout across 29 collision capsules.
#: Planning against the measured body rather than a nominal cylinder is the same principle
#: the clutter generator uses.
DEFAULT_BODY_HALF_WIDTH_M = 0.35

#: Extra clearance on top of the body. Tracking error moves the executed path off the
#: reference, and unlike the scene-around-motion direction there is no executed path to
#: build against here -- the room already exists, so the margin has to be paid up front.
DEFAULT_MARGIN_M = 0.25

#: Kimodo generates at 30 fps and a single segment caps at 300 frames (10 s).
GENERATION_FPS = 30
MAX_SEGMENT_FRAMES = 300


class RouteSamplingError(ValueError):
    """Raised when no route satisfying the stated clearance exists."""


@dataclass(frozen=True)
class Obstacle:
    """An axis-aligned footprint that blocks the floor, with its underside height."""

    rect: tuple[float, float, float, float]
    z_base: float = 0.0

    def blocks(self, swept_top_m: float) -> bool:
        """Whether the robot must go around rather than under this piece."""
        return self.z_base < swept_top_m


@dataclass
class SampledRoute:
    """A planned route with the evidence that it is walkable."""

    #: Waypoints in the scene frame, shape (N, 2).
    path_xy: np.ndarray
    #: Per-frame positions at ``GENERATION_FPS``, shape (T, 2).
    trajectory_xy: np.ndarray
    speed_mps: float
    #: Smallest clearance from the route to any blocking obstacle.
    min_clearance_m: float
    required_clearance_m: float
    path_length_m: float
    seed: int

    @property
    def frames(self) -> int:
        return int(self.trajectory_xy.shape[0])

    @property
    def duration_s(self) -> float:
        return self.frames / GENERATION_FPS

    def to_kimodo_root2d(self) -> np.ndarray:
        """Convert to Kimodo's ground plane, shape (T, 2) as (x, z).

        mujoco (x, y, z) = kimodo (z, x, y), so the scene's (X, Y) is kimodo's (Y, X).
        """
        return np.stack(
            [self.trajectory_xy[:, 1], self.trajectory_xy[:, 0]], axis=1
        ).astype(np.float64)


def _distance_field(
    obstacles: Sequence[Obstacle],
    bounds: tuple[float, float, float, float],
    resolution_m: float,
    swept_top_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clearance to the nearest blocking footprint, on a grid over the room."""
    min_x, min_y, max_x, max_y = bounds
    xs = np.arange(min_x, max_x + resolution_m, resolution_m)
    ys = np.arange(min_y, max_y + resolution_m, resolution_m)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")

    clearance = np.full(grid_x.shape, np.inf, dtype=np.float64)
    blocking = [o for o in obstacles if o.blocks(swept_top_m)]
    for obstacle in blocking:
        left, bottom, right, top = obstacle.rect
        dx = np.maximum(np.maximum(left - grid_x, grid_x - right), 0.0)
        dy = np.maximum(np.maximum(bottom - grid_y, grid_y - top), 0.0)
        clearance = np.minimum(clearance, np.hypot(dx, dy))

    # The walls bound the room, so distance to the boundary is a clearance too.
    to_wall = np.minimum.reduce(
        [grid_x - min_x, max_x - grid_x, grid_y - min_y, max_y - grid_y]
    )
    clearance = np.minimum(clearance, np.maximum(to_wall, 0.0))
    return xs, ys, clearance


def _plan(
    free: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    """A* on the 8-connected free grid, with Euclidean step costs."""
    if not free[start] or not free[goal]:
        return None
    rows, cols = free.shape
    neighbours = [
        (dx, dy, math.hypot(dx, dy))
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if (dx, dy) != (0, 0)
    ]

    def heuristic(node: tuple[int, int]) -> float:
        return math.hypot(node[0] - goal[0], node[1] - goal[1])

    open_heap: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best = {start: 0.0}
    closed: set[tuple[int, int]] = set()

    while open_heap:
        _, node = heapq.heappop(open_heap)
        if node == goal:
            path = [node]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            return path[::-1]
        if node in closed:
            continue
        closed.add(node)
        for dx, dy, step in neighbours:
            nxt = (node[0] + dx, node[1] + dy)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or not free[nxt]:
                continue
            # Disallow cutting a blocked corner diagonally -- the robot has width.
            if dx and dy and not (free[node[0] + dx, node[1]] and free[node[0], node[1] + dy]):
                continue
            tentative = best[node] + step
            if tentative < best.get(nxt, math.inf):
                best[nxt] = tentative
                came_from[nxt] = node
                heapq.heappush(open_heap, (tentative + heuristic(nxt), nxt))
    return None


def _simplify(path: np.ndarray, tolerance_m: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, so the route is waypoints rather than grid staircase."""
    if len(path) <= 2:
        return path
    start, end = path[0], path[-1]
    span = end - start
    length = float(np.linalg.norm(span))
    if length <= 0.0:
        deviations = np.linalg.norm(path - start, axis=1)
    else:
        normal = np.array([-span[1], span[0]]) / length
        deviations = np.abs((path - start) @ normal)
    index = int(np.argmax(deviations))
    if deviations[index] <= tolerance_m:
        return np.stack([start, end])
    left = _simplify(path[: index + 1], tolerance_m)
    right = _simplify(path[index:], tolerance_m)
    return np.concatenate([left[:-1], right])


def _resample_constant_speed(path: np.ndarray, speed_mps: float) -> np.ndarray:
    """Sample the polyline at ``GENERATION_FPS`` while moving at a constant speed."""
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    total = float(segments.sum())
    if total <= 0.0:
        raise RouteSamplingError("route has zero length")
    frames = max(int(round(total / speed_mps * GENERATION_FPS)), 2)
    cumulative = np.concatenate([[0.0], np.cumsum(segments)])
    wanted = np.linspace(0.0, total, frames)
    return np.stack(
        [np.interp(wanted, cumulative, path[:, 0]), np.interp(wanted, cumulative, path[:, 1])],
        axis=1,
    )


def route_clearance(path_xy: np.ndarray, xs: np.ndarray, ys: np.ndarray, clearance: np.ndarray) -> float:
    """Smallest clearance along a route, read off the distance field by nearest cell."""
    ix = np.clip(np.searchsorted(xs, path_xy[:, 0]), 0, len(xs) - 1)
    iy = np.clip(np.searchsorted(ys, path_xy[:, 1]), 0, len(ys) - 1)
    return float(clearance[ix, iy].min())


def sample_routes(
    obstacles: Iterable[Obstacle],
    room_size_xy: tuple[float, float],
    *,
    count: int = 4,
    seed: int = 0,
    body_half_width_m: float = DEFAULT_BODY_HALF_WIDTH_M,
    margin_m: float = DEFAULT_MARGIN_M,
    swept_top_m: float = 1.317,
    resolution_m: float = 0.10,
    speed_mps: float = 0.8,
    min_length_m: float = 2.0,
    max_attempts: int = 200,
) -> list[SampledRoute]:
    """Plan up to ``count`` distinct routes through the scene's free space.

    Raises rather than returning an empty list when the room admits no route at the
    requested clearance: a scene too cluttered to walk is a fact about the scene, and
    silently returning nothing would let a batch report success having generated nothing.
    """
    obstacles = list(obstacles)
    required = body_half_width_m + margin_m
    half_x, half_y = room_size_xy[0] / 2.0, room_size_xy[1] / 2.0
    bounds = (-half_x, -half_y, half_x, half_y)

    xs, ys, clearance = _distance_field(obstacles, bounds, resolution_m, swept_top_m)
    free = clearance >= required
    if not free.any():
        raise RouteSamplingError(
            f"no cell in the room has {required:.3f} m of clearance "
            f"(best is {clearance.max():.3f} m)"
        )

    free_cells = np.argwhere(free)
    rng = np.random.default_rng(seed)
    routes: list[SampledRoute] = []
    seen_endpoints: set[tuple[int, int, int, int]] = set()

    for attempt in range(max_attempts):
        if len(routes) >= count:
            break
        start_cell = tuple(free_cells[rng.integers(len(free_cells))])
        goal_cell = tuple(free_cells[rng.integers(len(free_cells))])
        key = (*start_cell, *goal_cell)
        if key in seen_endpoints:
            continue
        seen_endpoints.add(key)

        straight = math.hypot(
            xs[goal_cell[0]] - xs[start_cell[0]], ys[goal_cell[1]] - ys[start_cell[1]]
        )
        if straight < min_length_m:
            continue

        cells = _plan(free, start_cell, goal_cell)
        if cells is None:
            continue

        grid_path = np.array([[xs[i], ys[j]] for i, j in cells], dtype=np.float64)
        waypoints = _simplify(grid_path, tolerance_m=resolution_m)
        length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=1).sum())
        if length < min_length_m:
            continue

        trajectory = _resample_constant_speed(waypoints, speed_mps)
        if trajectory.shape[0] > MAX_SEGMENT_FRAMES:
            # Too long to generate as one segment. Skipping is right: silently trimming
            # would produce a motion that stops partway with no record of why.
            continue

        routes.append(
            SampledRoute(
                path_xy=waypoints,
                trajectory_xy=trajectory,
                speed_mps=speed_mps,
                min_clearance_m=route_clearance(trajectory, xs, ys, clearance),
                required_clearance_m=required,
                path_length_m=length,
                seed=seed * 1000 + attempt,
            )
        )

    if not routes:
        raise RouteSamplingError(
            f"no route of at least {min_length_m:.2f} m found at {required:.3f} m clearance "
            f"after {max_attempts} attempts"
        )
    return routes
