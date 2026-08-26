"""Exact semantic reach of executed capsules against finite LFH faces."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .keypoints import SEMANTIC_GROUPS, SemanticCapsuleTracks

DEFAULT_STATION_SPACING_M = 0.02


def _quadratic_roots(a: float, b: float, c: float) -> tuple[float, ...]:
    if abs(a) < 1e-15:
        return () if abs(b) < 1e-15 else (-c / b,)
    discriminant = b * b - 4 * a * c
    if -1e-12 < discriminant < 0:
        discriminant = 0.0
    if discriminant < 0:
        return ()
    root = math.sqrt(discriminant)
    return ((-b - root) / (2 * a), (-b + root) / (2 * a))


def capsule_height_over_rectangle(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rectangle_xy: tuple[float, float, float, float],
) -> float:
    """Highest capsule point whose XY projection lies in a finite rectangle.

    The capsule axis is split only where it crosses a rectangle edge.  Within each piece,
    squared XY distance to the rectangle is quadratic, so endpoints and analytic stationary
    points suffice; no spatial sampling or obstacle-height search is used.
    """

    lower = rectangle_xy[:2]
    upper = rectangle_xy[2:]
    origin = np.asarray(start, dtype=np.float64)
    delta = np.asarray(end, dtype=np.float64) - origin
    breaks = [0.0, 1.0]
    for axis in range(2):
        if abs(delta[axis]) > 1e-15:
            breaks.extend(
                (edge - origin[axis]) / delta[axis] for edge in (lower[axis], upper[axis])
            )
    breaks = sorted({max(0.0, min(1.0, value)) for value in breaks if -1e-12 <= value <= 1 + 1e-12})
    best = -math.inf
    for first, last in zip(breaks[:-1], breaks[1:]):
        middle = 0.5 * (first + last)
        quad_a = quad_b = quad_c = 0.0
        for axis in range(2):
            coordinate = origin[axis] + delta[axis] * middle
            if coordinate < lower[axis]:
                distance_0, distance_1 = lower[axis] - origin[axis], -delta[axis]
            elif coordinate > upper[axis]:
                distance_0, distance_1 = origin[axis] - upper[axis], delta[axis]
            else:
                continue
            quad_a += distance_1 * distance_1
            quad_b += 2 * distance_0 * distance_1
            quad_c += distance_0 * distance_0

        feasibility = sorted(
            [
                first,
                *(
                    root
                    for root in _quadratic_roots(quad_a, quad_b, quad_c - radius * radius)
                    if first < root < last
                ),
                last,
            ]
        )
        for feasible_first, feasible_last in zip(feasibility[:-1], feasibility[1:]):
            probe = 0.5 * (feasible_first + feasible_last)
            if quad_a * probe * probe + quad_b * probe + quad_c > radius * radius + 1e-12:
                continue
            candidates = [feasible_first, feasible_last]
            velocity_z = float(delta[2])
            if abs(velocity_z) < 1e-15:
                if quad_a > 1e-15:
                    candidates.append(-quad_b / (2 * quad_a))
            else:
                candidates.extend(
                    _quadratic_roots(
                        -(velocity_z * velocity_z * quad_a + quad_a * quad_a),
                        -(velocity_z * velocity_z * quad_b + quad_a * quad_b),
                        velocity_z * velocity_z * (radius * radius - quad_c) - quad_b * quad_b / 4,
                    )
                )
            for value in candidates:
                if value < feasible_first - 1e-10 or value > feasible_last + 1e-10:
                    continue
                distance_sq = quad_a * value * value + quad_b * value + quad_c
                if distance_sq > radius * radius + 1e-10:
                    continue
                if feasible_first + 1e-9 < value < feasible_last - 1e-9:
                    left = velocity_z * math.sqrt(max(0.0, radius * radius - distance_sq))
                    right = quad_a * value + quad_b / 2
                    if abs(left - right) > 1e-8:
                        continue
                best = max(
                    best,
                    origin[2]
                    + delta[2] * value
                    + math.sqrt(max(0.0, radius * radius - distance_sq)),
                )
    return best


@dataclass(frozen=True)
class FaceReach:
    """Per-group extrema and attribution for one finite face footprint."""

    axis_type: str
    station_xy_m: tuple[float, float]
    per_keypoint_reach_m: dict[str, float]
    per_keypoint_frame: dict[str, int]
    per_keypoint_owner: dict[str, str]

    @property
    def binding_keypoint(self) -> str:
        return max(self.per_keypoint_reach_m, key=self.per_keypoint_reach_m.__getitem__)

    @property
    def reach_m(self) -> float:
        return self.per_keypoint_reach_m[self.binding_keypoint]

    @property
    def critical_frame(self) -> int:
        return self.per_keypoint_frame[self.binding_keypoint]

    @property
    def binding_owner(self) -> str:
        return self.per_keypoint_owner[self.binding_keypoint]


@dataclass(frozen=True)
class ReachProfile:
    """Reach at fixed route stations; each point retains exact frame attribution."""

    axis_type: str
    route_axis: str
    spacing_m: float
    points: tuple[FaceReach, ...]


def overhead_face_reach(
    tracks: SemanticCapsuleTracks,
    station_xy_m: tuple[float, float],
    route_axis: str,
    along_route_m: float,
    across_route_m: float,
    *,
    require_all_groups: bool = True,
) -> FaceReach:
    """Compute semantic overhead reach for the exact finite face footprint.

    ``require_all_groups`` guards the single-face API, where a group that misses the footprint
    means the caller asked about the wrong face.  A reach *profile* sweeps stations past the
    ends of the route, where a face shorter than the body's along-route spread legitimately
    misses the wrists; there the group keeps ``-inf`` for that station instead of killing the
    whole profile.
    """

    if route_axis not in ("x", "y"):
        raise ValueError("route_axis must be x or y")
    if along_route_m <= 0 or across_route_m <= 0:
        raise ValueError("face extents must be positive")
    sizes = (
        (along_route_m, across_route_m) if route_axis == "x" else (across_route_m, along_route_m)
    )
    rectangle = (
        station_xy_m[0] - sizes[0] / 2,
        station_xy_m[1] - sizes[1] / 2,
        station_xy_m[0] + sizes[0] / 2,
        station_xy_m[1] + sizes[1] / 2,
    )
    reaches = {group: -math.inf for group in SEMANTIC_GROUPS}
    frames = {group: -1 for group in SEMANTIC_GROUPS}
    owners = {group: "" for group in SEMANTIC_GROUPS}
    for frame in range(tracks.frames):
        for capsule in range(tracks.capsules):
            value = capsule_height_over_rectangle(
                tracks.starts[frame, capsule],
                tracks.ends[frame, capsule],
                float(tracks.radii[capsule]),
                rectangle,
            )
            group = tracks.groups[capsule]
            if value > reaches[group]:
                reaches[group] = value
                frames[group] = frame
                owners[group] = tracks.owners[capsule]
    missing = [group for group, value in reaches.items() if not math.isfinite(value)]
    if missing and require_all_groups:
        raise ValueError(f"face footprint misses semantic groups: {missing}")
    return FaceReach("overhead", station_xy_m, reaches, frames, owners)


def overhead_reach_profile(
    tracks: SemanticCapsuleTracks,
    route_axis: str,
    along_route_m: float,
    across_route_m: float,
    *,
    station_spacing_m: float = DEFAULT_STATION_SPACING_M,
) -> ReachProfile:
    """Evaluate exact overhead reach at fixed stations along the executed route."""

    if station_spacing_m <= 0:
        raise ValueError("station_spacing_m must be positive")
    axis = 0 if route_axis == "x" else 1
    other = 1 - axis
    low, high = tracks.root_pos_w[:, axis].min(), tracks.root_pos_w[:, axis].max()
    stations = np.arange(low, high + station_spacing_m / 2, station_spacing_m)
    points = []
    for station in stations:
        frame = int(np.argmin(np.abs(tracks.root_pos_w[:, axis] - station)))
        xy = [0.0, 0.0]
        xy[axis] = float(station)
        xy[other] = float(tracks.root_pos_w[frame, other])
        points.append(
            overhead_face_reach(
                tracks,
                tuple(xy),
                route_axis,
                along_route_m,
                across_route_m,
                require_all_groups=False,
            )
        )
    return ReachProfile("overhead", route_axis, station_spacing_m, tuple(points))


def lateral_face_reach(
    tracks: SemanticCapsuleTracks,
    station_xy_m: tuple[float, float],
    route_axis: str,
    along_route_m: float,
    band_z_m: tuple[float, float],
    side: str,
) -> FaceReach:
    """One-sided capsule reach across heading within a route slab and height band."""

    if route_axis not in ("x", "y") or side not in ("left", "right"):
        raise ValueError("route_axis must be x/y and side must be left/right")
    if along_route_m <= 0 or band_z_m[1] <= band_z_m[0]:
        raise ValueError("face extent and height band must be positive")
    route = 0 if route_axis == "x" else 1
    slab_low = station_xy_m[route] - along_route_m / 2
    slab_high = station_xy_m[route] + along_route_m / 2
    rectangle = (slab_low, band_z_m[0], slab_high, band_z_m[1])
    quat = tracks.root_quat_w
    w, x, y, z = (quat[:, index] for index in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lateral = np.stack((-np.sin(yaw), np.cos(yaw)), axis=1)
    sign = 1.0 if side == "left" else -1.0
    reaches = {group: -math.inf for group in SEMANTIC_GROUPS}
    frames = {group: -1 for group in SEMANTIC_GROUPS}
    owners = {group: "" for group in SEMANTIC_GROUPS}
    for frame in range(tracks.frames):
        root_xy = tracks.root_pos_w[frame, :2]
        for capsule in range(tracks.capsules):
            radius = float(tracks.radii[capsule])
            start, end = tracks.starts[frame, capsule], tracks.ends[frame, capsule]
            route_low = min(start[route], end[route]) - radius
            route_high = max(start[route], end[route]) + radius
            vertical_low = min(start[2], end[2]) - radius
            vertical_high = max(start[2], end[2]) + radius
            if (
                route_high < slab_low
                or route_low > slab_high
                or vertical_high < band_z_m[0]
                or vertical_low > band_z_m[1]
            ):
                continue
            # The reach must be the extremum *of the part of the capsule that occupies the
            # face*, not of the whole capsule.  Taking whole-capsule endpoints after an
            # AABB overlap test reported an arm's far end as the lateral reach of a face the
            # arm only clipped -- an 8x overestimate on a wrist capsule crossing the slab
            # diagonally, and a finite reach for capsules whose in-slab portion lies entirely
            # outside the height band.  Projecting onto the per-frame heading is linear, so
            # the transformed segment is still a segment and the exact solver applies.
            offsets = np.stack((start[:2] - root_xy, end[:2] - root_xy))
            heights = sign * np.einsum("nd,d->n", offsets, lateral[frame])
            value = capsule_height_over_rectangle(
                np.asarray((start[route], start[2], heights[0])),
                np.asarray((end[route], end[2], heights[1])),
                radius,
                rectangle,
            )
            if not math.isfinite(value):
                continue
            group = tracks.groups[capsule]
            if value > reaches[group]:
                reaches[group] = value
                frames[group] = frame
                owners[group] = tracks.owners[capsule]
    if not any(math.isfinite(value) for value in reaches.values()):
        raise ValueError("no capsule occupies the requested route slab and height band")
    return FaceReach("lateral_one_sided", station_xy_m, reaches, frames, owners)


def lateral_gap_reach(
    tracks: SemanticCapsuleTracks,
    station_xy_m: tuple[float, float],
    route_axis: str,
    along_route_m: float,
    band_z_m: tuple[float, float],
) -> FaceReach:
    """Required width of a symmetric, world-fixed gap at one finite station.

    ``lateral_face_reach`` is a one-sided, root-relative response diagnostic.  A generated
    ``lateral_gap`` scene instead authors two world-fixed faces whose scalar coordinate is their
    full separation.  This function evaluates that scene contract exactly: for each semantic
    group it finds the furthest capsule surface from the gap centre on either side, conditioned on
    overlap with the finite route-by-height face, then doubles that half-width.

    Groups that never overlap the face rectangle receive zero reach.  They are therefore retained
    in the all-keypoint screen without being allowed to bind a height band they do not occupy.
    """

    if route_axis not in ("x", "y"):
        raise ValueError("route_axis must be x or y")
    if along_route_m <= 0 or band_z_m[1] <= band_z_m[0]:
        raise ValueError("face extent and height band must be positive")
    route = 0 if route_axis == "x" else 1
    cross = 1 - route
    rectangle = (
        station_xy_m[route] - along_route_m / 2,
        band_z_m[0],
        station_xy_m[route] + along_route_m / 2,
        band_z_m[1],
    )
    reaches = {group: 0.0 for group in SEMANTIC_GROUPS}
    frames = {group: -1 for group in SEMANTIC_GROUPS}
    owners = {group: "" for group in SEMANTIC_GROUPS}
    for frame in range(tracks.frames):
        for capsule in range(tracks.capsules):
            start = tracks.starts[frame, capsule]
            end = tracks.ends[frame, capsule]
            radius = float(tracks.radii[capsule])
            half_width = -math.inf
            for direction in (-1.0, 1.0):
                transformed_start = np.asarray(
                    (
                        start[route],
                        start[2],
                        direction * (start[cross] - station_xy_m[cross]),
                    )
                )
                transformed_end = np.asarray(
                    (
                        end[route],
                        end[2],
                        direction * (end[cross] - station_xy_m[cross]),
                    )
                )
                half_width = max(
                    half_width,
                    capsule_height_over_rectangle(
                        transformed_start,
                        transformed_end,
                        radius,
                        rectangle,
                    ),
                )
            if not math.isfinite(half_width):
                continue
            value = 2.0 * half_width
            group = tracks.groups[capsule]
            if value > reaches[group]:
                reaches[group] = value
                frames[group] = frame
                owners[group] = tracks.owners[capsule]
    if max(reaches.values()) <= 0:
        raise ValueError("no capsule occupies the requested route slab and height band")
    return FaceReach("lateral_gap", station_xy_m, reaches, frames, owners)


def lateral_reach_profile(
    tracks: SemanticCapsuleTracks,
    route_axis: str,
    along_route_m: float,
    band_z_m: tuple[float, float],
    side: str,
    *,
    station_spacing_m: float = DEFAULT_STATION_SPACING_M,
) -> ReachProfile:
    """Evaluate one-sided lateral group reach at fixed route stations."""

    if station_spacing_m <= 0:
        raise ValueError("station_spacing_m must be positive")
    axis = 0 if route_axis == "x" else 1
    other = 1 - axis
    low, high = tracks.root_pos_w[:, axis].min(), tracks.root_pos_w[:, axis].max()
    stations = np.arange(low, high + station_spacing_m / 2, station_spacing_m)
    points = []
    for station in stations:
        frame = int(np.argmin(np.abs(tracks.root_pos_w[:, axis] - station)))
        xy = [0.0, 0.0]
        xy[axis] = float(station)
        xy[other] = float(tracks.root_pos_w[frame, other])
        points.append(
            lateral_face_reach(tracks, tuple(xy), route_axis, along_route_m, band_z_m, side)
        )
    return ReachProfile("lateral_one_sided", route_axis, station_spacing_m, tuple(points))
