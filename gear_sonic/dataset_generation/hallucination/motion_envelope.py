"""Per-station directional body extents in the route frame.

The overhead reach solver answers one question — how high does the body reach at this station —
which is the only question a crouch can change. Explaining a *lateral* edit needs the body's
extent to the left and to the right, kept separate, and explaining any edit at all in a scene with
several obstacles needs those extents at every station rather than one.

This module produces the descriptor both the hallucinator and the differentiable decoder consume:
for each route station, how far the body reaches up, left and right, in the frame of the executed
route. Everything downstream is expressed against it, which is what lets an obstacle be placed on
the side the motion actually leans away from instead of always overhead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..local_adaptation import route_progress
from .keypoints import SemanticCapsuleTracks

#: Directions an obstacle face can come from, in the route frame.
DIRECTIONS = ("overhead", "left", "right")


@dataclass(frozen=True)
class MotionEnvelope:
    """Directional extents of one motion at fixed route stations.

    ``up_m`` is world height. ``left_m`` and ``right_m`` are signed distances from the route
    centreline, both reported positive: a larger value means the body sticks out further on that
    side. ``station_xy_m`` and ``yaw_rad`` carry the route frame each station was measured in, so
    an obstacle expressed in station-local coordinates can be placed back into the world.
    """

    motion_id: str
    fractions: np.ndarray
    station_xy_m: np.ndarray
    yaw_rad: np.ndarray
    up_m: np.ndarray
    left_m: np.ndarray
    right_m: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.fractions)
        for name in ("up_m", "left_m", "right_m", "yaw_rad"):
            if getattr(self, name).shape != (count,):
                raise ValueError(f"{name} must have one value per station")
        if self.station_xy_m.shape != (count, 2):
            raise ValueError("station_xy_m must be (stations, 2)")
        if not np.isfinite(self.up_m).all():
            raise ValueError("overhead extents must be finite")

    @property
    def stations(self) -> int:
        return len(self.fractions)

    def stack(self) -> np.ndarray:
        """(3, stations) descriptor in the fixed :data:`DIRECTIONS` order."""
        return np.stack((self.up_m, self.left_m, self.right_m), axis=0)


def _yaw_at(root_xy: np.ndarray, index: int, half_window: int = 3) -> float:
    low = max(0, index - half_window)
    high = min(len(root_xy) - 1, index + half_window)
    tangent = root_xy[high] - root_xy[low]
    if float(np.linalg.norm(tangent)) <= 1e-9:
        raise ValueError("route tangent is degenerate")
    return float(np.arctan2(tangent[1], tangent[0]))


def extract_envelope(
    tracks: SemanticCapsuleTracks,
    motion_id: str,
    *,
    fractions: np.ndarray,
    along_route_m: float = 0.20,
) -> MotionEnvelope:
    """Measure up/left/right extents at each requested route fraction.

    A station's window spans ``along_route_m`` of executed-route arclength, so the measurement
    remains local on curved routes instead of admitting distant frames with a similar tangent
    projection. Capsule radii are added, so the extents describe the body surface rather than
    its axis.
    """
    root = np.asarray(tracks.root_pos_w[:, :2], dtype=np.float64)
    progress = route_progress(root)
    segment_length = np.linalg.norm(np.diff(root, axis=0), axis=1)
    arclength = np.concatenate(([0.0], np.cumsum(segment_length)))
    up, left, right = [], [], []
    stations, yaws = [], []
    for fraction in np.asarray(fractions, dtype=np.float64):
        index = int(np.argmin(np.abs(progress - fraction)))
        station = root[index]
        yaw = _yaw_at(root, index)
        lateral = np.asarray((-np.sin(yaw), np.cos(yaw)))

        # Select frames by *route arclength*. Filtering all trajectory points only by their
        # projection onto this tangent aliases distant parts of a curved/looping route into
        # the same slab and can report metre-scale "body width" for an ordinary G1. The root
        # arclength window keeps the measurement local to this traversal event.
        frame_mask = np.abs(arclength - arclength[index]) <= along_route_m / 2.0
        if not frame_mask.any():  # Defensive: the nearest station frame must normally match.
            frame_mask[index] = True
        selected_starts = tracks.starts[frame_mask]
        selected_ends = tracks.ends[frame_mask]
        points = np.concatenate(
            (selected_starts.reshape(-1, 3), selected_ends.reshape(-1, 3)), axis=0
        )
        radii = np.tile(tracks.radii, int(frame_mask.sum()) * 2)
        offsets = points[:, :2] - station[None, :]
        across = offsets @ lateral
        up.append(float((points[:, 2] + radii).max()))
        left.append(float((across + radii).max()))
        right.append(float((-across + radii).max()))
        stations.append(station)
        yaws.append(yaw)
    return MotionEnvelope(
        motion_id=motion_id,
        fractions=np.asarray(fractions, dtype=np.float64),
        station_xy_m=np.asarray(stations, dtype=np.float64),
        yaw_rad=np.asarray(yaws, dtype=np.float64),
        up_m=np.asarray(up, dtype=np.float64),
        left_m=np.asarray(left, dtype=np.float64),
        right_m=np.asarray(right, dtype=np.float64),
    )


@dataclass(frozen=True)
class CandidateSet:
    """The motions a scene must choose between, and what each costs to perform.

    The decoder's job is to pick the cheapest candidate that the obstacles leave feasible. That is
    only a meaningful question when there is something to choose *between*: a nominal, and edits of
    different kinds and depths. With one candidate the inverse problem is trivial; with a nominal,
    crouches and one-sided arm tucks it is genuinely multimodal, because an overhead obstacle, a
    left obstacle and a right obstacle explain different observed edits.
    """

    envelopes: tuple[MotionEnvelope, ...]
    costs: np.ndarray
    labels: tuple[str, ...]
    observed_index: int

    def __post_init__(self) -> None:
        if len(self.envelopes) != len(self.costs) or len(self.envelopes) != len(self.labels):
            raise ValueError("envelopes, costs and labels must align")
        if not 0 <= self.observed_index < len(self.envelopes):
            raise ValueError("observed_index must select one of the candidates")
        if self.costs[0] != 0.0:
            raise ValueError("candidate 0 must be the nominal, at zero edit cost")
        stations = {envelope.stations for envelope in self.envelopes}
        if len(stations) != 1:
            raise ValueError("all candidates must share one station grid")

    @property
    def stations(self) -> int:
        return self.envelopes[0].stations

    def tensor(self) -> np.ndarray:
        """(candidates, 3, stations) extent block."""
        return np.stack([envelope.stack() for envelope in self.envelopes], axis=0)
