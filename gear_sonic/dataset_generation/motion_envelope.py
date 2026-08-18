"""Summarise what shape a motion sweeps, so pairs can be proposed without regenerating.

Every counterfactual family so far was built by picking two motions by hand and hoping their
swept volumes separated. That does not scale, and it wastes the corpus: 131 distinct executed
trajectories already exist, and the pair that makes a family is very likely among them.

The obstacle in each regime is a scalar, and what decides whether a motion clears it is one
number:

* **overhead** -- the lowest the robot's silhouette peak ever gets. A shelf below that stops
  it; a shelf above it does not.
* **lateral** -- the narrowest the robot ever is, measured across its direction of travel.
* **floor** -- the trailing foot's swing apex, which is what a bar on the ground has to clear.

Those three numbers are cheap. The expensive part -- binary-searching a real obstacle against
the full capsule model -- then runs only on pairs the cheap screen already likes, which is the
difference between minutes and hours over a corpus this size.

**A large gap is not enough.** Two motions only form a family if they are doing the *same
task*: comparable start, goal, route and duration. A duck walking 2 m straight and a brisk
walk turning through 5 m differ in head height, but a shelf that stops one and passes the
other says nothing about behaviour -- it says the two motions went to different places. The
compatibility check exists to refuse those, and it does most of the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .swept_volume import G1_COLLISION_CAPSULES, body_capsules_world

#: Regimes, and the signature field whose *spread* between two motions opens a window.
#: The sign says which direction the adapted motion must differ in.
REGIME_FIELD = {
    "overhead": ("min_silhouette_peak_m", -1),   # adapted must go lower
    "lateral": ("min_half_width_m", -1),         # adapted must be narrower
    "floor": ("min_foot_apex_m", +1),            # adapted must lift higher
}

#: Below this the two motions are doing the same thing and no obstacle separates them.
MIN_USEFUL_SPREAD_M = 0.03

#: Task-compatibility limits. A pair failing any of these is not a counterfactual, whatever
#: its geometric spread.
MAX_START_DISTANCE_M = 0.60
MAX_GOAL_DISTANCE_M = 1.50
MAX_PATH_LENGTH_RATIO = 1.60
MAX_DURATION_RATIO = 1.60
MAX_HEADING_DIFFERENCE_RAD = 0.90


@dataclass(frozen=True)
class EnvelopeSignature:
    """What one executed motion does, reduced to the numbers pairing needs."""

    episode_id: str
    behaviour: str
    frames: int
    duration_s: float
    start_xy: tuple[float, float]
    goal_xy: tuple[float, float]
    path_length_m: float
    net_displacement_m: float
    heading_change_rad: float
    mean_speed_mps: float
    #: Lowest the robot's highest collision point ever gets -- the overhead discriminator.
    min_silhouette_peak_m: float
    #: Narrowest half-width across the direction of travel -- the lateral discriminator.
    min_half_width_m: float
    #: Swing apex of the *lower* foot -- the floor discriminator. The trailing foot is what
    #: a bar catches; the leading one clearing says nothing.
    min_foot_apex_m: float


def _heading(root_quat: np.ndarray) -> np.ndarray:
    w, x, y, z = (root_quat[:, i] for i in range(4))
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compute_envelope(
    payload: Mapping,
    episode_id: str,
    behaviour: str = "",
    *,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> EnvelopeSignature:
    """Reduce one recorded trajectory to its signature."""
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    frames = int(root.shape[0])
    fps = float(payload.get("fps", 50.0))
    duration = frames / fps if fps else 0.0

    steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    path_length = float(steps.sum())
    displacement = float(np.linalg.norm(root[-1, :2] - root[0, :2]))
    yaw = _heading(np.asarray(payload["root_quat_w"], dtype=np.float64))
    heading_change = float(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0])

    starts, ends, radii, owners = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=capsules,
    )

    # Silhouette peak: the top of the highest capsule, per frame. The minimum over frames is
    # the lowest shelf this motion could pass under somewhere along its route.
    tops = np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]
    silhouette_peak = float(tops.max(axis=1).min())

    # Half-width across the direction of travel. Heading, not the world y axis: a motion that
    # walks along +y is not two metres wide.
    lateral_axis = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    centres = 0.5 * (starts + ends)
    offsets = centres[:, :, :2] - root[:, None, :2]
    lateral = np.abs(np.einsum("tcd,td->tc", offsets, lateral_axis)) + radii[None, :]
    half_width = float(lateral.max(axis=1).min())

    # Foot apex, taking the lower of the two feet at each foot's own peak.
    names = list(payload["body_names"])
    ankles = [i for i, n in enumerate(names) if "ankle_roll" in n]
    if len(ankles) >= 2:
        body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
        apexes = [float(body_pos[:, i, 2].max()) for i in ankles]
        foot_apex = float(min(apexes))
    else:
        foot_apex = float("nan")

    return EnvelopeSignature(
        episode_id=episode_id,
        behaviour=behaviour,
        frames=frames,
        duration_s=duration,
        start_xy=(float(root[0, 0]), float(root[0, 1])),
        goal_xy=(float(root[-1, 0]), float(root[-1, 1])),
        path_length_m=path_length,
        net_displacement_m=displacement,
        heading_change_rad=heading_change,
        mean_speed_mps=path_length / duration if duration else 0.0,
        min_silhouette_peak_m=silhouette_peak,
        min_half_width_m=half_width,
        min_foot_apex_m=foot_apex,
    )


@dataclass(frozen=True)
class PairCandidate:
    """A proposed nominal/adapted pair, and why it scored as it did."""

    nominal: str
    adapted: str
    regime: str
    #: Predicted window width from the signature alone, before any obstacle search.
    spread_m: float
    #: Task-compatibility penalty subtracted from the spread.
    penalty: float
    score: float
    reasons: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return not self.reasons


def _incompatibilities(a: EnvelopeSignature, b: EnvelopeSignature) -> tuple[str, ...]:
    """Why this pair is not the same task. Empty means it is."""
    problems = []
    start_gap = float(np.linalg.norm(np.subtract(a.start_xy, b.start_xy)))
    goal_gap = float(np.linalg.norm(np.subtract(a.goal_xy, b.goal_xy)))
    if start_gap > MAX_START_DISTANCE_M:
        problems.append(f"starts {start_gap:.2f} m apart")
    if goal_gap > MAX_GOAL_DISTANCE_M:
        problems.append(f"goals {goal_gap:.2f} m apart")
    for label, x, y, limit in (
        ("path length", a.path_length_m, b.path_length_m, MAX_PATH_LENGTH_RATIO),
        ("duration", a.duration_s, b.duration_s, MAX_DURATION_RATIO),
    ):
        low, high = sorted((x, y))
        if low <= 0 or high / low > limit:
            problems.append(f"{label} ratio {high / low:.2f}" if low > 0 else f"{label} zero")
    if abs(a.heading_change_rad - b.heading_change_rad) > MAX_HEADING_DIFFERENCE_RAD:
        problems.append(
            f"heading change differs by "
            f"{abs(a.heading_change_rad - b.heading_change_rad):.2f} rad"
        )
    return tuple(problems)


def score_pair(
    nominal: EnvelopeSignature, adapted: EnvelopeSignature, regime: str
) -> PairCandidate:
    """Score one ordered pair for a regime. Order matters: adapted must be the capable one."""
    if regime not in REGIME_FIELD:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(REGIME_FIELD)}")
    field, direction = REGIME_FIELD[regime]
    spread = direction * (getattr(adapted, field) - getattr(nominal, field))

    reasons = list(_incompatibilities(nominal, adapted))
    if not np.isfinite(spread):
        reasons.append(f"{field} not measurable")
        spread = 0.0
    elif spread < MIN_USEFUL_SPREAD_M:
        reasons.append(f"{field} spread {spread:.3f} m below {MIN_USEFUL_SPREAD_M:.3f} m")

    # Penalise task mismatch even among compatible pairs, so the ranking prefers the cleanest
    # comparison rather than merely the widest window.
    start_gap = float(np.linalg.norm(np.subtract(nominal.start_xy, adapted.start_xy)))
    goal_gap = float(np.linalg.norm(np.subtract(nominal.goal_xy, adapted.goal_xy)))
    duration_gap = abs(nominal.duration_s - adapted.duration_s)
    penalty = 0.05 * start_gap + 0.03 * goal_gap + 0.01 * duration_gap

    return PairCandidate(
        nominal=nominal.episode_id,
        adapted=adapted.episode_id,
        regime=regime,
        spread_m=float(spread),
        penalty=float(penalty),
        score=float(spread - penalty),
        reasons=tuple(reasons),
    )


def mine_pairs(
    signatures: Sequence[EnvelopeSignature], regime: str, *, limit: int = 20
) -> list[PairCandidate]:
    """Rank every ordered pair for one regime, keeping only the compatible ones.

    Ordered, not unordered: swapping nominal and adapted is a different proposal, and only
    one of the two directions can be right.
    """
    candidates = [
        score_pair(nominal, adapted, regime)
        for nominal in signatures
        for adapted in signatures
        if nominal.episode_id != adapted.episode_id
    ]
    viable = [candidate for candidate in candidates if candidate.compatible]
    viable.sort(key=lambda candidate: candidate.score, reverse=True)
    return viable[:limit]


# ---- station-aware refinement -------------------------------------------------------------
#
# The signature fields above are minima over the *whole* route, and that makes them
# optimistic: on the one family measured so far they predict a 0.197 m overhead window where
# the obstacle search found 0.053 m, a 3.7x overestimate. The reason is that an obstacle sits
# at one place. A motion whose deepest duck happens three metres past the shelf is no lower
# than a plain walk where the shelf actually is.
#
# Optimistic is the right direction for a screen -- it over-proposes rather than discarding
# real pairs -- but it is the wrong number to rank on, and it leaves the obstacle's position
# unchosen. The builder currently drops the shelf at the path midpoint, which is a guess.
# Scanning stations answers both: it predicts the window far more honestly, and it says where
# to put the obstacle to obtain it.

#: Depth of the obstacle along the direction of travel, in metres. Matches SHELF_SIZE in the
#: family builder: a station is only meaningful relative to how much route the obstacle covers.
DEFAULT_STATION_SPAN_M = 0.5


def silhouette_at_stations(
    payload: Mapping,
    stations: np.ndarray,
    *,
    span: float = DEFAULT_STATION_SPAN_M,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> np.ndarray:
    """Highest point the robot reaches while inside each station's x-slab.

    ``NaN`` where the motion never enters that slab, which is how a station outside one
    motion's route is excluded rather than silently scored as clearance.
    """
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=capsules,
    )
    centres_x = 0.5 * (starts[:, :, 0] + ends[:, :, 0])
    tops = np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]

    peaks = np.full(stations.shape, np.nan)
    for index, station in enumerate(stations):
        inside = np.abs(centres_x - station) <= span / 2.0
        if inside.any():
            peaks[index] = float(tops[inside].max())
    return peaks


def best_overhead_station(
    nominal_payload: Mapping,
    adapted_payload: Mapping,
    *,
    span: float = DEFAULT_STATION_SPAN_M,
    resolution_m: float = 0.05,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> tuple[float, float]:
    """Where along the shared route the overhead window is widest, and how wide.

    Returns ``(station_x, spread_m)``. A spread at or below zero means no station separates
    the two motions and the pair does not form an overhead family however different their
    global minima look.
    """
    def route_x(payload):
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        return float(root[:, 0].min()), float(root[:, 0].max())

    nominal_range, adapted_range = route_x(nominal_payload), route_x(adapted_payload)
    low = max(nominal_range[0], adapted_range[0])
    high = min(nominal_range[1], adapted_range[1])
    if high <= low:
        return float("nan"), 0.0

    stations = np.arange(low, high + resolution_m, resolution_m)
    nominal_peaks = silhouette_at_stations(
        nominal_payload, stations, span=span, capsules=capsules
    )
    adapted_peaks = silhouette_at_stations(
        adapted_payload, stations, span=span, capsules=capsules
    )

    spread = nominal_peaks - adapted_peaks
    if not np.isfinite(spread).any():
        return float("nan"), 0.0
    best = int(np.nanargmax(spread))
    return float(stations[best]), float(spread[best])


def half_width_at_stations(
    payload: Mapping,
    stations: np.ndarray,
    *,
    span: float = DEFAULT_STATION_SPAN_M,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> np.ndarray:
    """Widest the robot gets, across its direction of travel, inside each station's slab.

    The lateral analogue of :func:`silhouette_at_stations`, and it needs to be a separate
    measurement for the same reason: a motion that tucks its arms halfway along its route is
    no narrower than a full arm swing at the station where the gap actually is.

    Width is measured about the root and across the heading, not about the world y axis. The
    G1's measured half-width runs from 0.273 m with arms tucked to 0.664 m at peak arm
    swing, so which frame and which axis are used decides the answer.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    yaw = _heading(np.asarray(payload["root_quat_w"], dtype=np.float64))
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=capsules,
    )
    centres = 0.5 * (starts + ends)
    lateral_axis = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    offsets = centres[:, :, :2] - root[:, None, :2]
    widths = np.abs(np.einsum("tcd,td->tc", offsets, lateral_axis)) + radii[None, :]
    per_frame = widths.max(axis=1)

    peaks = np.full(stations.shape, np.nan)
    for index, station in enumerate(stations):
        inside = np.abs(root[:, 0] - station) <= span / 2.0
        if inside.any():
            peaks[index] = float(per_frame[inside].max())
    return peaks


def best_lateral_station(
    nominal_payload: Mapping,
    adapted_payload: Mapping,
    *,
    span: float = DEFAULT_STATION_SPAN_M,
    resolution_m: float = 0.05,
    capsules: Mapping[str, Sequence] = G1_COLLISION_CAPSULES,
) -> tuple[float, float]:
    """Where along the shared route the lateral window is widest, and how wide."""
    def route_x(payload):
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        return float(root[:, 0].min()), float(root[:, 0].max())

    nominal_range, adapted_range = route_x(nominal_payload), route_x(adapted_payload)
    low = max(nominal_range[0], adapted_range[0])
    high = min(nominal_range[1], adapted_range[1])
    if high <= low:
        return float("nan"), 0.0

    stations = np.arange(low, high + resolution_m, resolution_m)
    nominal_widths = half_width_at_stations(
        nominal_payload, stations, span=span, capsules=capsules
    )
    adapted_widths = half_width_at_stations(
        adapted_payload, stations, span=span, capsules=capsules
    )
    # The nominal motion must be the wider one for a gap to separate them.
    spread = nominal_widths - adapted_widths
    if not np.isfinite(spread).any():
        return float("nan"), 0.0
    best = int(np.nanargmax(spread))
    return float(stations[best]), float(spread[best])
