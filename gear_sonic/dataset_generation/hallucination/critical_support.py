"""Trajectory-conditioned support for LFH critical-obstacle proposals.

The support is deterministic geometry, not a physics-verdict model.  It records where an executed
nominal/adapted motion pair leaves a face-placement interval after conservative engineering
margins are deducted on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class CriticalSupport:
    """One finite-face placement interval for one executed motion pair."""

    record_id: str
    source_pair_id: str
    operator: str
    axis_type: str
    binding_keypoint: str
    route_progress: float
    face_along_route_m: float
    face_across_route_m: float
    nominal_reach_m: float
    adapted_reach_m: float
    clear_margin_m: float
    strike_margin_m: float
    context_status: str

    def __post_init__(self) -> None:
        numeric = (
            self.route_progress,
            self.face_along_route_m,
            self.face_across_route_m,
            self.nominal_reach_m,
            self.adapted_reach_m,
            self.clear_margin_m,
            self.strike_margin_m,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("critical-support values must be finite")
        if not 0.0 <= self.route_progress <= 1.0:
            raise ValueError("route_progress must lie in [0, 1]")
        if self.face_along_route_m <= 0 or self.face_across_route_m <= 0:
            raise ValueError("face extents must be positive")
        if self.clear_margin_m < 0 or self.strike_margin_m < 0:
            raise ValueError("engineering margins must be non-negative")
        if self.nominal_reach_m <= self.adapted_reach_m:
            raise ValueError("nominal reach must exceed adapted reach")

    @property
    def raw_width_m(self) -> float:
        return self.nominal_reach_m - self.adapted_reach_m

    @property
    def lower_m(self) -> float:
        """Lowest face coordinate retaining the adapted-clear engineering margin."""

        return self.adapted_reach_m + self.clear_margin_m

    @property
    def upper_m(self) -> float:
        """Highest face coordinate retaining the nominal-strike engineering margin."""

        return self.nominal_reach_m - self.strike_margin_m

    @property
    def engineering_width_m(self) -> float:
        return self.upper_m - self.lower_m

    def coordinate_at(self, quantile: float) -> float:
        """Map a normalized critical difficulty in [0, 1] to a face coordinate."""

        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError("quantile must lie in [0, 1]")
        if self.engineering_width_m <= 0:
            raise ValueError("engineering-margin support is empty")
        return self.lower_m + quantile * self.engineering_width_m

    def normalized_position(self, coordinate_m: float) -> float:
        """Return the trajectory-normalized face position, without clipping."""

        if not math.isfinite(coordinate_m):
            raise ValueError("coordinate_m must be finite")
        if self.engineering_width_m <= 0:
            raise ValueError("engineering-margin support is empty")
        return (coordinate_m - self.lower_m) / self.engineering_width_m

    def to_dict(self, *, quantile: float = 0.5) -> dict[str, object]:
        coordinate = self.coordinate_at(quantile)
        return {
            "record_id": self.record_id,
            "source_pair_id": self.source_pair_id,
            "operator": self.operator,
            "constraint_axis": self.axis_type,
            "binding_keypoint": self.binding_keypoint,
            "route_progress": self.route_progress,
            "face_along_route_m": self.face_along_route_m,
            "face_across_route_m": self.face_across_route_m,
            "nominal_reach_m": self.nominal_reach_m,
            "adapted_reach_m": self.adapted_reach_m,
            "raw_width_mm": 1000 * self.raw_width_m,
            "clear_engineering_margin_mm": 1000 * self.clear_margin_m,
            "strike_engineering_margin_mm": 1000 * self.strike_margin_m,
            "support_lower_m": self.lower_m,
            "support_upper_m": self.upper_m,
            "engineering_width_mm": 1000 * self.engineering_width_m,
            "proposal_quantile": quantile,
            "hard_coordinate_m": coordinate,
            "normalized_position": self.normalized_position(coordinate),
            "context_status": self.context_status,
        }


def widest_per_source(records: Iterable[CriticalSupport]) -> list[CriticalSupport]:
    """Choose one deterministic high-margin support atom per source.

    A shorter along-route face wins exact width ties, limiting temporal exposure while retaining
    the same geometric separation.
    """

    selected: dict[str, CriticalSupport] = {}
    for record in records:
        current = selected.get(record.source_pair_id)
        key = (record.engineering_width_m, -record.face_along_route_m, record.record_id)
        if current is None:
            selected[record.source_pair_id] = record
            continue
        current_key = (
            current.engineering_width_m,
            -current.face_along_route_m,
            current.record_id,
        )
        if key > current_key:
            selected[record.source_pair_id] = record
    return [selected[source] for source in sorted(selected)]


def source_balanced_weights(records: Iterable[CriticalSupport]) -> dict[str, float]:
    """Give every independent source equal total proposal mass."""

    rows = list(records)
    by_source: dict[str, list[CriticalSupport]] = {}
    for row in rows:
        by_source.setdefault(row.source_pair_id, []).append(row)
    if not by_source:
        return {}
    source_mass = 1.0 / len(by_source)
    return {row.record_id: source_mass / len(by_source[row.source_pair_id]) for row in rows}
