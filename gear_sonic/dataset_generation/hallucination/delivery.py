"""Small-data delivery-model contract for LFH keypoint responses.

Models predict proposal geometry only.  They never label a scene or replace the physics scorer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .keypoints import SEMANTIC_GROUPS

RESPONSE_COLUMNS = (
    "motion_id",
    "operator",
    "α",
    "keypoint",
    "axis",
    "commanded_mm",
    "executed_mm",
)


@dataclass(frozen=True)
class ResponseRow:
    motion_id: str
    operator: str
    alpha: float
    keypoint: str
    axis: str
    commanded_mm: float
    executed_mm: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ResponseRow":
        row = cls(
            motion_id=str(value.get("motion_id") or "").strip(),
            operator=str(value.get("operator") or "").strip(),
            alpha=float(value.get("α")),
            keypoint=str(value.get("keypoint") or "").strip(),
            axis=str(value.get("axis") or "").strip(),
            commanded_mm=float(value.get("commanded_mm")),
            executed_mm=float(value.get("executed_mm")),
        )
        row.validate()
        return row

    def validate(self) -> None:
        if not self.motion_id or not self.operator or not self.axis:
            raise ValueError("motion_id, operator, and axis are required")
        if self.keypoint not in SEMANTIC_GROUPS:
            raise ValueError(f"unknown semantic keypoint {self.keypoint!r}")
        for field, value in (
            ("alpha", self.alpha),
            ("commanded_mm", self.commanded_mm),
            ("executed_mm", self.executed_mm),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.commanded_mm < 0:
            raise ValueError("commanded_mm is a displacement magnitude and must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "motion_id": self.motion_id,
            "operator": self.operator,
            "α": self.alpha,
            "keypoint": self.keypoint,
            "axis": self.axis,
            "commanded_mm": self.commanded_mm,
            "executed_mm": self.executed_mm,
        }


def read_responses(path: Path) -> list[ResponseRow]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RESPONSE_COLUMNS:
            raise ValueError(f"{path}: expected response columns {RESPONSE_COLUMNS}")
        return [ResponseRow.from_mapping(row) for row in reader]


def append_responses(path: Path, rows: Iterable[ResponseRow]) -> int:
    """Append evidence-backed rows while preserving the exact sidecar schema."""

    batch = list(rows)
    for row in batch:
        row.validate()
    if not batch:
        return 0
    exists = path.exists()
    if exists:
        read_responses(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerows(row.to_dict() for row in batch)
    return len(batch)


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks = [
        [float(value), float(weight), index, index + 1]
        for index, (value, weight) in enumerate(zip(values, weights))
    ]
    cursor = 0
    while cursor < len(blocks) - 1:
        if blocks[cursor][0] <= blocks[cursor + 1][0]:
            cursor += 1
            continue
        left, right = blocks[cursor], blocks[cursor + 1]
        weight = left[1] + right[1]
        merged = [
            (left[0] * left[1] + right[0] * right[1]) / weight,
            weight,
            left[2],
            right[3],
        ]
        blocks[cursor : cursor + 2] = [merged]
        cursor = max(0, cursor - 1)
    fitted = np.empty(len(values), dtype=np.float64)
    for value, _, first, last in blocks:
        fitted[int(first) : int(last)] = value
    return fitted


class DeliveryOutOfSupport(ValueError):
    """A commanded amplitude lies outside the model's calibrated range."""


@dataclass(frozen=True)
class DeliveryModel:
    operator: str
    keypoint: str
    axis: str
    commanded_mm: tuple[float, ...]
    fitted_mm: tuple[float, ...]
    residual_q90_mm: float
    training_motions: tuple[str, ...]
    alpha_levels: tuple[float, ...]
    samples_per_level: tuple[int, ...]
    unsupported_alpha_levels: tuple[float, ...]

    def predict(self, commanded_mm: float) -> tuple[float, float, float]:
        if commanded_mm < 0:
            raise ValueError("commanded_mm must be non-negative")
        # ``np.interp`` clamps, so a 120 mm strong command silently received the 60 mm
        # response *and* an unchanged residual band -- a screening estimate presented with
        # calibrated confidence it does not have.  Out of range is out of support.
        low, high = min(self.commanded_mm), max(self.commanded_mm)
        if not low - 1e-9 <= commanded_mm <= high + 1e-9:
            raise DeliveryOutOfSupport(
                f"{commanded_mm:.3f} mm is outside the calibrated range "
                f"[{low:.3f}, {high:.3f}] mm for {self.operator}/{self.keypoint}/{self.axis}"
            )
        estimate = float(np.interp(commanded_mm, self.commanded_mm, self.fitted_mm))
        return (
            estimate,
            estimate - self.residual_q90_mm,
            estimate + self.residual_q90_mm,
        )


def fit_delivery_models(
    rows: Iterable[ResponseRow], *, min_distinct_commands: int = 3
) -> tuple[dict[tuple[str, str, str], DeliveryModel], dict[tuple[str, str, str], str]]:
    """Fit monotone v0 models, returning explicit refusals for undersupported groups.

    The registered ``alpha`` levels define matched replicates across motions. Grouping by exact
    commanded millimetres is invalid here: the same operator level produces slightly different
    keypoint displacements on each motion, and treating those values as adjacent independent x
    coordinates turns motion-to-motion variability into an artificial steep slope. We therefore
    pool commands and responses by alpha, fit the level means, and estimate residuals against the
    matched-level fit.
    """

    grouped: dict[tuple[str, str, str], list[ResponseRow]] = {}
    for row in rows:
        row.validate()
        grouped.setdefault((row.operator, row.keypoint, row.axis), []).append(row)
    models: dict[tuple[str, str, str], DeliveryModel] = {}
    refusals: dict[tuple[str, str, str], str] = {}
    for key, group in grouped.items():
        levels = sorted({row.alpha for row in group})
        motions = sorted({row.motion_id for row in group})
        if len(levels) < min_distinct_commands or len(motions) < 2:
            refusals[key] = (
                f"needs >= {min_distinct_commands} commanded levels and >= 2 motions; "
                f"has {len(levels)} levels / {len(motions)} motions"
            )
            continue
        all_samples = [[row for row in group if row.alpha == level] for level in levels]
        supported = [
            (level, rows)
            for level, rows in zip(levels, all_samples)
            if len({row.motion_id for row in rows}) >= 2
        ]
        unsupported = [
            level
            for level, rows in zip(levels, all_samples)
            if len({row.motion_id for row in rows}) < 2
        ]
        if len(supported) < min_distinct_commands:
            motions_per_level = [len({row.motion_id for row in rows}) for rows in all_samples]
            refusals[key] = (
                f"needs >= {min_distinct_commands} levels with >= 2 motions each; has "
                + "/".join(str(value) for value in motions_per_level)
            )
            continue
        fitted_levels = [level for level, _ in supported]
        samples = [rows for _, rows in supported]
        x = np.asarray(
            [np.mean([row.commanded_mm for row in rows]) for rows in samples],
            dtype=np.float64,
        )
        if np.any(np.diff(x) <= 1e-6):
            refusals[key] = "matched commanded levels are not strictly separated in millimetres"
            continue
        means = np.asarray(
            [np.mean([row.executed_mm for row in rows]) for rows in samples],
            dtype=np.float64,
        )
        weights = np.asarray([len(rows) for rows in samples], dtype=np.float64)
        fitted = _pava(means, weights)
        residuals = [
            abs(row.executed_mm - float(fitted[level_index]))
            for level_index, rows in enumerate(samples)
            for row in rows
        ]
        models[key] = DeliveryModel(
            operator=key[0],
            keypoint=key[1],
            axis=key[2],
            commanded_mm=tuple(float(value) for value in x),
            fitted_mm=tuple(float(value) for value in fitted),
            residual_q90_mm=float(np.quantile(residuals, 0.9)),
            training_motions=tuple(motions),
            alpha_levels=tuple(float(value) for value in fitted_levels),
            samples_per_level=tuple(len(rows) for rows in samples),
            unsupported_alpha_levels=tuple(float(value) for value in unsupported),
        )
    return models, refusals
