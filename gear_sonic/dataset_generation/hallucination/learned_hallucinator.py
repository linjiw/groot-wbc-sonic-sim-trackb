"""An LfLH-style learned hallucinator for the humanoid overhead case, and its honest baseline.

LfLH learns `q_psi(C | p)` by pushing sampled obstacles through a **fixed differentiable planner**
and reconstructing the trajectory. That design is what gives hallucinated obstacles their
semantics: obstacle parameters can only move the decoder's output by actually changing planning
cost, so encoder and decoder cannot collude.

Our real decoder -- a frozen SONIC policy in Isaac -- is neither differentiable, cheap, nor
deterministic, so the LfLH training loop cannot be run against it. This module runs the loop
against a **differentiable surrogate** of the one decision the real decoder makes, in order to
answer a methodological question rather than to ship a hallucinator:

    Given the same trajectories, does a learned Gaussian hallucinator recover the set of obstacle
    placements that explain a humanoid counterfactual, or does it collapse onto one of them?

The comparison is unusually clean here because, unlike in the 2-D mobile-robot case, **the exact
answer is known**. For the overhead axis the set of face coordinates that make the nominal strike
and the adaptation clear is the closed-form interval `[R_adapted + delta, R_nominal - delta]` at
each station. So we can measure coverage against ground truth instead of guessing at it.

Nothing here proposes a scene for physics. The surrogate is a study instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class ReachPair:
    """Nominal and adapted overhead reach sampled at shared route stations, in metres."""

    source_id: str
    stations_m: np.ndarray
    nominal_reach_m: np.ndarray
    adapted_reach_m: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            self.stations_m.shape,
            self.nominal_reach_m.shape,
            self.adapted_reach_m.shape,
        }
        if len(shapes) != 1 or self.stations_m.ndim != 1:
            raise ValueError("stations and both reach profiles must share one 1-D shape")
        if not np.isfinite(self.stations_m).all():
            raise ValueError("station coordinates must be finite")

    def window_at(self, index: int, margin_m: float) -> tuple[float, float]:
        """Closed-form interval of face coordinates that separate the pair at one station."""
        return (
            float(self.adapted_reach_m[index]) + margin_m,
            float(self.nominal_reach_m[index]) - margin_m,
        )

    def feasible_mask(self, margin_m: float) -> np.ndarray:
        lower = self.adapted_reach_m + margin_m
        upper = self.nominal_reach_m - margin_m
        return np.isfinite(lower) & np.isfinite(upper) & (upper > lower)

    def support_area_m2(self, margin_m: float) -> float:
        """Total (station x coordinate) area of the exact feasible set, for coverage scoring."""
        mask = self.feasible_mask(margin_m)
        if not mask.any():
            return 0.0
        widths = (self.nominal_reach_m - margin_m) - (self.adapted_reach_m + margin_m)
        spacing = float(np.median(np.diff(self.stations_m))) if len(self.stations_m) > 1 else 1.0
        return float(np.clip(widths[mask], 0.0, None).sum() * spacing)


class Hallucinator(nn.Module):
    """Conv1D encoder over the reach pair -> Gaussian over (station index, face coordinate).

    Deliberately the same shape as LfLH's: temporal convolutions over the trajectory, then a fully
    connected head emitting means and log-variances. The output is two scalars rather than ten
    ellipses because the humanoid overhead constraint has two free parameters, not thirty.
    """

    def __init__(self, stations: int, hidden: int = 32, min_log_sigma: float = -6.0):
        super().__init__()
        self.stations = stations
        self.min_log_sigma = min_log_sigma
        self.encoder = nn.Sequential(
            nn.Conv1d(2, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 4))

    def forward(self, profiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(profiles).mean(dim=-1)
        raw = self.head(features)
        mean = raw[:, :2]
        log_sigma = raw[:, 2:].clamp(min=self.min_log_sigma, max=1.0)
        return mean, log_sigma


def _soft_gather(profile: torch.Tensor, position: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Differentiably read a profile at a continuous station index."""
    indices = torch.arange(profile.shape[-1], device=profile.device, dtype=profile.dtype)
    weights = torch.softmax(-sharpness * (indices[None, :] - position[:, None]) ** 2, dim=-1)
    return (weights * profile).sum(dim=-1)


@dataclass
class SurrogateDecoder:
    """The fixed, parameter-free, differentiable stand-in for physics.

    It answers only what the real decoder answers: at this face, does the nominal strike and does
    the adaptation clear? Both are read from the executed reach profiles, so an obstacle parameter
    can only change the output by actually changing which body the face intersects -- the property
    that stops encoder and decoder colluding.
    """

    margin_m: float = 0.018044
    temperature_m: float = 0.004
    station_sharpness: float = 2.0

    def __call__(
        self,
        nominal: torch.Tensor,
        adapted: torch.Tensor,
        station: torch.Tensor,
        coordinate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nominal_at = _soft_gather(nominal, station, self.station_sharpness)
        adapted_at = _soft_gather(adapted, station, self.station_sharpness)
        strikes = torch.sigmoid((nominal_at - self.margin_m - coordinate) / self.temperature_m)
        clears = torch.sigmoid((coordinate - self.margin_m - adapted_at) / self.temperature_m)
        return strikes, clears


@dataclass
class TrainingReport:
    steps: int
    final_loss: float
    mean_sigma_station: float
    mean_sigma_coordinate_mm: float
    history: list[dict] = field(default_factory=list)


def train(
    pairs: list[ReachPair],
    *,
    steps: int = 600,
    learning_rate: float = 3e-3,
    samples: int = 8,
    prior_weight: float = 1.0,
    seed: int = 0,
    decoder: SurrogateDecoder | None = None,
) -> tuple[Hallucinator, TrainingReport]:
    """Run the LfLH objective: reconstruct through a fixed decoder, with prior and clearance terms."""
    torch.manual_seed(seed)
    decoder = decoder or SurrogateDecoder()
    stations = len(pairs[0].stations_m)
    nominal = torch.tensor(np.stack([pair.nominal_reach_m for pair in pairs]), dtype=torch.float32)
    adapted = torch.tensor(np.stack([pair.adapted_reach_m for pair in pairs]), dtype=torch.float32)
    profiles = torch.stack((nominal, adapted), dim=1)
    centre = torch.tensor((stations - 1) / 2.0, dtype=torch.float32)

    model = Hallucinator(stations)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    loss_value = float("nan")
    for step in range(steps):
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
        total = 0.0
        for _ in range(samples):
            noise = torch.randn_like(mean)
            sample = mean + sigma * noise
            station = sample[:, 0] * (stations / 6.0) + centre
            coordinate = sample[:, 1] * 0.1 + adapted.mean(dim=-1)
            strikes, clears = decoder(nominal, adapted, station, coordinate)
            # Reconstruction: the observed counterfactual is "the edit was necessary and enough".
            reconstruction = -(torch.log(strikes + 1e-8) + torch.log(clears + 1e-8))
            total = total + reconstruction.mean()
        loss = total / samples
        # Prior: keep the station on the route and the face near the body, as LfLH does.
        prior = ((station - centre) / stations).pow(2).mean() + (
            coordinate - adapted.mean(dim=-1)
        ).pow(2).mean()
        loss = loss + prior_weight * prior
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        loss_value = float(loss.detach())
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            history.append(
                {
                    "step": step,
                    "loss": loss_value,
                    "sigma_station": float(sigma[:, 0].mean().detach()),
                    "sigma_coordinate_mm": float(1000 * 0.1 * sigma[:, 1].mean().detach()),
                }
            )

    with torch.no_grad():
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
    return model, TrainingReport(
        steps=steps,
        final_loss=loss_value,
        mean_sigma_station=float(sigma[:, 0].mean()),
        mean_sigma_coordinate_mm=float(1000 * 0.1 * sigma[:, 1].mean()),
        history=history,
    )


def sample_placements(
    model: Hallucinator, pairs: list[ReachPair], *, count: int, seed: int = 0
) -> dict[str, np.ndarray]:
    """Draw obstacle placements per source: array of (station index, coordinate m)."""
    torch.manual_seed(seed)
    stations = len(pairs[0].stations_m)
    nominal = torch.tensor(np.stack([pair.nominal_reach_m for pair in pairs]), dtype=torch.float32)
    adapted = torch.tensor(np.stack([pair.adapted_reach_m for pair in pairs]), dtype=torch.float32)
    profiles = torch.stack((nominal, adapted), dim=1)
    centre = (stations - 1) / 2.0
    with torch.no_grad():
        mean, log_sigma = model(profiles)
        sigma = log_sigma.exp()
        draws = mean[:, None, :] + sigma[:, None, :] * torch.randn(mean.shape[0], count, 2)
    station = draws[:, :, 0].numpy() * (stations / 6.0) + centre
    coordinate = draws[:, :, 1].numpy() * 0.1 + adapted.mean(dim=-1).numpy()[:, None]
    return {
        pair.source_id: np.stack((station[row], coordinate[row]), axis=-1)
        for row, pair in enumerate(pairs)
    }


def coverage(
    placements: np.ndarray, pair: ReachPair, *, margin_m: float, bins: int = 24
) -> dict[str, float]:
    """How much of the exact feasible set a set of placements actually reaches.

    ``valid_rate`` is the share of samples that land inside the closed-form window at their own
    station -- the analogue of LfLH's reconstruction success. ``occupancy`` is the share of the
    feasible set's (station x coordinate) cells that any sample visits, which is the quantity a
    single Gaussian cannot keep high while also keeping ``valid_rate`` high.
    """
    stations = len(pair.stations_m)
    valid = 0
    visited: set[tuple[int, int]] = set()
    feasible_cells: set[tuple[int, int]] = set()
    lower = pair.adapted_reach_m + margin_m
    upper = pair.nominal_reach_m - margin_m
    finite = np.isfinite(lower) & np.isfinite(upper) & (upper > lower)
    if not finite.any():
        return {"valid_rate": 0.0, "occupancy": 0.0, "feasible_cells": 0}
    low_edge = float(lower[finite].min())
    high_edge = float(upper[finite].max())
    span = max(high_edge - low_edge, 1e-9)
    for index in np.flatnonzero(finite):
        for cell in range(bins):
            value = low_edge + (cell + 0.5) * span / bins
            if lower[index] <= value <= upper[index]:
                feasible_cells.add((int(index), cell))
    for station, coordinate in placements:
        index = int(np.clip(round(station), 0, stations - 1))
        if not finite[index]:
            continue
        if lower[index] <= coordinate <= upper[index]:
            valid += 1
            cell = int(np.clip((coordinate - low_edge) / span * bins, 0, bins - 1))
            visited.add((index, cell))
    return {
        "valid_rate": valid / max(len(placements), 1),
        "occupancy": len(visited) / max(len(feasible_cells), 1),
        "feasible_cells": len(feasible_cells),
    }
