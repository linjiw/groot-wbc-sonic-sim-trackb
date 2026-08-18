"""Gate-A trust region for composing LACE with SONIC's native sampler."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrustRegionConstraints:
    """Legal bounds for a LACE distribution relative to the native baseline."""

    max_kl: float = 0.05
    max_probability_ratio: float = 2.0
    min_effective_bins: float = 0.0
    min_mechanism_exposure: tuple[float, ...] | None = None

    def validate(self, mechanism_count: int) -> None:
        if not math.isfinite(self.max_kl) or self.max_kl < 0.0:
            raise ValueError("max_kl must be finite and nonnegative")
        if not math.isfinite(self.max_probability_ratio) or self.max_probability_ratio < 1.0:
            raise ValueError("max_probability_ratio must be finite and at least one")
        if not math.isfinite(self.min_effective_bins) or self.min_effective_bins < 0.0:
            raise ValueError("min_effective_bins must be finite and nonnegative")
        if self.min_mechanism_exposure is not None:
            if len(self.min_mechanism_exposure) != mechanism_count:
                raise ValueError("min_mechanism_exposure length must match mechanism count")
            if any(
                not math.isfinite(value) or value < 0.0 for value in self.min_mechanism_exposure
            ):
                raise ValueError("min_mechanism_exposure values must be finite and nonnegative")


@dataclass(frozen=True)
class TrustRegionDiagnostics:
    kl_from_base: float
    max_probability_ratio: float
    entropy: float
    effective_bins: float
    mechanism_exposure: tuple[float, ...]
    legal: bool


@dataclass(frozen=True)
class LaceSamplingResult:
    probabilities: FloatArray
    unprojected_probabilities: FloatArray
    applied_scale: float
    projected: bool
    enabled: bool
    diagnostics: TrustRegionDiagnostics


def _probability_vector(values: ArrayLike, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite, strictly positive probabilities")
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} has invalid total mass")
    return result / total


def _membership_matrix(values: ArrayLike, bin_count: int) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != bin_count or result.shape[1] == 0:
        raise ValueError("mechanism_membership must have shape [num_bins, num_mechanisms]")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("mechanism_membership must be finite and nonnegative")
    row_sums = result.sum(axis=1)
    if np.any(row_sums > 1.0 + 1e-8):
        raise ValueError("mechanism_membership rows may not sum above one")
    return result


def _tilt(base: FloatArray, score: FloatArray, scale: float) -> FloatArray:
    logits = np.log(base) + scale * score
    logits -= float(logits.max())
    result = np.exp(logits)
    return result / result.sum()


def distribution_diagnostics(
    base: FloatArray,
    probabilities: FloatArray,
    membership: FloatArray,
    constraints: TrustRegionConstraints,
) -> TrustRegionDiagnostics:
    kl = float(np.sum(probabilities * (np.log(probabilities) - np.log(base))))
    max_ratio = float(np.max(probabilities / base))
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    effective_bins = math.exp(entropy)
    exposure_array = membership.T @ probabilities
    exposure = tuple(float(value) for value in exposure_array)
    exposure_legal = True
    if constraints.min_mechanism_exposure is not None:
        exposure_legal = all(
            actual + 1e-12 >= minimum
            for actual, minimum in zip(
                exposure,
                constraints.min_mechanism_exposure,
                strict=True,
            )
        )
    legal = (
        kl <= constraints.max_kl + 1e-12
        and max_ratio <= constraints.max_probability_ratio + 1e-12
        and effective_bins + 1e-12 >= constraints.min_effective_bins
        and exposure_legal
    )
    return TrustRegionDiagnostics(
        kl_from_base=kl,
        max_probability_ratio=max_ratio,
        entropy=entropy,
        effective_bins=effective_bins,
        mechanism_exposure=exposure,
        legal=legal,
    )


def apply_lace_tilt(
    base_probabilities: ArrayLike,
    mechanism_membership: ArrayLike,
    utilities: Sequence[float],
    *,
    strength: float = 1.0,
    constraints: TrustRegionConstraints | None = None,
    enabled: bool = True,
    projection_iterations: int = 60,
) -> LaceSamplingResult:
    """Apply a mechanism utility tilt and project it onto the connected legal path.

    The input distribution is SONIC's native per-bin distribution. At scale zero
    the result is exactly that baseline; positive scale requests an exponential
    tilt using the soft mechanism membership. If the requested endpoint violates
    Gate A, bisection follows the convex mixture path from the baseline to that
    endpoint. Gate A's KL sublevel, entropy superlevel, ratio bounds, and linear
    exposure bounds are convex on this path, so the returned legal interval is
    connected.
    """

    base = _probability_vector(base_probabilities, "base_probabilities")
    membership = _membership_matrix(mechanism_membership, len(base))
    utility_array = np.asarray(utilities, dtype=np.float64)
    if utility_array.shape != (membership.shape[1],):
        raise ValueError("utilities length must match mechanism count")
    if not np.all(np.isfinite(utility_array)):
        raise ValueError("utilities must be finite")
    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("strength must be finite and nonnegative")
    if projection_iterations < 1:
        raise ValueError("projection_iterations must be positive")

    gate = constraints or TrustRegionConstraints()
    gate.validate(membership.shape[1])
    base_diagnostics = distribution_diagnostics(base, base, membership, gate)
    if not base_diagnostics.legal:
        raise ValueError("the native baseline distribution does not satisfy Gate A")

    if not enabled or strength == 0.0:
        return LaceSamplingResult(
            probabilities=base.copy(),
            unprojected_probabilities=base.copy(),
            applied_scale=0.0,
            projected=False,
            enabled=enabled,
            diagnostics=base_diagnostics,
        )

    score = membership @ utility_array
    requested = _tilt(base, score, strength)
    requested_diagnostics = distribution_diagnostics(base, requested, membership, gate)
    if requested_diagnostics.legal:
        return LaceSamplingResult(
            probabilities=requested,
            unprojected_probabilities=requested.copy(),
            applied_scale=strength,
            projected=False,
            enabled=True,
            diagnostics=requested_diagnostics,
        )

    low = 0.0
    high = strength
    best = base
    best_diagnostics = base_diagnostics
    for _ in range(projection_iterations):
        middle = (low + high) / 2.0
        mixture_fraction = middle / strength
        candidate = (1.0 - mixture_fraction) * base + mixture_fraction * requested
        candidate_diagnostics = distribution_diagnostics(base, candidate, membership, gate)
        if candidate_diagnostics.legal:
            low = middle
            best = candidate
            best_diagnostics = candidate_diagnostics
        else:
            high = middle

    if not best_diagnostics.legal:
        raise RuntimeError("trust-region projection failed to return a legal distribution")
    return LaceSamplingResult(
        probabilities=best,
        unprojected_probabilities=requested,
        applied_scale=low,
        projected=True,
        enabled=True,
        diagnostics=best_diagnostics,
    )
