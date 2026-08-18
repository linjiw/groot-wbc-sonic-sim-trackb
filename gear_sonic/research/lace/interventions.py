"""Pure fixed-budget source-panel interventions for the LACE transfer experiment."""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gear_sonic.research.lace.schema import canonical_sha256

FloatArray = NDArray[np.float64]

INTERVENTION_KIND = "lace_fixed_budget_intervention"
INTERVENTION_SCHEMA_VERSION = 1


def _finite_scalar(value: Any, name: str, *, minimum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _force_unit_sum(values: FloatArray, name: str) -> FloatArray:
    """Return a nonnegative copy whose accurate floating-point sum is exactly one."""

    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    if not np.any(result > 0.0):
        raise ValueError(f"{name} must contain positive mass")

    correction_index = int(np.argmax(result))
    for _ in range(4):
        residual = 1.0 - math.fsum(float(value) for value in result)
        if residual == 0.0:
            break
        result[correction_index] += residual
    if result[correction_index] < 0.0 or math.fsum(float(value) for value in result) != 1.0:
        raise RuntimeError(f"could not make {name} sum exactly to one")
    return result


def _probability_vector(values: ArrayLike, name: str, tolerance: float) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must be numeric, not boolean")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError(f"{name} must contain finite, nonnegative probabilities")
    total = math.fsum(float(value) for value in result)
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must contain positive total mass")
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} must sum to one within probability_tolerance")
    return _force_unit_sum(result / total, name)


def _panel_support(
    values: ArrayLike,
    *,
    bin_count: int,
    base: FloatArray,
    probability_tolerance: float,
) -> tuple[NDArray[np.bool_], str, list[bool] | list[float]]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.shape != (bin_count,):
        raise ValueError(f"panel must have shape [{bin_count}]")

    if np.issubdtype(raw.dtype, np.bool_):
        support = raw.astype(np.bool_, copy=True)
        input_kind = "boolean_mask"
        canonical_input: list[bool] | list[float] = [bool(value) for value in support]
    else:
        if not np.issubdtype(raw.dtype, np.number) or np.issubdtype(raw.dtype, np.complexfloating):
            raise ValueError(
                "panel must be a boolean mask, binary mask, or probability distribution"
            )
        numeric = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(numeric)) or np.any(numeric < 0.0):
            raise ValueError("panel must contain finite, nonnegative values")
        binary = np.all((numeric == 0.0) | (numeric == 1.0))
        if binary:
            support = numeric.astype(np.bool_)
            input_kind = "binary_mask"
            canonical_input = [bool(value) for value in support]
        else:
            distribution = _probability_vector(
                numeric,
                "panel distribution",
                probability_tolerance,
            )
            support = distribution > 0.0
            input_kind = "probability_distribution_support"
            canonical_input = [float(value) for value in distribution]

    if not np.any(support):
        raise ValueError("panel support must be non-empty")
    zero_base_members = np.flatnonzero(support & (base == 0.0))
    if zero_base_members.size:
        raise ValueError(
            "panel support includes zero-base bins and would request unsupported mass injection: "
            f"{zero_base_members.tolist()}"
        )
    return support, input_kind, canonical_input


def _kl_from_base(probabilities: FloatArray, base: FloatArray) -> float:
    active = probabilities > 0.0
    if np.any(base[active] <= 0.0):
        raise RuntimeError("intervention injected probability outside base support")
    return float(
        math.fsum(
            float(probability * math.log(probability / baseline))
            for probability, baseline in zip(
                probabilities[active],
                base[active],
                strict=True,
            )
        )
    )


def _total_variation(left: FloatArray, right: FloatArray) -> float:
    return 0.5 * math.fsum(float(value) for value in np.abs(left - right))


def _entropy_effective_bins(probabilities: FloatArray) -> float:
    active = probabilities > 0.0
    entropy = -math.fsum(
        float(probability * math.log(probability)) for probability in probabilities[active]
    )
    return math.exp(entropy)


def _inverse_simpson_effective_bins(probabilities: FloatArray) -> float:
    return 1.0 / math.fsum(float(value * value) for value in probabilities)


def _mixture(base: FloatArray, panel_distribution: FloatArray, rho: float) -> FloatArray:
    result = (1.0 - rho) * base + rho * panel_distribution
    return _force_unit_sum(result, "intervention probabilities")


def solve_fixed_budget_intervention(
    base_probabilities: ArrayLike,
    panel: ArrayLike,
    *,
    target_kl: float,
    max_probability_ratio: float,
    target_tv: float | None = None,
    tv_tolerance: float = 1e-10,
    kl_tolerance: float = 1e-12,
    probability_tolerance: float = 1e-12,
    bisection_iterations: int = 100,
) -> dict[str, Any]:
    """Solve a fixed-budget, representation-blind source-panel intervention.

    ``panel`` may be a boolean/binary membership mask or a probability
    distribution. In either case only its positive support is used: the source
    distribution is always ``r_s = p_base(. | panel)``. Consequently the
    intervention changes panel exposure without introducing a second, hidden
    within-panel sampling treatment.

    The returned distribution is ``p_plus = (1-rho) p_base + rho r_s``. KL is
    monotone in ``rho`` on this path, so deterministic bisection finds the unique
    solution subject to the upper per-bin probability-ratio cap. If ``target_tv``
    is supplied, the KL solution must also match it within ``tv_tolerance``.
    """

    probability_epsilon = _finite_scalar(
        probability_tolerance,
        "probability_tolerance",
        minimum=0.0,
    )
    if probability_epsilon == 0.0:
        raise ValueError("probability_tolerance must be positive")
    kl_epsilon = _finite_scalar(kl_tolerance, "kl_tolerance", minimum=0.0)
    if kl_epsilon == 0.0:
        raise ValueError("kl_tolerance must be positive")
    tv_epsilon = _finite_scalar(tv_tolerance, "tv_tolerance", minimum=0.0)
    iterations = _positive_integer(bisection_iterations, "bisection_iterations")
    requested_kl = _finite_scalar(target_kl, "target_kl", minimum=0.0)
    ratio_cap = _finite_scalar(
        max_probability_ratio,
        "max_probability_ratio",
        minimum=1.0,
    )
    requested_tv = (
        None if target_tv is None else _finite_scalar(target_tv, "target_tv", minimum=0.0)
    )
    if requested_tv is not None and requested_tv > 1.0:
        raise ValueError("target_tv must not exceed one")

    base = _probability_vector(base_probabilities, "base_probabilities", probability_epsilon)
    support, panel_input_kind, canonical_panel_input = _panel_support(
        panel,
        bin_count=len(base),
        base=base,
        probability_tolerance=probability_epsilon,
    )
    base_support = base > 0.0
    panel_base_exposure = math.fsum(float(value) for value in base[support])
    if panel_base_exposure <= 0.0:
        raise ValueError("panel has no positive mass under base_probabilities")

    panel_distribution = np.zeros_like(base)
    panel_distribution[support] = base[support] / panel_base_exposure
    panel_distribution = _force_unit_sum(panel_distribution, "base-conditional panel distribution")

    outside_mass = max(0.0, 1.0 - panel_base_exposure)
    if outside_mass == 0.0:
        rho_ratio_limit = 1.0
    else:
        rho_ratio_limit = min(
            1.0,
            max(0.0, (ratio_cap - 1.0) * panel_base_exposure / outside_mass),
        )
    maximum_rho = float(rho_ratio_limit)
    maximum_distribution = _mixture(base, panel_distribution, maximum_rho)
    maximum_kl = _kl_from_base(maximum_distribution, base)
    maximum_tv = _total_variation(maximum_distribution, base)

    if requested_kl > maximum_kl + kl_epsilon:
        raise ValueError(
            "target_kl is infeasible under the panel support and probability-ratio cap: "
            f"target={requested_kl}, maximum={maximum_kl}, rho_max={maximum_rho}"
        )

    if requested_kl <= kl_epsilon:
        rho = 0.0
        probabilities = base.copy()
        realized_kl = 0.0
    elif abs(requested_kl - maximum_kl) <= kl_epsilon:
        rho = maximum_rho
        probabilities = maximum_distribution
        realized_kl = maximum_kl
    else:
        low = 0.0
        high = maximum_rho
        low_distribution = base.copy()
        high_distribution = maximum_distribution
        low_kl = 0.0
        high_kl = maximum_kl
        for _ in range(iterations):
            middle = (low + high) / 2.0
            candidate = _mixture(base, panel_distribution, middle)
            candidate_kl = _kl_from_base(candidate, base)
            if candidate_kl < requested_kl:
                low = middle
                low_distribution = candidate
                low_kl = candidate_kl
            else:
                high = middle
                high_distribution = candidate
                high_kl = candidate_kl
        low_error = abs(low_kl - requested_kl)
        high_error = abs(high_kl - requested_kl)
        if low_error <= high_error:
            rho, probabilities, realized_kl = low, low_distribution, low_kl
        else:
            rho, probabilities, realized_kl = high, high_distribution, high_kl
        if abs(realized_kl - requested_kl) > kl_epsilon:
            raise RuntimeError(
                "bisection did not reach target_kl within kl_tolerance; "
                "increase bisection_iterations"
            )

    if np.any(probabilities[~base_support] != 0.0):
        raise RuntimeError("intervention injected probability outside base support")
    probability_sum = math.fsum(float(value) for value in probabilities)
    if probability_sum != 1.0:
        raise RuntimeError("intervention probabilities do not sum exactly to one")

    delta = probabilities - base
    realized_tv = _total_variation(probabilities, base)
    if requested_tv is not None and abs(realized_tv - requested_tv) > tv_epsilon:
        raise ValueError(
            "target_tv is incompatible with the target_kl solution: "
            f"target={requested_tv}, realized={realized_tv}, tolerance={tv_epsilon}"
        )

    ratios = probabilities[base_support] / base[base_support]
    maximum_ratio = float(np.max(ratios))
    minimum_ratio = float(np.min(ratios))
    ratio_numerical_tolerance = float(64.0 * np.finfo(np.float64).eps * max(1.0, ratio_cap))
    probability_numerical_tolerance = float(128.0 * np.finfo(np.float64).eps)
    if maximum_ratio > ratio_cap + ratio_numerical_tolerance:
        raise RuntimeError("intervention exceeds max_probability_ratio after solving")

    panel_new_exposure = math.fsum(float(value) for value in probabilities[support])
    added_panel_exposure = panel_new_exposure - panel_base_exposure
    if abs(added_panel_exposure - realized_tv) > probability_numerical_tolerance:
        raise RuntimeError("panel exposure shift is inconsistent with total variation")

    ratio_by_bin: list[float | None] = [
        float(probability / baseline) if baseline > 0.0 else None
        for probability, baseline in zip(probabilities, base, strict=True)
    ]
    importance_second_moment = math.fsum(
        float(baseline * ratio * ratio)
        for baseline, ratio in zip(base[base_support], ratios, strict=True)
    )

    manifest: dict[str, Any] = {
        "kind": INTERVENTION_KIND,
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "panel_rule": "base_conditional_on_positive_panel_support",
        "panel_input_kind": panel_input_kind,
        "panel_input": canonical_panel_input,
        "panel_support_mask": [bool(value) for value in support],
        "num_bins": int(len(base)),
        "base_probabilities": [float(value) for value in base],
        "panel_distribution": [float(value) for value in panel_distribution],
        "intervention_probabilities": [float(value) for value in probabilities],
        "p_base": [float(value) for value in base],
        "r_s": [float(value) for value in panel_distribution],
        "p_plus": [float(value) for value in probabilities],
        "delta_p": [float(value) for value in delta],
        "rho": float(rho),
        "target_kl": requested_kl,
        "realized_kl": float(realized_kl),
        "kl_tolerance": kl_epsilon,
        "target_tv": requested_tv,
        "realized_tv": float(realized_tv),
        "tv_tolerance": tv_epsilon if requested_tv is not None else None,
        "panel_base_exposure": float(panel_base_exposure),
        "panel_new_exposure": float(panel_new_exposure),
        "added_panel_exposure": float(added_panel_exposure),
        "ratio_diagnostics": {
            "max_probability_ratio_cap": ratio_cap,
            "maximum_probability_ratio": maximum_ratio,
            "minimum_probability_ratio_on_base_support": minimum_ratio,
            "ratio_by_bin": ratio_by_bin,
            "ratio_cap_active": bool(abs(maximum_ratio - ratio_cap) <= ratio_numerical_tolerance),
            "rho_limit_from_ratio_cap": maximum_rho,
            "maximum_feasible_kl": float(maximum_kl),
            "maximum_feasible_tv": float(maximum_tv),
            "zero_base_output_mass": float(probabilities[~base_support].sum()),
        },
        "support_stats": {
            "base_positive_bin_count": int(base_support.sum()),
            "zero_base_bin_count": int((~base_support).sum()),
            "panel_bin_count": int(support.sum()),
            "outside_panel_positive_bin_count": int((base_support & ~support).sum()),
            "intervention_positive_bin_count": int((probabilities > 0.0).sum()),
        },
        "effective_sample_stats": {
            "inverse_simpson_base": _inverse_simpson_effective_bins(base),
            "inverse_simpson_panel": _inverse_simpson_effective_bins(panel_distribution),
            "inverse_simpson_intervention": _inverse_simpson_effective_bins(probabilities),
            "entropy_effective_bins_base": _entropy_effective_bins(base),
            "entropy_effective_bins_panel": _entropy_effective_bins(panel_distribution),
            "entropy_effective_bins_intervention": _entropy_effective_bins(probabilities),
            "importance_weight_ess_fraction_from_base": 1.0 / importance_second_moment,
        },
        "fixed_budget_diagnostics": {
            "base_probability_sum": math.fsum(float(value) for value in base),
            "panel_probability_sum": math.fsum(float(value) for value in panel_distribution),
            "intervention_probability_sum": probability_sum,
            "signed_delta_sum": math.fsum(float(value) for value in delta),
            "l1_probability_shift": math.fsum(float(value) for value in np.abs(delta)),
        },
        "solver": {
            "method": "monotone_bisection_v1",
            "bisection_iterations": iterations,
            "probability_tolerance": probability_epsilon,
        },
    }
    manifest["intervention_sha256"] = canonical_sha256(manifest)
    return manifest
