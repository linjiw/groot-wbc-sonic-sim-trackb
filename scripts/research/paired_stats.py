#!/usr/bin/env python3
"""Small-n paired statistics for SONIC comparisons.

Provides an exact one-sided sign-flip permutation test and a paired bootstrap
confidence interval on the mean of per-seed deltas. Improvement is defined as a
NEGATIVE delta (lower MPJPE is better), so the one-sided p-value counts sign
assignments whose mean is <= the observed mean.

These statistics are reported for preregistration discipline; they do not
replace the preregistered effect gates. At n=3 the minimum achievable one-sided
permutation p is 1/8 = 0.125, so 3-seed designs are causal-sanity screens only.
"""

from __future__ import annotations

from itertools import product
import random
from typing import Any

_MAX_EXACT_N = 20


def _require_finite(deltas: list[float]) -> None:
    if not deltas:
        raise ValueError("at least one delta is required")
    for value in deltas:
        # NaN comparisons are all False, which would drive the permutation
        # count (and thus p) to 0.0 — reading as maximal significance on
        # invalid data. Refuse non-finite inputs outright.
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"non-finite delta in input: {deltas}")


def exact_sign_flip_permutation_p(deltas: list[float]) -> dict[str, Any]:
    """Exact one-sided sign-flip permutation test on paired deltas.

    Enumerates all 2^n sign assignments of the deltas and counts assignments
    whose mean is <= the observed mean (one-sided toward improvement, where
    improvement means negative delta). The identity assignment always counts,
    so p >= 1 / 2^n by construction.
    """
    _require_finite(deltas)
    n = len(deltas)
    if n > _MAX_EXACT_N:
        raise ValueError(f"exact enumeration limited to n <= {_MAX_EXACT_N}, got n = {n}")

    observed_mean = sum(deltas) / n
    total = 2**n
    count_leq = 0
    for signs in product((1.0, -1.0), repeat=n):
        perm_mean = sum(sign * delta for sign, delta in zip(signs, deltas)) / n
        if perm_mean <= observed_mean:
            count_leq += 1
    return {
        "n": n,
        "observed_mean": observed_mean,
        "permutation_p_one_sided": count_leq / total,
        "min_achievable_p": 1.0 / total,
        "direction": "improvement_is_negative_delta",
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list, q in [0, 1]."""
    if not sorted_values:
        raise ValueError("percentile of empty list")
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Percentile bootstrap CI on the mean of paired deltas (deterministic RNG)."""
    _require_finite(deltas)
    n = len(deltas)
    rng = random.Random(seed)
    means = []
    for _ in range(num_resamples):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": n,
        "mean": sum(deltas) / n,
        "confidence": confidence,
        "ci_low": _percentile(means, alpha),
        "ci_high": _percentile(means, 1.0 - alpha),
        "num_resamples": num_resamples,
        "rng_seed": seed,
    }


def power_note(n: int) -> str:
    """Human-readable power caveat for small-n paired designs."""
    min_p = 1.0 / 2**n
    if n < 5:
        return (
            f"n={n} paired design: minimum achievable one-sided permutation p = {min_p:g}; "
            "screen only, not an effect claim (effect claims need >= 5 seeds)."
        )
    return f"n={n} paired design: minimum achievable one-sided permutation p = {min_p:g}."


def paired_delta_statistics(
    deltas: list[float],
    *,
    num_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Bundle the permutation test, bootstrap CI, and power note for reporting."""
    permutation = exact_sign_flip_permutation_p(deltas)
    bootstrap = paired_bootstrap_ci(
        deltas, num_resamples=num_resamples, confidence=confidence, seed=seed
    )
    return {
        "n": permutation["n"],
        "mean_delta": permutation["observed_mean"],
        "permutation_p_one_sided": permutation["permutation_p_one_sided"],
        "min_achievable_p": permutation["min_achievable_p"],
        "direction": permutation["direction"],
        "bootstrap_ci_95": [bootstrap["ci_low"], bootstrap["ci_high"]],
        "bootstrap_num_resamples": bootstrap["num_resamples"],
        "bootstrap_rng_seed": bootstrap["rng_seed"],
        "power_note": power_note(permutation["n"]),
    }
