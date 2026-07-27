#!/usr/bin/env python3
"""Preregistered SIM-M5 activation-gate classifier (research_plan_zpd_teacher.md §4, v1.1).

Frozen BEFORE any M5 GPU run, mirroring classify_sampler_diagnosis.py's
preregistration-in-code pattern. Activation is signal-family-specific (the Z6
amendment, `docs/artifacts/sim_m5/zpd_forecast_preregistered.json`):

- **ZPD arms (M5-L learnability, M5-A advantage_mass):** PASS iff
  (a) frontier over-allocation — sampling mass on the middle difficulty tercile
      of evaluated bins divided by that tercile's uniform share — >= 1.2, AND
  (b) posterior sanity — the Beta-posterior survival mean rank-correlates with
      the SIM-D1 easiness ordering (Spearman >= 0.4).
  Peakedness (pmax/uniform >= 10 AND >= 1 concentrated bin) remains sufficient
  if it fires (not expected for ZPD utilities).
  Validity assertion (D9): the tripwire must bind in < 5% of iterations, else
  the run is *invalid-unstable* (not negative, not activated).
- **failure_rate arms:** PASS iff peakedness fires OR the hard-half/easy-half
  sampling-mass ratio >= 1.5 (the fable-next.md F2 amendment, unchanged).
- **M5-T (threshold schedule):** the manufactured-frontier existence check —
  the per-iteration fraction of episode ends where either scheduled Z-height
  term (``anchor_pos`` or ``ee_body_pos``) fired must lie in [0.15, 0.6] for
  >= 50% of iterations, else *invalid-inactive* (schedule mis-calibrated; one
  recalibration allowed). Every density series must contain one finite point at
  every ordered iteration 1..N; malformed instrumentation is
  *invalid-telemetry*, never negative evidence.

Arm-level activation requires the per-seed rule to PASS in >= 2/3 of seeds.

Inputs (all JSON, produced by existing tooling):
- checkpoint dump (`dump_sampler_checkpoint_state.py`) — per-bin episodes/failures,
  bin weights, bin-to-motion keys, and cumulative actual-draw
  `sampled_count`/`sampled_fraction`;
- SIM-D1 difficulty ranking — motion keys, EASIEST FIRST (frozen at D1 time);
- telemetry summary (`summarize_sampler_telemetry.py`) — per-iteration series for
  peakedness, the tripwire, and raw M5-T scheduled-term/episode-end densities.

ZPD frontier mass and peakedness use only the dump's empirical
`sampled_fraction`: selected target bins after global motion selection and
active-batch conditioning, but before the random frame/pre-failure-window shift.
It is not executed-start mass. Missing, insufficient, or inconsistent empirical
mass makes a seed incomplete; there is no unweighted utility-recompute fallback.
D9 additionally requires one finite binary point at every ordered iteration
1..N for exactly seeds 0-2. Posterior sanity remains independently count-derived.
The output binds every seed to checkpoint, dump, ranking, telemetry, and dataset
hashes, plus sampler schema/config for ZPD arms. Legacy failure-rate
classification retains its existing math.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from scripts.research.sampler_dynamics_sim import _rank_correlation  # noqa: E402

# Same convention as classify_sampler_diagnosis.py: telemetry records carry no
# seed field, so the seed is derived from the log path.
_SEED_RE = re.compile(r"seed[_\-]?(\d+)")

# ---------------------------------------------------------------------------
# Frozen thresholds (preregistration-in-code; changing any is a plan revision)
# ---------------------------------------------------------------------------
_ZPD_FRONTIER_OVER_ALLOC = 1.2  # plan §4 (Z6 amendment)
_ZPD_POSTERIOR_SPEARMAN = 0.4  # plan §4 posterior sanity
_PEAK_PMAX_OVER_UNIFORM = 10.0  # fable-next.md Phase 3 peakedness
_PEAK_MIN_CONCENTRATED = 1.0
_FR_HARD_HALF_RATIO = 1.5  # F2 amendment, failure_rate arms only
_TRIPWIRE_MAX_BINDING_FRACTION = 0.05  # D9: binding >=5% of iters => invalid-unstable
_MT_BAND = (0.15, 0.6)  # plan §4 M5-T manufactured-frontier band
_MT_MIN_FRACTION_IN_BAND = 0.5
_MT_TELEMETRY_KEYS = (
    "m5t_anchor_pos_termination_density",
    "m5t_ee_body_pos_termination_density",
    "m5t_height_termination_density",
    "m5t_episode_end_density",
)
_EXPECTED_SCREEN_SEEDS = frozenset((0, 1, 2))
_MIN_ACTUAL_DRAWS_DEFAULT = 1
_ZPD_SAMPLER_SCHEMA_KIND = "zpd_adaptive_sampler_state"
_ZPD_SAMPLER_SCHEMA_VERSION = 1
_ZPD_SAMPLING_MASS_SEMANTICS = "selected_target_bin_pre_window_shift"
_ZPD_SAMPLING_MASS_SOURCE = (
    "empirical_selected_target_bin_draw_counts_pre_window_shift"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_ZPD_SIGNALS = ("learnability", "advantage_mass")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest, got {value!r}")
    return text


def bin_difficulty_ranks(bin_motion_keys: list[str], ranking: list[str]) -> np.ndarray:
    """Per-bin difficulty rank (0 = easiest) inherited from the motion ranking.

    Every bin's motion must appear in the frozen ranking — a missing key means
    the ranking and the dump come from different datasets, which invalidates
    the gate rather than silently shrinking it.
    """
    rank_of = {key: i for i, key in enumerate(ranking)}
    missing = sorted({k for k in bin_motion_keys if k not in rank_of})
    if missing:
        raise ValueError(
            f"{len(missing)} bin motion keys absent from the difficulty ranking "
            f"(first: {missing[:3]}); dump and ranking must cover the same dataset"
        )
    return np.asarray([rank_of[k] for k in bin_motion_keys], dtype=float)


def frontier_tercile_mask(difficulty_ranks: np.ndarray) -> np.ndarray:
    """Middle difficulty tercile of evaluated bins (the preregistered frontier band).

    Bins sorted by inherited rank (stable order under motion-level ties); the
    middle third by count is the frontier.
    """
    n = len(difficulty_ranks)
    order = np.argsort(difficulty_ranks, kind="stable")
    lo, hi = n // 3, n - n // 3
    mask = np.zeros(n, dtype=bool)
    mask[order[lo:hi]] = True
    return mask


def _posterior_mean(episodes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    """Beta-posterior survival mean from checkpoint episode/failure counts."""
    fails = np.maximum(failures, 0.0)
    succ = np.maximum(episodes - fails, 0.0)
    return (1.0 + succ) / (2.0 + succ + fails)


def _empirical_sampling_mass(
    checkpoint_bins: list[dict[str, Any]],
    explicit_fractions: list[float] | None,
    *,
    min_actual_draws: int,
    explicit_draw_count: int | None,
) -> tuple[np.ndarray | None, str, int | None, str | None]:
    """Validate selected-target-bin mass and its positive draw evidence.

    Official checkpoint dumps must carry both ``sampled_count`` and
    ``sampled_fraction`` for every bin. These are pre-window-shift target-bin
    draws, not executed starts. Unit-level fixtures may explicitly provide a
    synthetic fraction vector and synthetic draw count; neither override exists
    on the CLI path.
    """
    if isinstance(min_actual_draws, bool) or not isinstance(min_actual_draws, int):
        return None, "unresolved", None, "minimum actual draws must be an integer"
    if min_actual_draws < 1:
        return None, "unresolved", None, "minimum actual draws must be positive"

    n = len(checkpoint_bins)
    if explicit_fractions is not None:
        source = "explicit_synthetic_selected_target_bin_fraction"
        raw_fractions: Any = explicit_fractions
        raw_counts = None
        if (
            isinstance(explicit_draw_count, bool)
            or not isinstance(explicit_draw_count, int)
            or explicit_draw_count < min_actual_draws
        ):
            return (
                None,
                source,
                explicit_draw_count,
                "explicit synthetic selected-target-bin fractions require an integer "
                f"draw count >= {min_actual_draws}",
            )
        draw_count_total = explicit_draw_count
    else:
        source = _ZPD_SAMPLING_MASS_SOURCE
        if any("sampled_fraction" not in bin_record for bin_record in checkpoint_bins):
            return (
                None,
                source,
                None,
                "empirical selected-target-bin sampled_fraction is missing for one or more bins",
            )
        if any("sampled_count" not in bin_record for bin_record in checkpoint_bins):
            return (
                None,
                source,
                None,
                "empirical selected-target-bin sampled_count is missing for one or more bins",
            )
        raw_fractions = [bin_record["sampled_fraction"] for bin_record in checkpoint_bins]
        raw_counts = [bin_record["sampled_count"] for bin_record in checkpoint_bins]
        draw_count_total = None

    if not isinstance(raw_fractions, (list, tuple, np.ndarray)) or len(raw_fractions) != n:
        actual_length = len(raw_fractions) if hasattr(raw_fractions, "__len__") else None
        return (
            None,
            source,
            draw_count_total,
            f"empirical sampled_fraction length mismatch: expected {n}, got {actual_length}",
        )
    try:
        fractions = np.asarray(raw_fractions, dtype=float)
    except (TypeError, ValueError):
        return None, source, draw_count_total, "empirical sampled_fraction values must be numeric"
    if fractions.ndim != 1 or not np.isfinite(fractions).all() or (fractions < 0).any():
        return (
            None,
            source,
            draw_count_total,
            "empirical sampled_fraction values must be a finite nonnegative vector",
        )

    fraction_sum = float(fractions.sum())
    if fraction_sum <= 0.0:
        return (
            None,
            source,
            draw_count_total,
            "empirical selected-target-bin sampling mass is zero; no actual draws were recorded",
        )
    if not np.isclose(fraction_sum, 1.0, rtol=1e-7, atol=1e-9):
        return (
            None,
            source,
            draw_count_total,
            f"empirical sampled_fraction is inconsistent: sum is {fraction_sum:.12g}, expected 1",
        )

    if raw_counts is not None:
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in raw_counts
        ):
            return (
                None,
                source,
                None,
                "empirical sampled_count values must be nonnegative integers",
            )
        draw_count_total = sum(raw_counts)
        if draw_count_total < min_actual_draws:
            return (
                None,
                source,
                draw_count_total,
                "empirical selected-target-bin draw evidence is insufficient: "
                f"observed {draw_count_total}, require >= {min_actual_draws}",
            )
        fractions_from_counts = np.asarray(raw_counts, dtype=float) / draw_count_total
        if not np.allclose(fractions, fractions_from_counts, rtol=1e-7, atol=1e-9):
            max_error = float(np.max(np.abs(fractions - fractions_from_counts)))
            return (
                None,
                source,
                draw_count_total,
                "empirical sampled_fraction is inconsistent with sampled_count "
                f"(max absolute error {max_error:.12g})",
            )

    return fractions, source, draw_count_total, None


def _validate_tripwire_points(
    points: list[list[Any]] | None,
    *,
    expected_iterations: int | None,
    seed_coverage_error: str | None,
) -> tuple[list[float] | None, dict[str, Any], str | None]:
    """Require a complete, indexed D9 binary series for one ZPD seed."""
    evidence: dict[str, Any] = {
        "tripwire_expected_iterations": expected_iterations,
        "tripwire_iteration_count": len(points) if isinstance(points, list) else 0,
        "tripwire_iteration_first": None,
        "tripwire_iteration_last": None,
    }
    if seed_coverage_error is not None:
        return None, evidence, seed_coverage_error
    if (
        isinstance(expected_iterations, bool)
        or not isinstance(expected_iterations, int)
        or expected_iterations < 2
    ):
        return (
            None,
            evidence,
            "D9 expected_iterations must be an integer >= 2; one-point certification is invalid",
        )
    if not isinstance(points, list):
        return None, evidence, "D9 tripwire telemetry series is missing"

    indices: list[int] = []
    values: list[float] = []
    for position, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None, evidence, f"D9 point {position} must be [iteration, binary_value]"
        iteration, value = point
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            return None, evidence, f"D9 point {position} has a non-integer iteration index"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, evidence, f"D9 point {position} has a non-numeric binary value"
        value = float(value)
        if not math.isfinite(value) or value not in (0.0, 1.0):
            return (
                None,
                evidence,
                f"D9 point {position} must be finite and binary (0 or 1), got {value!r}",
            )
        indices.append(iteration)
        values.append(value)

    if indices:
        evidence["tripwire_iteration_first"] = indices[0]
        evidence["tripwire_iteration_last"] = indices[-1]
    if len(indices) != len(set(indices)):
        return None, evidence, "D9 iteration indices contain duplicates"
    expected_indices = list(range(1, expected_iterations + 1))
    if indices != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))
        unexpected = sorted(set(indices) - set(expected_indices))
        return (
            None,
            evidence,
            "D9 iteration indices must be ordered, unique, and contiguous over "
            f"1..{expected_iterations}; missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}",
        )
    return values, evidence, None


def classify_zpd_seed(
    checkpoint_bins: list[dict[str, Any]],
    bin_motion_keys: list[str],
    ranking: list[str],
    *,
    signal: str,
    optimism_k: float = 0.0,
    advmass_n: int = 16,
    uniform_rate: float = 0.1,
    tripwire_binding_series: list[list[Any]] | None = None,
    expected_iterations: int | None = None,
    tripwire_seed_coverage_error: str | None = None,
    sampled_fractions: list[float] | None = None,
    synthetic_sampled_count_total: int | None = None,
    min_actual_draws: int = _MIN_ACTUAL_DRAWS_DEFAULT,
) -> dict[str, Any]:
    """One ZPD-arm seed against the empirical-mass Z6 activation rule."""
    if signal not in _ZPD_SIGNALS:
        raise ValueError(f"not a ZPD signal: {signal!r}")
    if len(checkpoint_bins) != len(bin_motion_keys):
        raise ValueError(
            f"dump has {len(checkpoint_bins)} bins but bin_motion_keys has "
            f"{len(bin_motion_keys)}; they must be the same dump"
        )
    episodes = np.asarray([b["num_episodes"] for b in checkpoint_bins], dtype=float)
    failures = np.asarray([b["num_failures"] for b in checkpoint_bins], dtype=float)
    ranks = bin_difficulty_ranks(bin_motion_keys, ranking)

    p_mean = _posterior_mean(episodes, failures)

    # Posterior sanity: survival mean should be HIGH on easy motions. Ranks are
    # 0 = easiest, so correlate p_mean against easiness = -rank. This remains
    # independently count-derived even if empirical sampling mass is unusable.
    posterior_spearman = _rank_correlation(p_mean, -ranks)
    evidence: dict[str, Any] = {
        "signal": signal,
        "num_bins": len(checkpoint_bins),
        "posterior_spearman_vs_easiness": posterior_spearman,
        "configured_optimism_k": optimism_k,
        "configured_advmass_n": advmass_n,
        "configured_uniform_rate": uniform_rate,
    }

    prob, mass_source, draw_count_total, mass_error = _empirical_sampling_mass(
        checkpoint_bins,
        sampled_fractions,
        min_actual_draws=min_actual_draws,
        explicit_draw_count=synthetic_sampled_count_total,
    )
    evidence["sampling_mass_source"] = mass_source
    evidence["sampling_mass_semantics"] = _ZPD_SAMPLING_MASS_SEMANTICS
    evidence["selected_target_bin_draw_count_total"] = draw_count_total
    evidence["minimum_required_actual_draws"] = min_actual_draws
    if mass_error is not None:
        evidence.update(
            {
                "frontier_share": None,
                "frontier_mass": None,
                "frontier_over_alloc": None,
                "reason": mass_error,
            }
        )
        return {"verdict": "incomplete", "evidence": evidence}
    assert prob is not None

    frontier = frontier_tercile_mask(ranks)
    frontier_share = float(frontier.mean())
    frontier_mass = float(prob[frontier].sum())
    frontier_over_alloc = frontier_mass / frontier_share if frontier_share > 0 else 0.0

    n = len(prob)
    pmax_over_uniform = float(prob.max() * n)
    num_concentrated = int((prob > 10.0 / n).sum())
    peakedness = (
        pmax_over_uniform >= _PEAK_PMAX_OVER_UNIFORM
        and num_concentrated >= _PEAK_MIN_CONCENTRATED
    )

    evidence.update(
        {
            "frontier_share": frontier_share,
            "frontier_mass": frontier_mass,
            "frontier_over_alloc": frontier_over_alloc,
            "pmax_over_uniform_empirical": pmax_over_uniform,
            "num_concentrated_bins_empirical": num_concentrated,
        }
    )

    # D9 validity assertion requires every indexed binary point for the frozen run.
    tripwire_values, tripwire_evidence, tripwire_error = _validate_tripwire_points(
        tripwire_binding_series,
        expected_iterations=expected_iterations,
        seed_coverage_error=tripwire_seed_coverage_error,
    )
    evidence.update(tripwire_evidence)
    if tripwire_error is not None:
        evidence["tripwire_binding_fraction"] = None
        evidence["reason"] = tripwire_error
        return {"verdict": "incomplete", "evidence": evidence}
    assert tripwire_values is not None
    binding_fraction = float(np.mean(tripwire_values))
    evidence["tripwire_binding_fraction"] = binding_fraction
    if binding_fraction >= _TRIPWIRE_MAX_BINDING_FRACTION:
        return {"verdict": "invalid-unstable", "evidence": evidence}

    targeted = (
        frontier_over_alloc >= _ZPD_FRONTIER_OVER_ALLOC
        and posterior_spearman >= _ZPD_POSTERIOR_SPEARMAN
    )
    return {"verdict": "activated" if (targeted or peakedness) else "inactive", "evidence": evidence}


def classify_failure_rate_seed(
    checkpoint_bins: list[dict[str, Any]],
    bin_motion_keys: list[str],
    ranking: list[str],
    *,
    telemetry_last: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One failure_rate-arm seed against the unchanged F2-amended rule.

    Peakedness prefers the training-time telemetry values (they include bin
    weights); falls back to the unweighted recompute when telemetry is absent.
    """
    if len(checkpoint_bins) != len(bin_motion_keys):
        raise ValueError("dump / bin_motion_keys length mismatch")
    ranks = bin_difficulty_ranks(bin_motion_keys, ranking)
    prob = np.asarray([b["recomputed_prob_unweighted"] for b in checkpoint_bins], dtype=float)

    order = np.argsort(ranks, kind="stable")
    n = len(prob)
    easy_mass = float(prob[order[: n // 2]].sum())
    hard_mass = float(prob[order[n - n // 2 :]].sum())
    hard_half_ratio = hard_mass / easy_mass if easy_mass > 0 else float("inf")

    if telemetry_last is not None:
        pmax = telemetry_last.get("prob_max_over_uniform")
        concentrated = telemetry_last.get("num_concentrated_bins")
        peak_source = "telemetry"
    else:
        pmax = float(prob.max() * n)
        concentrated = float((prob > 10.0 / n).sum())
        peak_source = "unweighted_recompute"
    peakedness = (
        pmax is not None
        and concentrated is not None
        and pmax >= _PEAK_PMAX_OVER_UNIFORM
        and concentrated >= _PEAK_MIN_CONCENTRATED
    )

    evidence = {
        "signal": "failure_rate",
        "hard_half_mass_ratio": hard_half_ratio,
        "pmax_over_uniform": pmax,
        "num_concentrated_bins": concentrated,
        "peakedness_source": peak_source,
    }
    activated = peakedness or hard_half_ratio >= _FR_HARD_HALF_RATIO
    return {"verdict": "activated" if activated else "inactive", "evidence": evidence}


def derive_termination_rate_series(
    num_failures_mean: list[float], num_episodes_mean: list[float]
) -> list[float]:
    """Legacy failure-rate hazard proxy retained for old diagnostic callers.

    delta(num_failures_mean) / delta(num_episodes_mean) per iteration: failures
    per episode-equivalent of bin traversal. M5-T activation does not use this
    global sampler signal: it uses the term-specific indexed telemetry validated
    by :func:`classify_m5t_telemetry_points`.
    """
    if len(num_failures_mean) != len(num_episodes_mean):
        raise ValueError("failures/episodes series length mismatch")
    rates: list[float] = []
    for i in range(1, len(num_episodes_mean)):
        d_eps = num_episodes_mean[i] - num_episodes_mean[i - 1]
        d_fail = num_failures_mean[i] - num_failures_mean[i - 1]
        if d_eps > 0:
            rates.append(max(d_fail, 0.0) / d_eps)
    return rates


def _validate_m5t_density_points(
    points: list[list[Any]] | None,
    *,
    key: str,
    expected_iterations: int,
) -> tuple[list[float] | None, str | None]:
    """Validate one M5-T density series over exact ordered indices 1..N."""
    if not isinstance(points, list):
        return None, f"M5-T telemetry key {key} is missing"
    if len(points) != expected_iterations:
        return (
            None,
            f"M5-T telemetry key {key} has {len(points)} points, "
            f"expected {expected_iterations}",
        )

    values: list[float] = []
    indices: list[int] = []
    for position, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None, f"M5-T telemetry key {key} point {position} must be [iteration, value]"
        iteration, value = point
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            return None, f"M5-T telemetry key {key} point {position} has a non-integer index"
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"M5-T telemetry key {key} point {position} has a non-numeric value"
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return (
                None,
                f"M5-T telemetry key {key} point {position} must be finite in [0, 1]",
            )
        indices.append(iteration)
        values.append(value)

    expected_indices = list(range(1, expected_iterations + 1))
    if indices != expected_indices:
        missing = sorted(set(expected_indices) - set(indices))
        unexpected = sorted(set(indices) - set(expected_indices))
        return (
            None,
            f"M5-T telemetry key {key} indices must be ordered and contiguous over "
            f"1..{expected_iterations}; missing={missing[:10]}, unexpected={unexpected[:10]}",
        )
    return values, None


def classify_m5t_telemetry_points(
    points_by_key: dict[str, list[list[Any]] | None],
    *,
    expected_iterations: int,
) -> dict[str, Any]:
    """Classify one M5-T seed from term-specific, exactly indexed densities."""
    if (
        isinstance(expected_iterations, bool)
        or not isinstance(expected_iterations, int)
        or expected_iterations < 2
    ):
        return {
            "verdict": "invalid-telemetry",
            "evidence": {"reason": "M5-T expected_iterations must be an integer >= 2"},
        }

    densities: dict[str, list[float]] = {}
    errors: list[str] = []
    for key in _MT_TELEMETRY_KEYS:
        values, error = _validate_m5t_density_points(
            points_by_key.get(key), key=key, expected_iterations=expected_iterations
        )
        if error is not None:
            errors.append(error)
        elif values is not None:
            densities[key] = values
    if errors:
        return {
            "verdict": "invalid-telemetry",
            "evidence": {
                "reason": "M5-T telemetry validation failed: " + "; ".join(errors),
                "telemetry_errors": errors,
                "expected_iterations": expected_iterations,
            },
        }

    anchor = densities["m5t_anchor_pos_termination_density"]
    ee = densities["m5t_ee_body_pos_termination_density"]
    height = densities["m5t_height_termination_density"]
    episode_end = densities["m5t_episode_end_density"]
    rates: list[float] = []
    consistency_errors: list[str] = []
    tolerance = 2e-4  # training logs render scalar densities to four decimals
    for iteration, (anchor_value, ee_value, height_value, episode_value) in enumerate(
        zip(anchor, ee, height, episode_end, strict=True), start=1
    ):
        if height_value < max(anchor_value, ee_value):
            consistency_errors.append(
                f"iteration {iteration}: union density is below an individual term density"
            )
        if height_value > anchor_value + ee_value + tolerance:
            consistency_errors.append(
                f"iteration {iteration}: union density exceeds the sum of term densities"
            )
        if height_value > episode_value:
            consistency_errors.append(
                f"iteration {iteration}: height termination density exceeds episode-end density"
            )
        if episode_value <= 0.0:
            consistency_errors.append(
                f"iteration {iteration}: no episode end, so the activation fraction is undefined"
            )
        else:
            rates.append(height_value / episode_value)

    if consistency_errors:
        return {
            "verdict": "invalid-telemetry",
            "evidence": {
                "reason": "M5-T telemetry consistency failed: "
                + "; ".join(consistency_errors),
                "telemetry_errors": consistency_errors,
                "expected_iterations": expected_iterations,
            },
        }

    result = classify_m5t_seed(rates)
    result["evidence"].update(
        {
            "telemetry_source": "anchor_pos_or_ee_body_pos_episode_end_fraction",
            "telemetry_iterations": expected_iterations,
            "anchor_pos_density_mean": float(np.mean(anchor)),
            "ee_body_pos_density_mean": float(np.mean(ee)),
            "height_union_density_mean": float(np.mean(height)),
            "episode_end_density_mean": float(np.mean(episode_end)),
        }
    )
    return result


def classify_m5t_seed(termination_rate_series: list[float]) -> dict[str, Any]:
    """M5-T manufactured-frontier existence check for one seed."""
    if not termination_rate_series:
        return {
            "verdict": "incomplete",
            "evidence": {"reason": "no term-specific termination-rate series"},
        }
    lo, hi = _MT_BAND
    in_band = [lo <= r <= hi for r in termination_rate_series]
    fraction = float(np.mean(in_band))
    evidence = {
        "iterations": len(termination_rate_series),
        "fraction_in_band": fraction,
        "band": list(_MT_BAND),
        "rate_median": float(np.median(termination_rate_series)),
    }
    verdict = "activated" if fraction >= _MT_MIN_FRACTION_IN_BAND else "invalid-inactive"
    return {"verdict": verdict, "evidence": evidence}


def aggregate_arm(seed_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Arm verdict over the preregistered seeds 0-2 (two activations required)."""
    verdicts = {seed: r["verdict"] for seed, r in seed_results.items()}
    actual_seeds = set(verdicts)
    missing_seeds = sorted(_EXPECTED_SCREEN_SEEDS - actual_seeds)
    unexpected_seeds = sorted(actual_seeds - _EXPECTED_SCREEN_SEEDS)
    if missing_seeds or unexpected_seeds:
        return {
            "arm_verdict": "incomplete",
            "seed_verdicts": verdicts,
            "missing_seeds": missing_seeds,
            "unexpected_seeds": unexpected_seeds,
            "seeds_activated": sum(v == "activated" for v in verdicts.values()),
            "seeds_total": len(verdicts),
        }
    invalid_verdicts = [v for v in verdicts.values() if v.startswith("invalid-")]
    if invalid_verdicts:
        arm = (
            "invalid-unstable"
            if "invalid-unstable" in invalid_verdicts
            else invalid_verdicts[0]
        )
    elif any(v == "incomplete" for v in verdicts.values()):
        arm = "incomplete"
    elif sum(v == "activated" for v in verdicts.values()) >= 2:
        arm = "activated"
    else:
        arm = "inactive"
    return {
        "arm_verdict": arm,
        "seed_verdicts": verdicts,
        "missing_seeds": [],
        "unexpected_seeds": [],
        "seeds_activated": sum(v == "activated" for v in verdicts.values()),
        "seeds_total": len(verdicts),
    }


def _checkpoint_summaries(paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Load official multi-checkpoint dumps and legacy one-summary JSON files."""
    summaries: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = _load_json(path)
        dump_sha256 = _sha256_file(path)
        records = payload.get("checkpoints") if "checkpoints" in payload else [payload]
        if not isinstance(records, list):
            raise ValueError(f"{path}: checkpoints must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"{path}: checkpoint summary must be an object")
            record = dict(record)
            seed = record.get("seed")
            if seed is None:
                match = _SEED_RE.search(str(record.get("checkpoint_path") or ""))
                if match is None:
                    raise ValueError(
                        f"{path}: no 'seed' field and no seedN in checkpoint_path"
                    )
                seed = match.group(1)
            seed = int(seed)
            if seed in summaries:
                raise ValueError(f"duplicate checkpoint summary for seed {seed}")
            if not isinstance(record.get("bins"), list):
                raise ValueError(f"{path}: checkpoint summary for seed {seed} has no bins list")
            record["_checkpoint_dump_path"] = str(path)
            record["_checkpoint_dump_sha256"] = dump_sha256
            summaries[seed] = record
    return summaries


def _telemetry_points(
    telemetry_summary: dict[str, Any], key: str
) -> dict[int, list[list[Any]]]:
    """Raw indexed telemetry points, preserving null/non-finite values for validation."""
    out: dict[int, list[list[Any]]] = {}
    for record in telemetry_summary.get("logs", []):
        if not isinstance(record, dict):
            continue
        match = _SEED_RE.search(str(record.get("log_path") or ""))
        series = ((record.get("keys") or {}).get(key) or {}).get("series")
        if match is None or not isinstance(series, list):
            continue
        seed = int(match.group(1))
        if seed in out:
            raise ValueError(f"duplicate telemetry series for seed {seed}, key {key}")
        out[seed] = series
    return out


def _telemetry_series(telemetry_summary: dict[str, Any], key: str) -> dict[int, list[float]]:
    """Per-seed value series for one adp_samp key from summarize_sampler_telemetry output."""
    out: dict[int, list[float]] = {}
    for record in telemetry_summary.get("logs", []):
        if not isinstance(record, dict) or not record.get("adaptive_telemetry_present"):
            continue
        match = _SEED_RE.search(str(record.get("log_path") or ""))
        series = ((record.get("keys") or {}).get(key) or {}).get("series")
        if match is None or not isinstance(series, list):
            continue
        seed = int(match.group(1))
        if seed in out:
            raise ValueError(f"duplicate telemetry series for seed {seed}, key {key}")
        out[seed] = [
            float(v) for _, v in series if isinstance(v, (int, float))
        ]
    return out


def _seed_provenance(
    dump: dict[str, Any] | None,
    *,
    include_zpd_sampler: bool,
    ranking_sha256: str | None,
    telemetry_sha256: str | None,
    dataset_manifest_sha256: str,
    dataset_manifest_sha256_kind: str,
    paired_dataset_sha256: str,
) -> dict[str, Any]:
    """Flat per-seed provenance contract consumed by multiseed finalization."""
    dump = dump or {}
    provenance = {
        "checkpoint_sha256": dump.get("checkpoint_sha256"),
        "checkpoint_global_step": dump.get("global_step"),
        "checkpoint_dump_sha256": dump.get("_checkpoint_dump_sha256"),
        "difficulty_ranking_sha256": ranking_sha256,
        "telemetry_summary_sha256": telemetry_sha256,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_manifest_sha256_kind": dataset_manifest_sha256_kind,
        "paired_dataset_sha256": paired_dataset_sha256,
    }
    if include_zpd_sampler:
        provenance.update(
            {
                "sampler_schema": dump.get("sampler_schema"),
                "sampler_config": dump.get("sampler_config"),
            }
        )
    return provenance


def _zpd_provenance_errors(
    dump: dict[str, Any],
    *,
    expected_signal: str,
    expected_iterations: int,
    expected_optimism_k: float,
    expected_advmass_n: int,
    paired_dataset_sha256: str,
) -> list[str]:
    """Validate the checkpoint/dump fields required to certify a ZPD seed."""
    errors: list[str] = []
    try:
        _validate_sha256(dump.get("checkpoint_sha256"), field="checkpoint_sha256")
    except ValueError as exc:
        errors.append(str(exc))
    global_step = dump.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        errors.append("checkpoint global_step must be an integer")
    elif global_step != expected_iterations:
        errors.append(
            f"checkpoint global_step {global_step} != expected_iterations {expected_iterations}"
        )

    expected_schema = {
        "kind": _ZPD_SAMPLER_SCHEMA_KIND,
        "version": _ZPD_SAMPLER_SCHEMA_VERSION,
        "sampling_mass_semantics": _ZPD_SAMPLING_MASS_SEMANTICS,
    }
    if dump.get("sampler_schema") != expected_schema:
        errors.append(
            f"sampler_schema mismatch: expected {expected_schema!r}, "
            f"got {dump.get('sampler_schema')!r}"
        )
    if dump.get("sampling_mass_semantics") != _ZPD_SAMPLING_MASS_SEMANTICS:
        errors.append("dump sampling_mass_semantics is missing or incorrect")
    if dump.get("sampling_mass_source") != _ZPD_SAMPLING_MASS_SOURCE:
        errors.append("dump sampling_mass_source is missing or incorrect")

    config = dump.get("sampler_config")
    required_config_keys = {
        "signal",
        "optimism_k",
        "evidence_half_life",
        "advmass_n",
        "uniform_sampling_rate",
        "tripwire_max_prob_over_uniform",
        "bin_size",
        "paired_dataset_sha256",
    }
    if not isinstance(config, dict) or set(config) != required_config_keys:
        errors.append(
            "sampler_config must contain exactly the frozen ZPD config fields: "
            f"{sorted(required_config_keys)}"
        )
        return errors

    if config["signal"] != expected_signal:
        errors.append(
            f"sampler signal {config['signal']!r} != classifier arm signal {expected_signal!r}"
        )
    if config["optimism_k"] != expected_optimism_k:
        errors.append(
            f"sampler optimism_k {config['optimism_k']!r} != expected {expected_optimism_k!r}"
        )
    if config["advmass_n"] != expected_advmass_n:
        errors.append(
            f"sampler advmass_n {config['advmass_n']!r} != expected {expected_advmass_n!r}"
        )
    if config["paired_dataset_sha256"] != paired_dataset_sha256:
        errors.append("sampler paired_dataset_sha256 does not match classifier provenance")
    numeric_checks = {
        "optimism_k": lambda value: not isinstance(value, bool)
        and math.isfinite(float(value)),
        "uniform_sampling_rate": lambda value: not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0,
        "tripwire_max_prob_over_uniform": lambda value: not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 1.0,
        "bin_size": lambda value: not isinstance(value, bool) and int(value) == value and value > 0,
        "advmass_n": lambda value: not isinstance(value, bool) and int(value) == value and value > 1,
    }
    for field, predicate in numeric_checks.items():
        try:
            valid = predicate(config[field])
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            errors.append(f"sampler_config.{field} is invalid: {config[field]!r}")
    half_life = config["evidence_half_life"]
    if half_life is not None:
        try:
            half_life_valid = math.isfinite(float(half_life)) and float(half_life) > 0
        except (TypeError, ValueError, OverflowError):
            half_life_valid = False
        if not half_life_valid:
            errors.append(
                f"sampler_config.evidence_half_life is invalid: {half_life!r}"
            )
    return errors


def _checkpoint_provenance_errors(
    dump: dict[str, Any], *, expected_iterations: int
) -> list[str]:
    """Validate checkpoint identity fields shared by non-ZPD M5 activation arms."""
    errors: list[str] = []
    try:
        _validate_sha256(dump.get("checkpoint_sha256"), field="checkpoint_sha256")
    except ValueError as exc:
        errors.append(str(exc))
    global_step = dump.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        errors.append("checkpoint global_step must be an integer")
    elif global_step != expected_iterations:
        errors.append(
            f"checkpoint global_step {global_step} != expected_iterations {expected_iterations}"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("m5_l", "m5_a", "m5_t", "failure_rate"))
    parser.add_argument("--checkpoint-dumps", type=Path, nargs="*", default=[])
    parser.add_argument("--bin-motions", type=Path, help="{'bin_motion_keys': [...]} JSON")
    parser.add_argument("--difficulty-ranking", type=Path, help="SIM-D1 ranking, easiest first")
    parser.add_argument("--telemetry-summary", type=Path)
    parser.add_argument("--optimism-k", type=float, default=0.0)
    parser.add_argument("--advmass-n", type=int, default=16)
    parser.add_argument(
        "--expected-iterations",
        type=int,
        help="Exact count N for ZPD D9 or M5-T telemetry indices 1..N and checkpoint step N.",
    )
    parser.add_argument("--min-actual-draws", type=int, default=_MIN_ACTUAL_DRAWS_DEFAULT)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument(
        "--dataset-manifest-sha256-kind",
        required=True,
        choices=("file_bytes", "canonical_json_without_source_locations_v1"),
    )
    parser.add_argument("--paired-dataset-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    try:
        dataset_manifest_sha256 = _validate_sha256(
            args.dataset_manifest_sha256, field="dataset_manifest_sha256"
        )
        paired_dataset_sha256 = _validate_sha256(
            args.paired_dataset_sha256, field="paired_dataset_sha256"
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.min_actual_draws < 1:
        parser.error("--min-actual-draws must be positive")
    is_zpd = args.arm in ("m5_l", "m5_a")
    if (is_zpd or args.arm == "m5_t") and (
        args.expected_iterations is None or args.expected_iterations < 2
    ):
        parser.error("ZPD and M5-T arms require --expected-iterations >= 2")
    if is_zpd and args.telemetry_summary is None:
        parser.error("ZPD arms require --telemetry-summary for complete D9 validation")
    if args.arm == "m5_t" and (
        args.telemetry_summary is None
        or not args.checkpoint_dumps
        or args.difficulty_ranking is None
    ):
        parser.error(
            "M5-T requires --telemetry-summary, --checkpoint-dumps, and "
            "--difficulty-ranking for finalizable evidence"
        )

    telemetry = _load_json(args.telemetry_summary) if args.telemetry_summary else None
    telemetry_sha256 = _sha256_file(args.telemetry_summary) if args.telemetry_summary else None
    ranking: list[str] | None = None
    ranking_sha256: str | None = None
    if args.difficulty_ranking:
        from scripts.research.summarize_sonic_logs import load_difficulty_ranking

        ranking = load_difficulty_ranking(args.difficulty_ranking)
        ranking_sha256 = _sha256_file(args.difficulty_ranking)
    bin_keys: list[str] | None = None
    if args.bin_motions:
        bin_keys = _load_json(args.bin_motions)["bin_motion_keys"]

    seed_results: dict[int, dict[str, Any]] = {}
    checkpoint_summaries = (
        _checkpoint_summaries(args.checkpoint_dumps) if args.checkpoint_dumps else {}
    )
    if args.arm == "m5_t":
        assert telemetry is not None
        telemetry_by_key: dict[str, dict[int, list[list[Any]]]] = {}
        telemetry_error: str | None = None
        try:
            telemetry_by_key = {
                key: _telemetry_points(telemetry, key) for key in _MT_TELEMETRY_KEYS
            }
        except ValueError as exc:
            telemetry_error = f"M5-T telemetry is ambiguous: {exc}"
        telemetry_seeds = {
            seed for by_seed in telemetry_by_key.values() for seed in by_seed
        }
        for seed in sorted(set(checkpoint_summaries) | telemetry_seeds):
            if telemetry_error is not None:
                seed_results[seed] = {
                    "verdict": "invalid-telemetry",
                    "evidence": {"reason": telemetry_error},
                }
            else:
                seed_results[seed] = classify_m5t_telemetry_points(
                    {
                        key: telemetry_by_key.get(key, {}).get(seed)
                        for key in _MT_TELEMETRY_KEYS
                    },
                    expected_iterations=args.expected_iterations,
                )
            dump = checkpoint_summaries.get(seed)
            provenance_errors = _checkpoint_provenance_errors(
                dump or {}, expected_iterations=args.expected_iterations
            )
            if provenance_errors:
                seed_results[seed]["verdict"] = "incomplete"
                seed_results[seed]["evidence"]["provenance_errors"] = provenance_errors
                seed_results[seed]["evidence"]["reason"] = (
                    "checkpoint provenance validation failed: "
                    + "; ".join(provenance_errors)
                )
            seed_results[seed]["provenance"] = _seed_provenance(
                dump,
                include_zpd_sampler=False,
                ranking_sha256=ranking_sha256,
                telemetry_sha256=telemetry_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                dataset_manifest_sha256_kind=args.dataset_manifest_sha256_kind,
                paired_dataset_sha256=paired_dataset_sha256,
            )
    else:
        if not args.checkpoint_dumps or ranking is None:
            parser.error("--checkpoint-dumps and --difficulty-ranking are required")
        tripwire_coverage_error = None
        try:
            tripwire = (
                _telemetry_points(telemetry, "tripwire_max_prob_binding")
                if is_zpd and telemetry
                else {}
            )
        except ValueError as exc:
            tripwire = {}
            tripwire_coverage_error = f"D9 telemetry is ambiguous: {exc}"
        if (
            is_zpd
            and tripwire_coverage_error is None
            and set(tripwire) != _EXPECTED_SCREEN_SEEDS
        ):
            missing = sorted(_EXPECTED_SCREEN_SEEDS - set(tripwire))
            unexpected = sorted(set(tripwire) - _EXPECTED_SCREEN_SEEDS)
            tripwire_coverage_error = (
                "D9 telemetry seed coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        telemetry_last: dict[int, dict[str, float]] = {}
        if telemetry:
            for key in ("prob_max_over_uniform", "num_concentrated_bins"):
                for seed, series in _telemetry_series(telemetry, key).items():
                    if series:
                        telemetry_last.setdefault(seed, {})[key] = series[-1]
        for seed, dump in sorted(checkpoint_summaries.items()):
            bins = dump["bins"]
            seed_bin_keys = bin_keys or dump.get("bin_motion_keys")
            if not isinstance(seed_bin_keys, list):
                raise ValueError(
                    f"seed {seed}: no bin-motion map in checkpoint dump; pass --bin-motions "
                    "or train a ZPD checkpoint containing the map"
                )
            if args.arm == "failure_rate":
                seed_results[seed] = classify_failure_rate_seed(
                    bins, seed_bin_keys, ranking, telemetry_last=telemetry_last.get(seed)
                )
            else:
                expected_signal = "learnability" if args.arm == "m5_l" else "advantage_mass"
                dump_sampler_config = dump.get("sampler_config")
                dump_uniform_rate = (
                    dump_sampler_config.get("uniform_sampling_rate", 0.1)
                    if isinstance(dump_sampler_config, dict)
                    else 0.1
                )
                if not isinstance(dump_uniform_rate, (int, float)) or not math.isfinite(
                    float(dump_uniform_rate)
                ):
                    dump_uniform_rate = 0.1
                seed_results[seed] = classify_zpd_seed(
                    bins,
                    seed_bin_keys,
                    ranking,
                    signal=expected_signal,
                    optimism_k=args.optimism_k,
                    advmass_n=args.advmass_n,
                    uniform_rate=float(dump_uniform_rate),
                    tripwire_binding_series=tripwire.get(seed),
                    expected_iterations=args.expected_iterations,
                    tripwire_seed_coverage_error=tripwire_coverage_error,
                    min_actual_draws=args.min_actual_draws,
                )
                provenance_errors = _zpd_provenance_errors(
                    dump,
                    expected_signal=expected_signal,
                    expected_iterations=args.expected_iterations,
                    expected_optimism_k=args.optimism_k,
                    expected_advmass_n=args.advmass_n,
                    paired_dataset_sha256=paired_dataset_sha256,
                )
                if provenance_errors:
                    seed_results[seed]["verdict"] = "incomplete"
                    seed_results[seed]["evidence"]["provenance_errors"] = provenance_errors
                    seed_results[seed]["evidence"]["reason"] = (
                        "ZPD provenance validation failed: " + "; ".join(provenance_errors)
                    )
            seed_results[seed]["provenance"] = _seed_provenance(
                dump,
                include_zpd_sampler=is_zpd,
                ranking_sha256=ranking_sha256,
                telemetry_sha256=telemetry_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                dataset_manifest_sha256_kind=args.dataset_manifest_sha256_kind,
                paired_dataset_sha256=paired_dataset_sha256,
            )

    result = {
        "schema_version": 2,
        "kind": "m5_activation_classification",
        "arm": args.arm,
        "expected_iterations": args.expected_iterations,
        "provenance": {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "dataset_manifest_sha256_kind": args.dataset_manifest_sha256_kind,
            "paired_dataset_sha256": paired_dataset_sha256,
            "difficulty_ranking_sha256": ranking_sha256,
            "telemetry_summary_sha256": telemetry_sha256,
            "checkpoint_dump_sha256_by_seed": {
                str(seed): dump.get("_checkpoint_dump_sha256")
                for seed, dump in sorted(checkpoint_summaries.items())
            },
        },
        "rule": (
            "Frozen in scripts/research/classify_m5_activation.py (plan v1.1 §4, Z6 amendment) "
            "before any M5 GPU run. ZPD: frontier over-alloc >= 1.2 AND posterior Spearman >= 0.4 "
            "(peakedness sufficient; tripwire binding >= 5% of iters => invalid-unstable). "
            "failure_rate: peakedness OR hard-half ratio >= 1.5. M5-T: exact indexed "
            "anchor_pos-or-ee_body_pos episode-end fraction in [0.15, 0.6] for >= 50% "
            "of iterations. Arm: >= 2/3 seeds."
        ),
        "seeds": {str(s): r for s, r in sorted(seed_results.items())},
        **aggregate_arm(seed_results),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"arm={args.arm} verdict={result['arm_verdict']} seeds={result['seed_verdicts']}")
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
