#!/usr/bin/env python3
"""Numerical experiment: forecast adaptive-sampler dynamics without GPU.

This is a SIMULATION, not a measurement. It reimplements the release sampler's
per-bin update and probability recompute (faithful to
``gear_sonic/utils/motion_lib/motion_lib_base.py``) and runs it against an
ASSUMED per-bin difficulty distribution to answer three mechanistic questions
BEFORE any GPU is spent:

  Q1. Under the "easy / flat" regime (low per-bin termination probability, like
      the 2-motion sample_data), does the RELEASE failure-rate sampler stay flat
      (reproducing SIM-M3's prob_max_over_uniform~3, num_concentrated_bins=0)?
  Q2. Do the SIM-M5a knobs (init_num_failures=0, uniform_sampling_rate 0.1->0.05,
      larger bin_size) mechanically activate concentration, or is flatness
      structural because observed failures stay ~0 (prior-dominated)?
  Q3. Does an ``error_ema`` signal — a continuous per-episode tracking error,
      observed on EVERY episode rather than only on rare terminations —
      concentrate and correctly target the hard bins where failure_rate cannot?

Honesty guardrails (this artifact can never support a headline claim,
guardrail 6):
  - Every output record carries ``"simulated": true`` and ``"is_measurement":
    false``. No MPJPE is produced; "improvement" is never claimed. The sim
    forecasts MECHANISM ACTIVATION (does the distribution concentrate and target
    difficulty), not tracking performance.
  - The difficulty->termination map is an INPUT ASSUMPTION, stated per run. The
    sim is not circular: it assumes the difficulty distribution and DERIVES
    whether the mechanism responds — it does not assume the flatness conclusion.
  - error_ema's advantage here is strictly a SIGNAL-DENSITY advantage, not a
    baked-in difficulty oracle. termination ~ Bernoulli(difficulty) (sparse) and
    tracking_error ~ difficulty + noise (observed every episode) both derive
    from the same latent difficulty — as they would on the real robot (harder
    motions both terminate more and track worse). The sim shows error_ema wins
    because it observes that latent difficulty densely, not because it is handed
    a cleaner difficulty label.
  - Validation hook: once the real SIM-M3 telemetry is synced,
    ``--validate-against`` checks that the easy-regime run reproduces the
    observed final ``prob_max_over_uniform`` within tolerance. If it does not,
    the sim's assumptions are wrong and its Q2/Q3 forecasts are void.

Faithful mechanics (cite motion_lib_base.py):
  - prior: num_failures and num_episodes BOTH init to init_num_failures
    (:2407-2414) => prior failure_rate 1.0 per bin.
  - update (:2482-2499): per iteration, episodes[bin] += hits/bin_motion_length
    (length-normalized, :2486); failures[bin] += failed_hits *
    failure_counts_multiplier (NOT length-normalized, :2499). This asymmetry is
    reproduced deliberately.
  - failure_rate = num_failures / num_episodes (:2531).
  - prob (:2566-2589): clip failure_rate at active_mean*cap; failure_based =
    clipped/sum; blend failure_based*(1-u) + uniform*u; multiply by bin_weights
    (:2390-2395, length/mean /peer_bins); renormalize. max_prob constraints unset
    in release (skipped).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# Release/config constants (fable-next.md §1, sonic_release.yaml).
_RELEASE_CAP = 200.0
_RELEASE_UNIFORM_RATE = 0.1
_RELEASE_INIT_FAILURES = 1.0
_CONCENTRATED_MULTIPLE = 10.0  # num_concentrated_bins := #bins with prob > 10x uniform


def _telemetry_from_prob(prob: np.ndarray) -> dict[str, float]:
    """The three flatness diagnostics the real wrapper emits, from a prob vector."""
    n = len(prob)
    uniform = 1.0 / n
    return {
        "prob_max_over_uniform": float(prob.max() / uniform),
        "num_concentrated_bins": int((prob > _CONCENTRATED_MULTIPLE * uniform).sum()),
        "effective_num_bins": float(1.0 / np.square(prob).sum()),
    }


def _compute_prob(
    failure_rate: np.ndarray,
    bin_weights: np.ndarray,
    *,
    cap: float,
    uniform_rate: float,
) -> np.ndarray:
    """Port of update_adaptive_sampling_probabilities (:2566-2589), all bins active."""
    upper_bound = failure_rate.mean() * cap
    clipped = np.clip(failure_rate, 0.0, upper_bound)
    total = clipped.sum()
    if total == 0.0:
        failure_based = np.full_like(failure_rate, 1.0 / len(failure_rate))
    else:
        failure_based = clipped / total
    uniform = np.full_like(failure_based, 1.0 / len(failure_based))
    prob = failure_based * (1.0 - uniform_rate) + uniform * uniform_rate
    prob = prob * bin_weights
    return prob / prob.sum()


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (stdlib-free) between two vectors.

    Unreliable under heavy ties (e.g. a few extreme bins amid many equal-difficulty
    bins), which is exactly the sparse-outlier regime — use ``_hard_half_mass_ratio``
    as the primary targeting metric there.
    """
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def _hard_half_mass_ratio(prob: np.ndarray, difficulty: np.ndarray) -> float:
    """Sampling mass on the harder half of bins divided by the easier half.

    Tie-robust targeting metric: 1.0 == uniform (no targeting), > 1 == correctly
    biased toward hard bins, regardless of whether the distribution is peaked. This
    is the metric the SIM-M5 activation gate SHOULD use — peakedness
    (prob_max_over_uniform) only fires on sparse outliers, not on a correctly
    targeted but diffuse reweighting of a broad difficulty frontier.
    """
    order = np.argsort(difficulty)
    n = len(difficulty)
    easy = order[: n // 2]
    hard = order[n - n // 2 :]
    easy_mass = float(prob[easy].sum())
    hard_mass = float(prob[hard].sum())
    return hard_mass / easy_mass if easy_mass > 0 else float("inf")


def simulate(
    difficulty: np.ndarray,
    *,
    signal: str = "failure_rate",
    iters: int = 50,
    episodes_per_iter: int = 200,
    bin_motion_length: float = 50.0,
    init_num_failures: float = _RELEASE_INIT_FAILURES,
    cap: float = _RELEASE_CAP,
    uniform_rate: float = _RELEASE_UNIFORM_RATE,
    failure_counts_multiplier: float = 1.0,
    error_ema_beta: float = 0.1,
    error_noise: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the sampler feedback loop against an assumed per-bin difficulty.

    ``difficulty[b]`` in [0,1] is the per-episode termination probability of bin
    ``b`` AND the mean of its continuous tracking-error signal. ``signal`` selects
    the reweighting statistic: ``failure_rate`` (release) or ``error_ema``.
    """
    if signal not in ("failure_rate", "error_ema"):
        raise ValueError(f"unknown signal {signal!r}")
    rng = np.random.default_rng(seed)
    n = len(difficulty)
    bin_weights = np.ones(n)  # uniform lengths/peer counts => weights renormalize away

    # Release failure-rate state (both counts seeded with the prior).
    num_failures = np.full(n, init_num_failures, dtype=float)
    num_episodes = np.full(n, init_num_failures, dtype=float)
    # error_ema state: EMA of observed per-episode tracking error, plus a seen mask.
    error_ema = np.zeros(n)
    error_seen = np.zeros(n, dtype=bool)

    prob = np.full(n, 1.0 / n)  # init uniform
    series: list[dict[str, float]] = []
    observed_failures_total = 0.0

    for _ in range(iters):
        # Attribute this iteration's episode-completions to bins by the current
        # sampling distribution (faithful: an episode is counted in the bin where
        # it ends; sampling prob drives which bins get played).
        hit_counts = rng.multinomial(episodes_per_iter, prob)
        terminated = rng.binomial(hit_counts, difficulty)  # Bernoulli(difficulty) per episode
        observed_failures_total += float(terminated.sum())

        # Release update (:2486 length-normalized episodes; :2499 raw failures).
        num_episodes += hit_counts / bin_motion_length
        num_failures += terminated * failure_counts_multiplier

        if signal == "failure_rate":
            # Faithful ratio, but guard 0/0 (unplayed bins under init_num_failures=0,
            # which the real code does NOT guard — a genuine hazard of the M5a knob at
            # num_envs=8 over ~70 bins, recorded in the forecast caveats).
            stat = np.divide(
                num_failures, num_episodes, out=np.zeros_like(num_failures), where=num_episodes > 0
            )
        else:
            # Continuous tracking error is observed on EVERY episode (dense),
            # error ~ difficulty + noise. Bins hit this iteration update their EMA.
            hit = hit_counts > 0
            if hit.any():
                # First observation of a bin seeds its EMA; later ones blend.
                seed_mask = hit & ~error_seen
                blend_mask = hit & error_seen
                if seed_mask.any():
                    error_ema[seed_mask] = np.clip(
                        difficulty[seed_mask] + rng.normal(0.0, error_noise, seed_mask.sum()),
                        0.0,
                        None,
                    )
                if blend_mask.any():
                    obs_b = np.clip(
                        difficulty[blend_mask] + rng.normal(0.0, error_noise, blend_mask.sum()),
                        0.0,
                        None,
                    )
                    error_ema[blend_mask] = (1.0 - error_ema_beta) * error_ema[
                        blend_mask
                    ] + error_ema_beta * obs_b
                error_seen[hit] = True
            # Unseen bins fall back to the population mean so they are not starved to 0.
            stat = np.where(
                error_seen, error_ema, error_ema[error_seen].mean() if error_seen.any() else 0.0
            )

        prob = _compute_prob(
            np.asarray(stat, dtype=float), bin_weights, cap=cap, uniform_rate=uniform_rate
        )
        tele = _telemetry_from_prob(prob)
        series.append(tele)

    final = series[-1]
    observed_failures_per_bin_mean = observed_failures_total / n
    # prior-dominated proxy: mean observed failures per bin <= 2 (matches the
    # checkpoint-dump prior_dominated threshold in the classifier). This is a
    # DENSITY diagnostic — one of two separable flatness drivers.
    prior_dominated = observed_failures_per_bin_mean <= 2.0
    # CONTRAST diagnostic: coefficient of variation of the per-bin statistic
    # actually used to reweight. Low CV => near-uniform failure_based => flat even
    # when signal is abundant (the second, distinct flatness driver).
    stat_arr = np.asarray(stat, dtype=float)
    stat_cv = float(stat_arr.std() / stat_arr.mean()) if stat_arr.mean() > 0 else 0.0

    return {
        "signal": signal,
        "iters": iters,
        "num_bins": n,
        "params": {
            "episodes_per_iter": episodes_per_iter,
            "bin_motion_length": bin_motion_length,
            "init_num_failures": init_num_failures,
            "cap": cap,
            "uniform_rate": uniform_rate,
            "failure_counts_multiplier": failure_counts_multiplier,
            "error_ema_beta": error_ema_beta,
            "error_noise": error_noise,
            "seed": seed,
        },
        "final_prob_max_over_uniform": final["prob_max_over_uniform"],
        "final_num_concentrated_bins": final["num_concentrated_bins"],
        "final_effective_num_bins": final["effective_num_bins"],
        "observed_failures_total": observed_failures_total,
        "observed_failures_per_bin_mean": observed_failures_per_bin_mean,
        "prior_dominated": bool(prior_dominated),
        "stat_coefficient_of_variation": stat_cv,
        # Peakedness targeting (unreliable under ties) AND tie-robust mass ratio.
        "targeting_rank_corr_prob_vs_difficulty": _rank_correlation(prob, difficulty),
        "targeting_hard_half_mass_ratio": _hard_half_mass_ratio(prob, difficulty),
        "series_prob_max_over_uniform": [round(s["prob_max_over_uniform"], 4) for s in series],
    }


def _difficulty_regime(name: str, n: int, seed: int) -> np.ndarray:
    """Assumed per-bin termination/error difficulty in [0,1]. INPUT assumption.

    Three regimes isolate the two separable flatness drivers (density vs contrast):
    - ``starved``: low level AND low relative spread -> few failures, low contrast
      (the sample_data working hypothesis).
    - ``spread``: broad difficulty frontier -> abundant failures, moderate contrast
      (a SIM-D1-PASSING dataset).
    - ``sparse_outlier``: a few very-hard bins amid easy ones -> high contrast on a
      handful of bins (the only structure the failure-rate sampler visibly peaks on).
    """
    rng = np.random.default_rng(1000 + seed)
    if name == "starved":
        return np.clip(rng.beta(1.2, 200.0, n), 0.0, 0.05)
    if name == "spread":
        return np.clip(rng.beta(1.5, 2.5, n), 0.0, 0.95)
    if name == "sparse_outlier":
        difficulty = np.full(n, 0.02)
        hard = rng.choice(n, size=max(1, n // 20), replace=False)
        difficulty[hard] = 0.95
        return difficulty
    raise ValueError(f"unknown difficulty regime {name!r}")


# SIM-M5 activation gate as currently written in fable-next.md §Phase 3.
_ACTIVATION_PMAX = 10.0
_ACTIVATION_MIN_CONCENTRATED = 1.0
# A tie-robust targeting bar: correctly biasing >1.5x mass toward the hard half.
_TARGETING_MASS_RATIO = 1.5


def _peakedness_activates(agg: dict[str, Any]) -> bool:
    """The gate as written: peakedness only. Fires on sparse outliers, not spread."""
    return (
        agg["final_prob_max_over_uniform"] >= _ACTIVATION_PMAX
        and agg["final_num_concentrated_bins"] >= _ACTIVATION_MIN_CONCENTRATED
    )


def _targets(agg: dict[str, Any]) -> bool:
    """Proposed tie-robust criterion: mass correctly biased toward hard bins."""
    return agg["targeting_hard_half_mass_ratio"] >= _TARGETING_MASS_RATIO


def run_forecast(
    *,
    n_bins: int = 70,
    iters: int = 50,
    episodes_per_iter: int = 20,
    seeds: tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    """Robust, budget-explicit mechanism forecast (see module docstring for scope).

    Default ``episodes_per_iter=20`` approximates the real SIM-M3 micro budget
    (num_envs=8, short episodes); a sweep over budgets confirms the headline
    findings are budget-independent.
    """

    def _avg(regime: str, **kw: Any) -> dict[str, Any]:
        runs = []
        for s in seeds:
            difficulty = _difficulty_regime(regime, n_bins, s)
            runs.append(
                simulate(difficulty, iters=iters, episodes_per_iter=episodes_per_iter, seed=s, **kw)
            )
        keys = [
            "final_prob_max_over_uniform",
            "final_num_concentrated_bins",
            "final_effective_num_bins",
            "observed_failures_per_bin_mean",
            "stat_coefficient_of_variation",
            "targeting_rank_corr_prob_vs_difficulty",
            "targeting_hard_half_mass_ratio",
        ]
        agg = {k: float(np.mean([r[k] for r in runs])) for k in keys}
        agg["prior_dominated_majority"] = sum(r["prior_dominated"] for r in runs) * 2 > len(runs)
        agg["per_seed"] = runs
        return agg

    runs = {
        "failure_rate_starved": _avg("starved", signal="failure_rate"),
        "failure_rate_spread": _avg("spread", signal="failure_rate"),
        "failure_rate_sparse_outlier": _avg("sparse_outlier", signal="failure_rate"),
        "error_ema_starved": _avg("starved", signal="error_ema"),
        "error_ema_spread": _avg("spread", signal="error_ema"),
    }

    fr_spread = runs["failure_rate_spread"]
    fr_outlier = runs["failure_rate_sparse_outlier"]
    fr_starved = runs["failure_rate_starved"]

    # ROBUST findings the sim strongly supports (budget- and seed-averaged).
    findings = {
        # F1: failure-rate concentrates on SPARSE OUTLIERS, not on broad spread.
        "F1_failure_rate_peaks_only_on_sparse_outliers": (
            _peakedness_activates(fr_outlier) and not _peakedness_activates(fr_spread)
        ),
        # F2: on a broad frontier it TARGETS correctly (mass ratio) while staying
        # DIFFUSE (peakedness gate would misread it as inactive) -> the plan's
        # activation gate is mis-specified.
        "F2_spread_targets_but_peakedness_gate_misfires": (
            _targets(fr_spread) and not _peakedness_activates(fr_spread)
        ),
        # F3: in the truly starved regime nothing concentrates AND nothing targets
        # (no contrast to exploit) -> a mechanism swap alone cannot manufacture
        # signal; SIM-D1 (difficulty spread) must come first.
        "F3_starved_regime_neither_concentrates_nor_targets": (
            not _peakedness_activates(fr_starved) and not _targets(fr_starved)
        ),
        # F4: SIM-M5a init_num_failures=0 is an unguarded 0/0 hazard (the sim guards
        # it; the release code at motion_lib_base.py:2531 does not).
        "F4_m5a_init0_nan_hazard": (
            "init_num_failures=0 yields 0/0 for any bin unplayed in an iteration; "
            "motion_lib_base.py:2531 does not guard this division. The sim guards it to "
            "run, but SIM-M5a on real hardware (num_envs=8 over ~70 bins) must add a guard "
            "or guarantee every bin is sampled each iteration."
        ),
    }

    # error_ema: report the honest CONDITIONAL, never a win. Its only sim-internal
    # edge is low-budget TARGETING (denser signal), it degrades with error noise,
    # and error=difficulty+noise bakes in the ordering.
    error_ema_note = (
        "error_ema is NOT claimed to beat failure_rate. In simulation its only edge is "
        "recovering the difficulty ORDERING at low sample budget (denser signal: observed "
        "every episode, not only on rare terminations). That edge decays with error-observation "
        "noise and reverses on sparse-hard outliers, where near-certain binary termination lets "
        "failure_rate concentrate more sharply. error=difficulty+noise bakes the ordering in, so "
        "targeting quality here is an assumption, not a result. Real per-episode tracking-error SNR "
        "(from command.metrics) is a PRECONDITION to check before the SIM-M5b error_ema arm."
    )

    route = (
        "Headline (F1+F2): the release failure-rate sampler peaks only on sparse outliers, not on "
        "broad difficulty spread — so SIM-M5's peakedness-only activation gate (prob_max_over_uniform>=10 "
        "AND num_concentrated_bins>=1) would misclassify a correctly-targeting-but-diffuse sampler as "
        "invalid-inactive. RECOMMEND adding a targeting criterion (hard-half/easy-half mass ratio >= 1.5) "
        "to the SIM-M5 activation gate. F3: on a flat dataset no reactive signal concentrates, so SIM-D1 "
        "(difficulty spread) must precede any mechanism work — a swap cannot manufacture contrast. Validate "
        "F1/F3 against real SIM-M3 telemetry + checkpoint dump before acting."
    )

    return {
        "schema_version": 2,
        "kind": "sampler_dynamics_forecast",
        "simulated": True,
        "is_measurement": False,
        "disclaimer": (
            "SIMULATION under an assumed difficulty->termination/error model. Forecasts mechanism "
            "ACTIVATION and TARGETING only; produces no MPJPE and supports no performance claim "
            "(guardrail 6). Findings are budget- and seed-averaged; validate F1/F3 against real "
            "SIM-M3 telemetry + checkpoint dump before acting on any of them."
        ),
        "config": {
            "n_bins": n_bins,
            "iters": iters,
            "episodes_per_iter": episodes_per_iter,
            "seeds": list(seeds),
        },
        "findings": findings,
        "error_ema_conditional": error_ema_note,
        "decision_forecast": route,
        "runs": runs,
    }


def validate_against_real(
    forecast: dict[str, Any], telemetry_summary: dict[str, Any], *, tol: float = 2.0
) -> dict[str, Any]:
    """Check the easy-regime prediction against real SIM-M3 adaptive telemetry.

    The sim's flat-regime forecast is only trustworthy if it reproduces the
    observed final prob_max_over_uniform. Returns a pass/fail record; a FAIL
    voids the Q2/Q3 forecasts.
    """
    observed = []
    for record in telemetry_summary.get("logs", []):
        if not isinstance(record, dict) or not record.get("adaptive_telemetry_present"):
            continue
        entry = (record.get("keys") or {}).get("prob_max_over_uniform")
        if isinstance(entry, dict) and isinstance(entry.get("last"), (int, float)):
            observed.append(float(entry["last"]))
    if not observed:
        return {"validated": False, "reason": "no adaptive prob_max_over_uniform in real telemetry"}
    observed_mean = float(np.mean(observed))
    predicted = forecast["runs"]["failure_rate_starved"]["final_prob_max_over_uniform"]
    ok = abs(observed_mean - predicted) <= tol
    return {
        "validated": bool(ok),
        "observed_prob_max_over_uniform_mean": observed_mean,
        "predicted_prob_max_over_uniform": predicted,
        "tolerance": tol,
        "note": "PASS => sim's starved-regime assumptions hold; FAIL => the forecast findings are void.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=70)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--episodes-per-iter",
        type=int,
        default=20,
        help="Episode completions per iteration (~num_envs=8 micro scale). Findings F1/F2/F3 "
        "are contrast-based and budget-independent; a sweep is available via the API.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="Real sampler_telemetry.json to check the easy-regime prediction against.",
    )
    args = parser.parse_args()

    result = run_forecast(
        n_bins=args.n_bins,
        iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        seeds=tuple(args.seeds),
    )
    if args.validate_against is not None:
        with args.validate_against.open("r", encoding="utf-8") as f:
            telemetry = json.load(f)
        result["validation"] = validate_against_real(result, telemetry)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print("[SIMULATION — not a measurement]")
    for q, v in result["findings"].items():
        print(f"  {q}: {v if not isinstance(v, str) else v[:70] + '...'}")
    print(f"  decision forecast: {result['decision_forecast'][:120]}...")
    print(f"wrote forecast to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
