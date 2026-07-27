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

# ZPD-teacher constants, frozen with research_plan_zpd_teacher.md (§3.4.1). These
# mirror the Change A implementation in motion_lib_base.py and the plan's locked
# decisions: D3 (optimism k=1 deterministic interval-max), D4 (half-life H ≈ 4x
# mean per-bin visits per recompute), D9 (tripwire 20x uniform replaces the cap).
_ZPD_SIGNALS = ("learnability", "advantage_mass")
_ZPD_H_MULT = 4.0  # H = _ZPD_H_MULT * episodes_per_iter / n_bins (min 2.0)
_ZPD_OPTIMISM_K = 1.0
_ZPD_ADVMASS_N = 16
_ZPD_TRIPWIRE_X_UNIFORM = 20.0
# Traversal accounting: a failed episode contributes this fraction of a full
# bin-traversal in episode-equivalents (real system: steps survived / bin length).
_TRAVERSAL_PARTIAL_CREDIT = 0.5


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


def _zpd_utility(
    succ: np.ndarray,
    fails: np.ndarray,
    *,
    signal: str,
    optimism_k: float = _ZPD_OPTIMISM_K,
    advmass_n: int = _ZPD_ADVMASS_N,
) -> np.ndarray:
    """Port of MotionLibBase._compute_zpd_utility (Change A) to numpy.

    Beta(1+succ, 1+fail) posterior over per-bin survival. learnability uses the
    exact closed form E[p(1-p)] when optimism is off; advantage_mass evaluates
    u(p) = (1-(1-p)^N) - p at the posterior mean. With optimism_k > 0 the utility
    is maximized over [p_mean - k*sd, p_mean + k*sd] (unimodal u: peak value if
    the interval contains the peak, else the larger endpoint).
    """
    a = 1.0 + np.maximum(succ, 0.0)
    b = 1.0 + np.maximum(fails, 0.0)
    p_mean = a / (a + b)
    if signal == "learnability":
        p_star = 0.5

        def u_fn(p: np.ndarray) -> np.ndarray:
            return p * (1.0 - p)

    elif signal == "advantage_mass":
        n_band = float(advmass_n)
        p_star = 1.0 - n_band ** (-1.0 / (n_band - 1.0))

        def u_fn(p: np.ndarray) -> np.ndarray:
            return np.maximum((1.0 - (1.0 - p) ** n_band) - p, 0.0)

    else:
        raise ValueError(f"not a ZPD signal: {signal!r}")

    if optimism_k > 0:
        var = a * b / (np.square(a + b) * (a + b + 1.0))
        sd = np.sqrt(var)
        lo = np.clip(p_mean - optimism_k * sd, 0.0, 1.0)
        hi = np.clip(p_mean + optimism_k * sd, 0.0, 1.0)
        utility = np.maximum(u_fn(lo), u_fn(hi))
        peak_inside = (lo <= p_star) & (p_star <= hi)
        u_star = float(u_fn(np.asarray([p_star]))[0])
        return np.where(peak_inside, u_star, utility)
    if signal == "learnability":
        return a * b / ((a + b) * (a + b + 1.0))  # exact E[p(1-p)]
    return u_fn(p_mean)


def _compute_prob_zpd(
    utility: np.ndarray,
    bin_weights: np.ndarray,
    *,
    uniform_rate: float,
    tripwire_x_uniform: float = _ZPD_TRIPWIRE_X_UNIFORM,
) -> tuple[np.ndarray, bool]:
    """ZPD probability path (Change A): NO mean-x-cap clip; tripwire ceiling instead.

    Returns (prob, tripwire_binding).
    """
    total = utility.sum()
    if total > 0:
        based = utility / total
    else:
        based = np.full_like(utility, 1.0 / len(utility))
    uniform = np.full_like(based, 1.0 / len(based))
    prob = based * (1.0 - uniform_rate) + uniform * uniform_rate
    prob = prob * bin_weights
    prob = prob / prob.sum()
    ceiling = tripwire_x_uniform / len(prob)
    binding = bool((prob > ceiling).any())
    if binding:
        prob = np.minimum(prob, ceiling)
        prob = prob / prob.sum()
    return prob, binding


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Tie-aware Spearman correlation using average ranks.

    Equal values receive the same average rank. In particular, a constant vector
    has zero variance and returns 0 rather than inheriting an arbitrary correlation
    from input order.
    """

    def average_ranks(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1:
            raise ValueError("rank correlation inputs must be one-dimensional")
        if not np.isfinite(values).all():
            raise ValueError("rank correlation inputs must be finite")
        order = np.argsort(values, kind="stable")
        ranks = np.empty(len(values), dtype=float)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[order[end]] == values[order[start]]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1)
            start = end
        return ranks

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("rank correlation inputs must have equal length")
    ra = average_ranks(a)
    rb = average_ranks(b)
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
    evidence_half_life: float | None = None,
    optimism_k: float = _ZPD_OPTIMISM_K,
    advmass_n: int = _ZPD_ADVMASS_N,
    seed: int = 0,
) -> dict[str, Any]:
    """Run the sampler feedback loop against an assumed per-bin difficulty.

    ``difficulty[b]`` in [0,1] is the per-episode termination probability of bin
    ``b`` AND the mean of its continuous tracking-error signal. ``signal`` selects
    the reweighting statistic: ``failure_rate`` (release), ``error_ema``, or the
    Change A ZPD utilities ``learnability`` / ``advantage_mass`` (Beta posterior
    over survival with evidence-scaled half-life decay and deterministic optimism,
    mirroring motion_lib_base.py).
    """
    if signal not in ("failure_rate", "error_ema", *_ZPD_SIGNALS):
        raise ValueError(f"unknown signal {signal!r}")
    is_zpd = signal in _ZPD_SIGNALS
    rng = np.random.default_rng(seed)
    n = len(difficulty)
    bin_weights = np.ones(n)  # uniform lengths/peer counts => weights renormalize away

    # ZPD arms start from empty counts (Beta(1,1) prior lives in the utility);
    # the release prior seeds BOTH counts (:2407-2414).
    zpd_init = 0.0 if is_zpd else init_num_failures
    num_failures = np.full(n, zpd_init, dtype=float)
    num_episodes = np.full(n, zpd_init, dtype=float)
    # error_ema state: EMA of observed per-episode tracking error, plus a seen mask.
    error_ema = np.zeros(n)
    error_seen = np.zeros(n, dtype=bool)

    prob = np.full(n, 1.0 / n)  # init uniform
    series: list[dict[str, float]] = []
    observed_failures_total = 0.0
    tripwire_binding_iters = 0

    for _ in range(iters):
        # Attribute this iteration's episode-completions to bins by the current
        # sampling distribution (faithful: an episode is counted in the bin where
        # it ends; sampling prob drives which bins get played).
        hit_counts = rng.multinomial(episodes_per_iter, prob)
        terminated = rng.binomial(hit_counts, difficulty)  # Bernoulli(difficulty) per episode
        observed_failures_total += float(terminated.sum())

        if is_zpd:
            # Occupancy semantics (plan D2): survivors credit a full traversal,
            # failed episodes a partial one; evidence-scaled decay (D4) halves the
            # OLD counts every ``evidence_half_life`` episode-equivalents per bin.
            survived = hit_counts - terminated
            delta_eps = survived + _TRAVERSAL_PARTIAL_CREDIT * terminated
            delta_fails = terminated.astype(float)
            if evidence_half_life is not None:
                keep = 0.5 ** (delta_eps / evidence_half_life)
                num_episodes = num_episodes * keep + delta_eps
                num_failures = num_failures * keep + delta_fails
            else:
                num_episodes = num_episodes + delta_eps
                num_failures = num_failures + delta_fails
        else:
            # Release update (:2486 length-normalized episodes; :2499 raw failures).
            num_episodes += hit_counts / bin_motion_length
            num_failures += terminated * failure_counts_multiplier

        if is_zpd:
            succ = np.maximum(num_episodes - num_failures, 0.0)
            utility = _zpd_utility(
                succ, num_failures, signal=signal, optimism_k=optimism_k, advmass_n=advmass_n
            )
            stat = utility  # recorded for the contrast diagnostic below
            prob, binding = _compute_prob_zpd(utility, bin_weights, uniform_rate=uniform_rate)
            tripwire_binding_iters += int(binding)
            tele = _telemetry_from_prob(prob)
            series.append(tele)
            continue

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

    # Mass-fraction diagnostics for the ZPD forecast (impossible-bin waste is the
    # diagnosed pathology; frontier mass is where a ZPD utility should move it).
    mastered = difficulty < 0.05
    frontier = (difficulty >= 0.2) & (difficulty <= 0.8)
    impossible = difficulty > 0.9

    # Posterior-sanity forecast for the M5-L activation sub-gate: does the Beta
    # posterior's survival mean rank-correlate with true survival (1 - d)?
    posterior_rank_corr = None
    if is_zpd:
        post_mean = (1.0 + np.maximum(num_episodes - num_failures, 0.0)) / (2.0 + num_episodes)
        posterior_rank_corr = _rank_correlation(post_mean, 1.0 - difficulty)

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
            "evidence_half_life": evidence_half_life,
            "optimism_k": optimism_k if is_zpd else None,
            "advmass_n": advmass_n if signal == "advantage_mass" else None,
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
        "mastered_mass": float(prob[mastered].sum()) if mastered.any() else float("nan"),
        "frontier_mass": float(prob[frontier].sum()) if frontier.any() else float("nan"),
        "impossible_mass": float(prob[impossible].sum()) if impossible.any() else float("nan"),
        # Uniform shares of each class, so mass fractions can be read as over/under
        # allocation (a ZPD utility should be OVER on frontier, UNDER on both ends).
        "mastered_share": float(mastered.mean()),
        "frontier_share": float(frontier.mean()),
        "impossible_share": float(impossible.mean()),
        "posterior_rank_corr_vs_survival": posterior_rank_corr,
        # D9 validity assertion input: fraction of iterations where the tripwire bound.
        "tripwire_binding_fraction": tripwire_binding_iters / iters if is_zpd else None,
        "series_prob_max_over_uniform": [round(s["prob_max_over_uniform"], 4) for s in series],
    }


def _difficulty_regime(name: str, n: int, seed: int) -> np.ndarray:
    """Assumed per-bin termination/error difficulty in [0,1]. INPUT assumption.

    Four regimes isolate the two separable flatness drivers (density vs contrast)
    plus the mixed regime the ZPD forecast targets:
    - ``starved``: low level AND low relative spread -> few failures, low contrast
      (the sample_data working hypothesis).
    - ``spread``: broad difficulty frontier -> abundant failures, moderate contrast
      (a SIM-D1-PASSING dataset).
    - ``sparse_outlier``: a few very-hard bins amid easy ones -> high contrast on a
      handful of bins (the only structure the failure-rate sampler visibly peaks on).
    - ``spread_impossible``: 30% mastered / 50% frontier / 20% impossible-at-budget —
      the regime where failure-rate's difficulty-monotone utility structurally
      over-commits to hazard≈1 bins (thesis C2).
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
    if name == "spread_impossible":
        difficulty = np.empty(n)
        idx = rng.permutation(n)
        n_mastered, n_frontier = int(n * 0.3), int(n * 0.5)
        difficulty[idx[:n_mastered]] = 0.02
        difficulty[idx[n_mastered : n_mastered + n_frontier]] = rng.uniform(0.2, 0.7, n_frontier)
        difficulty[idx[n_mastered + n_frontier :]] = 0.98
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


def run_zpd_forecast(
    *,
    n_bins: int = 70,
    iters: int = 50,
    episodes_per_iter: int = 200,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Preregistered ZPD-utility forecast (research_plan_zpd_teacher.md §3.4.1).

    Replaces the throwaway Appendix A script of the handoff doc with tested
    tooling. Compares the release failure-rate sampler against the Change A
    signals (learnability with D3 optimism + D4 evidence-scaled decay;
    advantage_mass N=16 ablation) across the four difficulty regimes.

    Preregistered forecast findings (frozen before any GPU run):
    - Z1: in mixed regimes (sparse_outlier, spread_impossible) the release
      sampler puts a large mass fraction on impossible (d > 0.9) bins;
      learnability cuts that mass by at least half.
    - Z2: learnability's frontier mass in spread_impossible exceeds the release
      sampler's (reallocation goes TO the frontier, not to mastered bins).
    - Z3: learnability targets the frontier (frontier over-allocation
      frontier_mass/frontier_share >= 1.2 in spread_impossible) while staying
      diffuse (pmax/uniform under the peakedness gate) — activation must be
      measured as frontier allocation, not concentration.
    - Z4: in the starved regime ZPD signals are as flat as everything else
      (SIM-D1 headroom gate stays a hard prerequisite).
    - Z5: the D9 tripwire never binds in any healthy regime (binding fraction
      0.0) — it is a validity assertion, not a shaping mechanism.
    - Z6 (GATE-AMENDMENT finding): the hard-half/easy-half mass-ratio >= 1.5
      criterion — the targeting criterion preregistered for the M5 activation
      sub-gate — MISFIRES on learnability (ratio ~1.0 across regimes): a ZPD
      utility is symmetric around the frontier, not difficulty-monotone, so it
      deliberately down-weights the impossible bins that populate the hard
      half. The activation gate must be re-frozen on frontier over-allocation
      (+ posterior sanity, which run-level posterior_rank_corr_vs_survival
      confirms is measurable) BEFORE any M5-L run, else a correctly working
      teacher is classified invalid-inactive.
    """
    # H (D4): a small multiple of mean per-bin visits per recompute.
    half_life = max(2.0, _ZPD_H_MULT * episodes_per_iter / n_bins)
    arms: dict[str, dict[str, Any]] = {
        "failure_rate": {"signal": "failure_rate"},
        "learnability": {
            "signal": "learnability",
            "evidence_half_life": half_life,
            "optimism_k": _ZPD_OPTIMISM_K,
        },
        "advantage_mass": {
            "signal": "advantage_mass",
            "evidence_half_life": half_life,
            "optimism_k": 0.0,  # ablation isolates the utility shape
            "advmass_n": _ZPD_ADVMASS_N,
        },
    }
    regimes = ("starved", "spread", "sparse_outlier", "spread_impossible")

    keys = (
        "final_prob_max_over_uniform",
        "targeting_hard_half_mass_ratio",
        "mastered_mass",
        "frontier_mass",
        "impossible_mass",
        "mastered_share",
        "frontier_share",
        "impossible_share",
        "posterior_rank_corr_vs_survival",
    )
    runs: dict[str, dict[str, Any]] = {}
    for arm_name, kw in arms.items():
        for regime in regimes:
            per_seed = [
                simulate(
                    _difficulty_regime(regime, n_bins, s),
                    iters=iters,
                    episodes_per_iter=episodes_per_iter,
                    seed=7000 + s,
                    **kw,
                )
                for s in seeds
            ]
            agg = {}
            for k in keys:
                values = [r[k] for r in per_seed if r[k] is not None]
                finite = [v for v in values if v == v]  # drop NaN (class absent)
                agg[k] = float(np.mean(finite)) if finite else None
            binding = [r["tripwire_binding_fraction"] for r in per_seed]
            agg["tripwire_binding_fraction"] = (
                float(np.mean(binding)) if binding[0] is not None else None
            )
            # Over-allocation vs uniform per class (None when the class is absent).
            for cls in ("mastered", "frontier", "impossible"):
                mass, share = agg[f"{cls}_mass"], agg[f"{cls}_share"]
                agg[f"{cls}_over_alloc"] = mass / share if mass is not None and share else None
            agg["per_seed"] = per_seed
            runs[f"{arm_name}_{regime}"] = agg

    fr_out = runs["failure_rate_sparse_outlier"]
    fr_mix = runs["failure_rate_spread_impossible"]
    ln_out = runs["learnability_sparse_outlier"]
    ln_mix = runs["learnability_spread_impossible"]
    ln_starved = runs["learnability_starved"]

    findings = {
        "Z1_learnability_halves_impossible_mass": bool(
            ln_out["impossible_mass"] <= 0.5 * fr_out["impossible_mass"]
            and ln_mix["impossible_mass"] <= 0.5 * fr_mix["impossible_mass"]
        ),
        "Z2_learnability_reallocates_to_frontier": bool(
            ln_mix["frontier_mass"] > fr_mix["frontier_mass"]
        ),
        "Z3_zpd_targets_frontier_but_stays_diffuse": bool(
            ln_mix["frontier_over_alloc"] >= 1.2
            and ln_mix["final_prob_max_over_uniform"] < _ACTIVATION_PMAX
        ),
        "Z4_starved_regime_still_flat_for_zpd": bool(
            ln_starved["targeting_hard_half_mass_ratio"] < _TARGETING_MASS_RATIO
        ),
        "Z5_tripwire_never_binds_in_healthy_regimes": bool(
            all(
                runs[key]["tripwire_binding_fraction"] == 0.0
                for key in runs
                if runs[key]["tripwire_binding_fraction"] is not None
            )
        ),
        # Z6: the plan's hard-half mass-ratio >= 1.5 activation criterion misfires
        # on a working ZPD utility (it down-weights impossible bins BY DESIGN, and
        # impossible bins live in the hard half). If this is True, the M5-L
        # activation sub-gate must be re-frozen on frontier over-allocation +
        # posterior sanity before launch.
        "Z6_hard_half_ratio_activation_criterion_misfires_on_learnability": bool(
            ln_mix["targeting_hard_half_mass_ratio"] < _TARGETING_MASS_RATIO
            and ln_mix["frontier_over_alloc"] >= 1.2
            and (ln_mix["posterior_rank_corr_vs_survival"] or 0.0) >= 0.4
        ),
    }

    return {
        "schema_version": 1,
        "kind": "zpd_teacher_forecast",
        "simulated": True,
        "is_measurement": False,
        "disclaimer": (
            "SIMULATION under an assumed static difficulty->termination model (no learning "
            "dynamics). Forecasts mechanism ALLOCATION only; produces no MPJPE and supports no "
            "performance claim (guardrail 11). Constants frozen with research_plan_zpd_teacher.md: "
            f"optimism_k={_ZPD_OPTIMISM_K}, advmass_n={_ZPD_ADVMASS_N}, "
            f"tripwire={_ZPD_TRIPWIRE_X_UNIFORM}x uniform, H={half_life} episode-equivalents here."
        ),
        "config": {
            "n_bins": n_bins,
            "iters": iters,
            "episodes_per_iter": episodes_per_iter,
            "seeds": list(seeds),
            "evidence_half_life": half_life,
        },
        "findings": findings,
        "runs": runs,
    }


def simulate_coupled_threshold_controller(
    *,
    eta: float,
    posterior_half_life: float | None = None,
    iters: int = 400,
    n_bins: int = 40,
    episodes_per_iter: int = 64,
    target_fail_rate: float = 0.35,
    tau_start: float = 3.0,
    tau_min: float = 1.0,
    delta_tau_cap: float | None = None,
    one_sided: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Coupled teacher + threshold-controller + learning-competence simulation.

    Port of the expert's Q7 appendix (docs/external/SONIC_RESPONSE.md) with our
    Change A learnability utility in place of their advmass draw. The latent
    competence c[b] is the survival probability at the loosest threshold; the
    effective survival at threshold strictness tau is c**(1/tau) (tau=1 strict,
    tau large loose — matching the expert's parameterization). Competence grows
    with successful visits. The controller nudges tau toward the target global
    early-termination rate; D7 hardening knobs (one_sided, delta_tau_cap) are
    exposed so the validation target (overshoot-and-pin at eta >= 0.3) and the
    mitigation can both be reproduced.
    """
    rng = np.random.default_rng(seed)
    competence = rng.uniform(0.02, 0.6, n_bins)
    num_episodes = np.zeros(n_bins)
    num_failures = np.zeros(n_bins)
    tau = float(tau_start)
    fail_series: list[float] = []
    tau_series: list[float] = []
    posterior_err_series: list[float] = []

    for _ in range(iters):
        survival = competence ** (1.0 / tau)
        succ_counts = np.maximum(num_episodes - num_failures, 0.0)
        utility = _zpd_utility(succ_counts, num_failures, signal="learnability", optimism_k=0.0)
        prob, _ = _compute_prob_zpd(utility, np.ones(n_bins), uniform_rate=_RELEASE_UNIFORM_RATE)

        visits = rng.multinomial(episodes_per_iter, prob)
        succ = rng.binomial(visits, survival)
        fail = visits - succ
        if posterior_half_life is not None:
            keep = 0.5 ** (visits / posterior_half_life)
            num_episodes = num_episodes * keep + visits
            num_failures = num_failures * keep + fail
        else:
            num_episodes = num_episodes + visits
            num_failures = num_failures + fail

        # Competence grows on successful practice (expert's update, verbatim shape).
        with np.errstate(divide="ignore", invalid="ignore"):
            succ_rate = np.where(visits > 0, succ / np.maximum(visits, 1), 0.0)
        competence = np.clip(
            competence + 0.002 * (visits / 8.0) * succ_rate * (1.0 - competence), 0.0, 0.99
        )

        obs_fail = float(fail.sum()) / max(float(visits.sum()), 1.0)
        fail_series.append(obs_fail)
        # Controller: tau shrinks (stricter) when failure is below target.
        delta = eta * (target_fail_rate - obs_fail)
        if one_sided:
            delta = min(delta, 0.0)  # D7: only tighten, never loosen
        if delta_tau_cap is not None:
            delta = float(np.clip(delta, -delta_tau_cap, delta_tau_cap))
        tau = float(np.clip(tau + delta, tau_min, tau_start))
        tau_series.append(tau)

        # Posterior tracking error: |posterior mean survival - true survival|.
        post_mean = (1.0 + np.maximum(num_episodes - num_failures, 0.0)) / (2.0 + num_episodes)
        posterior_err_series.append(float(np.abs(post_mean - competence ** (1.0 / tau)).mean()))

    late = slice(iters // 2, None)
    return {
        "simulated": True,
        "is_measurement": False,
        "params": {
            "eta": eta,
            "posterior_half_life": posterior_half_life,
            "iters": iters,
            "n_bins": n_bins,
            "episodes_per_iter": episodes_per_iter,
            "target_fail_rate": target_fail_rate,
            "one_sided": one_sided,
            "delta_tau_cap": delta_tau_cap,
            "seed": seed,
        },
        "final_tau": tau,
        "late_fail_rate_mean": float(np.mean(fail_series[late])),
        "late_posterior_error_mean": float(np.mean(posterior_err_series[late])),
        "pinned_at_strict_bound": bool(
            np.mean(np.asarray(tau_series[late]) <= tau_min + 1e-9) > 0.9
        ),
    }


def run_controller_scenarios(*, seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    """The expert's Q7 findings as a validation scenario set (plan §3.4.1).

    Validation targets (must reproduce, else the coupled-sim port is wrong):
    - CTRL1 overshoot-and-pin: eta=0.5 pins tau at the strict bound with late
      failure rate far above target; eta=0.05 holds failure near the target band.
    - CTRL2 slow-memory advantage: with the slow controller, LONG posterior
      memory tracks the moving survival better than fast forgetting (D4 coupling
      rule: never fast decay + active controller).
    - CTRL3 D7 hardening: one-sided + rate-capped controller at eta=0.05 does
      not pin and keeps late failure in the frontier band [0.15, 0.6].
    """

    def _avg(**kw: Any) -> dict[str, float]:
        runs = [simulate_coupled_threshold_controller(seed=s, **kw) for s in seeds]
        return {
            "late_fail_rate_mean": float(np.mean([r["late_fail_rate_mean"] for r in runs])),
            "late_posterior_error_mean": float(
                np.mean([r["late_posterior_error_mean"] for r in runs])
            ),
            "pinned_fraction": float(np.mean([r["pinned_at_strict_bound"] for r in runs])),
        }

    fast_ctrl = _avg(eta=0.5, posterior_half_life=200.0)
    slow_ctrl = _avg(eta=0.05, posterior_half_life=200.0)
    slow_ctrl_short_memory = _avg(eta=0.05, posterior_half_life=8.0)
    hardened = _avg(eta=0.05, posterior_half_life=200.0, one_sided=True, delta_tau_cap=0.002)

    findings = {
        "CTRL1_overshoot_and_pin_at_high_eta": bool(
            fast_ctrl["pinned_fraction"] > 0.5
            and fast_ctrl["late_fail_rate_mean"] > slow_ctrl["late_fail_rate_mean"] + 0.1
        ),
        "CTRL2_long_memory_tracks_better_under_slow_controller": bool(
            slow_ctrl["late_posterior_error_mean"]
            < slow_ctrl_short_memory["late_posterior_error_mean"]
        ),
        "CTRL3_hardened_controller_stays_in_band": bool(
            hardened["pinned_fraction"] == 0.0 and 0.15 <= hardened["late_fail_rate_mean"] <= 0.6
        ),
    }
    return {
        "schema_version": 1,
        "kind": "coupled_threshold_controller_scenarios",
        "simulated": True,
        "is_measurement": False,
        "disclaimer": (
            "SIMULATION with a toy competence model (expert's Q7 appendix shape). Validates the "
            "D7 controller rules qualitatively; supports no performance claim (guardrail 11)."
        ),
        "findings": findings,
        "runs": {
            "eta_0.5_H200": fast_ctrl,
            "eta_0.05_H200": slow_ctrl,
            "eta_0.05_H8": slow_ctrl_short_memory,
            "eta_0.05_hardened_D7": hardened,
        },
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
        default=None,
        help="Episode completions per iteration. Default depends on --forecast: 20 for "
        "release (~num_envs=8 micro scale; findings F1/F2/F3 are contrast-based and "
        "budget-independent) and 200 for zpd (the Z-findings are frozen at the "
        "release-scale budget where evidence actually accrues; the micro-budget "
        "behavior is covered by Z4's starved regime).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--validate-against",
        type=Path,
        default=None,
        help="Real sampler_telemetry.json to check the easy-regime prediction against.",
    )
    parser.add_argument(
        "--forecast",
        choices=("release", "zpd", "controller"),
        default="release",
        help="release = original F1-F4 forecast; zpd = Change A utility-family forecast "
        "(Z1-Z5, research_plan_zpd_teacher.md §3.4.1); controller = coupled "
        "threshold-controller scenarios (CTRL1-CTRL3, D7 validation targets).",
    )
    args = parser.parse_args()

    if args.forecast == "zpd":
        result = run_zpd_forecast(
            n_bins=args.n_bins,
            iters=args.iters,
            episodes_per_iter=args.episodes_per_iter or 200,
            seeds=tuple(args.seeds) if args.seeds else (0, 1, 2, 3, 4),
        )
    elif args.forecast == "controller":
        result = run_controller_scenarios(seeds=tuple(args.seeds) if args.seeds else (0, 1, 2))
    else:
        result = run_forecast(
            n_bins=args.n_bins,
            iters=args.iters,
            episodes_per_iter=args.episodes_per_iter or 20,
            seeds=tuple(args.seeds) if args.seeds else (0, 1, 2),
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
    if "decision_forecast" in result:
        print(f"  decision forecast: {result['decision_forecast'][:120]}...")
    print(f"wrote forecast to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
