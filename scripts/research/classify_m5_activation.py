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
  the per-iteration early-termination rate (derived from the cumulative
  adp_samp count telemetry as delta failures / delta episode-equivalents, a
  per-traversal hazard proxy) must lie in [0.15, 0.6] for >= 50% of
  iterations, else *invalid-inactive* (schedule mis-calibrated; one
  recalibration allowed).

Arm-level activation requires the per-seed rule to PASS in >= 2/3 of seeds.

Inputs (all JSON, produced by existing tooling):
- checkpoint dump (`dump_sampler_checkpoint_state.py`) — per-bin episodes/failures;
- bin->motion map — `{"bin_motion_keys": [key per bin, dump order]}`; produce on
  robotixx with the documented snippet (motion_lib `adp_samp_bins[:, 0]` ->
  `_motion_data_keys`), committed next to the dump;
- SIM-D1 difficulty ranking — motion keys, EASIEST FIRST (frozen at D1 time);
- telemetry summary (`summarize_sampler_telemetry.py`) — per-iteration series for
  peakedness, the tripwire, and the M5-T rate derivation.

The sampling distribution for mass criteria is recomputed from the dump with the
same utility math as training (imported from sampler_dynamics_sim, which is
parity-tested against motion_lib_base), minus length-based bin weights (not
checkpointed — same caveat as dump_sampler_checkpoint_state.py; mass *ratios*
over uniform shares are insensitive to a common weighting).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.sampler_dynamics_sim import (  # noqa: E402
    _compute_prob_zpd,
    _rank_correlation,
    _zpd_utility,
)

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
_MIN_SEED_PASS_FRACTION = 2.0 / 3.0

_ZPD_SIGNALS = ("learnability", "advantage_mass")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


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


def _posterior_and_prob(
    episodes: np.ndarray,
    failures: np.ndarray,
    *,
    signal: str,
    optimism_k: float,
    advmass_n: int,
    uniform_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Posterior survival mean + recomputed (unweighted) sampling distribution."""
    fails = np.maximum(failures, 0.0)
    succ = np.maximum(episodes - fails, 0.0)
    p_mean = (1.0 + succ) / (2.0 + succ + fails)
    utility = _zpd_utility(succ, fails, signal=signal, optimism_k=optimism_k, advmass_n=advmass_n)
    prob, _ = _compute_prob_zpd(utility, np.ones(len(utility)), uniform_rate=uniform_rate)
    return p_mean, prob


def classify_zpd_seed(
    checkpoint_bins: list[dict[str, Any]],
    bin_motion_keys: list[str],
    ranking: list[str],
    *,
    signal: str,
    optimism_k: float = 0.0,
    advmass_n: int = 16,
    uniform_rate: float = 0.1,
    tripwire_binding_series: list[float] | None = None,
) -> dict[str, Any]:
    """One ZPD-arm seed against the Z6-amended activation rule."""
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

    p_mean, prob = _posterior_and_prob(
        episodes,
        failures,
        signal=signal,
        optimism_k=optimism_k,
        advmass_n=advmass_n,
        uniform_rate=uniform_rate,
    )

    frontier = frontier_tercile_mask(ranks)
    frontier_share = float(frontier.mean())
    frontier_mass = float(prob[frontier].sum())
    frontier_over_alloc = frontier_mass / frontier_share if frontier_share > 0 else 0.0

    # Posterior sanity: survival mean should be HIGH on easy motions. Ranks are
    # 0 = easiest, so correlate p_mean against easiness = -rank.
    posterior_spearman = _rank_correlation(p_mean, -ranks)

    n = len(prob)
    pmax_over_uniform = float(prob.max() * n)
    num_concentrated = int((prob > 10.0 / n).sum())
    peakedness = (
        pmax_over_uniform >= _PEAK_PMAX_OVER_UNIFORM
        and num_concentrated >= _PEAK_MIN_CONCENTRATED
    )

    evidence: dict[str, Any] = {
        "signal": signal,
        "num_bins": n,
        "frontier_share": frontier_share,
        "frontier_mass": frontier_mass,
        "frontier_over_alloc": frontier_over_alloc,
        "posterior_spearman_vs_easiness": posterior_spearman,
        "pmax_over_uniform_unweighted": pmax_over_uniform,
        "num_concentrated_bins_unweighted": num_concentrated,
    }

    # D9 validity assertion, from the telemetry series when present.
    if tripwire_binding_series is not None and tripwire_binding_series:
        binding_fraction = float(np.mean([1.0 if v > 0 else 0.0 for v in tripwire_binding_series]))
        evidence["tripwire_binding_fraction"] = binding_fraction
        if binding_fraction > _TRIPWIRE_MAX_BINDING_FRACTION:
            return {"verdict": "invalid-unstable", "evidence": evidence}
    else:
        evidence["tripwire_binding_fraction"] = None  # telemetry absent: recorded, not gating

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
    """Per-iteration hazard proxy from the cumulative adp_samp count telemetry.

    delta(num_failures_mean) / delta(num_episodes_mean) per iteration: failures
    per episode-equivalent of bin traversal — the rate the sampler itself
    observes. Iterations with no new episode mass yield no sample. NOTE: this is
    a per-traversal hazard, not an episode-level early-termination fraction; the
    M5-T band was preregistered against THIS derivation.
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


def classify_m5t_seed(termination_rate_series: list[float]) -> dict[str, Any]:
    """M5-T manufactured-frontier existence check for one seed."""
    if not termination_rate_series:
        return {
            "verdict": "incomplete",
            "evidence": {"reason": "no derivable termination-rate series"},
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
    """Arm verdict over seeds: >= 2/3 activated; any invalid-unstable poisons the arm."""
    verdicts = {seed: r["verdict"] for seed, r in seed_results.items()}
    n = len(verdicts)
    if n == 0:
        return {"arm_verdict": "incomplete", "seed_verdicts": verdicts}
    if any(v == "invalid-unstable" for v in verdicts.values()):
        arm = "invalid-unstable"
    elif sum(v == "activated" for v in verdicts.values()) >= _MIN_SEED_PASS_FRACTION * n:
        arm = "activated"
    elif all(v == "incomplete" for v in verdicts.values()):
        arm = "incomplete"
    else:
        arm = "inactive"
    return {
        "arm_verdict": arm,
        "seed_verdicts": verdicts,
        "seeds_activated": sum(v == "activated" for v in verdicts.values()),
        "seeds_total": n,
    }


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
        out[int(match.group(1))] = [
            float(v) for _, v in series if isinstance(v, (int, float))
        ]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True, choices=("m5_l", "m5_a", "m5_t", "failure_rate"))
    parser.add_argument("--checkpoint-dumps", type=Path, nargs="*", default=[])
    parser.add_argument("--bin-motions", type=Path, help="{'bin_motion_keys': [...]} JSON")
    parser.add_argument("--difficulty-ranking", type=Path, help="SIM-D1 ranking, easiest first")
    parser.add_argument("--telemetry-summary", type=Path)
    parser.add_argument("--optimism-k", type=float, default=0.0)
    parser.add_argument("--advmass-n", type=int, default=16)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    telemetry = _load_json(args.telemetry_summary) if args.telemetry_summary else None
    ranking: list[str] | None = None
    if args.difficulty_ranking:
        from scripts.research.summarize_sonic_logs import load_difficulty_ranking

        ranking = load_difficulty_ranking(args.difficulty_ranking)
    bin_keys: list[str] | None = None
    if args.bin_motions:
        bin_keys = _load_json(args.bin_motions)["bin_motion_keys"]

    seed_results: dict[int, dict[str, Any]] = {}
    if args.arm == "m5_t":
        if telemetry is None:
            parser.error("--telemetry-summary is required for m5_t")
        fails = _telemetry_series(telemetry, "num_failures_mean")
        eps = _telemetry_series(telemetry, "num_episodes_mean")
        for seed in sorted(set(fails) & set(eps)):
            series = derive_termination_rate_series(fails[seed], eps[seed])
            seed_results[seed] = classify_m5t_seed(series)
    else:
        if not args.checkpoint_dumps or bin_keys is None or ranking is None:
            parser.error("--checkpoint-dumps, --bin-motions, --difficulty-ranking are required")
        tripwire = (
            _telemetry_series(telemetry, "tripwire_max_prob_binding") if telemetry else {}
        )
        telemetry_last: dict[int, dict[str, float]] = {}
        if telemetry:
            for key in ("prob_max_over_uniform", "num_concentrated_bins"):
                for seed, series in _telemetry_series(telemetry, key).items():
                    if series:
                        telemetry_last.setdefault(seed, {})[key] = series[-1]
        for dump_path in args.checkpoint_dumps:
            dump = _load_json(dump_path)
            seed = dump.get("seed")
            if seed is None:
                match = _SEED_RE.search(str(dump.get("checkpoint_path") or "")) or _SEED_RE.search(
                    str(dump_path)
                )
                if match is None:
                    raise ValueError(
                        f"{dump_path}: no 'seed' field and no seedN in checkpoint_path/filename"
                    )
                seed = match.group(1)
            seed = int(seed)
            bins = dump["bins"]
            if args.arm == "failure_rate":
                seed_results[seed] = classify_failure_rate_seed(
                    bins, bin_keys, ranking, telemetry_last=telemetry_last.get(seed)
                )
            else:
                seed_results[seed] = classify_zpd_seed(
                    bins,
                    bin_keys,
                    ranking,
                    signal="learnability" if args.arm == "m5_l" else "advantage_mass",
                    optimism_k=args.optimism_k,
                    advmass_n=args.advmass_n,
                    tripwire_binding_series=tripwire.get(seed),
                )

    result = {
        "schema_version": 1,
        "kind": "m5_activation_classification",
        "arm": args.arm,
        "rule": (
            "Frozen in scripts/research/classify_m5_activation.py (plan v1.1 §4, Z6 amendment) "
            "before any M5 GPU run. ZPD: frontier over-alloc >= 1.2 AND posterior Spearman >= 0.4 "
            "(peakedness sufficient; tripwire binding >= 5% of iters => invalid-unstable). "
            "failure_rate: peakedness OR hard-half ratio >= 1.5. M5-T: derived termination rate "
            "in [0.15, 0.6] for >= 50% of iterations. Arm: >= 2/3 seeds."
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
