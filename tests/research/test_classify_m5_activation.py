"""Tests freezing the SIM-M5 activation-gate rules (plan v1.1 §4, Z6 amendment).

The synthetic fixtures encode the forecast's scenarios so the classifier's
behavior is pinned to what the gate re-freeze promised: a correctly-working
learnability teacher (frontier-loaded, sane posterior, hard-half ratio < 1.5)
must ACTIVATE under the ZPD rule and would have FAILED the old ratio rule.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.classify_m5_activation import (
    aggregate_arm,
    bin_difficulty_ranks,
    classify_failure_rate_seed,
    classify_m5t_seed,
    classify_zpd_seed,
    derive_termination_rate_series,
    frontier_tercile_mask,
)

# 12 motions, easiest first; one bin per motion keeps fixtures readable.
_RANKING = [f"m{i:02d}" for i in range(12)]
_BIN_KEYS = list(_RANKING)


def _bins(episodes, failures):
    n = len(episodes)
    prob_uniform = 1.0 / n
    out = []
    for e, f in zip(episodes, failures):
        rate = f / e if e > 0 else 0.0
        out.append(
            {
                "num_episodes": float(e),
                "num_failures": float(f),
                "failure_rate": rate,
                "recomputed_prob_unweighted": prob_uniform,  # overwritten where needed
            }
        )
    return out


def _working_teacher_bins():
    """Mastered easy tercile, ~50% frontier tercile, impossible hard tercile."""
    episodes, failures = [], []
    for i in range(12):
        if i < 4:  # easy: high survival
            episodes.append(100.0)
            failures.append(2.0)
        elif i < 8:  # frontier: ~50%
            episodes.append(100.0)
            failures.append(50.0)
        else:  # impossible: hazard ~1
            episodes.append(100.0)
            failures.append(98.0)
    return _bins(episodes, failures)


def test_ranks_and_frontier_tercile() -> None:
    ranks = bin_difficulty_ranks(_BIN_KEYS, _RANKING)
    mask = frontier_tercile_mask(ranks)
    assert mask.sum() == 4
    assert list(np.nonzero(mask)[0]) == [4, 5, 6, 7]


def test_ranks_reject_unknown_motion_keys() -> None:
    with pytest.raises(ValueError, match="absent from the difficulty ranking"):
        bin_difficulty_ranks(["m00", "not_in_ranking"], _RANKING)


def test_working_learnability_teacher_activates_under_zpd_rule() -> None:
    result = classify_zpd_seed(
        _working_teacher_bins(), _BIN_KEYS, _RANKING, signal="learnability"
    )
    assert result["verdict"] == "activated"
    ev = result["evidence"]
    assert ev["frontier_over_alloc"] >= 1.2
    assert ev["posterior_spearman_vs_easiness"] >= 0.4
    # The Z6 point: this same working teacher would FAIL the old hard-half rule
    # (utility near-zero on the impossible bins that fill the hard half).
    assert ev["pmax_over_uniform_unweighted"] < 10.0  # diffuse, as forecast


def test_flat_posterior_is_inactive() -> None:
    # No evidence: uniform posterior everywhere -> no targeting, no sanity.
    result = classify_zpd_seed(
        _bins([0.0] * 12, [0.0] * 12), _BIN_KEYS, _RANKING, signal="learnability"
    )
    assert result["verdict"] == "inactive"


def test_insane_posterior_fails_even_if_frontier_loaded() -> None:
    # Difficulty INVERTED vs the ranking (easy motions fail, hard survive):
    # posterior sanity must catch it regardless of allocation shape.
    episodes = [100.0] * 12
    failures = [98.0] * 4 + [50.0] * 4 + [2.0] * 4
    result = classify_zpd_seed(
        _bins(episodes, failures), _BIN_KEYS, _RANKING, signal="learnability"
    )
    assert result["evidence"]["posterior_spearman_vs_easiness"] < 0
    assert result["verdict"] == "inactive"


def test_tripwire_binding_over_5pct_is_invalid_unstable() -> None:
    series_bad = [0.0] * 90 + [1.0] * 10  # 10% binding
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=series_bad,
    )
    assert result["verdict"] == "invalid-unstable"
    series_ok = [0.0] * 98 + [1.0] * 2  # 2% binding
    result = classify_zpd_seed(
        _working_teacher_bins(),
        _BIN_KEYS,
        _RANKING,
        signal="learnability",
        tripwire_binding_series=series_ok,
    )
    assert result["verdict"] == "activated"


def test_failure_rate_rule_unchanged_hard_half_path() -> None:
    bins = _working_teacher_bins()
    # Failure-rate-shaped allocation: mass proportional to failure rate.
    rates = np.asarray([b["failure_rate"] for b in bins])
    prob = rates / rates.sum()
    for b, p in zip(bins, prob):
        b["recomputed_prob_unweighted"] = float(p)
    result = classify_failure_rate_seed(bins, _BIN_KEYS, _RANKING)
    assert result["evidence"]["hard_half_mass_ratio"] >= 1.5
    assert result["verdict"] == "activated"


def test_failure_rate_prefers_telemetry_peakedness() -> None:
    bins = _bins([100.0] * 12, [10.0] * 12)  # flat recompute
    result = classify_failure_rate_seed(
        bins,
        _BIN_KEYS,
        _RANKING,
        telemetry_last={"prob_max_over_uniform": 15.0, "num_concentrated_bins": 2.0},
    )
    assert result["evidence"]["peakedness_source"] == "telemetry"
    assert result["verdict"] == "activated"


def test_m5t_band_check() -> None:
    in_band = [0.3] * 60 + [0.05] * 40  # 60% in [0.15, 0.6]
    assert classify_m5t_seed(in_band)["verdict"] == "activated"
    out_of_band = [0.02] * 80 + [0.3] * 20  # schedule never bit
    assert classify_m5t_seed(out_of_band)["verdict"] == "invalid-inactive"
    assert classify_m5t_seed([])["verdict"] == "incomplete"


def test_derive_termination_rate_from_cumulative_counts() -> None:
    eps = [1.0, 2.0, 4.0, 4.0, 8.0]  # per-iteration deltas: 1, 2, 0, 4
    fails = [1.0, 1.5, 2.5, 2.5, 3.5]  # deltas: 0.5, 1.0, 0.0, 1.0
    rates = derive_termination_rate_series(fails, eps)
    assert rates == pytest.approx([0.5, 0.5, 0.25])  # zero-delta iteration skipped


def test_aggregate_arm_two_thirds_rule() -> None:
    ok = {"verdict": "activated"}
    no = {"verdict": "inactive"}
    assert aggregate_arm({0: ok, 1: ok, 2: no})["arm_verdict"] == "activated"
    assert aggregate_arm({0: ok, 1: no, 2: no})["arm_verdict"] == "inactive"
    bad = {"verdict": "invalid-unstable"}
    assert aggregate_arm({0: ok, 1: ok, 2: bad})["arm_verdict"] == "invalid-unstable"
    assert aggregate_arm({})["arm_verdict"] == "incomplete"
