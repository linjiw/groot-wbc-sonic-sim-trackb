from __future__ import annotations

import pytest

from scripts.research.paired_stats import (
    exact_sign_flip_permutation_p,
    paired_bootstrap_ci,
    paired_delta_statistics,
    power_note,
)

_SIM_M3_DELTAS = [0.202, 0.142, -0.012]


def test_permutation_p_reproduces_recorded_sim_m3_value() -> None:
    result = exact_sign_flip_permutation_p(_SIM_M3_DELTAS)

    assert result["n"] == 3
    assert result["observed_mean"] == pytest.approx(0.110667, abs=1e-6)
    # Only the {+,+,-} -> {+,+,+} flip (mean 0.118667) exceeds the observed mean,
    # so 7 of 8 sign assignments have mean <= observed.
    assert result["permutation_p_one_sided"] == pytest.approx(7 / 8)
    assert result["min_achievable_p"] == pytest.approx(1 / 8)
    assert result["direction"] == "improvement_is_negative_delta"


def test_permutation_p_is_small_for_consistent_improvement() -> None:
    result = exact_sign_flip_permutation_p([-1.0, -0.9, -1.1])

    # All-negative deltas: only the identity assignment has mean <= observed.
    assert result["permutation_p_one_sided"] == pytest.approx(1 / 8)


def test_permutation_p_counts_identity_so_p_never_below_min() -> None:
    for deltas in ([-5.0], [-2.0, -3.0], [0.0, 0.0, 0.0]):
        result = exact_sign_flip_permutation_p(deltas)
        assert result["permutation_p_one_sided"] >= result["min_achievable_p"]


def test_permutation_p_rejects_empty_and_oversized_inputs() -> None:
    with pytest.raises(ValueError, match="at least one delta"):
        exact_sign_flip_permutation_p([])
    with pytest.raises(ValueError, match="n <= 20"):
        exact_sign_flip_permutation_p([0.1] * 21)


def test_statistics_reject_non_finite_deltas() -> None:
    # NaN comparisons are all False, which would yield p = 0.0 — reading as
    # maximally significant on invalid data. Must raise instead.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="non-finite"):
            exact_sign_flip_permutation_p([0.1, bad])
        with pytest.raises(ValueError, match="non-finite"):
            paired_bootstrap_ci([0.1, bad])


def test_bootstrap_ci_is_deterministic_and_brackets_mean() -> None:
    first = paired_bootstrap_ci(_SIM_M3_DELTAS)
    second = paired_bootstrap_ci(_SIM_M3_DELTAS)

    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]
    assert first["ci_low"] >= min(_SIM_M3_DELTAS) - 1e-12
    assert first["ci_high"] <= max(_SIM_M3_DELTAS) + 1e-12


def test_bootstrap_ci_seed_changes_resamples() -> None:
    assert paired_bootstrap_ci(_SIM_M3_DELTAS, seed=0) != paired_bootstrap_ci(
        _SIM_M3_DELTAS, seed=1
    )


def test_power_note_marks_small_n_as_screen_only() -> None:
    assert "screen only" in power_note(3)
    assert "0.125" in power_note(3)
    assert "screen only" not in power_note(5)


def test_paired_delta_statistics_bundles_all_fields() -> None:
    stats = paired_delta_statistics(_SIM_M3_DELTAS)

    assert stats["n"] == 3
    assert stats["mean_delta"] == pytest.approx(0.110667, abs=1e-6)
    assert stats["permutation_p_one_sided"] == pytest.approx(7 / 8)
    assert len(stats["bootstrap_ci_95"]) == 2
    assert stats["bootstrap_rng_seed"] == 0
    assert "screen only" in stats["power_note"]
