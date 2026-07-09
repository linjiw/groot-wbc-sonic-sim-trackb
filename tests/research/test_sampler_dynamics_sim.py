from __future__ import annotations

import numpy as np
import pytest

from scripts.research.sampler_dynamics_sim import (
    _compute_prob,
    _difficulty_regime,
    _hard_half_mass_ratio,
    run_forecast,
    simulate,
    validate_against_real,
)


def test_compute_prob_matches_release_math_uniform_input() -> None:
    # Equal failure rates -> uniform distribution after blend+renorm.
    fr = np.ones(10)
    prob = _compute_prob(fr, np.ones(10), cap=200.0, uniform_rate=0.1)
    assert np.allclose(prob, 0.1)
    assert prob.sum() == pytest.approx(1.0)


def test_compute_prob_respects_clip_cap() -> None:
    # One elevated bin: a tight cap clips it, a loose cap does not. Mean of
    # [1000, 1x69] ~ 15.3, so cap=2 (bound ~30.5) clips 1000; cap=200 (bound ~3054)
    # does not.
    fr = np.ones(70)
    fr[0] = 1000.0
    tight = _compute_prob(fr, np.ones(70), cap=2.0, uniform_rate=0.1)
    loose = _compute_prob(fr, np.ones(70), cap=200.0, uniform_rate=0.1)
    assert tight[0] < loose[0]


def test_compute_prob_all_zero_falls_back_to_uniform() -> None:
    prob = _compute_prob(np.zeros(5), np.ones(5), cap=200.0, uniform_rate=0.1)
    assert np.allclose(prob, 0.2)


def test_hard_half_mass_ratio_uniform_is_one() -> None:
    difficulty = np.linspace(0, 1, 10)
    uniform = np.full(10, 0.1)
    assert _hard_half_mass_ratio(uniform, difficulty) == pytest.approx(1.0)


def test_hard_half_mass_ratio_detects_correct_targeting() -> None:
    difficulty = np.linspace(0, 1, 10)
    prob = difficulty / difficulty.sum()  # mass concentrated on hard bins
    assert _hard_half_mass_ratio(prob, difficulty) > 1.5


def test_simulate_failure_rate_flat_in_starved_regime() -> None:
    difficulty = _difficulty_regime("starved", 70, 0)
    r = simulate(difficulty, signal="failure_rate", seed=0, iters=50, episodes_per_iter=20)
    assert r["final_prob_max_over_uniform"] < 5.0
    assert r["final_num_concentrated_bins"] == 0


def test_simulate_failure_rate_concentrates_on_sparse_outliers() -> None:
    difficulty = _difficulty_regime("sparse_outlier", 70, 0)
    r = simulate(difficulty, signal="failure_rate", seed=0, iters=50, episodes_per_iter=200)
    assert r["final_prob_max_over_uniform"] >= 10.0
    assert r["final_num_concentrated_bins"] >= 1


def test_simulate_is_deterministic() -> None:
    difficulty = _difficulty_regime("spread", 70, 0)
    a = simulate(difficulty, signal="failure_rate", seed=0)
    b = simulate(difficulty, signal="failure_rate", seed=0)
    assert a["final_prob_max_over_uniform"] == b["final_prob_max_over_uniform"]


def test_simulate_init0_does_not_nan() -> None:
    # The M5a hazard: the sim must guard 0/0 to run (and records the caveat).
    difficulty = _difficulty_regime("starved", 70, 0)
    r = simulate(
        difficulty, signal="failure_rate", init_num_failures=0.0, seed=0, episodes_per_iter=20
    )
    assert np.isfinite(r["final_prob_max_over_uniform"])


def test_run_forecast_robust_findings_hold() -> None:
    result = run_forecast()
    f = result["findings"]
    assert f["F1_failure_rate_peaks_only_on_sparse_outliers"] is True
    assert f["F2_spread_targets_but_peakedness_gate_misfires"] is True
    assert f["F3_starved_regime_neither_concentrates_nor_targets"] is True
    assert result["simulated"] is True
    assert result["is_measurement"] is False
    assert "no MPJPE" in result["disclaimer"]


def test_run_forecast_findings_budget_independent() -> None:
    for budget in (8, 50):
        f = run_forecast(episodes_per_iter=budget)["findings"]
        assert f["F1_failure_rate_peaks_only_on_sparse_outliers"] is True
        assert f["F3_starved_regime_neither_concentrates_nor_targets"] is True


def test_error_ema_is_not_claimed_as_a_win() -> None:
    result = run_forecast()
    note = result["error_ema_conditional"]
    assert "NOT claimed to beat" in note
    # error_ema does not activate the peakedness gate either -> no accidental win claim.
    assert result["runs"]["error_ema_spread"]["final_num_concentrated_bins"] == 0


def test_validate_against_real_passes_when_prediction_matches() -> None:
    result = run_forecast()
    predicted = result["runs"]["failure_rate_starved"]["final_prob_max_over_uniform"]
    telemetry = {
        "logs": [
            {
                "adaptive_telemetry_present": True,
                "keys": {"prob_max_over_uniform": {"last": predicted + 0.5}},
            }
        ]
    }
    v = validate_against_real(result, telemetry, tol=2.0)
    assert v["validated"] is True


def test_validate_against_real_fails_when_prediction_diverges() -> None:
    result = run_forecast()
    telemetry = {
        "logs": [
            {"adaptive_telemetry_present": True, "keys": {"prob_max_over_uniform": {"last": 50.0}}}
        ]
    }
    v = validate_against_real(result, telemetry, tol=2.0)
    assert v["validated"] is False
