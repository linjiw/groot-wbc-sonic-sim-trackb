"""Tests for the ZPD modes of sampler_dynamics_sim (research_plan_zpd_teacher.md §3.4.1).

The Z-findings of ``run_zpd_forecast`` and the CTRL-findings of
``run_controller_scenarios`` are preregistered forecast artifacts: these tests
freeze them (a code change that flips one is a preregistration change and must
be deliberate). The utility/prob helpers are additionally checked against the
Change A implementation in motion_lib_base.py so the sim cannot drift from the
mechanism it forecasts.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase
from scripts.research.sampler_dynamics_sim import (
    _compute_prob_zpd,
    _difficulty_regime,
    _zpd_utility,
    run_controller_scenarios,
    run_zpd_forecast,
    simulate,
    simulate_coupled_threshold_controller,
)

# ---------------------------------------------------------------------------
# Parity with the Change A implementation (sim must not drift from the code)
# ---------------------------------------------------------------------------


def _motion_lib_utility(succ, fails, *, signal, optimism_k, advmass_n=16):
    stub = object.__new__(MotionLibBase)
    stub.adaptive_sampling_cfg = {
        "signal": signal,
        "optimism_k": optimism_k,
        "advmass_n": advmass_n,
    }
    stub.adp_samp_num_episodes = torch.tensor(succ, dtype=torch.float64) + torch.tensor(
        fails, dtype=torch.float64
    )
    stub.adp_samp_num_failures = torch.tensor(fails, dtype=torch.float64)
    _, utility = stub._compute_zpd_utility()
    return utility.numpy()


@pytest.mark.parametrize("signal", ["learnability", "advantage_mass"])
@pytest.mark.parametrize("optimism_k", [0.0, 1.0])
def test_sim_utility_matches_motion_lib_implementation(signal: str, optimism_k: float) -> None:
    rng = np.random.default_rng(11)
    succ = rng.integers(0, 80, 50).astype(float)
    fails = rng.integers(0, 80, 50).astype(float)
    sim_u = _zpd_utility(succ, fails, signal=signal, optimism_k=optimism_k, advmass_n=16)
    lib_u = _motion_lib_utility(succ, fails, signal=signal, optimism_k=optimism_k)
    assert np.allclose(sim_u, lib_u, atol=1e-10)


def test_compute_prob_zpd_uniform_floor_and_tripwire() -> None:
    # Flat utility -> uniform; extreme utility -> tripwire binds and reports it.
    prob, binding = _compute_prob_zpd(np.ones(10), np.ones(10), uniform_rate=0.1)
    assert np.allclose(prob, 0.1)
    assert binding is False
    utility = np.zeros(100)
    utility[0] = 1.0
    prob, binding = _compute_prob_zpd(utility, np.ones(100), uniform_rate=0.1)
    assert binding is True
    # Single-pass clamp + renorm (legacy max_prob_per_bin semantics): the renorm
    # can push back above the raw ceiling under extreme concentration — the
    # tripwire's job is the binding FLAG (validity assertion), not a hard cap.
    assert prob.max() < 0.75  # reduced from the unconstrained ~0.9
    assert prob.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# simulate() ZPD modes
# ---------------------------------------------------------------------------


def test_simulate_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError, match="unknown signal"):
        simulate(np.full(10, 0.5), signal="thompson")


def test_simulate_learnability_deterministic() -> None:
    d = _difficulty_regime("spread_impossible", 70, 0)
    kw = dict(signal="learnability", evidence_half_life=11.4, seed=0, episodes_per_iter=200)
    a = simulate(d, **kw)
    b = simulate(d, **kw)
    assert a["final_prob_max_over_uniform"] == b["final_prob_max_over_uniform"]
    assert a["impossible_mass"] == b["impossible_mass"]


def test_simulate_learnability_starves_impossible_bins() -> None:
    # The thesis mechanism at unit granularity: on the mixed regime the release
    # sampler over-allocates to impossible bins, learnability under-allocates.
    d = _difficulty_regime("spread_impossible", 70, 0)
    fr = simulate(d, signal="failure_rate", seed=0, episodes_per_iter=200)
    ln = simulate(d, signal="learnability", evidence_half_life=11.4, seed=0, episodes_per_iter=200)
    assert ln["impossible_mass"] < 0.5 * fr["impossible_mass"]
    assert ln["frontier_mass"] > fr["frontier_mass"]


def test_simulate_zpd_reports_tripwire_and_posterior_fields() -> None:
    d = _difficulty_regime("spread", 70, 0)
    r = simulate(d, signal="learnability", seed=0, episodes_per_iter=200)
    assert r["tripwire_binding_fraction"] == 0.0
    assert r["posterior_rank_corr_vs_survival"] is not None
    assert r["posterior_rank_corr_vs_survival"] > 0.4  # posterior-sanity analogue
    r_fr = simulate(d, signal="failure_rate", seed=0, episodes_per_iter=200)
    assert r_fr["tripwire_binding_fraction"] is None
    assert r_fr["posterior_rank_corr_vs_survival"] is None


def test_simulate_release_signal_unchanged_by_extension() -> None:
    # The F1 regression must still hold with the extended simulate().
    d = _difficulty_regime("sparse_outlier", 70, 0)
    r = simulate(d, signal="failure_rate", seed=0, iters=50, episodes_per_iter=200)
    assert r["final_prob_max_over_uniform"] >= 10.0


# ---------------------------------------------------------------------------
# Preregistered Z-findings (frozen forecast artifact)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zpd_forecast() -> dict:
    return run_zpd_forecast(seeds=(0, 1, 2, 3, 4))


def test_zpd_forecast_all_findings_hold(zpd_forecast: dict) -> None:
    f = zpd_forecast["findings"]
    assert f["Z1_learnability_halves_impossible_mass"] is True
    assert f["Z2_learnability_reallocates_to_frontier"] is True
    assert f["Z3_zpd_targets_frontier_but_stays_diffuse"] is True
    assert f["Z4_starved_regime_still_flat_for_zpd"] is True
    assert f["Z5_tripwire_never_binds_in_healthy_regimes"] is True
    assert f["Z6_hard_half_ratio_activation_criterion_misfires_on_learnability"] is True


def test_zpd_forecast_is_labeled_simulation(zpd_forecast: dict) -> None:
    assert zpd_forecast["simulated"] is True
    assert zpd_forecast["is_measurement"] is False
    assert "no MPJPE" in zpd_forecast["disclaimer"]


def test_zpd_forecast_impossible_mass_direction(zpd_forecast: dict) -> None:
    runs = zpd_forecast["runs"]
    # The C2 forecast band: release wastes a large fraction on impossible bins in
    # mixed regimes; learnability cuts it to a small one.
    assert runs["failure_rate_spread_impossible"]["impossible_mass"] > 0.4
    assert runs["learnability_spread_impossible"]["impossible_mass"] < 0.2


# ---------------------------------------------------------------------------
# Coupled threshold-controller scenarios (D7 validation targets)
# ---------------------------------------------------------------------------


def test_controller_overshoot_and_pin_at_high_eta() -> None:
    r = simulate_coupled_threshold_controller(eta=0.5, posterior_half_life=200.0, seed=0)
    assert r["pinned_at_strict_bound"] is True
    assert r["late_fail_rate_mean"] > 0.5  # far above the 0.35 target


def test_controller_slow_eta_stays_near_target() -> None:
    r = simulate_coupled_threshold_controller(eta=0.05, posterior_half_life=200.0, seed=0)
    assert r["pinned_at_strict_bound"] is False
    assert 0.15 <= r["late_fail_rate_mean"] <= 0.6


def test_controller_scenarios_findings_hold() -> None:
    result = run_controller_scenarios(seeds=(0, 1, 2))
    f = result["findings"]
    assert f["CTRL1_overshoot_and_pin_at_high_eta"] is True
    assert f["CTRL2_long_memory_tracks_better_under_slow_controller"] is True
    assert f["CTRL3_hardened_controller_stays_in_band"] is True
    assert result["is_measurement"] is False
