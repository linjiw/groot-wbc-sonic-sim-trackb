"""Change A (ZPD teacher) unit tests — research_plan_zpd_teacher.md §3.1.

Preregistered test list from the plan: byte-identity when ``signal`` unset
(golden-tensor comparison across a scripted count sequence); learnability closed
form vs Monte Carlo; decay half-life invariance (old evidence retention exactly
invariant to recompute chunking; full posterior approximately invariant when the
half-life dominates the window); optimism interval-max correctness incl. the
peak-inside case; tripwire binding; NaN-guard regression (init=0 path stays
fixed); family-kernel neutrality when null.

Uses the same minimal MotionLibBase-shaped stub pattern as
``test_adaptive_sampling_prob.py`` — CPU-only, no Isaac Lab.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase


def _sampler_stub(
    num_episodes,
    num_failures,
    *,
    cfg=None,
    cap=200.0,
    uniform_rate=0.1,
):
    """Minimal object exposing exactly what the sampling-prob math reads."""
    stub = object.__new__(MotionLibBase)
    n = len(num_episodes)
    stub.use_adaptive_sampling = True
    stub.adaptive_sampling_cfg = dict(cfg or {})
    stub.adp_samp_num_episodes = torch.tensor(num_episodes, dtype=torch.float32)
    stub.adp_samp_num_failures = torch.tensor(num_failures, dtype=torch.float32)
    stub.adp_samp_active_motion_bins = torch.arange(n)
    stub.adp_samp_failure_rate_max_over_mean = cap
    stub.uniform_sampling_rate = uniform_rate
    stub.adp_samp_bin_weights = torch.ones(n)
    stub.max_prob_per_bin_cfg = None
    stub.max_prob_per_motion_cfg = None
    return stub


def _release_reference_prob(num_episodes, num_failures, *, cap=200.0, uniform_rate=0.1):
    """Independent reimplementation of the release prob math (golden reference)."""
    eps = torch.tensor(num_episodes, dtype=torch.float32)
    fails = torch.tensor(num_failures, dtype=torch.float32)
    failure_rate = torch.where(eps > 0, fails / eps, torch.zeros_like(eps)).double()
    upper = failure_rate.mean() * cap
    clipped = torch.clip(failure_rate, 0.0, upper)
    total = clipped.sum()
    if total > 0:
        based = clipped / total
    else:
        based = torch.ones_like(clipped) / len(clipped)
    uniform = torch.ones_like(based) / len(based)
    prob = based * (1 - uniform_rate) + uniform * uniform_rate
    prob = prob * torch.ones(len(based))
    return (prob / prob.sum()).float()


# ---------------------------------------------------------------------------
# Byte-identity when signal unset (guardrail 7)
# ---------------------------------------------------------------------------


def test_release_path_byte_identical_over_scripted_count_sequence() -> None:
    rng = np.random.default_rng(42)
    n = 12
    eps = np.ones(n)
    fails = np.ones(n)
    for _ in range(30):
        eps = eps + rng.integers(0, 5, n) / 50.0
        fails = fails + rng.binomial(3, 0.2, n).astype(float)
        stub = _sampler_stub(list(eps), list(fails))
        stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
        golden = _release_reference_prob(list(eps), list(fails))
        assert torch.equal(stub.adp_sampling_active_prob, golden)


def test_release_path_leaves_zpd_state_untouched() -> None:
    stub = _sampler_stub([5.0, 3.0], [2.0, 1.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    # No decay clock, no posterior/telemetry attributes on the release signal.
    assert getattr(stub, "_adp_samp_eps_at_last_recompute", None) is None
    assert getattr(stub, "adp_samp_posterior_p_mean", None) is None
    assert getattr(stub, "_adp_samp_tripwire_binding", None) is None
    # Checkpoint keys stay exactly the legacy pair.
    assert set(stub.get_state_dict().keys()) == {
        "adp_samp_num_episodes",
        "adp_samp_num_failures",
    }


def test_unknown_signal_rejected() -> None:
    stub = _sampler_stub([1.0], [1.0], cfg={"signal": "banana"})
    with pytest.raises(ValueError, match="signal"):
        stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)


# ---------------------------------------------------------------------------
# Learnability closed form (D1)
# ---------------------------------------------------------------------------


def test_learnability_closed_form_matches_monte_carlo() -> None:
    rng = np.random.default_rng(7)
    for succ, fail in ((0, 0), (3, 1), (10, 40), (200, 5)):
        a, b = 1.0 + succ, 1.0 + fail
        samples = rng.beta(a, b, size=400_000)
        mc = float((samples * (1.0 - samples)).mean())
        closed = a * b / ((a + b) * (a + b + 1.0))
        assert closed == pytest.approx(mc, abs=2e-4)


def test_learnability_prefers_frontier_over_mastered_and_impossible() -> None:
    # Bin 0: mastered (many successes), bin 1: frontier (~50%), bin 2: impossible.
    eps = [101.0, 100.0, 100.0]
    fails = [1.0, 50.0, 100.0]
    stub = _sampler_stub(eps, fails, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert prob[1] > prob[0]
    assert prob[1] > prob[2]
    # Posterior sanity telemetry exposed for the activation gate.
    p = stub.adp_samp_posterior_p_mean
    assert p[0] > p[1] > p[2]


def test_learnability_uses_exact_expectation_not_plug_in_mean() -> None:
    # Jensen: E[p(1-p)] < p_mean(1-p_mean) at finite counts. The wide-posterior
    # bin (1/1 counts, uniform posterior E=1/6) must be distinguishable from the
    # plug-in value 0.25.
    stub = _sampler_stub([1.0, 41.0], [1.0, 21.0], cfg={"signal": "learnability"})
    _, utility = stub._compute_zpd_utility()
    assert utility[0].item() == pytest.approx(1.0 / 6.0, abs=1e-9)
    a, b = 21.0, 21.0  # bin 1: succ=20, fail=20
    assert utility[1].item() == pytest.approx(a * b / ((a + b) * (a + b + 1.0)), abs=1e-9)


# ---------------------------------------------------------------------------
# Evidence-scaled decay (D4)
# ---------------------------------------------------------------------------


def _run_decay_sequence(chunks, *, half_life, eps0, fails0):
    """Feed evidence in the given per-recompute chunks; return final counts."""
    stub = _sampler_stub([eps0], [fails0], cfg={"evidence_half_life": half_life})
    # First recompute initializes the decay clock at the current counts.
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    for delta in chunks:
        stub.adp_samp_num_episodes = stub.adp_samp_num_episodes + delta
        stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    return (
        float(stub.adp_samp_num_episodes[0]),
        float(stub.adp_samp_num_failures[0]),
    )


def test_decay_old_evidence_retention_invariant_to_recompute_chunking() -> None:
    # Failures live only in the pre-existing counts (no new failures arrive), so
    # their retention isolates the old-evidence decay: 0.5**(total_delta/H)
    # exactly, no matter how the 30 episode-equivalents are chunked.
    h = 10.0
    _, fails_one = _run_decay_sequence([30.0], half_life=h, eps0=20.0, fails0=8.0)
    _, fails_ten = _run_decay_sequence([3.0] * 10, half_life=h, eps0=20.0, fails0=8.0)
    expected = 8.0 * 0.5 ** (30.0 / h)
    assert fails_one == pytest.approx(expected, rel=1e-6)
    assert fails_ten == pytest.approx(expected, rel=1e-6)


def test_decay_posterior_approximately_invariant_when_half_life_dominates_window() -> None:
    # Within-window evidence is not decayed at its own recompute, so exact
    # invariance holds only for pre-window counts. With H >> window the full
    # posterior is approximately chunking-invariant (documented semantics).
    h = 100.0
    eps_one, _ = _run_decay_sequence([30.0], half_life=h, eps0=20.0, fails0=8.0)
    eps_ten, _ = _run_decay_sequence([3.0] * 10, half_life=h, eps0=20.0, fails0=8.0)
    assert abs(eps_one - eps_ten) / eps_one < 0.10


def test_decay_disabled_is_noop_and_keeps_counts_cumulative() -> None:
    stub = _sampler_stub([4.0], [2.0], cfg={})
    for _ in range(5):
        stub.adp_samp_num_episodes = stub.adp_samp_num_episodes + 10.0
        stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    assert float(stub.adp_samp_num_episodes[0]) == pytest.approx(54.0)
    assert getattr(stub, "_adp_samp_eps_at_last_recompute", None) is None


def test_decay_state_roundtrips_through_state_dict() -> None:
    stub = _sampler_stub([4.0, 6.0], [2.0, 1.0], cfg={"evidence_half_life": 10.0})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    state = stub.get_state_dict()
    assert "adp_samp_eps_at_last_recompute" in state
    assert "adp_samp_fails_at_last_recompute" in state

    restored = _sampler_stub([0.0, 0.0], [0.0, 0.0], cfg={"evidence_half_life": 10.0})
    restored._device = "cpu"
    restored.load_state_dict(state)
    assert torch.allclose(restored.adp_samp_num_episodes, stub.adp_samp_num_episodes)
    assert torch.allclose(
        restored._adp_samp_eps_at_last_recompute, stub._adp_samp_eps_at_last_recompute
    )


def test_invalid_half_life_rejected() -> None:
    stub = _sampler_stub([1.0], [1.0], cfg={"evidence_half_life": -3})
    with pytest.raises(ValueError, match="evidence_half_life"):
        stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)


# ---------------------------------------------------------------------------
# Deterministic optimism (D3)
# ---------------------------------------------------------------------------


def test_optimism_peak_inside_interval_yields_peak_utility() -> None:
    # succ=fail=4 => a=b=5 => p_mean=0.5, sd>0 => the learnability peak 0.25 is
    # inside [lo, hi].
    stub = _sampler_stub([9.0], [4.0], cfg={"signal": "learnability", "optimism_k": 1.0})
    _, utility = stub._compute_zpd_utility()
    assert utility[0].item() == pytest.approx(0.25, abs=1e-12)


def test_optimism_interval_below_peak_takes_upper_endpoint() -> None:
    # succ=0, fail=19 => a=1, b=20: interval entirely below p*=0.5 => the max of
    # the unimodal utility over the interval is at the upper endpoint hi.
    stub = _sampler_stub([19.0], [19.0], cfg={"signal": "learnability", "optimism_k": 1.0})
    a, b = 1.0, 20.0
    p_mean = a / (a + b)
    sd = (a * b / ((a + b) ** 2 * (a + b + 1.0))) ** 0.5
    hi = min(1.0, p_mean + sd)
    assert hi < 0.5
    _, utility = stub._compute_zpd_utility()
    assert utility[0].item() == pytest.approx(hi * (1.0 - hi), abs=1e-9)


def test_optimism_never_below_posterior_mean_utility_for_learnability() -> None:
    # max over an interval containing p_mean >= u(p_mean) >= E[u] (Jensen,
    # concave u), so optimism-on utility dominates optimism-off per bin.
    rng = np.random.default_rng(3)
    eps = list(1.0 + rng.integers(0, 60, 20).astype(float))
    fails = [min(e, float(rng.integers(0, 30))) for e in eps]
    on = _sampler_stub(eps, fails, cfg={"signal": "learnability", "optimism_k": 1.0})
    off = _sampler_stub(eps, fails, cfg={"signal": "learnability", "optimism_k": 0.0})
    _, u_on = on._compute_zpd_utility()
    _, u_off = off._compute_zpd_utility()
    assert (u_on >= u_off - 1e-12).all()


def test_optimism_widens_with_uncertainty() -> None:
    # Same posterior mean (p=0.2), different confidence: the wide posterior must
    # get at least as much optimism-boosted utility as the tight one.
    wide = _sampler_stub([9.0], [8.0], cfg={"signal": "learnability", "optimism_k": 1.0})
    tight = _sampler_stub([499.0], [400.0], cfg={"signal": "learnability", "optimism_k": 1.0})
    _, u_wide = wide._compute_zpd_utility()
    _, u_tight = tight._compute_zpd_utility()
    assert u_wide[0].item() > u_tight[0].item()


# ---------------------------------------------------------------------------
# advantage_mass ablation arm (M5-A)
# ---------------------------------------------------------------------------


def test_advantage_mass_zero_at_both_ends_peak_in_between() -> None:
    # Mastered (p~1) and impossible (p~0) bins get ~0 utility; frontier wins.
    eps = [1001.0, 100.0, 1000.0]
    fails = [1.0, 50.0, 1000.0]
    stub = _sampler_stub(eps, fails, cfg={"signal": "advantage_mass", "advmass_n": 16})
    _, utility = stub._compute_zpd_utility()
    assert utility[1] > utility[0]
    assert utility[1] > utility[2]


def test_advantage_mass_matches_formula_at_posterior_mean() -> None:
    stub = _sampler_stub([100.0], [40.0], cfg={"signal": "advantage_mass", "advmass_n": 8})
    succ, fail = 60.0, 40.0
    a, b = 1.0 + succ, 1.0 + fail
    p = a / (a + b)
    expected = max((1.0 - (1.0 - p) ** 8) - p, 0.0)
    _, utility = stub._compute_zpd_utility()
    assert utility[0].item() == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Tripwire (D9)
# ---------------------------------------------------------------------------


def test_tripwire_binds_on_pathological_concentration_and_is_recorded() -> None:
    # 99 hyper-mastered bins + 1 frontier bin: unconstrained ZPD prob would put
    # >20x uniform on the frontier bin; the tripwire clamps and records binding.
    n = 100
    eps = [1001.0] * (n - 1) + [100.0]
    fails = [1.0] * (n - 1) + [50.0]
    stub = _sampler_stub(eps, fails, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert stub._adp_samp_tripwire_binding == 1.0
    # Post-clamp renorm can push slightly above the raw ceiling; assert the
    # binding materially reduced concentration below the unconstrained level.
    assert prob.max().item() < 0.30  # unconstrained would be ~0.64
    assert torch.isfinite(prob).all()
    assert prob.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_tripwire_not_binding_on_flat_posterior() -> None:
    stub = _sampler_stub([10.0] * 8, [5.0] * 8, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    assert stub._adp_samp_tripwire_binding == 0.0
    assert torch.allclose(stub.adp_sampling_active_prob, torch.full((8,), 0.125), atol=1e-6)


# ---------------------------------------------------------------------------
# NaN-guard regression (init=0 path stays fixed; ZPD path never NaNs)
# ---------------------------------------------------------------------------


def test_zpd_with_zero_counts_is_uniform_and_finite() -> None:
    # init_num_failures=0 + nothing observed: Beta(1,1) everywhere -> E[p(1-p)]
    # = 1/6 for every bin -> uniform, finite, no 0/0 anywhere.
    stub = _sampler_stub([0.0] * 6, [0.0] * 6, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert torch.isfinite(prob).all()
    assert torch.allclose(prob, torch.full((6,), 1.0 / 6.0), atol=1e-6)


def test_zpd_excess_failures_clamped_not_negative() -> None:
    # Failure counts are not length-normalized, so fails can exceed episode
    # mass; successes clamp at 0 rather than going negative.
    stub = _sampler_stub([1.0], [5.0], cfg={"signal": "learnability"})
    a, b = stub._zpd_posterior_counts()
    assert a[0].item() == pytest.approx(1.0)  # succ clamped to 0
    assert b[0].item() == pytest.approx(6.0)


def test_release_init_zero_nan_guard_still_fixed_with_new_code() -> None:
    # The 2026-07-09 fix must survive Change A: release signal, init 0, unplayed bins.
    stub = _sampler_stub([0.0, 4.0], [0.0, 2.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    assert torch.isfinite(stub.adp_sampling_active_prob).all()


# ---------------------------------------------------------------------------
# Family kernel (D6 remnant)
# ---------------------------------------------------------------------------


def test_family_kernel_null_is_neutral() -> None:
    stub = _sampler_stub([10.0, 20.0], [4.0, 8.0], cfg={"signal": "learnability"})
    a, b = stub._zpd_posterior_counts()
    assert a[0].item() == pytest.approx(1.0 + 6.0)
    assert b[0].item() == pytest.approx(1.0 + 4.0)
    assert a[1].item() == pytest.approx(1.0 + 12.0)
    assert b[1].item() == pytest.approx(1.0 + 8.0)


def _kernel_stub():
    """Stub with 3 motions (walk x1.0 [2 bins], walk x1.5 [2 bins], run x1.0 [1 bin])."""
    stub = _sampler_stub(
        [10.0, 10.0, 10.0, 10.0, 10.0],
        [2.0, 2.0, 6.0, 6.0, 4.0],
        cfg={"signal": "learnability", "family_kernel": {"weights": {1: 0.25}}},
    )
    stub._device = "cpu"
    stub._motion_data_keys = ["walk_x1.0", "walk_x1.5", "run_x1.0"]
    stub.orig_motion_id_to_bins = [
        torch.tensor([0, 1]),
        torch.tensor([2, 3]),
        torch.tensor([4]),
    ]
    stub.adp_samp_family_kernel_cfg = stub.adaptive_sampling_cfg["family_kernel"]
    stub._init_family_kernel()
    return stub


def test_family_kernel_shares_evidence_between_speed_neighbors_only() -> None:
    stub = _kernel_stub()
    a, b = stub._zpd_posterior_counts()
    # walk_x1.0 bin 0 receives 0.25x pseudo-counts from walk_x1.5 bin 0 (phase 0).
    # raw: succ=8, fail=2; neighbor bin 2: succ=4, fail=6.
    assert a[0].item() == pytest.approx(1.0 + 8.0 + 0.25 * 4.0)
    assert b[0].item() == pytest.approx(1.0 + 2.0 + 0.25 * 6.0)
    # run_x1.0 has no family neighbor: untouched.
    assert a[4].item() == pytest.approx(1.0 + 6.0)
    assert b[4].item() == pytest.approx(1.0 + 4.0)


def test_family_kernel_never_mutates_raw_counts() -> None:
    stub = _kernel_stub()
    eps_before = stub.adp_samp_num_episodes.clone()
    fails_before = stub.adp_samp_num_failures.clone()
    stub._zpd_posterior_counts()
    assert torch.equal(stub.adp_samp_num_episodes, eps_before)
    assert torch.equal(stub.adp_samp_num_failures, fails_before)


# ---------------------------------------------------------------------------
# Full-path sanity for the ZPD signal (blend, floor, telemetry)
# ---------------------------------------------------------------------------


def test_zpd_prob_keeps_uniform_floor() -> None:
    n = 10
    eps = [1001.0] * (n - 1) + [100.0]
    fails = [1.0] * (n - 1) + [50.0]
    stub = _sampler_stub(eps, fails, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    # Every bin retains at least ~uniform_rate/n mass (floor survives the swap).
    assert stub.adp_sampling_active_prob.min().item() >= 0.1 / n * 0.9


def test_zpd_telemetry_attributes_populated() -> None:
    stub = _sampler_stub([10.0] * 4, [5.0] * 4, cfg={"signal": "learnability"})
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    assert stub.adp_samp_posterior_p_mean is not None
    assert stub._adp_samp_utility_max_over_uniform is not None
    assert stub._adp_samp_utility_entropy is not None
    assert stub._adp_samp_tripwire_binding is not None
