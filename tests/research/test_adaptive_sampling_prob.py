"""Prob-computation tests for the adaptive sampler's 0/0 guard.

These construct a minimal MotionLibBase-shaped stub (the prob methods reference
only ``adp_samp_*`` attributes plus a few scalars) so the sampling-probability
math can be exercised without instantiating a full motion library / Isaac Lab.
The guard added at sync_and_compute_adaptive_sampling must be byte-identical for
the release config (init_num_failures=1) and only fix the init_num_failures=0
NaN hazard that the SIM-M5a knob would otherwise trigger.
"""

from __future__ import annotations

import torch

from gear_sonic.utils.motion_lib.motion_lib_base import MotionLibBase


def _sampler_stub(num_episodes, num_failures, *, cap=200.0, uniform_rate=0.1):
    """Minimal object exposing exactly what the prob computation reads."""
    stub = object.__new__(MotionLibBase)
    n = len(num_episodes)
    stub.use_adaptive_sampling = True
    stub.adaptive_sampling_cfg = {}
    stub.adp_samp_num_episodes = torch.tensor(num_episodes, dtype=torch.float32)
    stub.adp_samp_num_failures = torch.tensor(num_failures, dtype=torch.float32)
    stub.adp_samp_active_motion_bins = torch.arange(n)
    stub.adp_samp_failure_rate_max_over_mean = cap
    stub.uniform_sampling_rate = uniform_rate
    stub.adp_samp_bin_weights = torch.ones(n)
    stub.max_prob_per_bin_cfg = None
    stub.max_prob_per_motion_cfg = None
    return stub


def test_release_config_uniform_counts_give_uniform_prob() -> None:
    # init_num_failures=1 => every bin 1/1 => failure_rate 1.0 => uniform.
    stub = _sampler_stub([1.0] * 10, [1.0] * 10)
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert torch.allclose(prob, torch.full((10,), 0.1), atol=1e-6)
    assert torch.isfinite(prob).all()


def test_release_config_never_divides_by_zero() -> None:
    # Realistic release state: episodes always >= init (1), some failures accrued.
    stub = _sampler_stub([5.0, 3.0, 8.0, 1.0], [2.0, 1.0, 1.0, 1.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    assert torch.isfinite(stub.adp_sampling_active_prob).all()
    # Guard is a no-op here (all episodes > 0): failure_rate matches raw division.
    expected = stub.adp_samp_num_failures / stub.adp_samp_num_episodes
    assert torch.allclose(stub.adp_samp_failure_rate_raw.float(), expected, atol=1e-6)


def test_init_zero_unplayed_bins_do_not_nan() -> None:
    # SIM-M5a knob init_num_failures=0: a bin never played is 0/0. The guard must
    # yield a finite (zero-signal) failure_rate, not NaN, and a valid distribution.
    stub = _sampler_stub([0.0, 4.0, 0.0, 6.0], [0.0, 2.0, 0.0, 1.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert torch.isfinite(prob).all()
    assert abs(prob.sum().item() - 1.0) < 1e-5
    # Unplayed bins get failure_rate 0.
    assert stub.adp_samp_failure_rate_raw[0].item() == 0.0
    assert stub.adp_samp_failure_rate_raw[2].item() == 0.0


def test_init_zero_all_bins_zero_failures_falls_back_to_uniform() -> None:
    # The REAL SIM-M5a starved regime: init_num_failures=0 and no early termination
    # anywhere, so every failure_rate is legitimately 0. The per-bin guard alone
    # leaves clipped.sum()==0; the normalization guard must fall back to uniform
    # rather than produce NaN probabilities (which would trip the >=0 assertion).
    stub = _sampler_stub([4.0, 3.0, 5.0, 2.0], [0.0, 0.0, 0.0, 0.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert torch.isfinite(prob).all()
    assert torch.allclose(prob, torch.full((4,), 0.25), atol=1e-6)


def test_higher_failure_rate_gets_more_mass() -> None:
    # A bin with a higher failure rate should be sampled more (mechanism sanity).
    stub = _sampler_stub([10.0, 10.0, 10.0], [1.0, 5.0, 1.0])
    stub.sync_and_compute_adaptive_sampling(sync_across_gpus=False)
    prob = stub.adp_sampling_active_prob
    assert prob[1] > prob[0]
    assert prob[1] > prob[2]
