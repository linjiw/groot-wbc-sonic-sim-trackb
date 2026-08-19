"""Pin what SONIC's shipped adaptive sampler actually does with training exposure.

WHY THESE TESTS EXIST
---------------------
``gear_sonic/research/hygiene/sampler_diagnostics.py`` makes one claim that is easy to
get wrong and expensive to get wrong: SONIC already ships the normalised uniform mixture
at ``a = 0.10``, the floor is approximately delivered, and it still does not stop a single
impossible clip from absorbing tens of times its fair share of sampling mass. A uniform
floor bounds probability FROM BELOW; nothing in the release config bounds it from above.

Every probability below comes out of the real
``MotionLibBase.sync_and_compute_adaptive_sampling`` (the diagnostics module only stubs the
attributes it reads), so these tests are assertions about the shipped sampler, not about a
re-implementation of it. The shipped fixture
``tests/research/test_adaptive_sampling_prob.py:18-32`` stubs ``adp_samp_bin_weights`` to
ones; the length-derived-weight regime exercised here is the one training actually runs and
the one that was previously untested.

The motion bank is a deterministic lognormal QUANTILE GRID -- no RNG, no seed. It lands
within 1% of the numbers measured on a randomly sampled 800-clip bank, which is itself the
point: the concentration result is a property of realistic clip-length spread, not of a seed.

CPU only. No Isaac Lab, no GPU.
"""

from __future__ import annotations

import json
import math
from statistics import NormalDist

import numpy as np
import pytest
import torch

from gear_sonic.research.hygiene import sampler_diagnostics as sd

BANK_SIZE = 800
FAIR_SHARE = 1.0 / BANK_SIZE
BASELINE_FAILURE_RATE = 0.02


def _realistic_lengths_frames(num_motions: int = BANK_SIZE) -> np.ndarray:
    """A retargeted-mocap length mix: mostly 2-10 s, a tail to 60 s, at the 50 Hz target fps.

    Deterministic by construction (the lognormal inverse CDF on a fixed quantile grid), so
    the pinned numbers below cannot drift with a torch/numpy RNG change.
    """
    quantiles = (np.arange(num_motions) + 0.5) / num_motions
    normal = np.array([NormalDist().inv_cdf(q) for q in quantiles])
    seconds = np.clip(np.exp(math.log(6.0) + 0.9 * normal), 1.5, 60.0)
    return (seconds * 50).astype(np.int64)


@pytest.fixture(scope="module")
def bank() -> sd.BinLayout:
    return sd.build_bins(_realistic_lengths_frames())


@pytest.fixture(scope="module")
def bank_keys() -> list[str]:
    return [f"clip_{index:04d}" for index in range(BANK_SIZE)]


def _one_impossible_clip(layout: sd.BinLayout, motion_id: int) -> np.ndarray:
    """Per-bin failure rates for a converged bank with exactly one clip that always fails."""
    rates = np.full(layout.num_bins, BASELINE_FAILURE_RATE)
    rates[layout.motion_index == motion_id] = 1.0
    return rates


# ---------------------------------------------------------------------------
# build_bins -- motion_lib_base.py:2400-2451
# ---------------------------------------------------------------------------


def test_build_bins_reproduces_shipped_bin_edges() -> None:
    # 120 frames at bin_size 50 -> [0,50), [50,100), [100,120): the last bin is a short
    # remainder bin (motion_lib_base.py:2404-2405 clamps bin_ends at num_frames).
    layout = sd.build_bins([120, 100], bin_size=50)
    assert layout.num_bins == 5
    assert layout.bin_starts.tolist() == [0, 50, 100, 0, 50]
    assert layout.bin_ends.tolist() == [50, 100, 120, 50, 100]
    assert layout.motion_index.tolist() == [0, 0, 0, 1, 1]
    assert layout.num_peer_bins.tolist() == [3, 3, 3, 2, 2]


def test_build_bins_weight_formula_matches_motion_lib() -> None:
    # :2447-2451 -- w = (bin_length / mean(bin_length)) / num_peer_bins.
    lengths = np.array([120, 100, 275], dtype=np.int64)
    layout = sd.build_bins(lengths, bin_size=50)
    bin_lengths = (layout.bin_ends - layout.bin_starts).astype(np.float64)
    expected = bin_lengths / np.float64(np.float32(bin_lengths.mean())) / layout.num_peer_bins
    assert np.allclose(layout.bin_weights, expected, rtol=1e-6)


def test_sequence_length_agnostic_equalises_per_motion_weight() -> None:
    # The shipped comment at :2450 says "this will make sure each sequence is sampled
    # equally": with the peer-bin divide, every clip that is an exact multiple of bin_size
    # carries the same total bin weight regardless of how long it is.
    layout = sd.build_bins([50, 500, 3000], bin_size=50)
    sums = layout.motion_weight_sums()
    assert np.allclose(sums, sums[0], rtol=1e-6)
    # Without it, weight is proportional to clip length -- long clips dominate.
    raw = sd.build_bins([50, 500, 3000], bin_size=50, sequence_length_agnostic=False)
    raw_sums = raw.motion_weight_sums()
    assert raw_sums[2] / raw_sums[0] == pytest.approx(60.0, rel=1e-6)


def test_bin_weight_scales_as_one_over_bin_count(bank: sd.BinLayout) -> None:
    # w ~ 1/num_peer_bins is what makes a SHORT clip's single bin heavy. On the realistic
    # mix that is a 450x spread between the heaviest and lightest bin.
    spread = float(bank.bin_weights.max() / bank.bin_weights.min())
    assert spread == pytest.approx(450.0, rel=0.02)


# ---------------------------------------------------------------------------
# The release prior: t = 0, before any episode is played
# ---------------------------------------------------------------------------


def test_release_prior_uniform_counts_give_exactly_uniform_prob() -> None:
    # Matches tests/research/test_adaptive_sampling_prob.py::
    # test_release_config_uniform_counts_give_uniform_prob, but reached through
    # build_bins: 10 clips of exactly one bin each => every bin weight is exactly 1.0,
    # init_num_failures=1 => every failure rate is 1.0 => the distribution is uniform.
    layout = sd.build_bins([50] * 10, bin_size=50)
    assert np.array_equal(layout.bin_weights, np.ones(10))
    prob = sd.sampling_distribution(layout)
    assert np.allclose(prob, 0.1, atol=1e-6)
    assert np.isfinite(prob).all()


def test_release_prior_uniform_mass_is_exactly_nominal(bank: sd.BinLayout) -> None:
    # At the prior the failure-based component IS the uniform component, so the blend is a
    # no-op and the realized uniform share is analytically exactly `a`. The decomposition
    # must report that as degenerate rather than fitting noise.
    decomposition = sd.uniform_mass_decomposition(bank)
    assert decomposition.degenerate is True
    assert decomposition.realized_mass == pytest.approx(0.1, abs=1e-12)
    assert decomposition.residual_l1 == pytest.approx(0.0, abs=1e-12)


def test_release_prior_distribution_is_bin_weights_not_uniform(bank: sd.BinLayout) -> None:
    # A caution against reading the prior as "uniform over bins": the peer-bin divide aims
    # it at uniform over MOTIONS, so it is very far from uniform over BINS, whose weights
    # span 450x on this length mix.
    prob = sd.sampling_distribution(bank)
    exposure = sd.per_motion_exposure(prob, bank.motion_index)
    assert exposure.ratio.max() < 1.2
    assert sd.normalized_shannon_entropy(prob) < 0.98
    assert sd.effective_num_bins(prob) == pytest.approx(4166.0, rel=0.02)


def test_remainder_bin_breaks_each_sequence_sampled_equally() -> None:
    # CORRECTION TO THE FOLK STATEMENT. motion_lib_base.py:2450 says the peer-bin divide
    # "will make sure each sequence is sampled equally". It does so only for clips whose
    # frame count is an exact multiple of bin_size. A clip of 101 frames gets bins of
    # 50/50/1, so its per-motion weight sum is 101/3 against 100/2 for a 100-frame clip --
    # a third less exposure for one extra frame of mocap.
    layout = sd.build_bins([100, 101], bin_size=50)
    sums = layout.motion_weight_sums()
    assert sums[1] / sums[0] == pytest.approx((101 / 3) / (100 / 2), rel=1e-6)
    assert sums[1] / sums[0] == pytest.approx(0.673, abs=0.002)


def test_release_prior_is_not_exactly_fair_across_motions(bank: sd.BinLayout) -> None:
    # The same effect at bank scale, BEFORE any adaptive signal exists: on a realistic
    # length mix only 18 of 800 clips are exact multiples of bin_size, so the t=0 prior
    # already spans 1.49x between the least- and most-exposed clip (0.74x to 1.10x fair
    # share). The floor is delivered; "equal per sequence" is delivered only to within ~1.5x.
    prob = sd.sampling_distribution(bank)
    exposure = sd.per_motion_exposure(prob, bank.motion_index)
    exact_multiples = int((bank.motion_lengths_frames % bank.bin_size == 0).sum())
    assert exact_multiples == 18
    assert exposure.ratio.min() == pytest.approx(0.740, abs=0.005)
    assert exposure.ratio.max() == pytest.approx(1.099, abs=0.005)
    assert exposure.ratio.max() / exposure.ratio.min() == pytest.approx(1.485, abs=0.01)
    # It is a bias, not noise: the under-exposed clips are the short ones.
    assert exposure.share[:80].sum() < exposure.share[-80:].sum()


# ---------------------------------------------------------------------------
# Does the nominal a = 0.10 floor survive the bin-weight multiply?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("which", "expected_uniform_mass"),
    [("shortest", 0.0968), ("longest", 0.1266)],
)
def test_realized_uniform_mass_stays_near_nominal(
    bank: sd.BinLayout, which: str, expected_uniform_mass: float
) -> None:
    # THE FLOOR IS APPROXIMATELY DELIVERED. Whether the impossible clip is the shortest
    # (highest bin weight) or the longest (lowest bin weight) in the bank, the share of
    # post-renormalisation mass traceable to the uniform component stays within ~27% of the
    # nominal 0.10 -- i.e. "add a uniform floor" is not an available intervention, because
    # the floor is already there and already working.
    target = int(
        np.argmin(bank.motion_lengths_frames)
        if which == "shortest"
        else np.argmax(bank.motion_lengths_frames)
    )
    decomposition = sd.uniform_mass_decomposition(
        bank, failure_rates=_one_impossible_clip(bank, target)
    )
    assert decomposition.degenerate is False
    assert decomposition.caps_active is False
    # The two-component identity is exact when no cap is active.
    assert decomposition.residual_l1 < 1e-6
    assert decomposition.realized_mass == pytest.approx(expected_uniform_mass, abs=2e-3)
    assert 0.09 < decomposition.realized_mass < 0.13


def test_realized_uniform_mass_tracks_the_nominal_knob(bank: sd.BinLayout) -> None:
    # Sanity that the decomposition measures the knob and not an artefact: raising
    # uniform_sampling_rate must raise the realized share monotonically, and hit 1.0 at a=1.
    rates = _one_impossible_clip(bank, int(np.argmax(bank.motion_lengths_frames)))
    realized = [
        sd.realized_uniform_mass(bank, failure_rates=rates, config=sd.SamplerConfig(uniform_rate=a))
        for a in (0.0, 0.1, 0.5, 1.0)
    ]
    assert realized[0] == pytest.approx(0.0, abs=1e-9)
    assert realized[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(b > a for a, b in zip(realized, realized[1:]))


# ---------------------------------------------------------------------------
# ... and yet one impossible clip still eats the bank
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("which", "expected_ratio"),
    [("shortest", 35.5), ("longest", 45.1)],
)
def test_single_impossible_clip_exceeds_30x_fair_share(
    bank: sd.BinLayout, bank_keys: list[str], which: str, expected_ratio: float
) -> None:
    # THE HEADLINE. With the SHIPPED config (a=0.1, cap=200, both max_prob_* = None), one
    # clip that fails every episode takes 4.4-5.6% of ALL sampling mass in an 800-clip bank
    # -- 35-45x its fair share. The uniform floor is intact (previous test) and irrelevant
    # here, because a floor is a lower bound.
    target = int(
        np.argmin(bank.motion_lengths_frames)
        if which == "shortest"
        else np.argmax(bank.motion_lengths_frames)
    )
    prob = sd.sampling_distribution(bank, failure_rates=_one_impossible_clip(bank, target))
    exposure = sd.per_motion_exposure(prob, bank.motion_index, motion_keys=bank_keys)
    assert exposure.top1_index == target
    assert exposure.top1_ratio > 30.0
    assert exposure.top1_ratio == pytest.approx(expected_ratio, rel=0.02)
    assert 0.04 < exposure.top1_share < 0.06
    assert exposure.share_for(bank_keys[target]) == pytest.approx(exposure.top1_share)


def test_per_motion_exposure_sums_to_one_and_covers_silent_motions() -> None:
    # Motions with no bins in the active subset must appear with share 0 rather than be
    # dropped -- otherwise fair share is computed against the wrong denominator.
    layout = sd.build_bins([50, 50, 50, 50])
    prob = sd.sampling_distribution(layout, active_bins=[0, 1, 2])
    exposure = sd.per_motion_exposure(
        prob, layout.motion_index[[0, 1, 2]], num_motions=layout.num_motions
    )
    assert exposure.share.sum() == pytest.approx(1.0)
    assert exposure.share[3] == 0.0
    assert exposure.fair_share == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# The knobs that exist in code but are None in every shipped yaml
# ---------------------------------------------------------------------------


def test_max_prob_per_motion_5x_fair_brings_the_clip_under_6x(bank: sd.BinLayout) -> None:
    # motion_lib_base.py:3318-3346. An EXPLICIT small multiple of fair share is the only
    # setting that binds, and it lands almost exactly where it is aimed (the small overshoot
    # is the renormalisation that follows the clamp at :3344-3347).
    target = int(np.argmax(bank.motion_lengths_frames))
    rates = _one_impossible_clip(bank, target)
    config = sd.SamplerConfig(max_prob_per_motion=5.0 * FAIR_SHARE)
    prob = sd.sampling_distribution(bank, failure_rates=rates, config=config)
    exposure = sd.per_motion_exposure(prob, bank.motion_index)
    assert exposure.ratio[target] < 6.0
    assert exposure.ratio[target] == pytest.approx(5.26, rel=0.02)

    tight = sd.SamplerConfig(max_prob_per_motion=2.0 * FAIR_SHARE)
    tight_prob = sd.sampling_distribution(bank, failure_rates=rates, config=tight)
    tight_exposure = sd.per_motion_exposure(tight_prob, bank.motion_index)
    assert tight_exposure.ratio[target] == pytest.approx(2.11, rel=0.03)


@pytest.mark.parametrize(
    "config",
    [
        sd.SamplerConfig(),
        sd.SamplerConfig(max_prob_per_motion="auto"),
        sd.SamplerConfig(max_prob_per_bin="auto"),
        sd.SamplerConfig(max_prob_per_bin="auto", max_prob_per_motion="auto"),
    ],
    ids=["shipped_none", "motion_auto", "bin_auto", "both_auto"],
)
def test_auto_concentration_caps_are_inert(bank: sd.BinLayout, config: sd.SamplerConfig) -> None:
    # "auto" resolves to failure_rate_cap / num_active_{bins,motions} (:3302-3305, :3321-3325).
    # At the release cap of 200 that is 200/800 = 0.25 per motion -- five times the mass the
    # worst clip takes. So the only two settings shipped or suggested by the code, None and
    # "auto", both leave the 45x concentration completely untouched.
    target = int(np.argmax(bank.motion_lengths_frames))
    rates = _one_impossible_clip(bank, target)
    baseline = sd.sampling_distribution(bank, failure_rates=rates)
    prob = sd.sampling_distribution(bank, failure_rates=rates, config=config)
    assert np.abs(prob - baseline).max() < 1e-9
    exposure = sd.per_motion_exposure(prob, bank.motion_index)
    assert exposure.ratio[target] > 30.0


def test_uniform_mass_decomposition_flags_an_active_cap(bank: sd.BinLayout) -> None:
    # When a cap actually fires, the two-component identity no longer holds and the reported
    # "uniform mass" is only a least-squares projection. The dataclass must say so.
    rates = _one_impossible_clip(bank, int(np.argmax(bank.motion_lengths_frames)))
    inert = sd.uniform_mass_decomposition(
        bank, failure_rates=rates, config=sd.SamplerConfig(max_prob_per_motion="auto")
    )
    assert inert.caps_active is False
    active = sd.uniform_mass_decomposition(
        bank, failure_rates=rates, config=sd.SamplerConfig(max_prob_per_motion=2.0 * FAIR_SHARE)
    )
    assert active.caps_active is True


# ---------------------------------------------------------------------------
# The 200x-mean heavy-tail clip
# ---------------------------------------------------------------------------


def test_failure_rate_cap_is_inert_at_realistic_mean_failure_rate(bank: sd.BinLayout) -> None:
    # motion_lib_base.py:3216-3219 clips failure rates at mean * 200. Failure rates live in
    # [0, 1], so the bound only bites when the mean is below 1/200 = 0.005. At a mean of
    # ~0.02 the bound is 4.05 -- above every attainable value -- so nothing is clipped and
    # the distribution is bit-identical to an uncapped sampler. This is the mechanism that
    # was supposed to stop the heavy tail, and it is switched off by its own arithmetic.
    rates = _one_impossible_clip(bank, int(np.argmax(bank.motion_lengths_frames)))
    diagnostic = sd.failure_rate_cap_diagnostic(bank, failure_rates=rates)
    assert diagnostic.cap == 200.0
    assert diagnostic.mean_failure_rate == pytest.approx(0.0274, abs=2e-3)
    assert diagnostic.mean_failure_rate_for_binding == pytest.approx(0.005)
    assert diagnostic.upper_bound > 1.0
    assert diagnostic.num_bins_clipped == 0
    assert diagnostic.binds is False
    assert diagnostic.max_abs_prob_delta_vs_uncapped == 0.0


def test_failure_rate_cap_only_binds_on_an_almost_solved_bank() -> None:
    # The complementary case, so the previous test is a measurement and not a tautology:
    # 1000 bins whose mean failure rate is 0.002 (< 0.005). Two bins sit above the bound and
    # are flattened onto it, which does change the distribution.
    layout = sd.build_bins([50] * 1000, bin_size=50)
    rates = np.full(1000, 0.0005)
    rates[0] = 1.0
    rates[1] = 0.5
    diagnostic = sd.failure_rate_cap_diagnostic(layout, failure_rates=rates)
    assert diagnostic.mean_failure_rate < 0.005
    assert diagnostic.upper_bound < 1.0
    assert diagnostic.num_bins_clipped == 2
    assert diagnostic.binds is True
    assert diagnostic.max_abs_prob_delta_vs_uncapped > 1e-3


def test_clipping_a_lone_outlier_fires_but_changes_nothing() -> None:
    # Why `binds` is measured against the uncapped distribution rather than reported as
    # num_bins_clipped > 0: with a single positive-signal bin, the clip scales the only
    # nonzero entry and the renormalisation immediately undoes it.
    layout = sd.build_bins([50] * 1000, bin_size=50)
    rates = np.zeros(1000)
    rates[0] = 1.0
    diagnostic = sd.failure_rate_cap_diagnostic(layout, failure_rates=rates)
    assert diagnostic.num_bins_clipped == 1
    assert diagnostic.binds is False


# ---------------------------------------------------------------------------
# Shannon is not Renyi-2
# ---------------------------------------------------------------------------


def test_shannon_and_effective_num_bins_agree_only_on_the_uniform() -> None:
    uniform = np.full(512, 1.0 / 512)
    assert sd.normalized_shannon_entropy(uniform) == pytest.approx(1.0)
    assert sd.effective_num_bins(uniform) == pytest.approx(512.0)
    # exp(H_shannon) == effective_num_bins == N, only here.
    assert math.exp(sd.normalized_shannon_entropy(uniform) * math.log(512)) == pytest.approx(
        sd.effective_num_bins(uniform), rel=1e-9
    )


def test_shannon_and_renyi2_disagree_badly_on_a_skewed_distribution(bank: sd.BinLayout) -> None:
    # THE ONE-CLIP COLLAPSE, read two ways. SONIC logs `adp_samp/effective_num_bins`
    # (manager_env_wrapper.py:1023) which is 1/sum(p^2) = exp(Renyi-2), NOT exp(Shannon).
    # On the collapsed distribution the Shannon reading looks nearly healthy (0.94 of max,
    # ~4.6k "effective" bins) while the Renyi-2 reading -- the one that is actually logged --
    # says ~760 of 7499. Quoting them side by side as if interchangeable would understate
    # the collapse by a factor of six.
    rates = _one_impossible_clip(bank, int(np.argmin(bank.motion_lengths_frames)))
    prob = sd.sampling_distribution(bank, failure_rates=rates)
    shannon = sd.normalized_shannon_entropy(prob)
    renyi2 = sd.effective_num_bins(prob)
    exp_shannon = math.exp(shannon * math.log(prob.size))
    assert shannon == pytest.approx(0.945, abs=0.01)
    assert renyi2 == pytest.approx(761.0, rel=0.02)
    assert exp_shannon == pytest.approx(4590.0, rel=0.03)
    assert renyi2 < exp_shannon / 5.0


def test_entropy_metrics_reject_degenerate_input() -> None:
    with pytest.raises(ValueError):
        sd.normalized_shannon_entropy([0.0, 0.0])
    with pytest.raises(ValueError):
        sd.effective_num_bins([1.0, -0.5])
    assert sd.normalized_shannon_entropy([3.0]) == 0.0


# ---------------------------------------------------------------------------
# The ledger: screen verdict -> training cost
# ---------------------------------------------------------------------------


def test_wasted_exposure_measures_mass_spent_on_flagged_clips(
    bank: sd.BinLayout, bank_keys: list[str]
) -> None:
    target = int(np.argmax(bank.motion_lengths_frames))
    prob = sd.sampling_distribution(bank, failure_rates=_one_impossible_clip(bank, target))
    exposure = sd.per_motion_exposure(prob, bank.motion_index, motion_keys=bank_keys)
    ledger = sd.wasted_exposure(exposure, [bank_keys[target]])
    # 0.125% of the bank, 5.6% of the exposure: a 45x concentration of wasted compute.
    assert ledger.flagged_motion_count == 1
    assert ledger.flagged_bank_fraction == pytest.approx(FAIR_SHARE)
    assert ledger.wasted_fraction == pytest.approx(0.0564, rel=0.02)
    assert ledger.concentration_ratio == pytest.approx(45.1, rel=0.02)


def test_wasted_exposure_is_the_bank_fraction_when_nothing_concentrates(
    bank: sd.BinLayout, bank_keys: list[str]
) -> None:
    # Control: at the release prior there is no adaptive signal at all, so wasted mass is
    # (nearly) the flagged share of the bank and concentration_ratio is (nearly) 1. Anything
    # well above 1 in the real ledger is exposure the sampler actively pulled toward
    # infeasible clips.
    #
    # "Nearly", not "exactly", because of the remainder-bin bias measured in
    # test_release_prior_is_not_exactly_fair_across_motions: the 80 SHORTEST clips carry
    # 8.65% of the mass, not 10%, and the 80 longest carry 10.8%. Which end of the length
    # distribution the screen flags therefore shifts the baseline by ~15% before any
    # adaptive concentration is involved -- a confound worth stating whenever a wasted-
    # exposure number is quoted against a length-correlated failure mode.
    prob = sd.sampling_distribution(bank)
    exposure = sd.per_motion_exposure(prob, bank.motion_index, motion_keys=bank_keys)
    shortest = sd.wasted_exposure(exposure, bank_keys[:80])
    longest = sd.wasted_exposure(exposure, bank_keys[-80:])
    assert shortest.flagged_bank_fraction == pytest.approx(0.1)
    assert shortest.wasted_fraction == pytest.approx(0.0865, abs=2e-3)
    assert shortest.concentration_ratio == pytest.approx(0.865, abs=0.02)
    assert longest.wasted_fraction == pytest.approx(0.1081, abs=2e-3)
    # A length-blind flag set recovers concentration_ratio == 1 exactly.
    every_tenth = sd.wasted_exposure(exposure, bank_keys[::10])
    assert every_tenth.concentration_ratio == pytest.approx(1.0, abs=0.02)


def test_wasted_exposure_refuses_unknown_keys_by_default(
    bank: sd.BinLayout, bank_keys: list[str]
) -> None:
    # A typo'd key would silently UNDERSTATE waste, the one direction this ledger must not
    # fail in, so strict is the default.
    prob = sd.sampling_distribution(bank)
    exposure = sd.per_motion_exposure(prob, bank.motion_index, motion_keys=bank_keys)
    with pytest.raises(ValueError, match="not in the bank"):
        sd.wasted_exposure(exposure, [bank_keys[0], "clip_not_in_bank"])
    lenient = sd.wasted_exposure(exposure, [bank_keys[0], "clip_not_in_bank"], strict=False)
    assert lenient.unknown_keys == ("clip_not_in_bank",)
    assert lenient.flagged_motion_count == 1


def test_wasted_exposure_needs_keys() -> None:
    layout = sd.build_bins([50, 50])
    exposure = sd.per_motion_exposure(sd.sampling_distribution(layout), layout.motion_index)
    with pytest.raises(ValueError, match="motion_keys"):
        sd.wasted_exposure(exposure, ["anything"])


# ---------------------------------------------------------------------------
# Checkpoint state -- reuse of scripts/research/dump_sampler_checkpoint_state.py
# ---------------------------------------------------------------------------


def _checkpoint_state() -> dict:
    return {
        "checkpoint_path": "runs/hygiene_a/last.pt",
        "checkpoint_sha256": "b" * 64,
        "checkpoint_size_bytes": 4096,
        "global_step": 1000,
        "adp_samp_num_episodes": [10.0, 10.0, 10.0, 10.0],
        "adp_samp_num_failures": [10.0, 1.0, 1.0, 1.0],
        "adp_samp_bin_motion_ids": [0, 0, 1, 1],
        "adp_samp_motion_data_keys": ["broken_clip", "fine_clip"],
        "adp_samp_bin_weights": [0.5, 0.5, 0.5, 0.5],
        "adp_samp_bin_draw_counts": [600, 200, 100, 100],
    }


def test_checkpoint_summary_aggregates_bins_to_motions() -> None:
    state = sd.summarize_checkpoint_sampler_state(_checkpoint_state())
    assert state.adaptive_state_present is True
    assert state.num_bins == 4
    assert state.num_motions == 2
    assert state.realized_bin_share is not None
    assert np.allclose(state.realized_bin_share, [0.6, 0.2, 0.1, 0.1])
    assert np.allclose(state.realized_motion_share, [0.8, 0.2])
    exposure = state.realized_motion_exposure()
    assert exposure.top1_key == "broken_clip"
    assert exposure.top1_ratio == pytest.approx(1.6)


def test_checkpoint_summary_carries_the_window_shift_caveat_verbatim() -> None:
    # The single most misreadable number in the dump: realized mass is the SELECTED TARGET
    # bin, before pre_failure_sample_window moves the executed start earlier. Carried
    # through byte-for-byte rather than paraphrased.
    state = sd.summarize_checkpoint_sampler_state(_checkpoint_state())
    assert sd.SAMPLING_MASS_CAVEAT in state.caveats
    assert state.sampling_mass_semantics == "selected_target_bin_pre_window_shift"
    assert any("use_failure_rate_decay=false" in caveat for caveat in state.caveats)
    assert "not executed-start-bin mass" in sd.SAMPLING_MASS_CAVEAT


def test_checkpoint_summary_delegates_to_the_dump_script() -> None:
    # Not a copy: the derived quantities must be the dump script's own, so the two artifacts
    # can never disagree about what a checkpoint says.
    from scripts.research.dump_sampler_checkpoint_state import build_sampler_state_summary

    raw = build_sampler_state_summary(_checkpoint_state())
    state = sd.summarize_checkpoint_sampler_state(_checkpoint_state())
    assert np.allclose(
        state.recomputed_bin_prob_unweighted,
        [row["recomputed_prob_unweighted"] for row in raw["bins"]],
    )
    assert np.allclose(state.failure_rate, [row["failure_rate"] for row in raw["bins"]])
    assert sd.SAMPLING_MASS_CAVEAT in raw["caveats"]


def test_checkpoint_summary_handles_a_checkpoint_without_sampler_state() -> None:
    state = sd.summarize_checkpoint_sampler_state({"checkpoint_path": "x.pt"})
    assert state.adaptive_state_present is False
    assert state.num_bins == 0
    with pytest.raises(ValueError, match="no bin draw counts"):
        state.realized_motion_exposure()


def test_read_checkpoint_sampler_state_round_trips_a_file(tmp_path) -> None:
    checkpoint = {
        "env_state_dict": {
            "motion_lib": {
                "adp_samp_num_episodes": torch.tensor([10.0, 10.0, 10.0, 10.0]),
                "adp_samp_num_failures": torch.tensor([10.0, 1.0, 1.0, 1.0]),
                "adp_samp_bin_motion_ids": torch.tensor([0, 0, 1, 1]),
                "adp_samp_motion_data_keys": ["broken_clip", "fine_clip"],
                "adp_samp_bin_draw_counts": torch.tensor([600, 200, 100, 100]),
            }
        }
    }
    path = tmp_path / "last.pt"
    torch.save(checkpoint, path)
    state = sd.read_checkpoint_sampler_state(path)
    assert state.num_bins == 4
    assert state.motion_keys == ("broken_clip", "fine_clip")
    assert np.allclose(state.realized_motion_share, [0.8, 0.2])
    assert state.checkpoint_sha256 is not None and len(state.checkpoint_sha256) == 64
    assert sd.SAMPLING_MASS_CAVEAT in state.caveats


# ---------------------------------------------------------------------------
# One row per configuration
# ---------------------------------------------------------------------------


def test_concentration_report_carries_every_metric_and_serialises(
    bank: sd.BinLayout, bank_keys: list[str]
) -> None:
    target = int(np.argmax(bank.motion_lengths_frames))
    report = sd.concentration_report(
        bank,
        failure_rates=_one_impossible_clip(bank, target),
        motion_keys=bank_keys,
        infeasible_keys=[bank_keys[target]],
        label="shipped_release",
    )
    payload = report.to_dict()
    # Must survive json.dumps: the analysis script tabulates these rows.
    json.dumps(payload)
    assert payload["label"] == "shipped_release"
    assert payload["schema_version"] == sd.SAMPLER_DIAGNOSTICS_SCHEMA_VERSION
    assert payload["num_motions"] == BANK_SIZE
    assert payload["realized_uniform_mass"] == pytest.approx(0.1266, abs=2e-3)
    assert payload["exposure"]["top1_share_over_fair"] == pytest.approx(45.1, rel=0.02)
    assert payload["exposure"]["top1_motion_key"] == bank_keys[target]
    assert payload["failure_rate_cap"]["binds"] is False
    assert payload["wasted_exposure"]["concentration_ratio"] == pytest.approx(45.1, rel=0.02)
    # Shannon and Renyi-2 are both present and are labelled distinctly.
    assert payload["normalized_shannon_entropy"] != payload["effective_num_bins_frac"]
    assert payload["effective_num_bins"] == pytest.approx(3740.0, rel=0.02)
    assert payload["bin_prob_max_over_uniform"] > 1.0
    assert len(payload["exposure"]["top_k"]) == 10


def test_concentration_report_rows_are_comparable_across_configs(
    bank: sd.BinLayout, bank_keys: list[str]
) -> None:
    # The intended use: one row per intervention, tabulated. The capped row must show the
    # collapse gone while the uniform floor is unchanged.
    target = int(np.argmax(bank.motion_lengths_frames))
    rates = _one_impossible_clip(bank, target)
    rows = {
        label: sd.concentration_report(
            bank,
            failure_rates=rates,
            config=config,
            motion_keys=bank_keys,
            infeasible_keys=[bank_keys[target]],
            label=label,
        )
        for label, config in (
            ("shipped", sd.SamplerConfig()),
            ("cap_5x", sd.SamplerConfig(max_prob_per_motion=5.0 * FAIR_SHARE)),
        )
    }
    assert rows["shipped"].wasted.wasted_fraction > 8.0 * rows["cap_5x"].wasted.wasted_fraction
    assert rows["cap_5x"].exposure.top1_ratio < 6.0
    assert rows["cap_5x"].effective_num_bins > rows["shipped"].effective_num_bins


def test_sampler_config_validates_its_knobs() -> None:
    with pytest.raises(ValueError, match="uniform_rate"):
        sd.SamplerConfig(uniform_rate=1.5).validate()
    with pytest.raises(ValueError, match="failure_rate_cap"):
        sd.SamplerConfig(failure_rate_cap=0.0).validate()
    with pytest.raises(ValueError, match="max_prob_per_bin"):
        sd.SamplerConfig(max_prob_per_bin="sometimes").validate()
    with pytest.raises(ValueError, match="unknown SamplerConfig"):
        sd.SamplerConfig().replace(nope=1)


def test_counts_and_rates_are_two_views_of_the_same_state(bank: sd.BinLayout) -> None:
    # The sampler only ever reads failures/episodes (:2952-2956), so specifying rates and
    # specifying counts must produce identical probabilities.
    rates = _one_impossible_clip(bank, 3)
    failures, episodes = sd.counts_from_failure_rates(rates, num_episodes=7.0)
    from_rates = sd.sampling_distribution(bank, failure_rates=rates)
    from_counts = sd.sampling_distribution(bank, num_failures=failures, num_episodes=episodes)
    assert np.allclose(from_rates, from_counts, atol=1e-9)
