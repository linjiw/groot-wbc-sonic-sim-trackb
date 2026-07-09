from __future__ import annotations

import pytest

from scripts.research.dump_sampler_checkpoint_state import build_sampler_state_summary


def _state(episodes: list[float], failures: list[float]) -> dict:
    return {
        "checkpoint_path": "runs/adaptive_seed0/last.pt",
        "global_step": 50,
        "adp_samp_num_episodes": episodes,
        "adp_samp_num_failures": failures,
    }


def test_prior_dominated_state_is_classified_and_stays_near_uniform() -> None:
    # 4 bins, all still at the init prior (1 episode / 1 failure): failure rate 1.0
    # everywhere, zero observed failures -> fully prior-dominated, uniform-ish prob.
    summary = build_sampler_state_summary(_state([1.0] * 4, [1.0] * 4))

    assert summary["adaptive_state_present"] is True
    assert summary["num_bins"] == 4
    aggregates = summary["aggregates"]
    assert aggregates["observed_failures_total"] == 0.0
    assert aggregates["prior_dominated_bin_count"] == 4
    assert aggregates["prior_dominated_fraction"] == 1.0
    assert aggregates["prob_max_over_uniform_unweighted"] == pytest.approx(1.0)
    assert aggregates["effective_num_bins_unweighted"] == pytest.approx(4.0)
    assert aggregates["num_concentrated_bins_unweighted"] == 0


def test_discriminative_failures_concentrate_probability() -> None:
    # Bin 0 accumulated real failures; the others only timed out.
    episodes = [11.0, 11.0, 11.0, 11.0]
    failures = [11.0, 1.0, 1.0, 1.0]

    summary = build_sampler_state_summary(_state(episodes, failures))

    bins = summary["bins"]
    assert bins[0]["observed_failures"] == 10.0
    assert bins[0]["prior_dominated"] is False
    assert bins[1]["prior_dominated"] is True
    assert bins[0]["failure_rate"] == pytest.approx(1.0)
    assert bins[1]["failure_rate"] == pytest.approx(1.0 / 11.0)
    aggregates = summary["aggregates"]
    assert aggregates["prior_dominated_fraction"] == pytest.approx(0.75)
    assert aggregates["prob_max_over_uniform_unweighted"] > 2.0
    assert bins[0]["recomputed_prob_unweighted"] > bins[1]["recomputed_prob_unweighted"]


def test_recompute_respects_failure_rate_cap() -> None:
    # One extreme bin: without the cap it would take ~everything; with cap = 2x mean
    # its clipped rate is bounded.
    episodes = [100.0, 100.0]
    failures = [100.0, 1.0]

    capped = build_sampler_state_summary(_state(episodes, failures), failure_rate_cap=2.0)
    uncapped = build_sampler_state_summary(_state(episodes, failures), failure_rate_cap=200.0)

    assert (
        capped["aggregates"]["prob_max_over_uniform_unweighted"]
        <= uncapped["aggregates"]["prob_max_over_uniform_unweighted"]
    )
    assert capped["params"]["failure_rate_cap"] == 2.0


def test_zero_prior_state_handles_zero_episode_bins() -> None:
    # SIM-M5a candidate config: init_num_failures=0 seeds bins at 0/0.
    summary = build_sampler_state_summary(_state([0.0, 2.0], [0.0, 1.0]), init_num_failures=0.0)

    bins = summary["bins"]
    assert bins[0]["failure_rate"] == 0.0  # 0/0 guarded, not NaN
    assert bins[1]["failure_rate"] == pytest.approx(0.5)
    assert summary["aggregates"]["observed_failures_total"] == 1.0


def test_missing_sampler_state_is_reported_not_fatal() -> None:
    summary = build_sampler_state_summary(
        {"checkpoint_path": "runs/uniform_seed0/last.pt", "global_step": 50}
    )

    assert summary["adaptive_state_present"] is False
    assert "bins" not in summary
    assert summary["caveats"]


def test_mismatched_tensor_lengths_raise() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        build_sampler_state_summary(_state([1.0, 1.0], [1.0]))


def test_output_records_caveats_about_weights_and_decay() -> None:
    summary = build_sampler_state_summary(_state([1.0], [1.0]))

    joined = " ".join(summary["caveats"])
    assert "bin weights" in joined
    assert "decay" in joined
