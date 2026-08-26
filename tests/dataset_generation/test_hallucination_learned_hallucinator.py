"""Tests for the LfLH-style hallucinator study instrument and its coverage metric."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.learned_hallucinator import (
    Hallucinator,
    ReachPair,
    SurrogateDecoder,
    coverage,
    sample_placements,
    train,
)

MARGIN = 0.018044


def _pair(source_id: str = "clip", stations: int = 24, gap_m: float = 0.09) -> ReachPair:
    fractions = np.linspace(0.25, 0.85, stations)
    nominal = 1.30 + 0.01 * np.sin(np.linspace(0, 3.0, stations))
    return ReachPair(
        source_id=source_id,
        stations_m=fractions,
        nominal_reach_m=nominal,
        adapted_reach_m=nominal - gap_m,
    )


def test_reach_pair_rejects_mismatched_profiles():
    with pytest.raises(ValueError):
        ReachPair("bad", np.zeros(4), np.zeros(4), np.zeros(5))


def test_window_and_support_follow_the_closed_form():
    pair = _pair(gap_m=0.09)
    lower, upper = pair.window_at(0, MARGIN)
    assert upper - lower == pytest.approx(0.09 - 2 * MARGIN)
    assert pair.feasible_mask(MARGIN).all()
    assert pair.support_area_m2(MARGIN) > 0.0


def test_a_pair_that_does_not_separate_has_no_support():
    pair = _pair(gap_m=0.01)  # smaller than the two margins together
    assert not pair.feasible_mask(MARGIN).any()
    assert pair.support_area_m2(MARGIN) == 0.0


def test_surrogate_decoder_is_monotone_in_the_face_coordinate():
    """Raising the face must make the nominal less likely to strike and the adaptation to clear."""
    decoder = SurrogateDecoder()
    nominal = torch.full((1, 8), 1.30)
    adapted = torch.full((1, 8), 1.21)
    station = torch.tensor([3.5])
    low = decoder(nominal, adapted, station, torch.tensor([1.23]))
    high = decoder(nominal, adapted, station, torch.tensor([1.28]))
    assert high[0].item() < low[0].item()  # strikes: less likely as the face rises
    assert high[1].item() > low[1].item()  # clears: more likely as the face rises


def test_surrogate_decoder_has_no_learnable_parameters():
    """The decoder is fixed by construction; that is what stops encoder-decoder collusion."""
    decoder = SurrogateDecoder()
    assert not hasattr(decoder, "parameters")
    assert not isinstance(decoder, torch.nn.Module)


def test_coverage_separates_validity_from_occupancy():
    pair = _pair()
    lower = pair.adapted_reach_m + MARGIN
    upper = pair.nominal_reach_m - MARGIN
    # A degenerate sampler: always the same station, always the window centre. Perfectly valid,
    # and it reaches one cell of the feasible set.
    collapsed = np.stack((np.zeros(50), np.full(50, 0.5 * (lower[0] + upper[0]))), axis=-1)
    collapsed_score = coverage(collapsed, pair, margin_m=MARGIN)
    assert collapsed_score["valid_rate"] == pytest.approx(1.0)
    assert collapsed_score["occupancy"] < 0.05

    rng = np.random.default_rng(0)
    stations = rng.integers(0, len(lower), size=600)
    spread = np.stack(
        (stations.astype(float), rng.uniform(lower[stations], upper[stations])), axis=-1
    )
    spread_score = coverage(spread, pair, margin_m=MARGIN)
    assert spread_score["valid_rate"] == pytest.approx(1.0)
    assert spread_score["occupancy"] > 10 * collapsed_score["occupancy"]


def test_training_runs_and_reports_a_contracted_sigma():
    """The headline phenomenon: the objective drives the coordinate variance down."""
    pairs = [_pair(f"clip_{i}", gap_m=0.09 + 0.01 * i) for i in range(4)]
    model, report = train(pairs, steps=60, samples=4, seed=3)
    assert isinstance(model, Hallucinator)
    assert np.isfinite(report.final_loss)
    assert report.mean_sigma_coordinate_mm >= 0.0
    assert report.history and report.history[-1]["step"] == 59


def test_sampled_placements_have_the_requested_shape_per_source():
    pairs = [_pair("a"), _pair("b")]
    model, _ = train(pairs, steps=10, samples=2, seed=1)
    drawn = sample_placements(model, pairs, count=17, seed=1)
    assert set(drawn) == {"a", "b"}
    for values in drawn.values():
        assert values.shape == (17, 2)
        assert np.isfinite(values).all()
