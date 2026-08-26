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
    """Anti-collusion: an nn.Module decoder could learn to invert meaningless encoder outputs."""
    decoder = SurrogateDecoder()
    assert not isinstance(decoder, torch.nn.Module)
    # The previous version of this test asserted `not hasattr(decoder, "parameters")` on a plain
    # dataclass, which is vacuously true. Assert the fields are plain floats instead.
    for name in ("margin_m", "temperature_m", "station_sharpness"):
        value = getattr(decoder, name)
        assert isinstance(value, float) and not isinstance(value, torch.Tensor)


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


def test_occupancy_can_never_exceed_one():
    """The retracted metric used different cell rules per side and measured 124.7% occupancy."""
    pair = _pair()
    lower = pair.adapted_reach_m + MARGIN
    upper = pair.nominal_reach_m - MARGIN
    rng = np.random.default_rng(0)
    for bins in (4, 6, 12, 24, 48):
        stations = rng.integers(0, len(lower), size=4000)
        placements = np.stack(
            (stations.astype(float), rng.uniform(lower[stations], upper[stations])), axis=-1
        )
        score = coverage(placements, pair, margin_m=MARGIN, bins=bins)
        assert 0.0 <= score["occupancy"] <= 1.0, f"occupancy out of range at bins={bins}"


def test_occupancy_is_budget_dependent_and_must_be_read_as_a_curve():
    """A uniform sampler's occupancy saturates with N, so a single budget is not a property."""
    pair = _pair()
    lower = pair.adapted_reach_m + MARGIN
    upper = pair.nominal_reach_m - MARGIN
    rng = np.random.default_rng(1)

    def occupancy_at(count: int) -> float:
        stations = rng.integers(0, len(lower), size=count)
        placements = np.stack(
            (stations.astype(float), rng.uniform(lower[stations], upper[stations])), axis=-1
        )
        return coverage(placements, pair, margin_m=MARGIN)["occupancy"]

    assert occupancy_at(25) < occupancy_at(200) < occupancy_at(2000)


def test_the_bare_objective_applies_unbounded_contraction_pressure():
    """Pins the defect that invalidated the first result.

    ``-log sigmoid(z)`` is ``softplus(-z)``, which is convex, so the expected reconstruction loss
    is strictly increasing in sigma at every temperature. With no entropy or KL term opposing it,
    sigma decreases monotonically until it hits ``min_log_sigma`` -- so any reported "final sigma"
    is the clamp floor or a snapshot of however many steps were run, never a property of the data.
    """
    pairs = [_pair(f"clip_{i}") for i in range(4)]
    _, report = train(pairs, steps=400, samples=4, seed=5)
    sigmas = [row["sigma_coordinate_mm"] for row in report.history]
    assert sigmas[-1] < 0.5 * sigmas[0], "sigma must contract without a KL term"
    # Monotone in the large: no sustained recovery once contracted.
    assert min(sigmas[len(sigmas) // 2 :]) <= sigmas[len(sigmas) // 2]
    floor_mm = 1000 * 0.1 * float(np.exp(-6.0))
    assert all(value >= floor_mm - 1e-9 for value in sigmas), "sigma is clamped from below"


def test_a_kl_term_prevents_the_collapse_that_the_bare_objective_forces():
    """The collapse is a property of a convex loss with no entropy term, not of the model class."""
    pairs = [_pair(f"clip_{i}") for i in range(4)]
    _, bare = train(pairs, steps=200, samples=4, seed=5, kl_weight=0.0)
    _, regularised = train(pairs, steps=200, samples=4, seed=5, kl_weight=0.1)
    assert regularised.mean_sigma_coordinate_mm > 5 * bare.mean_sigma_coordinate_mm


def test_a_constant_predictor_scores_a_perfect_valid_rate():
    """Why valid_rate alone is misleading: a constant achieves 100% of it."""
    pair = _pair()
    lower = pair.adapted_reach_m + MARGIN
    constant = np.stack((np.zeros(200), np.full(200, lower[0] + 0.004)), axis=-1)
    score = coverage(constant, pair, margin_m=MARGIN)
    assert score["valid_rate"] == pytest.approx(1.0)
    assert score["occupancy"] < 0.02
