"""Verify prefix equivalence, independent stopping and charged query counts."""

import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_budgeted_pattern import (
    budgeted_pattern,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_distillation import pattern_five
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search


def query(scenes):
    target = 0.015 - (scenes[:, 0] - 0.4).square() - (scenes[:, 1] - 1.25).square()
    return torch.stack((-target, target), dim=-1)[:, None].expand(-1, 17, -1)


def test_exact_five_and_seventeen_equivalence():
    initial = torch.tensor([[0.42, 1.25], [0.2, 1.2], [0.9, 1.45]], dtype=torch.float64)
    for budget, reference in [
        (5, pattern_five(initial, query)[0]),
        (17, station_search(initial, query, "pattern")[0]),
    ]:
        output, trace = budgeted_pattern(initial, query, budget)
        assert torch.equal(output, reference)
        assert trace["queries"] == budget * 3 * 34


def test_adaptive_spends_only_on_search_failures():
    initial = torch.tensor([[0.42, 1.25], [0.1, 1.1]], dtype=torch.float64)
    output, trace = budgeted_pattern(initial, query, "adaptive")
    assert trace["evaluations_per_output"].tolist() == [5, 17]
    assert trace["queries"] == 22 * 34
    fixed, _ = budgeted_pattern(initial[1:], query, 17)
    assert torch.equal(output[1:], fixed)


def test_nine_is_exact_prefix_with_original_trust_bounds():
    initial = torch.tensor([[0.12, 1.1], [0.9, 1.45]], dtype=torch.float64)
    nine, trace = budgeted_pattern(initial, query, 9)
    _, old = station_search(initial, query, "pattern")
    from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import margin_slack

    best = torch.stack([margin_slack(v) for v in old["clearances"][:9]]).argmax(0)
    assert torch.equal(nine, old["scenes"][best, torch.arange(2)])
    assert trace["evaluations_per_output"].tolist() == [9, 9]
