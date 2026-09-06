"""Trust region and flat-gradient tests using synthetic, non-corpus geometry."""

import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import (
    bounded_refine,
    margin_slack,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import (
    METHODS,
    station_search,
)


@pytest.mark.parametrize("method", METHODS)
def test_station_exploration_escapes_a_flat_gradient(method):
    initial = torch.tensor([[0.5, 1.2]], dtype=torch.float64)
    calls = []

    def query(scenes):
        calls.append(scenes.detach().clone())
        return torch.stack((0.005 - 2 * torch.relu(scenes[:, 0] - 0.51), scenes[:, 1] - 1.18), -1)[
            :, None
        ]

    original, _ = bounded_refine(initial, query)
    assert margin_slack(query(original)).item() < 0
    calls.clear()
    result, trace = station_search(initial, query, method)
    assert len(calls) == 17
    assert margin_slack(query(result)).item() >= 0
    scores = torch.stack([margin_slack(v) for v in trace["clearances"]])
    torch.testing.assert_close(margin_slack(query(result)), scores.max(0).values)


@pytest.mark.parametrize("method", METHODS)
def test_original_bounds_and_infeasible_outputs_are_retained(method):
    initial = torch.tensor([[0.1, 1.1], [0.9, 1.45]], dtype=torch.float64)

    def query(scenes):
        return torch.stack((scenes[:, 1] - 1.42, scenes[:, 1] - 1.38), -1)[:, None]

    result, trace = station_search(initial, query, method)
    assert margin_slack(query(result))[0] < 0
    assert (abs(trace["scenes"] - initial) <= initial.new_tensor([0.05, 0.03]) + 1e-14).all()
    assert (trace["scenes"] >= initial.new_tensor([0.1, 1.1]) - 1e-14).all()
    assert (trace["scenes"] <= initial.new_tensor([0.9, 1.45]) + 1e-14).all()
    assert len(trace["scenes"]) == 17


def test_invalid_inputs_and_nonfinite_queries_stop():
    initial = torch.tensor([[0.5, 1.2]], dtype=torch.float64)
    with pytest.raises(ValueError):
        station_search(initial, lambda _: None, "unknown")
    with pytest.raises(ValueError):
        station_search(initial + 10, lambda _: None, "pattern")
    with pytest.raises(FloatingPointError):
        station_search(initial, lambda _: torch.full((1, 1, 2), torch.nan), "pattern")
