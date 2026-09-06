"""Synthetic constraint and trust-region checks independent of the study motions."""

import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import (
    bounded_refine,
    margin_slack,
)


def test_refinement_rescues_clearance_and_keeps_incumbent():
    initial = torch.tensor([[0.5, 1.175], [0.5, 1.225]], dtype=torch.float64)
    calls = []

    def query(scenes):
        calls.append(scenes.detach().clone())
        return torch.stack((scenes[:, 1] - 1.22, scenes[:, 1] - 1.18), -1)[:, None]

    result, trace = bounded_refine(initial, query)
    assert len(calls) == 17
    assert (margin_slack(query(result)) >= 0).all()
    scores = torch.stack([margin_slack(v) for v in trace["clearances"]])
    torch.testing.assert_close(margin_slack(query(result)), scores.max(0).values)
    assert (abs(trace["scenes"] - initial)[..., 0] <= 0.05 + 1e-14).all()
    assert (abs(trace["scenes"] - initial)[..., 1] <= 0.03 + 1e-14).all()


def test_unreachable_region_and_domain_bounds_are_retained():
    initial = torch.tensor([[0.1, 1.1], [0.9, 1.45]], dtype=torch.float64)

    def query(scenes):
        return torch.stack((scenes[:, 1] - 1.42, scenes[:, 1] - 1.38), -1)[:, None]

    result, trace = bounded_refine(initial, query)
    assert margin_slack(query(result))[0] < 0
    assert (trace["scenes"] >= initial.new_tensor([0.1, 1.1]) - 1e-14).all()
    assert (trace["scenes"] <= initial.new_tensor([0.9, 1.45]) + 1e-14).all()
    assert result[0, 1] <= 1.13 + 1e-14
    with pytest.raises(ValueError):
        bounded_refine(initial + 10, query)


def test_worst_placement_and_both_margins_control_slack():
    values = torch.tensor(
        [[[-0.02, 0.02], [-0.005, 0.03]], [[-0.04, 0.009], [-0.02, 0.02]]], dtype=torch.float64
    )
    torch.testing.assert_close(
        margin_slack(values), torch.tensor([-0.005, -0.001], dtype=torch.float64)
    )
    with pytest.raises(ValueError):
        margin_slack(torch.zeros(2, 0, 2))
