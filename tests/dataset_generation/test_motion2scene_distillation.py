"""Distribution matching, empty sets and actual reduced query spend."""

import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_distillation import (
    pattern_five,
    set_distance,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_refinement import margin_slack
from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import station_search


def test_matching_weighted_distribution_has_zero_distance():
    points = torch.tensor([[0.2, 1.2], [0.7, 1.4]], dtype=torch.float64, requires_grad=True)
    loss = set_distance(points, torch.tensor([0.5, 0.5], dtype=torch.float64), points.detach())
    torch.testing.assert_close(loss, torch.tensor(0.0, dtype=torch.float64), atol=1e-12, rtol=0)
    loss.backward()
    assert torch.isfinite(points.grad).all()


def test_gradient_moves_proposal_toward_teacher():
    point = torch.tensor([[0.3, 1.2]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[0.6, 1.3]], dtype=torch.float64)
    loss = set_distance(point, torch.ones(1, dtype=torch.float64), target)
    loss.backward()
    assert (point.grad < 0).all()
    assert (
        set_distance(point - 0.01 * point.grad, torch.ones(1, dtype=torch.float64), target) < loss
    )


def test_weights_can_learn_which_component_matches():
    points = torch.tensor([[0.3, 1.2], [0.8, 1.4]], dtype=torch.float64)
    logits = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    set_distance(points, logits.softmax(0), points[:1]).backward()
    assert logits.grad[0] < 0 < logits.grad[1]


def test_empty_teacher_is_finite_zero_with_zero_gradient():
    points = torch.tensor([[0.3, 1.2]], dtype=torch.float64, requires_grad=True)
    loss = set_distance(
        points, torch.ones(1, dtype=torch.float64), torch.empty((0, 2), dtype=torch.float64)
    )
    loss.backward()
    assert loss == 0 and torch.equal(points.grad, torch.zeros_like(points))


def test_reduced_search_really_spends_five_evaluations_and_matches_prefix():
    initial = torch.tensor([[0.1, 1.1], [0.5, 1.25], [0.9, 1.45]], dtype=torch.float64)
    calls = []

    def query(x):
        calls.append(x.clone())
        return torch.stack((0.02 - 2 * torch.relu(x[:, 0] - 0.51), x[:, 1] - 1.18), -1)[
            :, None
        ].expand(-1, 17, -1)

    output, trace = pattern_five(initial, query)
    assert len(calls) == 5
    _, full = station_search(initial, query, "pattern")
    torch.testing.assert_close(trace["scenes"], full["scenes"][:5])
    scores = torch.stack([margin_slack(v) for v in full["clearances"][:5]])
    expected = full["scenes"][:5][scores.argmax(0), torch.arange(3)]
    torch.testing.assert_close(output, expected)
    assert (abs(trace["scenes"] - initial) <= torch.tensor([0.05, 0.03]) + 1e-9).all()
