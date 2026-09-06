"""Counterexample regression, exact margin semantics, and physical gradient directions."""

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import BeamDomain
from gear_sonic.dataset_generation.hallucination.motion2scene_margins import margin_terms
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def test_old_low_preference_loss_cannot_hide_inadequate_overlap():
    select, target, upright, _, _ = margin_terms(
        t([[[-0.005, 0.030]]]), t([0, 1]), torch.tensor([False, True])
    )
    assert select.item() < 0.001
    assert target.item() == 0
    assert upright.item() == pytest.approx(0.25)


def test_zero_penalties_iff_all_supplied_recordings_and_offsets_meet_margins():
    rng = np.random.default_rng(1123)
    values = t(rng.uniform(-0.05, 0.05, (100, 3, 4)))
    mask = torch.tensor([False, True, False, True])
    _, target, upright, _, _ = margin_terms(values, t([0, 1, 0, 1]), mask)
    expected = (values[..., mask] >= 0.01).all(-1).all(-1) & (values[..., ~mask] <= -0.01).all(
        -1
    ).all(-1)
    torch.testing.assert_close((target == 0) & (upright == 0), expected)
    boundary = t([[[-0.01, 0.01]]])
    _, a, b, _, _ = margin_terms(boundary, t([0, 1]), torch.tensor([False, True]))
    assert (a + b).item() == 0


def test_worst_offset_gradients_push_both_classes_toward_their_margin():
    values = t([[[-0.03, 0.04], [0.004, 0.003]]], True)
    _, target, upright, _, _ = margin_terms(values, t([0, 1]), torch.tensor([False, True]))
    (target + upright).sum().backward()
    assert values.grad[0, 1, 0] > 0
    assert values.grad[0, 1, 1] < 0
    assert values.grad[0, 0].abs().sum() == 0


def test_interference_penalty_moves_a_high_beam_down():
    clouds = [(t([[0, 0, z]]), t([[0, 0, z]]), t([0.03])) for z in (1.30, 1.20)]
    scenes = t([[0.5, 1.325]], True)
    values = perturbed_clearances(
        scenes,
        t([[0, 0, 0, 0], [0, 0, 0.005, 0]]),
        clouds,
        t([[-1, 0], [1, 0]]),
        t([0, 1]),
        t(0),
        BeamDomain(),
    )
    _, target, upright, _, _ = margin_terms(values, t([0, 1]), torch.tensor([False, True]))
    assert target.item() == 0
    upright.sum().backward()
    assert torch.isfinite(scenes.grad).all()
    assert scenes.grad[0, 1] > 0


@pytest.mark.parametrize("margin", [0, -0.01, float("nan"), float("inf")])
def test_invalid_margin_rejected(margin):
    with pytest.raises(ValueError, match="positive and finite"):
        margin_terms(t([[[-0.02, 0.02]]]), t([0, 1]), torch.tensor([False, True]), margin=margin)
