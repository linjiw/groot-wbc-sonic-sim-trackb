"""Uncertainty culling, independent pose queries, and worst-offset loss contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamDomain,
    domain_capsules,
    inverse_terms,
    scene_clearances,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_uncertainty import (
    perturbed_clearances,
    uncertainty_capsules,
    uncertainty_terms,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_uncertainty_learning import numpy_perturbed, offset_sets  # noqa: E402


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


@pytest.mark.parametrize("z,height,dz", [(0.98, 1.10, -0.01), (1.67, 1.45, 0.01)])
def test_vertical_offset_culling_retains_capsules_nominal_domain_excludes(z, height, dz):
    domain = BeamDomain()
    state = {"starts": t([[[0, 0, z]]]), "ends": t([[[0, 0, z]]]), "radii": t([0.015])}
    nominal = domain_capsules(state, domain)
    expanded = uncertainty_capsules(state, domain, 0.01)
    assert len(nominal[0]) == 0
    assert len(expanded[0]) == 1
    result = perturbed_clearances(
        t([[0.5, height]]),
        t([[0, 0, dz, 0]]),
        [expanded],
        t([[-1, 0], [1, 0]]),
        t([0, 1]),
        t(0),
        domain,
    )
    assert result.item() == pytest.approx(0.095)


def test_zero_offset_matches_frozen_nominal_query_and_loss():
    domain = BeamDomain()
    clouds = [(t([[0, 0, z]]), t([[0, 0, z]]), t([0.03])) for z in (1.3, 1.2)]
    scenes = t([[0.5, 1.27]])
    args = (clouds, t([[-1, 0], [1, 0]]), t([0, 1]), t(0.2), domain)
    nominal = scene_clearances(scenes, *args)
    perturbed = perturbed_clearances(scenes, t([[0, 0, 0, 0]]), *args)
    torch.testing.assert_close(perturbed[:, 0], nominal)
    mask, costs = torch.tensor([False, True]), t([0, 1])
    for actual, expected in zip(
        uncertainty_terms(perturbed, costs, mask), inverse_terms(nominal, costs, mask)
    ):
        torch.testing.assert_close(actual, expected)


def test_worst_offset_cannot_hide_target_or_alternative_failure_and_has_gradient():
    values = t([[[-0.04, 0.05], [-0.03, -0.005], [0.02, 0.06]]], True)
    mask, costs = torch.tensor([False, True]), t([0, 1])
    selection, feasibility, target, weaker = uncertainty_terms(values, costs, mask)
    assert target.item() == pytest.approx(-0.005)
    assert weaker.item() == pytest.approx(0.02)
    (selection + feasibility).sum().backward()
    assert torch.isfinite(values.grad).all()
    assert values.grad[0, 1, 1] < 0
    assert values.grad[0, 2, 0] > 0
    assert values.grad[0, 0].abs().sum() == 0


def test_all_offsets_match_independent_numpy_with_full_frames():
    rng = np.random.default_rng(761)
    starts = rng.uniform([-0.3, -0.5, 0.8], [0.7, 0.5, 1.8], (4, 5, 3))
    ends = starts + rng.normal(0, 0.12, starts.shape)
    state = {"starts": starts, "ends": ends, "radii": rng.uniform(0.01, 0.08, 5)}
    route = np.array([[-0.4, -0.1], [0.1, 0.2], [0.8, 0.2]])
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=-1))]
    arc /= arc[-1]
    scenes = np.array([[0.35, 1.1], [0.65, 1.45], [0.5, 1.26]])
    _, offsets = offset_sets()
    cloud = uncertainty_capsules(
        {key: t(value) for key, value in state.items()}, BeamDomain(), 0.01
    )
    actual = perturbed_clearances(
        t(scenes), t(offsets), [cloud], t(route), t(arc), t(0.37), BeamDomain()
    )
    expected = numpy_perturbed(scenes, offsets, {"test": state}, route, 0.37)
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=0)


def test_empty_offset_set_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        perturbed_clearances(
            t([[0.5, 1.2]]), t(np.empty((0, 4))), [], None, None, None, BeamDomain()
        )
