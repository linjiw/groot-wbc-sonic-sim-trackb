"""Meaningful training contracts: fixed costs, mixture density, gradients, and culling."""

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import (
    BeamDomain,
    BeamMixture,
    choice_energies,
    domain_capsules,
    inverse_terms,
    mixture_log_prob,
    scene_clearances,
    stratified_latents,
)


def t(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def test_mixture_density_agrees_with_distribution_library():
    torch.manual_seed(19)
    model = BeamMixture(2).double()
    parameters = model(torch.randn(10, 2, 7, dtype=torch.float64))
    logits, means, factor = parameters
    distribution = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(logits=logits),
        torch.distributions.MultivariateNormal(means, scale_tril=factor),
    )
    values = torch.randn(30, 2, dtype=torch.float64)
    torch.testing.assert_close(mixture_log_prob(values, *parameters), distribution.log_prob(values))


def test_stratification_backpropagates_all_component_weights_and_shapes():
    model = BeamMixture(2, motion_conditioned=False).double()
    latent, weights, kl = stratified_latents(model(None), 10, torch.Generator().manual_seed(12))
    assert weights.sum().item() == pytest.approx(1)
    loss = (weights * ((latent[..., 0] - 0.7).square() + latent[..., 1].square() + 0.1 * kl)).sum()
    loss.backward()
    assert torch.isfinite(model.base.grad).all()
    assert (model.base.grad[:, 0].abs() > 1e-8).all()
    assert (model.base.grad[:, 1:5].abs() > 1e-8).all()


def test_standard_normal_component_mixture_has_zero_kl():
    parameters = (t([0, 0, 0, 0]), t(np.zeros((4, 2))), t(np.tile(np.eye(2), (4, 1, 1))))
    _, _, kl = stratified_latents(parameters, 15, torch.Generator().manual_seed(20))
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-12, rtol=0)


def test_decoder_geometry_and_costs_permute_together():
    clearances, costs = t([[0.05, 0.05, -0.03, 0.02]]), t([0, 0, 1, 1])
    mask = torch.tensor([False, False, True, True])
    order = torch.tensor([2, 0, 3, 1])
    expected = inverse_terms(clearances, costs, mask)
    actual = inverse_terms(clearances[:, order], costs[order], mask[order])
    for first, second in zip(actual, expected):
        torch.testing.assert_close(first, second)
    # The scene evaluator has no desired-target argument; equal clearances preserve fixed costs.
    energy = choice_energies(t([[0.05, 0.05]]), t([0, 1]))
    assert (energy[0, 1] - energy[0, 0]).item() == pytest.approx(1)


def test_low_beam_target_gradient_moves_up_and_high_beam_preference_moves_down():
    domain = BeamDomain()
    clouds = [(t([[0, 0, z]]), t([[0, 0, z]]), t([0.03])) for z in [1.3, 1.2]]
    route, progress, yaw = t([[-1, 0], [1, 0]]), t([0, 1]), t(0.0)
    mask, costs = torch.tensor([False, True]), t([0, 1])
    scene = t([[0.5, 1.21]], True)
    values = scene_clearances(scene, clouds, route, progress, yaw, domain)
    select, feasibility, _, _ = inverse_terms(values, costs, mask)
    feasibility.sum().backward()
    assert scene.grad[0, 1] < 0
    scene = t([[0.5, 1.35]], True)
    select, _, _, _ = inverse_terms(
        scene_clearances(scene, clouds, route, progress, yaw, domain), costs, mask
    )
    select.sum().backward()
    assert scene.grad[0, 1] > 0


def test_domain_culling_equals_full_capsules_at_interior_and_domain_edges():
    domain = BeamDomain()
    generator = torch.Generator().manual_seed(93)
    starts = torch.rand(8, 20, 3, dtype=torch.float64, generator=generator) * 2
    ends = starts + torch.rand(8, 20, 3, dtype=torch.float64, generator=generator) * 0.3
    radii = torch.rand(20, dtype=torch.float64, generator=generator) * 0.1
    state = {"starts": starts, "ends": ends, "radii": radii}
    full = (starts.reshape(-1, 3), ends.reshape(-1, 3), radii.expand(8, 20).flatten())
    reduced = domain_capsules(state, domain)
    scene = t([[0.35, 1.10], [0.65, 1.45], [0.45, 1.28]])
    args = (t([[0, 0], [2, 1]]), t([0, 1]), t(0.3), domain)
    torch.testing.assert_close(
        scene_clearances(scene, [full], *args), scene_clearances(scene, [reduced], *args)
    )


def test_output_bounds_have_no_teacher_interval():
    samples = BeamDomain().physical(t([[-100, -100], [100, 100]]))
    torch.testing.assert_close(samples, t([[0.35, 1.10], [0.65, 1.45]]), atol=1e-12, rtol=0)
