"""Small motion-only inverse beam model and a fixed geometric choice objective.

This development instrument learns station and height, with all other beam
dimensions fixed. No analytic interval or obstacle label enters the loss.
The finite, sampled-motion capsule model is not a simulator collision certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..capsule_box_torch import capsule_yaw_box_clearance


@dataclass(frozen=True)
class BeamDomain:
    station_low: float = 0.35
    station_high: float = 0.65
    height_low: float = 1.10
    height_high: float = 1.45
    depth: float = 0.10
    width: float = 1.20
    thickness: float = 0.10
    clearance_cap: float = 0.10

    def physical(self, latent):
        low = latent.new_tensor([self.station_low, self.height_low])
        span = latent.new_tensor(
            [self.station_high - self.station_low, self.height_high - self.height_low]
        )
        return low + span * latent.sigmoid()


class BeamMixture(nn.Module):
    """Four correlated Gaussian components in logit coordinates.

    With motion_conditioned=False the same distribution is optimized per motion,
    providing an explicit non-amortized comparison with the encoder variant.
    """

    def __init__(self, capsule_count, components=4, *, motion_conditioned=True):
        super().__init__()
        self.components = components
        self.motion_conditioned = motion_conditioned
        initial = torch.zeros(components, 6)
        initial[:, 1] = torch.linspace(-1.2, 1.2, components)
        initial[:, 3:5] = math.log(math.expm1(0.75 - 0.03))
        self.base = nn.Parameter(initial)
        if motion_conditioned:
            self.body = nn.Sequential(nn.Linear(7, 16), nn.Tanh())
            self.temporal = nn.Sequential(
                nn.Conv1d(capsule_count * 16, 64, 5, padding=2),
                nn.Tanh(),
                nn.Conv1d(64, 64, 5, padding=2),
                nn.Tanh(),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(64, components * 6)
            nn.init.normal_(self.head.weight, std=0.001)
            nn.init.zeros_(self.head.bias)

    def forward(self, motion):
        """motion: (time, capsules, 7), midpoint/axis/radius in canonical metres."""
        raw = self.base
        if self.motion_conditioned:
            features = self.body(motion).flatten(1).T.unsqueeze(0)
            raw = raw + self.head(self.temporal(features).flatten()).reshape(self.components, 6)
        logits, means = raw[:, 0], raw[:, 1:3]
        scales = F.softplus(raw[:, 3:5]) + 0.03
        correlation = 0.95 * raw[:, 5].tanh()
        zero = torch.zeros_like(correlation)
        factor = torch.stack(
            (
                scales[:, 0],
                zero,
                correlation * scales[:, 1],
                torch.sqrt(1 - correlation.square()) * scales[:, 1],
            ),
            dim=-1,
        ).reshape(-1, 2, 2)
        return logits, means, factor


def mixture_log_prob(value, logits, means, factor):
    """Marginal mixture density, including every component (not assigned-component KL)."""
    difference = value[..., None, :] - means
    first = difference[..., 0] / factor[:, 0, 0]
    second = (difference[..., 1] - factor[:, 1, 0] * first) / factor[:, 1, 1]
    log_density = (
        -0.5 * (first.square() + second.square())
        - math.log(2 * math.pi)
        - factor[:, 0, 0].log()
        - factor[:, 1, 1].log()
    )
    return torch.logsumexp(log_density + logits.log_softmax(-1), dim=-1)


def stratified_latents(parameters, count, generator):
    """Reparameterized samples from every component; categorical weights stay differentiable."""
    logits, means, factor = parameters
    noise = torch.randn(
        (len(means), count, 2), dtype=means.dtype, device=means.device, generator=generator
    )
    samples = means[:, None] + torch.einsum("kij,ksj->ksi", factor, noise)
    weights = logits.softmax(-1)[:, None].expand(-1, count) / count
    log_q = mixture_log_prob(samples, *parameters)
    log_prior = -0.5 * samples.square().sum(-1) - math.log(2 * math.pi)
    return samples, weights, log_q - log_prior


def sample_latents(parameters, count, generator):
    """Actual categorical-mixture samples for a fixed-budget evaluation."""
    logits, means, factor = parameters
    selected = torch.multinomial(logits.softmax(-1), count, replacement=True, generator=generator)
    noise = torch.randn((count, 2), dtype=means.dtype, device=means.device, generator=generator)
    return means[selected] + torch.einsum("nij,nj->ni", factor[selected], noise)


def route_centers(stations, route, progress):
    """Piecewise-linear route lookup: differentiable within segments, fixed world frame."""
    if route.ndim != 2 or route.shape[-1] != 2 or progress.shape != route.shape[:1]:
        raise ValueError("route/progress must be (T,2)/(T,)")
    if torch.any(progress[1:] <= progress[:-1]):
        raise ValueError("route progress must be strictly increasing")
    index = torch.searchsorted(progress, stations.contiguous()).clamp(1, len(progress) - 1)
    weight = (stations - progress[index - 1]) / (progress[index] - progress[index - 1])
    return route[index - 1] + weight[:, None] * (route[index] - route[index - 1])


def domain_capsules(state, domain):
    """Exact exclusion below/above the entire allowed height domain plus clearance cap.

    All retained time samples and capsule axes are evaluated. Excluded capsules
    cannot lower a minimum that is capped at domain.clearance_cap.
    """
    starts, ends = state["starts"].reshape(-1, 3), state["ends"].reshape(-1, 3)
    radii = state["radii"].expand(state["starts"].shape[:-1]).reshape(-1)
    top = torch.maximum(starts[:, 2], ends[:, 2]) + radii
    bottom = torch.minimum(starts[:, 2], ends[:, 2]) - radii
    keep = (top >= domain.height_low - domain.clearance_cap) & (
        bottom <= domain.height_high + domain.thickness + domain.clearance_cap
    )
    return starts[keep], ends[keep], radii[keep]


def scene_clearances(scene, clouds, route, progress, yaw, domain):
    """(proposals, recordings) capped minima over all retained frames and capsule axes."""
    xy = route_centers(scene[:, 0], route, progress)
    center = torch.cat((xy, (scene[:, 1] + domain.thickness / 2)[:, None]), dim=-1)
    half = scene.new_tensor([domain.depth, domain.width, domain.thickness]) / 2
    results = []
    for starts, ends, radii in clouds:
        if len(starts) == 0:
            results.append(scene[:, 0] * 0 + domain.clearance_cap)
            continue
        clearance = capsule_yaw_box_clearance(
            starts[None], ends[None], radii[None], center[:, None], half, yaw
        )
        results.append(clearance.amin(-1).clamp(max=domain.clearance_cap))
    return torch.stack(results, dim=-1)


def choice_energies(clearances, costs, *, margin=0.01):
    """Fixed decoder: cost per recording supplied independently of the target label."""
    return costs[None] + 200 * F.softplus((margin - clearances) / 0.005) * 0.005


def inverse_terms(clearances, costs, target_mask, *, margin=0.01):
    """Robust binary decoder and target barrier; target mask belongs to loss/evaluation.

    Costs are fixed at 0 for upright and 1 for crouch. Select against the easiest
    upright recording and the hardest target recording. Distances are in metres.
    """
    target = clearances[:, target_mask].amin(-1)
    weaker = clearances[:, ~target_mask].amax(-1)
    energies = choice_energies(clearances, costs, margin=margin)
    target_energy = energies[:, target_mask].amax(-1)
    weaker_energy = energies[:, ~target_mask].amin(-1)
    select = F.softplus((target_energy - weaker_energy) / 0.25)
    feasibility = ((margin - target).clamp(min=0) / 0.01).square()
    return select, feasibility, target, weaker
