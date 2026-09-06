"""Motion-only local proposals with explicit route anchors and training-only scaling."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ANCHORS = 8
FRAMES = 64


def event_features(capsules, route):
    """Encode on route progress; the collision checker must still use all source frames.

    Subtract the root's horizontal position at each frame. Absolute vertical body
    geometry remains visible. No alternative motion or construction event is used.
    Repeated route samples keep their first frame for the encoder only.
    """
    capsules, route = np.asarray(capsules), np.asarray(route)
    if capsules.ndim != 3 or capsules.shape[2] != 7 or route.shape != (len(capsules), 2):
        raise ValueError("expected capsules (T,C,7) and route (T,2)")
    if len(route) < 2 or not np.isfinite(capsules).all() or not np.isfinite(route).all():
        raise ValueError("requires finite motion with at least two frames")
    distance = np.r_[0.0, np.linalg.norm(np.diff(route, axis=0), axis=1).cumsum()]
    if distance[-1] <= 1e-8:
        raise ValueError("stationary routes require a separate time-anchor model")
    progress = distance / distance[-1]
    keep = np.r_[True, np.diff(progress) > 1e-12]
    local = capsules.copy()
    local[..., :2] -= route[:, None, :]
    feature = np.c_[local.reshape(len(route), -1), progress, np.linspace(0, 1, len(route))]
    grid = np.linspace(0, 1, FRAMES)
    return np.stack([np.interp(grid, progress[keep], f[keep]) for f in feature.T], axis=1)


def fit_normalization(features, training_ids):
    """Only explicitly supplied training IDs influence feature scale."""
    if not training_ids or len(set(training_ids)) != len(training_ids):
        raise ValueError("requires unique nonempty training IDs")
    data = np.concatenate([features[key] for key in training_ids])
    if not np.isfinite(data).all():
        raise ValueError("nonfinite training features")
    return data.mean(0), np.maximum(data.std(0), 0.02)


class EventMixture(nn.Module):
    """Shared normalized temporal encoder; local or globally pooled proposal features.

    All variants retain eight anchor-specific base distributions, including the
    constant-input control. This makes hand-designed station coverage explicit.
    """

    def __init__(self, features=205, width=32, *, pooled=False):
        super().__init__()
        self.pooled = pooled
        self.input = nn.Sequential(nn.Linear(features, width), nn.LayerNorm(width), nn.SiLU())
        self.temporal = nn.Conv1d(width, width, 5, padding=2)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 6)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.zeros_(self.head.bias)
        base = torch.zeros(ANCHORS, 6)
        base[:, 3:5] = math.log(math.expm1(0.75 - 0.03))
        self.base = nn.Parameter(base)
        self.register_buffer("anchor_progress", torch.linspace(0.15, 0.85, ANCHORS))

    def forward(self, motion):
        if motion.ndim != 2 or len(motion) != FRAMES:
            raise ValueError("requires a 64-frame normalized feature sequence")
        h = self.input(motion)
        h = self.norm(h + F.silu(self.temporal(h.T[None])[0].T))
        if self.pooled:
            h = h.mean(0).expand(ANCHORS, -1)
        else:
            index = self.anchor_progress * (FRAMES - 1)
            low = index.long()
            alpha = (index - low)[:, None]
            h = (1 - alpha) * h[low] + alpha * h[low + 1]
        raw = self.base + self.head(h)
        scales = F.softplus(raw[:, 3:5]) + 0.03
        rho = raw[:, 5].tanh() * 0.95
        factor = torch.stack(
            (
                scales[:, 0],
                torch.zeros_like(rho),
                rho * scales[:, 1],
                (1 - rho.square()).sqrt() * scales[:, 1],
            ),
            dim=-1,
        ).reshape(ANCHORS, 2, 2)
        return raw[:, 0], raw[:, 1:3], factor


def physical_at_anchor(latent, anchor_ids):
    """Eight adjoining 0.1-wide station intervals cover [0.10,0.90]."""
    station = 0.10 + 0.10 * (anchor_ids.to(latent) + latent[..., 0].sigmoid())
    height = 1.10 + 0.35 * latent[..., 1].sigmoid()
    return torch.stack((station, height), dim=-1)


def stratified_scenes(parameters, generator):
    """One reparameterized draw per anchor; exactly eight geometry proposals."""
    logits, means, factor = parameters
    noise = torch.randn(means.shape, dtype=means.dtype, generator=generator)
    latent = means + torch.einsum("kij,kj->ki", factor, noise)
    return physical_at_anchor(latent, torch.arange(ANCHORS)), logits.softmax(-1)


def sample_scenes(parameters, count, generator):
    logits, means, factor = parameters
    selected = torch.multinomial(logits.softmax(-1), count, replacement=True, generator=generator)
    noise = torch.randn((count, 2), dtype=means.dtype, generator=generator)
    latent = means[selected] + torch.einsum("nij,nj->ni", factor[selected], noise)
    return physical_at_anchor(latent, selected)
