"""Shared small outcome predictor; data arms change examples, never this learner."""

import numpy as np
import torch
from torch import nn

from .motion2scene_policy_features import packet_features

FEATURE_COUNT = 214


def decision_features(packet, state, phase_s, active, observation_age_s):
    """144 rays + phase/skill/age + gravity/velocities + joint positions/velocities."""
    extras = np.concatenate(
        [
            np.array([phase_s / 4, active, observation_age_s]),
            np.asarray(state["projected_gravity_b"]).reshape(3),
            np.asarray(state["root_lin_vel_w"]).reshape(3),
            np.asarray(state["root_ang_vel_w"]).reshape(3),
            np.asarray(state["dof_pos"]).reshape(29),
            np.asarray(state["dof_vel"]).reshape(29),
        ]
    )
    x = np.r_[packet_features(packet), extras].astype(np.float32)
    if x.shape != (FEATURE_COUNT,) or not np.isfinite(x).all() or active not in (0, 1):
        raise ValueError("invalid causal decision features")
    if phase_s < 0 or observation_age_s < 0:
        raise ValueError("negative phase or observation age")
    return x


class OutcomePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_COUNT, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)


def masked_loss(logits, labels, mask):
    if logits.shape != labels.shape or mask.shape != labels.shape or labels.shape[-1] != 2:
        raise ValueError("two outcome heads and corresponding label mask required")
    known = mask.bool()
    if not known.any():
        raise ValueError("no observed action outcomes")
    if not torch.isfinite(labels[known]).all() or not torch.all(
        (labels[known] == 0) | (labels[known] == 1)
    ):
        raise ValueError("known outcomes must be binary")
    # Index before computing loss: unknown NaN targets must not contaminate gradients.
    return nn.functional.binary_cross_entropy_with_logits(logits[known], labels[known])


def select_action(probabilities, threshold=0.5):
    """Prefer feasible walk; neither-head-positive is a refusal, not a stop command."""
    p = np.asarray(probabilities)
    if p.shape[-1] != 2 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("requires two finite probabilities")
    walk, crouch = p[..., 0] >= threshold, p[..., 1] >= threshold
    return np.where(walk, 0, np.where(crouch, 1, -1))


def fit(x, labels, mask, seed, steps=1000):
    """Train-only normalization; caller must enforce registered split/eligibility."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or not torch.isfinite(x).all():
        raise ValueError("invalid training features")
    if len(x) != len(y) or not len(x) or steps <= 0:
        raise ValueError("invalid training budget or labels")
    torch.manual_seed(seed)
    mean = x.mean(0)
    std = x.std(0, correction=0).clamp_min(1e-3)
    normalized = (x - mean) / std
    model = OutcomePredictor()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(model(normalized), y, mask)
        loss.backward()
        optimizer.step()
    model.eval()
    return model, mean, std, float(loss.detach())
