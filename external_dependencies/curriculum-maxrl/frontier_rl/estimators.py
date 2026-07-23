"""Group advantage weights. MaxRL is the framework's estimator (P5: ~N x
RLOO's signal on frontier tasks; H6: the only one of the three that is safe
under a frontier curriculum). GRPO/RLOO are included for baselines."""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def maxrl_weights(rewards: np.ndarray) -> np.ndarray:
    """w_i = r_i/K − 1/N; zero vector when K = 0 (paper eq. 10)."""
    n = len(rewards)
    k = rewards.sum()
    if k == 0:
        return np.zeros(n)
    return rewards / k - 1.0 / n


def rloo_weights(rewards: np.ndarray) -> np.ndarray:
    n = len(rewards)
    if n < 2:
        return rewards.copy()
    loo = (rewards.sum() - rewards) / (n - 1)
    return (rewards - loo) / n


def grpo_weights(rewards: np.ndarray) -> np.ndarray:
    n = len(rewards)
    std = rewards.std(ddof=1) if n > 1 else 1.0
    return (rewards - rewards.mean()) / (std + EPS) / n
