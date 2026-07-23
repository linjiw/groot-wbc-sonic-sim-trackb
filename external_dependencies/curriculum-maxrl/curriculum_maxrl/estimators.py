"""Per-group advantage weights for REINFORCE / RLOO / GRPO / MaxRL.

Each function maps a binary reward vector r (n rollouts of one prompt) to
per-rollout scalar weights w such that the gradient estimate is
sum_j w_j * S_j, with S_j = grad log pi(z_j).  These mirror the formulas in
verl/trainer/ppo/core_algos.py of the MaxRL codebase, minus token masking.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def weights_reinforce(r: np.ndarray) -> np.ndarray:
    return r / len(r)


def weights_rloo(r: np.ndarray) -> np.ndarray:
    n = len(r)
    if n < 2:
        return r.copy()
    loo_mean = (r.sum() - r) / (n - 1)
    return (r - loo_mean) / n


def weights_grpo(r: np.ndarray) -> np.ndarray:
    n = len(r)
    std = r.std(ddof=1) if n > 1 else 1.0
    return (r - r.mean()) / (std + EPS) / n


def weights_maxrl(r: np.ndarray) -> np.ndarray:
    """Variance-reduced MaxRL estimator, eq. (10)/Algorithm 1 of the paper.

    w_j = (r_j / K - 1/N); the whole group is dropped when K = 0.
    Unbiased for the truncated ML objective with T = N.
    """
    n = len(r)
    k = r.sum()
    if k == 0:
        return np.zeros(n)
    return r / k - 1.0 / n


def _c_TN(K: int, N: int, T: int) -> float:
    """Per-success weight of the subset estimator (maclaurin.py c_sub_TN,
    paper appendix eq. 51): unbiased for the T-truncated objective with N
    rollouts, any T <= N.  c_{N,N}(K) = 1/K recovers Algorithm 1."""
    from math import lgamma, log, exp
    if K == 0 or T <= 0:
        return 0.0

    def logcomb(a, kk):
        if kk < 0 or a < kk or a < 0:
            return float("-inf")
        return lgamma(a + 1) - lgamma(kk + 1) - lgamma(a - kk + 1)

    F = N - K
    logC_NT = logcomb(N, T)
    s = 0.0
    for k in range(1, min(T, K) + 1):
        lt = logcomb(K - 1, k - 1) + logcomb(F, T - k) - logC_NT - log(k)
        if lt > float("-inf"):
            s += exp(lt)
    return s


def weights_maxrl_t(r: np.ndarray, T: int) -> np.ndarray:
    """MaxRL subset estimator with decoupled truncation order T <= N.

    w_succ = c_{T,N}(K) - 1/N, w_fail = -1/N (same zero-mean control variate
    as eq. 10); group dropped at K = 0.  T = N recovers weights_maxrl; T = 1
    gives E[w*] matching plain RL weighting.
    """
    n = len(r)
    k = int(r.sum())
    if k == 0:
        return np.zeros(n)
    c = _c_TN(k, n, min(T, n))
    return r * c - 1.0 / n
