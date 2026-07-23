"""StreamingFrontierTeacher: the advantage-mass curriculum over a CONTINUOUS
difficulty axis, for procedural/generative task sources where no fixed task
pool exists (every task is fresh: generated mazes, sampled goals, synthetic
problems parameterized by difficulty d ∈ [0, 1]).

Design — a kernel pass-rate model instead of per-task Beta rows:

  p̂(d) = Σᵢ K((d−dᵢ)/h)·kᵢ / Σᵢ K((d−dᵢ)/h)·nᵢ        (Nadaraya-Watson)

over a decayed sliding window of (difficulty, successes, trials)
observations.  Pseudo-counts from the same kernel sums give a Beta
posterior at any query point, so Thompson sampling works exactly as in the
discrete teacher:

  α(d) = 1 + Σ K·kᵢ·wᵢ ,   β(d) = 1 + Σ K·(nᵢ−kᵢ)·wᵢ   (wᵢ = age decay)

Sampling: draw a difficulty grid, Thompson-sample p̃(d) per grid point,
score u(p̃) = (1−(1−p̃)^N) − p̃, sample d ∝ u^γ + floor, then jitter within
the grid cell — a continuous analogue of ALP-GMM's GMM-over-(task, LP)
resampling, but with the *derived* utility instead of learning progress.

Monotone option: if difficulty is known to order pass rates (usually true
by construction of d), apply isotonic (PAV) projection to the Thompson
draws across the grid — VALIDATION.md V2b showed this shares statistical
strength across the axis and closes part of the oracle gap.
"""

from __future__ import annotations

from collections import deque

import numpy as np


def _pav_nonincreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators projection onto nonincreasing sequences."""
    vals = list(y.astype(float))
    wts = [1.0] * len(vals)
    idx = [[i] for i in range(len(vals))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] < vals[i + 1] - 1e-12:
            tot = wts[i] + wts[i + 1]
            v = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            vals[i] = v
            wts[i] = tot
            idx[i] += idx[i + 1]
            del vals[i + 1], wts[i + 1], idx[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(len(y))
    for v, ii in zip(vals, idx):
        for j in ii:
            out[j] = v
    return out


class StreamingFrontierTeacher:
    """Continuous-difficulty frontier teacher.

    Args:
      n_rollouts: group size N (sets the utility's difficulty band).
      bandwidth: kernel bandwidth h on the difficulty axis (default 0.08 —
        ~12 effective bins over [0,1]; the resolution/variance knob).
      window: max observations kept (sliding).
      decay: per-observation age decay applied at query time.
      gamma, floor: as in the discrete teacher.
      monotone: apply isotonic projection (difficulty orders pass rates).
      grid: number of difficulty grid points for sampling.
    """

    def __init__(self, n_rollouts: int = 16, *, bandwidth: float = 0.08,
                 window: int = 512, decay: float = 0.995, gamma: float = 1.0,
                 floor: float = 0.1, monotone: bool = True, grid: int = 64,
                 seed: int = 0):
        self.n_rollouts = n_rollouts
        self.h = bandwidth
        self.window = window
        self.decay = decay
        self.gamma = gamma
        self.floor = floor
        self.monotone = monotone
        self.grid = np.linspace(0.0, 1.0, grid)
        self.rng = np.random.default_rng(seed)
        self.obs = deque(maxlen=window)   # (d, k, n, t)
        self.t = 0

    # -- evidence ----------------------------------------------------------
    def observe(self, difficulty: float, rewards: np.ndarray) -> None:
        self.t += 1
        self.obs.append((float(difficulty), float(np.sum(rewards)),
                         float(len(rewards)), self.t))

    # -- posterior ---------------------------------------------------------
    def _pseudo_counts(self, ds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kernel-weighted (alpha, beta) at each query difficulty."""
        alpha = np.ones_like(ds)
        beta = np.ones_like(ds)
        if not self.obs:
            return alpha, beta
        d_o = np.array([o[0] for o in self.obs])
        k_o = np.array([o[1] for o in self.obs])
        n_o = np.array([o[2] for o in self.obs])
        t_o = np.array([o[3] for o in self.obs])
        age_w = self.decay ** (self.t - t_o)
        # Gaussian kernel, (grid, obs)
        K = np.exp(-0.5 * ((ds[:, None] - d_o[None, :]) / self.h) ** 2)
        alpha = 1.0 + (K * (k_o * age_w)[None, :]).sum(axis=1)
        beta = 1.0 + (K * ((n_o - k_o) * age_w)[None, :]).sum(axis=1)
        return alpha, beta

    def pass_rate_estimate(self, ds: np.ndarray) -> np.ndarray:
        a, b = self._pseudo_counts(np.asarray(ds, dtype=float))
        return a / (a + b)

    # -- sampling ----------------------------------------------------------
    def sample_difficulties(self, batch: int) -> np.ndarray:
        a, b = self._pseudo_counts(self.grid)
        p = self.rng.beta(a, b)
        if self.monotone:
            p = _pav_nonincreasing(p)   # difficulty ↑ ⇒ pass rate ↓
        u = np.maximum((1.0 - (1.0 - p) ** self.n_rollouts) - p, 0.0)
        u = u ** self.gamma
        if u.sum() <= 1e-12:
            u = np.ones_like(u)
        probs = u / u.sum()
        probs = (1 - self.floor) * probs + self.floor / len(self.grid)
        idx = self.rng.choice(len(self.grid), size=batch, p=probs)
        cell = (self.grid[1] - self.grid[0]) if len(self.grid) > 1 else 0.0
        return np.clip(self.grid[idx]
                       + self.rng.uniform(-cell / 2, cell / 2, size=batch),
                       0.0, 1.0)

    def metrics(self) -> dict:
        p = self.pass_rate_estimate(self.grid)
        return {"teacher/frontier_d": float(self.grid[int(np.argmax(
                    np.maximum((1 - (1 - p) ** self.n_rollouts) - p, 0)))]),
                "teacher/mastered_below_d": float(
                    self.grid[np.searchsorted(-p, -0.9)]
                    if (p > 0.9).any() else 0.0)}
