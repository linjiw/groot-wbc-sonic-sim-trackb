"""Curriculum teachers: map per-task statistics -> a sampling distribution.

All teachers see only what a real RLVR trainer sees: the empirical rewards of
past rollouts (never the true pass rate).  Statistics are tracked with an
exponential moving average plus a Beta posterior for uncertainty.

Teachers
--------
UniformTeacher       no curriculum (baseline).
ZPDBandTeacher       samples tasks whose estimated pass rate lies in a target
                     band (ADARFT/DAPO-style difficulty targeting).
ALPTeacher           absolute-learning-progress bandit (TSCL / ALP-GMM style):
                     samples proportional to |d p̂ / dt|.
MaxRLFrontierTeacher MaxRL-native: the effective per-prompt signal of the
                     MaxRL estimator with N rollouts is
                         w_N(p) * p = 1 - (1-p)^N = pass@N,
                     i.e. the probability the group produces >=1 success and
                     is not dropped.  Utility = pass@N * (1 - p): probability
                     of receiving likelihood-weighted signal times headroom.
                     Thompson sampling over a Beta posterior supplies optimism
                     on rarely-visited tasks, so the frontier keeps advancing.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class TaskStats:
    """Online statistics for one task, from observed rollout rewards only."""
    ema_pass: float = 0.0
    ema_initialized: bool = False
    alpha_beta: tuple[float, float] = (1.0, 1.0)  # Beta posterior params
    prev_ema: float = 0.0
    visits: int = 0

    def update(self, rewards: np.ndarray, ema_w: float = 0.3):
        mean_r = float(rewards.mean())
        self.prev_ema = self.ema_pass if self.ema_initialized else mean_r
        if not self.ema_initialized:
            self.ema_pass = mean_r
            self.ema_initialized = True
        else:
            self.ema_pass = (1 - ema_w) * self.ema_pass + ema_w * mean_r
        a, b = self.alpha_beta
        # decay old evidence so the posterior tracks the moving policy
        decay = 0.9
        self.alpha_beta = (1.0 + (a - 1.0) * decay + rewards.sum(),
                           1.0 + (b - 1.0) * decay + (len(rewards) - rewards.sum()))
        self.visits += 1


class Teacher:
    def __init__(self, n_tasks: int, seed: int = 0):
        self.n_tasks = n_tasks
        self.stats = [TaskStats() for _ in range(n_tasks)]
        self.rng = np.random.default_rng(seed)

    def sample_tasks(self, batch_size: int) -> np.ndarray:
        probs = self.distribution()
        return self.rng.choice(self.n_tasks, size=batch_size, p=probs)

    def distribution(self) -> np.ndarray:
        raise NotImplementedError

    def observe(self, task_id: int, rewards: np.ndarray):
        self.stats[task_id].update(rewards)


class UniformTeacher(Teacher):
    def distribution(self) -> np.ndarray:
        return np.full(self.n_tasks, 1.0 / self.n_tasks)


class ZPDBandTeacher(Teacher):
    """Prefer tasks with estimated pass rate inside [lo, hi]."""

    def __init__(self, n_tasks: int, seed: int = 0, lo: float = 0.05, hi: float = 0.85,
                 explore_frac: float = 0.15):
        super().__init__(n_tasks, seed)
        self.lo, self.hi, self.explore_frac = lo, hi, explore_frac

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            if not st.ema_initialized:
                w[i] = 1.0  # unvisited counts as in-band (optimism)
            elif self.lo <= st.ema_pass <= self.hi:
                w[i] = 1.0
        if w.sum() == 0:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class ALPTeacher(Teacher):
    """Absolute learning progress bandit (softmax over |Δ ema_pass|)."""

    def __init__(self, n_tasks: int, seed: int = 0, explore_frac: float = 0.2):
        super().__init__(n_tasks, seed)
        self.explore_frac = explore_frac
        self.alp = np.zeros(n_tasks)

    def observe(self, task_id: int, rewards: np.ndarray):
        st = self.stats[task_id]
        st.update(rewards)
        self.alp[task_id] = 0.7 * self.alp[task_id] + 0.3 * abs(st.ema_pass - st.prev_ema)

    def distribution(self) -> np.ndarray:
        w = self.alp.copy()
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class MaxRLFrontierTeacher(Teacher):
    """MaxRL-native curriculum.

    For the MaxRL estimator with group size N, a prompt contributes gradient
    signal only when K >= 1, which happens with probability
    pass@N = 1 - (1-p)^N — exactly w_N(p) * p, the estimator's effective
    weight on the pass-rate gradient.  Utility:

        u(p) = (1 - (1-p)^N) * (1 - p)

    -> 0 for mastered tasks (p ~ 1, no headroom) and for tasks far beyond the
    frontier (p << 1/N, group almost surely dropped); maximal on the widest
    band of "hard but reachable" tasks.  p is drawn from the task's Beta
    posterior (Thompson sampling) so that uncertain tasks are probed.
    """

    def __init__(self, n_tasks: int, seed: int = 0, n_rollouts: int = 16,
                 explore_frac: float = 0.1):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.explore_frac = explore_frac

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            a, b = st.alpha_beta
            p = self.rng.beta(a, b)
            w[i] = (1.0 - (1.0 - p) ** self.n_rollouts) * (1.0 - p)
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


class AdvMassTeacher(Teacher):
    """Teacher driven by the *exact* expected MaxRL advantage mass.

    THEORY.md section 2: for a group of N rollouts, the expected total
    |advantage| the MaxRL estimator emits on a prompt with pass rate p is
    2*(pass@N(p) - p) = 2*((1-(1-p)^N) - p) — the probability the prompt is
    solvable within N attempts but not within one.  Sampling proportional to
    this quantity maximizes expected learning signal per group.  Thompson
    sampling over the Beta posterior supplies optimism.

    ``power`` (VALIDATION.md V6): sample ∝ u^power.  Proportional sampling
    (power=1) maximizes mass per draw, but learning compounds — steps on the
    highest-mass task unlock the next — so sharper concentration wins on
    chain-structured pools (γ≈4 saturates); use 1–2 on flat pools.
    """

    def __init__(self, n_tasks: int, seed: int = 0, n_rollouts: int = 16,
                 explore_frac: float = 0.1, power: float = 1.0):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.explore_frac = explore_frac
        self.power = power

    def distribution(self) -> np.ndarray:
        w = np.zeros(self.n_tasks)
        for i, st in enumerate(self.stats):
            a, b = st.alpha_beta
            p = self.rng.beta(a, b)
            w[i] = (1.0 - (1.0 - p) ** self.n_rollouts) - p
        w = np.maximum(w, 0.0) ** self.power
        if w.sum() <= 1e-12:
            w[:] = 1.0
        probs = w / w.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.explore_frac) * probs + self.explore_frac * uniform


def allocate_rollouts_greedy(p_hat: np.ndarray, total_budget: int,
                             n_min: int = 4, n_max: int = 64) -> np.ndarray:
    """Optimal rollout allocation for the advantage-mass objective.

    Maximizing sum_i [(1-(1-p_i)^{N_i}) - p_i] s.t. sum N_i = B is concave in
    each N_i, so greedy water-filling is optimal: the marginal mass of the
    (N+1)-th rollout on prompt i is p_i(1-p_i)^N — the probability that
    rollout is the group's *first success*.  Repeatedly award the next
    rollout to the prompt with the largest marginal.
    """
    m = len(p_hat)
    p = np.clip(p_hat, 1e-4, 1 - 1e-4)
    n = np.full(m, n_min, dtype=int)
    remaining = total_budget - n_min * m
    if remaining < 0:  # budget can't even cover minimums
        n = np.full(m, max(total_budget // m, 1), dtype=int)
        return n
    marginal = p * (1 - p) ** n  # value of the next rollout per prompt
    for _ in range(remaining):
        i = int(np.argmax(marginal))
        n[i] += 1
        if n[i] >= n_max:
            marginal[i] = -1.0
        else:
            marginal[i] = p[i] * (1 - p[i]) ** n[i]
    return n


def allocate_rollouts_adaptive(teacher: Teacher, task_ids: np.ndarray,
                               total_budget: int, n_min: int = 4,
                               n_max: int = 64) -> np.ndarray:
    """Compute-indexed curriculum: split a fixed rollout budget across the
    selected tasks so that harder tasks get more rollouts.

    MaxRL's truncation order equals the group size (T = N), so giving a hard
    task a larger N simultaneously (a) raises pass@N, the chance its group is
    not dropped, and (b) raises the fidelity of the ML approximation on that
    task.  Allocation ~ 1/max(p̂, 1/n_max), clipped to [n_min, n_max] and
    renormalized to the budget.
    """
    p_hat = np.array([
        max(teacher.stats[t].ema_pass if teacher.stats[t].ema_initialized else 0.5, 1e-3)
        for t in task_ids
    ])
    raw = 1.0 / np.maximum(p_hat, 1.0 / n_max)
    scaled = raw / raw.sum() * total_budget
    n = np.clip(np.round(scaled), n_min, n_max).astype(int)
    # settle rounding drift: take from easiest, give to hardest
    order = np.argsort(-raw)
    while n.sum() > total_budget:
        movable = [i for i in order[::-1] if n[i] > n_min]
        if not movable:
            break
        n[movable[0]] -= 1
    while n.sum() < total_budget:
        movable = [i for i in order if n[i] < n_max]
        if not movable:
            break
        n[movable[0]] += 1
    return n
