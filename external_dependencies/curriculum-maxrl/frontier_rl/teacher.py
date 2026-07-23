"""FrontierTeacher: the validated curriculum sampler.

Utility (PROOFS.md P1): u(p) = (1-(1-p)^N) - p — the exact expected
advantage mass of the MaxRL estimator per group, peaking at p* ≈ ln(N)/N.
Posterior: decayed Beta per task (decay 0.7, VALIDATION.md V2b).
Sampling: Thompson draw → u^gamma (V6: gamma tracks task-graph
connectivity — 4 for chained/shared-skill pools, 1 for flat pools) →
mix with uniform floor (P7: the floor bounds posterior staleness).

Validated defaults are the constructor defaults; every knob's provenance
is in its docstring line.
"""

from __future__ import annotations

import numpy as np


class FrontierTeacher:
    def __init__(self, n_tasks: int, n_rollouts: int = 16, *,
                 decay: float = 0.7,      # V2b: tracking > memory
                 floor: float = 0.1,      # P7/V3: staleness insurance
                 gamma: float = 1.0,      # V6: raise to ~4 on chained pools
                 seed: int = 0):
        self.n_tasks = n_tasks
        self.n_rollouts = n_rollouts
        self.decay = decay
        self.floor = floor
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(n_tasks)
        self.beta = np.ones(n_tasks)
        self.visits = np.zeros(n_tasks, dtype=np.int64)

    # -- evidence ---------------------------------------------------------
    def observe(self, task_id: int, rewards: np.ndarray) -> None:
        """Update the task's posterior from one group's binary rewards.

        Only requested-task evidence belongs here — feeding relabeled
        successes back inflates the posterior (V4 + GPU A/B/C config C).
        """
        k = float(np.sum(rewards))
        n = float(len(rewards))
        self.alpha[task_id] = 1.0 + (self.alpha[task_id] - 1.0) * self.decay + k
        self.beta[task_id] = 1.0 + (self.beta[task_id] - 1.0) * self.decay + (n - k)
        self.visits[task_id] += 1

    # -- sampling ---------------------------------------------------------
    def utility(self, p: np.ndarray) -> np.ndarray:
        return np.maximum((1.0 - (1.0 - p) ** self.n_rollouts) - p, 0.0)

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = self.utility(p) ** self.gamma
        if u.sum() <= 1e-12:
            u = np.ones(self.n_tasks)
        probs = u / u.sum()
        uniform = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1.0 - self.floor) * probs + self.floor * uniform

    def sample_tasks(self, batch: int) -> np.ndarray:
        return self.rng.choice(self.n_tasks, size=batch, p=self.distribution())

    # -- introspection / persistence --------------------------------------
    def pass_rate_estimates(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def metrics(self) -> dict:
        p = self.pass_rate_estimates()
        seen = self.visits > 0
        out = {"teacher/visited_frac": float(seen.mean())}
        if seen.any():
            out["teacher/frac_dead"] = float((p[seen] < 0.05).mean())
            out["teacher/frac_mastered"] = float((p[seen] > 0.9).mean())
        return out

    def state_dict(self) -> dict:
        return {"alpha": self.alpha.copy(), "beta": self.beta.copy(),
                "visits": self.visits.copy()}

    def load_state_dict(self, state: dict) -> None:
        self.alpha = np.asarray(state["alpha"], dtype=float)
        self.beta = np.asarray(state["beta"], dtype=float)
        self.visits = np.asarray(state["visits"], dtype=np.int64)
