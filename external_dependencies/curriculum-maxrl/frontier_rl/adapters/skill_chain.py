"""The validated CPU testbed as a TaskSpace + Policy pair (regression anchor).

Identical math to curriculum_maxrl/testbed.py: chains of skills, task
(chain c, level l) requires skills c_1..c_l, exact softmax policy gradients.
Running examples/run_skill_chain.py must reproduce the REPORT.md numbers
(uniform+maxrl AUC ~0.65, full stack ~0.89 at 400 steps).
"""

from __future__ import annotations

import numpy as np

from frontier_rl.interfaces import GroupResult


class SkillChainSpace:
    """TaskSpace + Policy in one object (the policy IS the env's tables)."""

    def __init__(self, n_chains: int = 3, n_levels: int = 12,
                 n_actions: int = 10, lr: float = 0.5, seed: int = 0):
        self.n_chains, self.n_levels, self.n_actions = n_chains, n_levels, n_actions
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.theta = np.zeros((n_chains * n_levels, n_actions))
        self.tasks = []
        for c in range(n_chains):
            base = c * n_levels
            for l in range(1, n_levels + 1):
                self.tasks.append(np.arange(base, base + l))

    # ---- TaskSpace ----
    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    def _probs(self, req):
        z = self.theta[req] - self.theta[req].max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        req = self.tasks[task_id]
        probs = self._probs(req)
        cum = probs.cumsum(axis=1)
        u = self.rng.random((n_rollouts, len(req), 1))
        actions = (u > cum[None]).sum(axis=2)
        rewards = (actions == 0).all(axis=1).astype(float)
        return GroupResult(task_id, rewards, trajectories=list(actions))

    def relabel(self, group: GroupResult):
        actions = np.stack(group.trajectories)
        wrong = actions != 0
        prefixes = np.where(wrong.any(axis=1), wrong.argmax(axis=1),
                            actions.shape[1])
        j = int(prefixes.max())
        if j < 1:
            return None
        chain0 = (group.task_id // self.n_levels) * self.n_levels
        return chain0 + (j - 1), (prefixes >= j).astype(float)

    # ---- Policy ----
    def update(self, task_id: int, trajectories, weights) -> None:
        req = self.tasks[task_id]
        actions = np.stack(trajectories)[:, :len(req)]
        probs = self._probs(req)
        n = actions.shape[0]
        onehot = np.zeros((n, len(req), self.n_actions))
        onehot[np.arange(n)[:, None], np.arange(len(req))[None, :], actions] = 1.0
        grad = np.einsum("j,jla->la", np.asarray(weights), onehot - probs[None])
        self.theta[req] += self.lr * grad

    # ---- eval ----
    def true_pass_rates(self) -> np.ndarray:
        q = np.stack([self._probs(np.array([s]))[0, 0] for s in
                      range(self.theta.shape[0])])
        return np.array([q[req].prod() for req in self.tasks])
