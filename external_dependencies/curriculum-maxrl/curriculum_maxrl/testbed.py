"""Skill-chain testbed for studying curriculum x advantage-estimator interactions.

This is a minimal instantiation of the latent-generation model from the MaxRL
paper (arXiv:2602.02710): the policy samples a latent trajectory z, a
deterministic verifier checks correctness, and the trainer only observes a
binary reward.  Tasks are factored into shared *skills* so that a curriculum
matters: learning an easy task strengthens skills reused by harder tasks.

Environment
-----------
- S skills, each a categorical choice over A actions with logits theta[s] in R^A.
  Action 0 is always the "correct" action; q_s = softmax(theta[s])[0].
- A task is a set of required skills.  A rollout samples one action per
  required skill; the rollout succeeds iff every sampled action is correct.
  True pass rate: p(task) = prod_{s in task} q_s.
- Tasks are organized in chains: task (chain c, level l) requires skills
  c_1..c_l (nested).  Initial pass rate at level l is A^-l, so deep levels are
  essentially unsolvable without either curriculum or likelihood reweighting.

The score function of a rollout is the exact gradient of its log-probability,
so REINFORCE / RLOO / GRPO / MaxRL estimators can be applied verbatim.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class SkillChainEnv:
    n_chains: int = 3
    n_levels: int = 12
    n_actions: int = 10
    init_logit_correct: float = 0.0  # uniform init => q_s = 1/A
    seed: int = 0

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.n_skills = self.n_chains * self.n_levels
        # theta[s] in R^A, action 0 is correct
        self.theta = np.zeros((self.n_skills, self.n_actions), dtype=np.float64)
        self.theta[:, 0] = self.init_logit_correct
        # task -> list of required skill ids
        self.tasks: list[np.ndarray] = []
        self.task_level: list[int] = []
        for c in range(self.n_chains):
            base = c * self.n_levels
            for l in range(1, self.n_levels + 1):
                self.tasks.append(np.arange(base, base + l))
                self.task_level.append(l)
        self.n_tasks = len(self.tasks)

    # ---------- policy ----------
    def skill_probs(self) -> np.ndarray:
        """(S, A) softmax over actions for every skill."""
        z = self.theta - self.theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def true_pass_rates(self) -> np.ndarray:
        """Exact p(task) for every task."""
        q = self.skill_probs()[:, 0]
        return np.array([q[req].prod() for req in self.tasks])

    # ---------- rollouts ----------
    def rollout(self, task_id: int, n: int):
        """Sample n rollouts for a task.

        Returns:
          actions: (n, L) sampled action per required skill
          rewards: (n,) binary success
        """
        req = self.tasks[task_id]
        probs = self.skill_probs()[req]  # (L, A)
        cum = probs.cumsum(axis=1)
        u = self.rng.random((n, len(req), 1))
        actions = (u > cum[None]).sum(axis=2)  # inverse-CDF sampling
        rewards = (actions == 0).all(axis=1).astype(np.float64)
        return actions, rewards

    # ---------- learning ----------
    def apply_gradient(self, task_id: int, actions: np.ndarray, weights: np.ndarray, lr: float):
        """SGD step: theta += lr * sum_j weights[j] * grad log pi(actions_j).

        grad_theta[s] log softmax(theta[s])[a] = onehot(a) - softmax(theta[s])
        """
        req = self.tasks[task_id]
        probs = self.skill_probs()[req]  # (L, A)
        n = actions.shape[0]
        onehot = np.zeros((n, len(req), self.n_actions))
        onehot[np.arange(n)[:, None], np.arange(len(req))[None, :], actions] = 1.0
        score = onehot - probs[None]  # (n, L, A)
        grad = np.einsum("j,jla->la", weights, score)
        self.theta[req] += lr * grad
