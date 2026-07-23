"""Gridworld reach task — the goal-conditioned robotics pattern.

An agent starts at the center of a (2R+1)² grid and must reach a goal cell
within a step budget; tasks = goal cells binned by Chebyshev ring distance
(ring r ∈ 1..R), sampled uniformly within the ring.  The policy is a
goal-conditioned softmax over 4 moves with a tabular feature (relative goal
direction × distance bucket) — deliberately simple so the framework, not
the function approximator, is what's under test.

Hindsight: a failed episode's final cell IS a reached goal; relabel the
group to the ring of the farthest final cell, marking rollouts that ended
in that ring as successes (exact under the env's own verifier — P6's
contract).

This mirrors Fetch-style sparse-reward reach tasks: replace `step()` with a
mujoco/bullet sim and `ring of final cell` with `distance-band of achieved
end-effector pose` and the adapter carries over unchanged.
"""

from __future__ import annotations

import numpy as np

from frontier_rl.interfaces import GroupResult

MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])


class GridReachSpace:
    def __init__(self, radius: int = 8, step_budget_factor: float = 2.0,
                 lr: float = 0.3, seed: int = 0):
        self.R = radius
        self.step_budget_factor = step_budget_factor
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # features: (goal direction octant x distance bucket) -> move logits
        self.n_buckets = radius
        self.theta = np.zeros((8, self.n_buckets, 4))

    @property
    def n_tasks(self) -> int:
        return self.R  # task r = "reach ring r+1"

    # ---- internals ----
    @staticmethod
    def _octant(dx, dy):
        ang = np.arctan2(dy, dx)
        return int(((ang + np.pi) / (2 * np.pi / 8)) % 8)

    def _feat(self, pos, goal):
        dx, dy = goal[0] - pos[0], goal[1] - pos[1]
        d = max(abs(dx), abs(dy))
        return self._octant(dx, dy), min(d - 1, self.n_buckets - 1) if d > 0 else 0

    def _policy_probs(self, feat):
        z = self.theta[feat] - self.theta[feat].max()
        e = np.exp(z)
        return e / e.sum()

    def _sample_goal(self, ring: int):
        while True:
            g = self.rng.integers(-ring, ring + 1, size=2)
            if max(abs(g[0]), abs(g[1])) == ring:
                return g

    def _episode(self, ring: int):
        goal = self._sample_goal(ring)
        pos = np.zeros(2, dtype=int)
        budget = int(self.step_budget_factor * ring) + 2
        positions, actions = [], []  # raw trajectory; features recomputed in update
        for _ in range(budget):
            feat = self._feat(pos, goal)
            p = self._policy_probs(feat)
            a = int(self.rng.choice(4, p=p))
            positions.append(pos.copy())
            actions.append(a)
            pos = pos + MOVES[a]
            if np.array_equal(pos, goal):
                return positions, actions, pos, goal, True
        return positions, actions, pos, goal, False

    # ---- TaskSpace ----
    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        ring = task_id + 1
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            positions, actions, final_pos, goal, ok = self._episode(ring)
            trajs.append({"positions": positions, "actions": actions, "goal": goal})
            rewards.append(float(ok))
            infos.append({"final_pos": final_pos, "goal": goal})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def relabel(self, group: GroupResult):
        """Relabel to the farthest reached ring, REWRITING the goal in each
        successful trajectory to its own final cell (interfaces.py contract 2:
        goal-relative features must be recomputed against the achieved goal,
        else goal-A-conditioned actions get credited to goal B)."""
        rings = [max(abs(i["final_pos"][0]), abs(i["final_pos"][1]))
                 for i in group.infos]
        best = max((r for r in rings if 1 <= r <= self.R), default=0)
        if best < 1:
            return None
        new_rewards = np.array([1.0 if r == best else 0.0 for r in rings])
        new_trajs = []
        for traj, info, r in zip(group.trajectories, group.infos, rings):
            if r == best:
                nt = dict(traj)
                nt["goal"] = info["final_pos"]   # achieved goal
                new_trajs.append(nt)
            else:
                new_trajs.append(traj)
        return best - 1, new_rewards, new_trajs

    # ---- Policy ----
    def update(self, task_id: int, trajectories, weights) -> None:
        for traj, w in zip(trajectories, np.asarray(weights)):
            if w == 0.0:
                continue
            goal = traj["goal"]
            for pos, a in zip(traj["positions"], traj["actions"]):
                feat = self._feat(pos, goal)
                p = self._policy_probs(feat)
                g = -p
                g[a] += 1.0
                self.theta[feat] += self.lr * w * g

    # ---- eval ----
    def eval_pass_rates(self, n: int = 64) -> np.ndarray:
        out = []
        for task in range(self.n_tasks):
            g = self.rollout_group(task, n)
            out.append(float(np.mean(g.rewards)))
        return np.array(out)
