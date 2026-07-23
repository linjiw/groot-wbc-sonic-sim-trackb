"""Gymnasium classic-control adapters with real difficulty curricula.

Two tasks, chosen to test different claims:

MountainCarSpace — THE hard-exploration classic. Sparse binary success
  ("reach x >= 0.5 within budget") has pass rate ~0 for a random policy, so
  plain sparse RL flatlines: the real-world replica of our V5 frontier-heavy
  regime where uniform/DAPO/plain-teacher all scored exactly 0.  Curriculum
  axis: target position x* from just-right-of-valley out to the flag at 0.5
  (task j = "reach x >= x*_j").  Hindsight: a failed rollout's max x reached
  IS a success for the bin containing it — exact under the env's own
  dynamics (contract 1) — and the policy is conditioned on the task bin, so
  relabeling rewrites the conditioning to the achieved bin (contract 2).

CartPoleSurviveSpace — control (non-goal-conditioned): task j = "survive
  >= T_j steps".  Relabel = longest survival bin actually reached.  Tests
  that the schedule behaves on a task where difficulty is a scalar
  threshold rather than a spatial goal.

Both use REINFORCE with a tiny tile-coded linear softmax policy — weak on
purpose: the *schedule* is what's under test, and a policy strong enough to
solve the task instantly would hide scheduling differences.
"""

from __future__ import annotations

import numpy as np

from frontier_rl.interfaces import GroupResult

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("pip install gymnasium") from e


# --------------------------------------------------------------------------
# small tile-coded linear policy, shared by both adapters
# --------------------------------------------------------------------------
class TilePolicy:
    """Softmax over actions; features = one-hot tile of (obs bins x task bin)."""

    def __init__(self, n_bins_per_dim, obs_low, obs_high, n_tasks, n_actions,
                 lr=0.15, seed=0):
        self.bins = np.asarray(n_bins_per_dim)
        self.low = np.asarray(obs_low, dtype=float)
        self.high = np.asarray(obs_high, dtype=float)
        self.n_tasks = n_tasks
        self.n_actions = n_actions
        self.lr = lr
        n_tiles = int(np.prod(self.bins)) * n_tasks
        self.theta = np.zeros((n_tiles, n_actions))
        self.rng = np.random.default_rng(seed)

    def tile(self, obs, task_id):
        frac = np.clip((np.asarray(obs) - self.low) / (self.high - self.low),
                       0, 1 - 1e-9)
        idx = (frac * self.bins).astype(int)
        flat = 0
        for i, b in zip(idx, self.bins):
            flat = flat * b + i
        return int(flat) * self.n_tasks + task_id

    def probs(self, tile):
        z = self.theta[tile] - self.theta[tile].max()
        e = np.exp(z)
        return e / e.sum()

    def act(self, obs, task_id):
        t = self.tile(obs, task_id)
        return int(self.rng.choice(self.n_actions, p=self.probs(t))), t

    # Policy protocol: trajectories are lists of (tile, action);
    # task-conditioning is inside the tile, so relabeled trajectories arrive
    # already rewritten by the adapter.
    def update(self, task_id, trajectories, weights):
        for traj, w in zip(trajectories, np.asarray(weights)):
            if w == 0.0:
                continue
            for tile, a in traj:
                p = self.probs(tile)
                g = -p
                g[a] += 1.0
                self.theta[tile] += self.lr * w * g


# --------------------------------------------------------------------------
# MountainCar: positional curriculum
# --------------------------------------------------------------------------
class MountainCarSpace:
    """Task j = "reach x >= targets[j]"; targets walk from valley to flag."""

    def __init__(self, n_tasks: int = 10, max_steps: int = 200, seed: int = 0,
                 lr: float = 0.15):
        self.env = gym.make("MountainCar-v0").unwrapped
        self.env.reset(seed=seed)
        self._n_tasks = n_tasks
        self.max_steps = max_steps
        # valley bottom ~ -0.5; flag at 0.5. Space targets in between.
        self.targets = np.linspace(-0.35, 0.5, n_tasks)
        self.policy = TilePolicy(n_bins_per_dim=[12, 12],
                                 obs_low=[-1.2, -0.07], obs_high=[0.6, 0.07],
                                 n_tasks=n_tasks, n_actions=3, lr=lr, seed=seed)
        self.rng = np.random.default_rng(seed + 1)

    @property
    def n_tasks(self):
        return self._n_tasks

    def _episode(self, task_id):
        obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        steps, max_x = [], -1.2
        target = self.targets[task_id]
        for _ in range(self.max_steps):
            a, tile = self.policy.act(obs, task_id)
            steps.append((obs.copy(), a))
            obs, _, term, trunc, _ = self.env.step(a)
            max_x = max(max_x, float(obs[0]))
            if obs[0] >= target:
                return steps, max_x, True
            if term or trunc:
                break
        return steps, max_x, False

    def rollout_group(self, task_id, n_rollouts):
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            steps, max_x, ok = self._episode(task_id)
            # store raw (obs, action); tiles are recomputed per credited task
            trajs.append(steps)
            rewards.append(float(ok))
            infos.append({"max_x": max_x})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def _bin_of_x(self, x):
        """Largest task bin whose target is <= x (i.e. truly achieved)."""
        reached = np.nonzero(self.targets <= x)[0]
        return int(reached[-1]) if len(reached) else -1

    def relabel(self, group):
        bins = [self._bin_of_x(i["max_x"]) for i in group.infos]
        best = max(bins)
        if best < 0:
            return None
        new_rewards = np.array([1.0 if b == best else 0.0 for b in bins])
        # conditioning rewrite: recompute tiles against the achieved bin
        new_trajs = [[(self.policy.tile(obs, best), a) for obs, a in traj]
                     for traj in group.trajectories]
        return best, new_rewards, new_trajs

    # Policy protocol shim: convert raw (obs, action) to tiles for the
    # *credited* task, then delegate (live groups come through here).
    def update(self, task_id, trajectories, weights):
        tiled = []
        for traj in trajectories:
            if traj and isinstance(traj[0][0], (int, np.integer)):
                tiled.append(traj)          # already tiled (relabeled path)
            else:
                tiled.append([(self.policy.tile(obs, task_id), a)
                              for obs, a in traj])
        self.policy.update(task_id, tiled, weights)

    def eval_pass_rates(self, n=24):
        return np.array([np.mean(self.rollout_group(t, n).rewards)
                         for t in range(self._n_tasks)])


# --------------------------------------------------------------------------
# CartPole: survival-duration curriculum (non-goal-conditioned control)
# --------------------------------------------------------------------------
class CartPoleSurviveSpace:
    """Task j = "survive >= durations[j] steps"."""

    def __init__(self, n_tasks: int = 8, seed: int = 0, lr: float = 0.1):
        self.env = gym.make("CartPole-v1").unwrapped
        self.env.reset(seed=seed)
        self._n_tasks = n_tasks
        self.durations = np.unique(np.geomspace(20, 400, n_tasks).astype(int))
        self._n_tasks = len(self.durations)
        self.policy = TilePolicy(n_bins_per_dim=[6, 6, 8, 8],
                                 obs_low=[-2.4, -3.0, -0.21, -3.0],
                                 obs_high=[2.4, 3.0, 0.21, 3.0],
                                 n_tasks=self._n_tasks, n_actions=2,
                                 lr=lr, seed=seed)
        self.rng = np.random.default_rng(seed + 1)

    @property
    def n_tasks(self):
        return self._n_tasks

    def rollout_group(self, task_id, n_rollouts):
        need = int(self.durations[task_id])
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            obs, _ = self.env.reset(seed=int(self.rng.integers(1 << 30)))
            steps, alive = [], 0
            for _ in range(need):
                a, tile = self.policy.act(obs, task_id)
                steps.append((obs.copy(), a))
                obs, _, term, trunc, _ = self.env.step(a)
                alive += 1
                if term or trunc:
                    break
            trajs.append(steps)
            rewards.append(1.0 if alive >= need else 0.0)
            infos.append({"alive": alive})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def relabel(self, group):
        alive = np.array([i["alive"] for i in group.infos])
        reached = [np.nonzero(self.durations <= a)[0] for a in alive]
        bins = [int(r[-1]) if len(r) else -1 for r in reached]
        best = max(bins)
        if best < 0:
            return None
        new_rewards = np.array([1.0 if b >= best else 0.0 for b in bins])
        new_trajs = [[(self.policy.tile(obs, best), a) for obs, a in traj]
                     for traj in group.trajectories]
        return best, new_rewards, new_trajs

    def update(self, task_id, trajectories, weights):
        tiled = []
        for traj in trajectories:
            if traj and isinstance(traj[0][0], (int, np.integer)):
                tiled.append(traj)
            else:
                tiled.append([(self.policy.tile(obs, task_id), a)
                              for obs, a in traj])
        self.policy.update(task_id, tiled, weights)

    def eval_pass_rates(self, n=16):
        return np.array([np.mean(self.rollout_group(t, n).rewards)
                         for t in range(self._n_tasks)])
