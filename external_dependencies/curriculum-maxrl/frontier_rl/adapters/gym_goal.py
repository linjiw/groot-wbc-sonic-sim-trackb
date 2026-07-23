"""Gym/Gymnasium adapter skeleton for goal-conditioned environments.

Pattern for Fetch/HandReach-style or custom robotics sims:

  tasks    = bins over the goal space (here: distance bands; use whatever
             parameterization your curriculum should walk along)
  rollout  = reset with a goal sampled from the band, run the policy, binary
             success from the env's own `info["is_success"]`
  relabel  = HER's achieved_goal, mapped back to its bin

The framework never imports gym; this file does, lazily, so the package
stays importable without it.  The Policy protocol is satisfied by whatever
learner you have — the example below sketches a REINFORCE-style update over
stored (obs, action) trajectories; swap in your PPO/SAC update keyed by the
same weights.

Usage sketch:

    import gymnasium as gym
    env = gym.make("FetchReach-v3")             # any GoalEnv-like task
    space = GymGoalSpace(env, n_bands=10, policy=my_policy)
    trainer = FrontierTrainer(space, my_policy,
                              TrainerConfig(n_rollouts=16, hindsight=True))
    trainer.train(steps=500)
"""

from __future__ import annotations

import numpy as np

from frontier_rl.interfaces import GroupResult


class GymGoalSpace:
    """TaskSpace over distance-banded goals of a GoalEnv-style gym env.

    Requirements on `env` (the gymnasium GoalEnv convention):
      - reset(options={"goal": g}) or a settable goal (see _set_goal)
      - obs dict with "achieved_goal" / "desired_goal"
      - info["is_success"] at episode end (or compute_reward-based check)
      - env.unwrapped.goal_space_sample(band) OR pass goal_sampler=
    """

    def __init__(self, env, n_bands: int, policy, *, goal_sampler=None,
                 band_of_goal=None, max_steps: int = 50, seed: int = 0):
        self.env = env
        self._n_bands = n_bands
        self.policy = policy
        self.goal_sampler = goal_sampler
        self.band_of_goal = band_of_goal
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    @property
    def n_tasks(self) -> int:
        return self._n_bands

    # --- override points for concrete envs -------------------------------
    def _sample_goal(self, band: int):
        if self.goal_sampler is not None:
            return self.goal_sampler(band, self.rng)
        raise NotImplementedError("pass goal_sampler=(band, rng) -> goal")

    def _band(self, achieved_goal) -> int:
        if self.band_of_goal is not None:
            return self.band_of_goal(achieved_goal)
        raise NotImplementedError("pass band_of_goal=goal -> band index")

    # --- TaskSpace --------------------------------------------------------
    def rollout_group(self, task_id: int, n_rollouts: int) -> GroupResult:
        trajs, rewards, infos = [], [], []
        for _ in range(n_rollouts):
            goal = self._sample_goal(task_id)
            obs, _ = self.env.reset(options={"goal": goal})
            steps, success, achieved = [], False, None
            for _ in range(self.max_steps):
                action = self.policy.act(obs)
                steps.append((obs, action))
                obs, _, terminated, truncated, info = self.env.step(action)
                achieved = obs.get("achieved_goal") if isinstance(obs, dict) else None
                if info.get("is_success"):
                    success = True
                    break
                if terminated or truncated:
                    break
            trajs.append(steps)
            rewards.append(float(success))
            infos.append({"achieved_goal": achieved, "goal": goal})
        return GroupResult(task_id, np.array(rewards), trajs, infos)

    def relabel(self, group: GroupResult):
        """HER: credit each rollout to the band of its achieved goal."""
        bands = [self._band(i["achieved_goal"]) for i in group.infos
                 if i["achieved_goal"] is not None]
        if not bands:
            return None
        best = int(max(bands))
        if best < 0:
            return None
        new_rewards = np.array([
            1.0 if (i["achieved_goal"] is not None
                    and self._band(i["achieved_goal"]) == best) else 0.0
            for i in group.infos])
        return best, new_rewards
