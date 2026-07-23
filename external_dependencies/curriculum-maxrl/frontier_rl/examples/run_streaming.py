"""P4 validation: streaming (continuous-difficulty) teacher vs discrete bins.

Env: continuous-goal gridworld reach — every episode samples a FRESH goal at
real-valued ring radius r = d * R (no fixed task pool; the procedural
setting the streaming teacher exists for). Policy is the shared tile-coded
softmax (the MountainCar transfer lesson applied).

Compared at matched episode budgets, 5 seeds:
  uniform-d            : d ~ U[0,1]
  discrete-teacher     : FrontierTeacher over 8 bins (the old way)
  streaming-teacher    : StreamingFrontierTeacher, kernel posterior
  streaming+monotone=off ablation

Run: python3 frontier_rl/examples/run_streaming.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTeacher
from frontier_rl.streaming import StreamingFrontierTeacher
from frontier_rl.estimators import maxrl_weights
from frontier_rl.adapters.grid_reach import GridReachSpace, MOVES


class ContinuousGridReach(GridReachSpace):
    """Goal at continuous ring radius r ∈ [1, R] (d ∈ [0,1] → r = 1+d(R−1))."""

    def rollout_group_d(self, d: float, n_rollouts: int):
        ring = 1 + d * (self.R - 1)
        ring_int = max(1, int(round(ring)))
        return self.rollout_group(ring_int - 1, n_rollouts), ring_int

    def eval_curve(self, n: int = 24) -> np.ndarray:
        return self.eval_pass_rates(n)


def run(method: str, seed: int, steps: int = 150, n_roll: int = 16,
        tasks_per_step: int = 4):
    env = ContinuousGridReach(radius=8, seed=seed)
    if method == "discrete":
        teacher = FrontierTeacher(env.n_tasks, n_roll, seed=seed + 1000, gamma=4.0)
    elif method.startswith("streaming"):
        teacher = StreamingFrontierTeacher(n_roll, seed=seed + 1000, gamma=4.0,
                                           monotone=("nomono" not in method))
    else:
        teacher = None
    rng = np.random.default_rng(seed + 7)

    curve = []
    for step in range(steps):
        for _ in range(tasks_per_step):
            if method == "uniform":
                d = float(rng.uniform(0, 1))
            elif method == "discrete":
                b = int(teacher.sample_tasks(1)[0])
                d = (b + 0.5) / env.n_tasks
            else:
                d = float(teacher.sample_difficulties(1)[0])
            group, ring = None, None
            group, ring = ContinuousGridReach.rollout_group_d(env, d, n_roll)
            r = np.asarray(group.rewards, dtype=float)
            if method == "discrete":
                teacher.observe(ring - 1, r)
            elif method.startswith("streaming"):
                teacher.observe(d, r)
            w = maxrl_weights(r)
            if np.any(w != 0):
                env.update(group.task_id, group.trajectories, w)
                continue
            rel = env.relabel(group)
            if rel is not None:
                t2, r2, tr2 = rel
                w2 = maxrl_weights(np.asarray(r2, dtype=float))
                if np.any(w2 != 0):
                    env.update(int(t2), tr2, w2)
        if step % 15 == 0 or step == steps - 1:
            curve.append(env.eval_curve(n=16).mean())
    return np.array(curve)


def main():
    for method in ["uniform", "discrete", "streaming", "streaming-nomono"]:
        aucs, finals = [], []
        for seed in range(5):
            c = run(method, seed)
            aucs.append(c.mean()); finals.append(c[-1])
        print(f"{method:18s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f}", flush=True)


if __name__ == "__main__":
    main()
