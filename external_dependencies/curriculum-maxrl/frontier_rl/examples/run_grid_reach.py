"""Goal-conditioned gridworld reach — the robotics-style demo.

Compares uniform / teacher / teacher+hindsight at equal group budgets.
Run: python3 -m frontier_rl.examples.run_grid_reach
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.grid_reach import GridReachSpace


def run(hindsight: bool, uniform: bool, seed: int, steps: int = 150):
    env = GridReachSpace(radius=8, seed=seed)
    cfg = TrainerConfig(n_rollouts=16, tasks_per_step=4, hindsight=hindsight,
                        teacher_gamma=4.0, seed=seed)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              gamma=cfg.teacher_gamma)
    if uniform:
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve = []
    trainer.train(steps, on_eval=lambda i: curve.append(env.eval_pass_rates(n=32).mean()),
                  eval_every=15)
    return np.array(curve)


def main():
    for label, hs, uni in [("uniform + maxrl", False, True),
                           ("teacher + maxrl", False, False),
                           ("teacher + maxrl + hindsight", True, False)]:
        aucs, finals = [], []
        for seed in range(3):
            h = run(hs, uni, seed)
            aucs.append(h.mean()); finals.append(h[-1])
        print(f"{label:30s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f}", flush=True)


if __name__ == "__main__":
    main()
