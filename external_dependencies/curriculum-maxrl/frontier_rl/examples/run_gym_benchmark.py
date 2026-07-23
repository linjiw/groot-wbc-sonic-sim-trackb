"""Gymnasium classic-control benchmark of the frontier_rl schedule.

MountainCar (hard exploration, goal-conditioned positional curriculum) and
CartPole (survival-duration curriculum). uniform vs teacher vs
teacher+hindsight at matched group budgets.

Headline metric per env: mean pass rate over all task bins (AUC + final),
plus the hardest bin's final pass rate (did the curriculum reach the top?).

Run: python3 frontier_rl/examples/run_gym_benchmark.py [--quick]
"""

from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.gym_classic import MountainCarSpace, CartPoleSurviveSpace


def run(env_cls, hindsight, uniform, seed, steps, n_rollouts=16, tasks_per_step=4,
        gamma=4.0, eval_every=None, eval_n=16):
    env = env_cls(seed=seed)
    cfg = TrainerConfig(n_rollouts=n_rollouts, tasks_per_step=tasks_per_step,
                        hindsight=hindsight, teacher_gamma=gamma, seed=seed)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              gamma=cfg.teacher_gamma)
    if uniform:
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve, hard = [], []
    ee = eval_every or max(steps // 8, 1)
    def on_eval(i):
        p = env.eval_pass_rates(n=eval_n)
        curve.append(p.mean())
        hard.append(p[-1])
    trainer.train(steps, on_eval=on_eval, eval_every=ee)
    return np.array(curve), np.array(hard)


def bench(env_cls, name, steps, seeds, eval_n):
    print(f"\n=== {name} ({steps} steps x {seeds} seeds) ===", flush=True)
    for label, hs, uni in [("uniform + maxrl", False, True),
                           ("teacher + maxrl", False, False),
                           ("teacher + maxrl + hindsight", True, False)]:
        aucs, finals, hards = [], [], []
        for seed in range(seeds):
            c, h = run(env_cls, hs, uni, seed, steps, eval_n=eval_n)
            aucs.append(c.mean()); finals.append(c[-1]); hards.append(h[-1])
        print(f"  {label:29s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f} hardest-bin={np.mean(hards):.3f}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds = 2 if args.quick else 3
    bench(MountainCarSpace, "MountainCar positional curriculum",
          steps=60 if args.quick else 120, seeds=seeds, eval_n=12)
    bench(CartPoleSurviveSpace, "CartPole survival curriculum",
          steps=40 if args.quick else 80, seeds=seeds, eval_n=10)


if __name__ == "__main__":
    main()
