"""P2: scaled MountainCar benchmark — can the full stack reach the flag?

600 steps (5x the demo), 5 seeds, plus a gamma ablation. The question that
matters: does any method get the HARDEST bin (the actual flag at x>=0.5)
off zero? MountainCar's flag is unreachable by undirected exploration, so
this is the external-env analog of the frontier-heavy categorical result.

Run: python3 frontier_rl/examples/run_mountaincar_scaled.py
"""

from __future__ import annotations

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.gym_classic import MountainCarSpace


def run(hindsight, uniform, gamma, seed, steps=600, eval_every=75, eval_n=16):
    env = MountainCarSpace(seed=seed)
    cfg = TrainerConfig(n_rollouts=16, tasks_per_step=4, hindsight=hindsight,
                        teacher_gamma=gamma, seed=seed)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              gamma=gamma)
    if uniform:
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve, hard = [], []
    def on_eval(i):
        p = env.eval_pass_rates(n=eval_n)
        curve.append(float(p.mean()))
        hard.append(float(p[-1]))
    trainer.train(steps, on_eval=on_eval, eval_every=eval_every)
    return np.array(curve), np.array(hard)


def main():
    results = {}
    cases = [
        ("uniform+maxrl",            dict(hindsight=False, uniform=True,  gamma=1.0)),
        ("teacher(g1)+maxrl",        dict(hindsight=False, uniform=False, gamma=1.0)),
        ("teacher(g4)+maxrl",        dict(hindsight=False, uniform=False, gamma=4.0)),
        ("teacher(g1)+maxrl+hs",     dict(hindsight=True,  uniform=False, gamma=1.0)),
        ("teacher(g4)+maxrl+hs",     dict(hindsight=True,  uniform=False, gamma=4.0)),
    ]
    for label, kw in cases:
        aucs, finals, flags = [], [], []
        for seed in range(5):
            c, h = run(seed=seed, **kw)
            aucs.append(c.mean()); finals.append(c[-1]); flags.append(h[-1])
        results[label] = {"auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
                          "final": float(np.mean(finals)),
                          "flag_bin": float(np.mean(flags))}
        print(f"{label:24s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f} FLAG-bin={np.mean(flags):.3f}", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "mountaincar_scaled.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
