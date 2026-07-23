"""Export per-step training curves + teacher-distribution snapshots for the
website's real-data charts.

For each env x method: eval curve (mean pass over bins) every few steps, and
for teacher methods a snapshot of the sampling distribution + posterior p̂ —
so the site can show performance rising WHILE the curriculum visibly walks.

Output: docs/curves.json  (kept small: ~3 seeds x 4 envs x 3 methods)

Run: python3 frontier_rl/examples/export_curves.py
"""

from __future__ import annotations

import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace
from frontier_rl.adapters.grid_reach import GridReachSpace
from frontier_rl.adapters.gym_classic import MountainCarSpace, CartPoleSurviveSpace

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "docs", "curves.json")


def run_one(env_cls, env_kw, method, seed, steps, eval_every, eval_n, gamma):
    env = env_cls(seed=seed, **env_kw)
    cfg = TrainerConfig(n_rollouts=16, tasks_per_step=4,
                        hindsight=(method == "hindsight"),
                        teacher_gamma=gamma, seed=seed)
    teacher = FrontierTeacher(env.n_tasks, cfg.n_rollouts, seed=seed + 1000,
                              gamma=gamma)
    if method == "uniform":
        teacher.distribution = lambda n=env.n_tasks: np.full(n, 1.0 / n)
    trainer = FrontierTrainer(env, env, cfg, teacher=teacher)
    curve, dists, phats = [], [], []
    def on_eval(i):
        if hasattr(env, "true_pass_rates"):
            curve.append(float(env.true_pass_rates().mean()))
        else:
            curve.append(float(env.eval_pass_rates(n=eval_n).mean()))
        if method != "uniform":
            dists.append(np.round(teacher.distribution(), 4).tolist())
            phats.append(np.round(teacher.pass_rate_estimates(), 3).tolist())
    trainer.train(steps, on_eval=on_eval, eval_every=eval_every)
    return curve, dists, phats


ENVS = [
    ("skill_chain", SkillChainSpace, {}, 400, 20, 0, 4.0, 3),
    ("grid_reach", GridReachSpace, {"radius": 8}, 150, 10, 24, 4.0, 3),
    ("mountaincar", MountainCarSpace, {}, 120, 10, 12, 4.0, 2),
    ("cartpole", CartPoleSurviveSpace, {}, 80, 8, 10, 4.0, 2),
]


def main():
    out = {}
    for name, cls, kw, steps, ee, en, gamma, seeds in ENVS:
        out[name] = {"eval_every": ee, "methods": {}}
        for method in ["uniform", "teacher", "hindsight"]:
            curves, dists, phats = [], None, None
            for seed in range(seeds):
                c, d, p = run_one(cls, kw, method, seed, steps, ee, en, gamma)
                curves.append(c)
                if seed == 0 and method != "uniform":
                    dists, phats = d, p
            L = min(len(c) for c in curves)
            arr = np.array([c[:L] for c in curves])
            entry = {"mean": np.round(arr.mean(0), 4).tolist(),
                     "lo": np.round(arr.min(0), 4).tolist(),
                     "hi": np.round(arr.max(0), 4).tolist()}
            if dists:
                entry["dist"] = dists[:L]
                entry["phat"] = phats[:L]
            out[name]["methods"][method] = entry
            print(f"{name}/{method}: final {entry['mean'][-1]:.3f}", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f)
    print("wrote", OUT, os.path.getsize(OUT)//1024, "KB")


if __name__ == "__main__":
    main()
