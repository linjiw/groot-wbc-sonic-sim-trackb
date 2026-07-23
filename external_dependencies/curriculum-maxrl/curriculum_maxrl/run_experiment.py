"""Compare {uniform, ZPD, ALP, MaxRL-frontier} teachers x {GRPO, RLOO, MaxRL}
estimators on the skill-chain testbed.

Usage:
    python3 run_experiment.py [--steps 400] [--seeds 5] [--adaptive-n]

Metrics: mean true pass rate over all tasks, frontier level (deepest level
with p > 0.5) per chain, and fraction of tasks with p > 0.9 ("solved").
"""

from __future__ import annotations

import argparse
import json
import sys
import numpy as np

from testbed import SkillChainEnv
from estimators import (weights_grpo, weights_rloo, weights_maxrl,
                        weights_reinforce, weights_maxrl_t)
from teachers import (
    UniformTeacher, ZPDBandTeacher, ALPTeacher, MaxRLFrontierTeacher,
    AdvMassTeacher, allocate_rollouts_adaptive, allocate_rollouts_greedy,
)

ESTIMATORS = {
    "reinforce": weights_reinforce,
    "rloo": weights_rloo,
    "grpo": weights_grpo,
    "maxrl": weights_maxrl,
    "maxrl_adaptive_t": weights_maxrl,  # dispatched specially in run()
}

TEACHERS = {
    "uniform": UniformTeacher,
    "zpd": ZPDBandTeacher,
    "alp": ALPTeacher,
    "maxrl_frontier": MaxRLFrontierTeacher,
    "advmass": AdvMassTeacher,
}


def run(teacher_name: str, estimator_name: str, seed: int, steps: int,
        tasks_per_batch: int = 8, n_rollouts: int = 16, lr: float = 0.5,
        adaptive_n: bool = False):
    env = SkillChainEnv(seed=seed)
    kwargs = ({"n_rollouts": n_rollouts}
              if teacher_name in ("maxrl_frontier", "advmass") else {})
    teacher = TEACHERS[teacher_name](env.n_tasks, seed=seed + 1000, **kwargs)
    est = ESTIMATORS[estimator_name]

    history = []
    budget = tasks_per_batch * n_rollouts
    for step in range(steps):
        task_ids = teacher.sample_tasks(tasks_per_batch)
        if adaptive_n == "greedy":
            p_hat = np.array([
                teacher.stats[t].ema_pass if teacher.stats[t].ema_initialized else 0.5
                for t in task_ids])
            n_per_task = allocate_rollouts_greedy(p_hat, budget)
        elif adaptive_n:
            n_per_task = allocate_rollouts_adaptive(teacher, task_ids, budget)
        else:
            n_per_task = np.full(tasks_per_batch, n_rollouts)

        for t, n in zip(task_ids, n_per_task):
            actions, rewards = env.rollout(int(t), int(n))
            teacher.observe(int(t), rewards)
            if estimator_name == "maxrl_adaptive_t":
                st = teacher.stats[int(t)]
                p_hat = st.ema_pass if st.ema_initialized else 0.5
                T = int(np.clip(round(1.0 / max(p_hat, 1e-3)), 1, int(n)))
                w = weights_maxrl_t(rewards, T)
            else:
                w = est(rewards)
            if np.any(w != 0):
                env.apply_gradient(int(t), actions, w, lr)

        if step % 10 == 0 or step == steps - 1:
            p = env.true_pass_rates()
            levels = np.array(env.task_level)
            frontier = 0
            for l in range(1, env.n_levels + 1):
                if p[levels == l].mean() > 0.5:
                    frontier = l
            history.append({
                "step": step,
                "mean_pass": float(p.mean()),
                "solved_frac": float((p > 0.9).mean()),
                "frontier": frontier,
            })
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--adaptive-n", action="store_true")
    ap.add_argument("--out", type=str, default="results.json")
    args = ap.parse_args()

    combos = [(t, e) for t in TEACHERS for e in ESTIMATORS]
    results = {}
    for teacher_name, est_name in combos:
        finals = []
        for seed in range(args.seeds):
            hist = run(teacher_name, est_name, seed, args.steps,
                       adaptive_n=args.adaptive_n)
            finals.append(hist[-1])
        key = f"{teacher_name}+{est_name}"
        results[key] = {
            "mean_pass": float(np.mean([f["mean_pass"] for f in finals])),
            "mean_pass_std": float(np.std([f["mean_pass"] for f in finals])),
            "solved_frac": float(np.mean([f["solved_frac"] for f in finals])),
            "frontier": float(np.mean([f["frontier"] for f in finals])),
        }
        print(f"{key:28s} mean_pass={results[key]['mean_pass']:.3f} "
              f"(±{results[key]['mean_pass_std']:.3f}) "
              f"solved={results[key]['solved_frac']:.2f} "
              f"frontier={results[key]['frontier']:.1f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
