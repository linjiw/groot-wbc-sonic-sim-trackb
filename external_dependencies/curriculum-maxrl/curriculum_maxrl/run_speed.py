"""Learning-speed comparison: AUC of mean_pass and steps-to-frontier-12,
plus fixed-N vs adaptive-N for MaxRL."""

from __future__ import annotations

import json
import numpy as np
from run_experiment import run, TEACHERS, ESTIMATORS


def summarize(hist):
    steps = np.array([h["step"] for h in hist])
    mp = np.array([h["mean_pass"] for h in hist])
    fr = np.array([h["frontier"] for h in hist])
    auc = float(np.trapz(mp, steps) / (steps[-1] - steps[0]))
    hit = steps[fr >= 12]
    return auc, (int(hit[0]) if len(hit) else -1)


def main():
    combos = [
        ("uniform", "grpo", False), ("uniform", "maxrl", False),
        ("zpd", "grpo", False), ("zpd", "maxrl", False),
        ("alp", "maxrl", False),
        ("maxrl_frontier", "grpo", False), ("maxrl_frontier", "maxrl", False),
        ("maxrl_frontier", "maxrl", True),   # adaptive rollout allocation
        ("zpd", "maxrl", True),
    ]
    out = {}
    for teacher, est, adaptive in combos:
        aucs, hits = [], []
        for seed in range(5):
            hist = run(teacher, est, seed, steps=400, adaptive_n=adaptive)
            a, h = summarize(hist)
            aucs.append(a)
            hits.append(h if h >= 0 else 400)
        key = f"{teacher}+{est}" + ("+adaptiveN" if adaptive else "")
        out[key] = {
            "auc": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "steps_to_frontier12": float(np.mean(hits)),
        }
        print(f"{key:34s} AUC={out[key]['auc']:.3f} (±{out[key]['auc_std']:.3f}) "
              f"steps_to_frontier12={out[key]['steps_to_frontier12']:.0f}", flush=True)
    with open("results_speed.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
