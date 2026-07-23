"""Regression anchor: the framework must reproduce REPORT.md numbers on the
skill-chain testbed (uniform+maxrl AUC ~0.65; full stack ~0.88-0.89).

Run: python3 -m frontier_rl.examples.run_skill_chain
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import numpy as np

from frontier_rl import FrontierTrainer, TrainerConfig, FrontierTeacher
from frontier_rl.adapters.skill_chain import SkillChainSpace


def run(config: TrainerConfig, uniform: bool, seed: int, steps: int = 400):
    env = SkillChainSpace(seed=seed)
    teacher = FrontierTeacher(env.n_tasks, config.n_rollouts, seed=seed + 1000,
                              decay=config.teacher_decay, floor=config.teacher_floor,
                              gamma=config.teacher_gamma)
    if uniform:
        teacher.distribution = lambda: np.full(env.n_tasks, 1.0 / env.n_tasks)
    trainer = FrontierTrainer(env, env, config, teacher=teacher)
    curve = []
    def on_eval(i):
        curve.append(env.true_pass_rates().mean())
    trainer.train(steps, on_eval=on_eval, eval_every=10)
    return np.array(curve)


def main():
    cases = [
        ("uniform + maxrl (no hindsight)", dict(hindsight=False), True),
        ("teacher + maxrl (no hindsight)", dict(hindsight=False), False),
        ("teacher gamma=4 + hindsight (full stack)",
         dict(hindsight=True, teacher_gamma=4.0), False),
    ]
    for label, over, uniform in cases:
        aucs, finals = [], []
        for seed in range(5):
            cfg = TrainerConfig(seed=seed, **over)
            h = run(cfg, uniform, seed)
            aucs.append(h.mean()); finals.append(h[-1])
        print(f"{label:44s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f}", flush=True)
    print("\nexpected (REPORT.md): uniform ~0.65 | teacher ~0.72-0.73 | full ~0.88-0.89")


if __name__ == "__main__":
    main()
