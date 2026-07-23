"""Head-to-head vs DAPO-style dynamic sampling + regime map.

Study 1 (DAPO comparison, matched GENERATION budget):
  DAPO's dynamic sampling fixes the zero-gradient problem by oversampling:
  keep drawing prompts, discard degenerate groups (all-fail or all-pass),
  until the batch holds `tasks_per_batch` live groups.  Cost model: every
  generated group (kept or discarded) costs its rollouts.  Our teacher
  instead *predicts* liveness and rarely draws dead groups.  Both are run
  to the same total-group budget so compute is equal.

Study 2 (regime map):
  Where does teacher+hindsight win biggest?  Vary the task-distribution
  shape by choosing which levels exist in the pool:
    easy-heavy   : levels 1..6  of 12 (most tasks already learnable)
    balanced     : levels 1..12 (standard)
    frontier-heavy: levels 5..12 (most tasks beyond initial reach)
  Compare uniform+maxrl, dapo+maxrl, teacher+maxrl, teacher+maxrl+hindsight
  at matched generation budget in each regime.
"""

from __future__ import annotations

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import AdvMassTeacher, UniformTeacher
from run_hindsight import correct_prefix_len


def run(method: str, seed: int, total_groups: int = 3200, n_rollouts: int = 16,
        lr: float = 0.5, level_range=None, eval_every: int = 400):
    """One run with a fixed generation budget (total groups, live or dead).

    level_range: (lo, hi) 1-based inclusive level filter for the task pool.
    """
    env = SkillChainEnv(seed=seed)
    levels = np.array(env.task_level)
    if level_range is None:
        pool = np.arange(env.n_tasks)
    else:
        lo, hi = level_range
        pool = np.array([t for t in range(env.n_tasks) if lo <= levels[t] <= hi])

    teacher = AdvMassTeacher(len(pool), seed=seed + 1000, n_rollouts=n_rollouts)
    rng = np.random.default_rng(seed + 5)
    chain_len = env.n_levels

    hist, used = [], 0
    while used < total_groups:
        if method.startswith("dapo"):
            # dynamic sampling: draw until a live group appears (paying for
            # every draw); DAPO also discards all-pass groups (0<K<N filter)
            for _ in range(64):
                t = int(pool[rng.integers(len(pool))])
                actions, rewards = env.rollout(t, n_rollouts)
                used += 1
                k = rewards.sum()
                if 0 < k < n_rollouts:
                    env.apply_gradient(t, actions, weights_maxrl(rewards), lr)
                    break
                if used >= total_groups:
                    break
        elif method.startswith("uniform"):
            t = int(pool[rng.integers(len(pool))])
            actions, rewards = env.rollout(t, n_rollouts)
            used += 1
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
        else:  # teacher variants
            i = int(teacher.sample_tasks(1)[0])
            t = int(pool[i])
            actions, rewards = env.rollout(t, n_rollouts)
            used += 1
            teacher.observe(i, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
            elif method.endswith("hindsight"):
                prefixes = np.array([correct_prefix_len(a) for a in actions])
                j = int(prefixes.max())
                if j >= 1:
                    target = (t // chain_len) * chain_len + (j - 1)
                    r2 = (prefixes >= j).astype(float)
                    w2 = weights_maxrl(r2)
                    if np.any(w2 != 0):
                        env.apply_gradient(target, actions[:, :j], w2, lr)
        if used % eval_every < 1:
            hist.append(env.true_pass_rates()[pool].mean())
    hist.append(env.true_pass_rates()[pool].mean())
    return np.array(hist)


METHODS = ["uniform+maxrl", "dapo+maxrl", "teacher+maxrl", "teacher+maxrl+hindsight"]
REGIMES = {"easy-heavy": (1, 6), "balanced": (1, 12), "frontier-heavy": (5, 12)}


def main():
    for regime, rng_ in REGIMES.items():
        print(f"\n--- regime: {regime} (levels {rng_[0]}..{rng_[1]}) ---", flush=True)
        for m in METHODS:
            finals, aucs = [], []
            for seed in range(5):
                h = run(m, seed, level_range=rng_)
                finals.append(h[-1])
                aucs.append(h.mean())
            print(f"  {m:26s} final={np.mean(finals):.3f}(±{np.std(finals):.3f}) "
                  f"AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f})", flush=True)


if __name__ == "__main__":
    main()
