"""Hindsight relabeling for dead MaxRL groups on the skill-chain testbed.

Idea: when a group for task (chain c, level l) comes back all-fail, each
rollout still has a *correct prefix* of length j < l (the first j skills were
executed correctly).  Under the chain structure, that rollout IS a success
for the nested task (c, j) — the same trajectory, truncated.  We relabel the
best-prefix rollouts as successes of the deepest prefix task j* achieved by
the group and apply MaxRL's success-conditioned weights there.

This is Hindsight Experience Replay mapped into MaxRL: the estimator only
learns from successes, and relabeling manufactures successes on the frontier
from compute that would otherwise be dropped (K=0).

Caveat re bias: the relabeled group is conditioned on the achieved outcome,
so it is NOT an unbiased estimator of task-j*'s truncated-ML gradient; it is
an auxiliary imitation-style term (like HER).  We test empirically whether
it helps or hurts.
"""

from __future__ import annotations

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import UniformTeacher, AdvMassTeacher

TEACHERS = {"uniform": UniformTeacher, "advmass": AdvMassTeacher}


def correct_prefix_len(actions_row: np.ndarray) -> int:
    """Number of leading correct actions (action 0 is always correct)."""
    wrong = np.nonzero(actions_row != 0)[0]
    return int(wrong[0]) if len(wrong) else len(actions_row)


def run(teacher_name: str, seed: int, steps: int = 400, hindsight: bool = True,
        tasks_per_batch: int = 8, n_rollouts: int = 16, lr: float = 0.5,
        hindsight_scale: float = 1.0):
    env = SkillChainEnv(seed=seed)
    kwargs = {"n_rollouts": n_rollouts} if teacher_name == "advmass" else {}
    teacher = TEACHERS[teacher_name](env.n_tasks, seed=seed + 1000, **kwargs)

    # map (chain, level) -> task_id for relabeling
    level_of = env.task_level
    chain_of = [t // env.n_levels for t in range(env.n_tasks)]
    task_of = {}
    for tid in range(env.n_tasks):
        task_of[(chain_of[tid], level_of[tid])] = tid

    history = []
    relabeled_groups = 0
    for step in range(steps):
        task_ids = teacher.sample_tasks(tasks_per_batch)
        for t in task_ids:
            t = int(t)
            actions, rewards = env.rollout(t, n_rollouts)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, lr)
                continue
            if not hindsight:
                continue
            # dead group: relabel to the deepest prefix level achieved
            prefixes = np.array([correct_prefix_len(a) for a in actions])
            jstar = int(prefixes.max())
            if jstar < 1:
                continue
            target = task_of[(chain_of[t], jstar)]
            # rollouts achieving the full prefix are successes of the nested
            # task; the others are failures of it (their truncated actions
            # are valid attempts at the same subtask)
            r2 = (prefixes >= jstar).astype(float)
            a2 = actions[:, :jstar]
            w2 = weights_maxrl(r2) * hindsight_scale
            if np.any(w2 != 0):
                env.apply_gradient(target, a2, w2, lr)
                relabeled_groups += 1

        if step % 10 == 0 or step == steps - 1:
            p = env.true_pass_rates()
            levels = np.array(env.task_level)
            frontier = 0
            for l in range(1, env.n_levels + 1):
                if p[levels == l].mean() > 0.5:
                    frontier = l
            history.append({"step": step, "mean_pass": float(p.mean()),
                            "frontier": frontier,
                            "relabeled": relabeled_groups})
    return history


def main():
    import sys
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for teacher in ["uniform", "advmass"]:
        for hs in [False, True]:
            finals, aucs, rels = [], [], []
            for seed in range(5):
                h = run(teacher, seed, steps=steps, hindsight=hs)
                s = np.array([x["step"] for x in h])
                mp = np.array([x["mean_pass"] for x in h])
                finals.append(mp[-1])
                aucs.append(float(np.trapz(mp, s) / (s[-1] - s[0])))
                rels.append(h[-1]["relabeled"])
            tag = f"{teacher}+maxrl" + ("+hindsight" if hs else "")
            print(f"{tag:32s} final={np.mean(finals):.3f}(±{np.std(finals):.3f}) "
                  f"AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
                  f"relabeled_groups={np.mean(rels):.0f}", flush=True)


if __name__ == "__main__":
    main()
