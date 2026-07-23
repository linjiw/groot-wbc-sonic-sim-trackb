"""Validation suite for the curriculum-MaxRL proposals (CPU, exact gradients).

V1  Hindsight gradient fidelity: cosine similarity between the hindsight
    (relabeled dead-group) gradient and the TRUE success-conditioned ML
    gradient of the relabeled task — computable exactly on the skill chain.
    Also the unbiased fresh-group reference for scale.

V2  Oracle gap: teacher driven by TRUE pass rates (oracle) vs Thompson
    posterior vs uniform.  The oracle-Thompson gap is the price of
    exploration; oracle-uniform is the total value of the curriculum.

V3  Explore/exploit curve: AUC as a function of the uniform floor
    (exploration fraction) of the advmass teacher.

V4  Feedback-loop stability: hindsight successes feeding the teacher
    posterior (CPU analog of --hindsight-to-teacher).  Watch dead-group
    rate and AUC for runaway optimism.
"""

from __future__ import annotations

import numpy as np

from testbed import SkillChainEnv
from estimators import weights_maxrl
from teachers import AdvMassTeacher, Teacher, UniformTeacher
from run_hindsight import correct_prefix_len


# ---------------------------------------------------------------- helpers
def true_ml_gradient(env: SkillChainEnv, task_id: int) -> np.ndarray:
    """Exact success-conditioned ML gradient for a task, flattened over its
    skills: grad log p = sum_s (e_0 - softmax(theta_s))."""
    req = env.tasks[task_id]
    probs = env.skill_probs()[req]
    g = -probs
    g[:, 0] += 1.0
    return g.flatten()


def group_gradient(env: SkillChainEnv, task_id: int, actions: np.ndarray,
                   weights: np.ndarray, n_prefix: int | None = None) -> np.ndarray:
    """Gradient estimate sum_i w_i * grad log pi(a_i) over the (prefix of a)
    task's skills, flattened to align with true_ml_gradient(prefix task)."""
    req = env.tasks[task_id]
    if n_prefix is not None:
        req = req[:n_prefix]
        actions = actions[:, :n_prefix]
    probs = env.skill_probs()[req]
    n, L = actions.shape
    onehot = np.zeros((n, L, env.n_actions))
    onehot[np.arange(n)[:, None], np.arange(L)[None, :], actions] = 1.0
    score = onehot - probs[None]
    return np.einsum("j,jla->la", weights, score).flatten()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(a @ b / (na * nb))


def pretrain(env: SkillChainEnv, rng_seed: int = 0, steps: int = 120):
    """Light training so pass rates look mid-run (frontier ~ level 4-6)."""
    teacher = AdvMassTeacher(env.n_tasks, seed=rng_seed + 1, n_rollouts=16)
    for _ in range(steps):
        for t in teacher.sample_tasks(8):
            t = int(t)
            actions, rewards = env.rollout(t, 16)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, 0.5)


# ---------------------------------------------------------------- V1
def v1_hindsight_fidelity(n_groups: int = 4000, N: int = 16):
    print("\n=== V1: hindsight gradient fidelity (exact, skill chain) ===")
    env = SkillChainEnv(seed=0)
    pretrain(env)
    p = env.true_pass_rates()
    levels = np.array(env.task_level)
    # pick a beyond-frontier task: deepest level with 1e-4 < p < 2e-2
    cands = [t for t in range(env.n_tasks) if 1e-4 < p[t] < 2e-2]
    t_hard = max(cands, key=lambda t: levels[t])
    chain0 = (t_hard // env.n_levels) * env.n_levels
    print(f"  probe task: level {levels[t_hard]}, true p = {p[t_hard]:.4f}")

    per_group_cos = {}   # j -> list of per-group cosines
    mean_grad = {}       # j -> accumulated hindsight gradient
    fresh_cos = {}       # j -> per-group cosines of unbiased fresh groups
    counts = {}
    rng = np.random.default_rng(7)
    for _ in range(n_groups):
        actions, rewards = env.rollout(t_hard, N)
        if rewards.sum() > 0:
            continue  # only dead groups feed hindsight
        prefixes = np.array([correct_prefix_len(a) for a in actions])
        j = int(prefixes.max())
        if j < 1:
            continue
        target = chain0 + (j - 1)  # task id of (chain, level j)
        r2 = (prefixes >= j).astype(float)
        w2 = weights_maxrl(r2)
        g_hs = group_gradient(env, t_hard, actions, w2, n_prefix=j)
        g_true = true_ml_gradient(env, target)
        per_group_cos.setdefault(j, []).append(cosine(g_hs, g_true))
        mean_grad[j] = mean_grad.get(j, 0) + g_hs
        counts[j] = counts.get(j, 0) + 1
        # unbiased reference: fresh on-policy group for the same prefix task
        fa, fr = env.rollout(target, N)
        fw = weights_maxrl(fr)
        if np.any(fw != 0):
            fresh_cos.setdefault(j, []).append(
                cosine(group_gradient(env, target, fa, fw), g_true))

    print(f"  dead-group rate at probe task: "
          f"{sum(counts.values())/n_groups:.2f} of sampled groups usable")
    print(f"  {'j':>3s} {'n':>5s} {'hindsight cos (per-group)':>28s} "
          f"{'fresh cos (per-group)':>24s} {'cos(MEAN hs grad, true)':>24s}")
    for j in sorted(counts):
        hs = np.array(per_group_cos[j])
        fr = np.array(fresh_cos.get(j, [np.nan]))
        cm = cosine(mean_grad[j] / counts[j], true_ml_gradient(env, chain0 + j - 1))
        print(f"  {j:3d} {counts[j]:5d} {np.nanmean(hs):14.3f} ± {np.nanstd(hs):.3f} "
              f"{np.nanmean(fr):12.3f} ± {np.nanstd(fr):.3f} {cm:24.3f}")


# ---------------------------------------------------------------- V2
class OracleTeacher(Teacher):
    """Cheating teacher: computes advantage-mass utility from TRUE pass rates."""

    def __init__(self, n_tasks, seed=0, n_rollouts=16, env=None, floor=0.1):
        super().__init__(n_tasks, seed)
        self.n_rollouts = n_rollouts
        self.env = env
        self.floor = floor

    def distribution(self):
        p = self.env.true_pass_rates()
        u = np.maximum((1 - (1 - p) ** self.n_rollouts) - p, 0.0)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(self.n_tasks, 1.0 / self.n_tasks)
        return (1 - self.floor) * probs + self.floor * unif


def run_teacher(kind: str, seed: int, steps: int = 400, floor: float = 0.1):
    env = SkillChainEnv(seed=seed)
    if kind == "oracle":
        teacher = OracleTeacher(env.n_tasks, seed=seed + 1000, n_rollouts=16,
                                env=env, floor=floor)
    elif kind == "advmass":
        teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000, n_rollouts=16,
                                 explore_frac=floor)
    else:
        teacher = UniformTeacher(env.n_tasks, seed=seed + 1000)
    hist, mass_collected = [], 0.0
    for step in range(steps):
        for t in teacher.sample_tasks(8):
            t = int(t)
            actions, rewards = env.rollout(t, 16)
            teacher.observe(t, rewards)
            w = weights_maxrl(rewards)
            mass_collected += np.abs(w).sum()
            if np.any(w != 0):
                env.apply_gradient(t, actions, w, 0.5)
        if step % 10 == 0 or step == steps - 1:
            hist.append(env.true_pass_rates().mean())
    return np.array(hist), mass_collected


def v2_oracle_gap():
    print("\n=== V2: oracle vs Thompson vs uniform (5 seeds) ===")
    for kind in ["uniform", "advmass", "oracle"]:
        aucs, finals, masses = [], [], []
        for seed in range(5):
            h, m = run_teacher(kind, seed)
            aucs.append(h.mean())
            finals.append(h[-1])
            masses.append(m)
        print(f"  {kind:8s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f} "
              f"advantage-mass collected={np.mean(masses):7.0f}")


# ---------------------------------------------------------------- V3
def v3_floor_curve():
    print("\n=== V3: explore/exploit curve — uniform floor ablation (5 seeds) ===")
    for floor in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0]:
        aucs, finals = [], []
        for seed in range(5):
            h, _ = run_teacher("advmass", seed, floor=floor)
            aucs.append(h.mean())
            finals.append(h[-1])
        print(f"  floor={floor:4.2f} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f}")


# ---------------------------------------------------------------- V4
def v4_feedback_loop(steps: int = 400):
    print("\n=== V4: hindsight->teacher feedback stability (5 seeds) ===")
    for feed in [False, True]:
        aucs, finals, dead_rates = [], [], []
        for seed in range(5):
            env = SkillChainEnv(seed=seed)
            teacher = AdvMassTeacher(env.n_tasks, seed=seed + 1000, n_rollouts=16)
            chain_len = env.n_levels
            hist, dead = [], 0
            total_groups = 0
            for step in range(steps):
                for t in teacher.sample_tasks(8):
                    t = int(t)
                    actions, rewards = env.rollout(t, 16)
                    teacher.observe(t, rewards)
                    total_groups += 1
                    w = weights_maxrl(rewards)
                    if np.any(w != 0):
                        env.apply_gradient(t, actions, w, 0.5)
                        continue
                    dead += 1
                    prefixes = np.array([correct_prefix_len(a) for a in actions])
                    j = int(prefixes.max())
                    if j < 1:
                        continue
                    target = (t // chain_len) * chain_len + (j - 1)
                    r2 = (prefixes >= j).astype(float)
                    w2 = weights_maxrl(r2)
                    if np.any(w2 != 0):
                        env.apply_gradient(target, actions[:, :j], w2, 0.5)
                        if feed:
                            teacher.observe(target, r2)
                if step % 10 == 0 or step == steps - 1:
                    hist.append(env.true_pass_rates().mean())
            aucs.append(np.mean(hist))
            finals.append(hist[-1])
            dead_rates.append(dead / total_groups)
        print(f"  feed={str(feed):5s} AUC={np.mean(aucs):.3f}(±{np.std(aucs):.3f}) "
              f"final={np.mean(finals):.3f} dead-group rate={np.mean(dead_rates):.3f}")


if __name__ == "__main__":
    v1_hindsight_fidelity()
    v2_oracle_gap()
    v3_floor_curve()
    v4_feedback_loop()
