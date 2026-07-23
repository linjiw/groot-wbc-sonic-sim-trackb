"""GPU curriculum x MaxRL experiment on multi-size mazes.

Pipeline per run:
  1. SFT warmstart on level-0/1 BFS solutions only (so deeper levels start
     near p=0 and the curriculum question is real).
  2. RL loop: teacher picks levels -> sample fresh mazes (infinite-data
     regime, as in the paper's maze experiment) -> group rollouts -> binary
     verifier -> estimator advantages -> policy-gradient step.
  3. Periodic eval on a fixed held-out set per level (pass@1 greedy-free,
     sampled) -> results JSONL.

Usage:
  python3 train.py --teacher uniform --estimator maxrl --steps 300 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maze_env import (LEVELS, MOVE_BUDGET, PAD, EOS,
                      sample_task, sft_example, verify, simulate_prefix,
                      encode_prompt)
from model import TinyTransformer
from estimators import weights_reinforce, weights_rloo, weights_grpo, weights_maxrl

ESTIMATORS = {
    "reinforce": weights_reinforce,
    "rloo": weights_rloo,
    "grpo": weights_grpo,
    "maxrl": weights_maxrl,
}

DEVICE = "cuda"


# ------------------------------------------------------------------ teachers
class Teacher:
    """Level-based teacher: curriculum over the 7 maze sizes."""

    def __init__(self, n_rollouts: int, seed: int):
        self.rng = np.random.default_rng(seed)
        self.n_rollouts = n_rollouts
        self.alpha = np.ones(len(LEVELS))
        self.beta = np.ones(len(LEVELS))

    def observe(self, level: int, rewards: np.ndarray, decay: float = 0.7):
        k, n = rewards.sum(), len(rewards)
        self.alpha[level] = 1.0 + (self.alpha[level] - 1.0) * decay + k
        self.beta[level] = 1.0 + (self.beta[level] - 1.0) * decay + (n - k)

    def p_hat(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def distribution(self) -> np.ndarray:
        raise NotImplementedError

    def sample_levels(self, m: int) -> np.ndarray:
        return self.rng.choice(len(LEVELS), size=m, p=self.distribution())


class UniformTeacher(Teacher):
    def distribution(self) -> np.ndarray:
        return np.full(len(LEVELS), 1.0 / len(LEVELS))


class FrontierTeacher(Teacher):
    """u_N(p) = (1-(1-p)^N)(1-p) with Thompson-sampled p, uniform floor."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.15):
        super().__init__(n_rollouts, seed)
        self.floor = floor

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = (1.0 - (1.0 - p) ** self.n_rollouts) * (1.0 - p)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class LearnabilityTeacher(Teacher):
    """SFL-style u(p) = p(1-p) — the N=1 special case of frontier utility."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.15):
        super().__init__(n_rollouts, seed)
        self.floor = floor

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = p * (1.0 - p)
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


class FrontierALPTeacher(FrontierTeacher):
    """Frontier utility + ALP-GMM-style anti-forgetting term.

    utility = u_N(p) + alp_coef * |Δ ema_pass|.  The |ΔLP| term re-injects
    levels whose competence is *changing* — including regressions on mastered
    levels, which pure u_N(p) would retire (its u -> 0 as p -> 1).

    power (VALIDATION.md V6): sample ∝ utility^power; sharper-than-
    proportional concentration compounds on ordered level structures."""

    def __init__(self, n_rollouts: int, seed: int, floor: float = 0.1,
                 alp_coef: float = 2.0, power: float = 1.0):
        super().__init__(n_rollouts, seed, floor)
        self.alp_coef = alp_coef
        self.power = power
        self.ema = np.zeros(len(LEVELS))
        self.alp = np.zeros(len(LEVELS))
        self.seen = np.zeros(len(LEVELS), dtype=bool)

    def observe(self, level: int, rewards: np.ndarray, decay: float = 0.7):
        super().observe(level, rewards, decay)
        m = rewards.mean()
        prev = self.ema[level] if self.seen[level] else m
        self.ema[level] = 0.7 * prev + 0.3 * m
        self.seen[level] = True
        self.alp[level] = 0.7 * self.alp[level] + 0.3 * abs(self.ema[level] - prev)

    def distribution(self) -> np.ndarray:
        p = self.rng.beta(self.alpha, self.beta)
        u = (1.0 - (1.0 - p) ** self.n_rollouts) * (1.0 - p) + self.alp_coef * self.alp
        u = np.maximum(u, 0.0) ** self.power
        if u.sum() <= 1e-12:
            u[:] = 1.0
        probs = u / u.sum()
        unif = np.full(len(LEVELS), 1.0 / len(LEVELS))
        return (1 - self.floor) * probs + self.floor * unif


TEACHERS = {
    "uniform": UniformTeacher,
    "frontier": FrontierTeacher,
    "learnability": LearnabilityTeacher,
    "frontier_alp": FrontierALPTeacher,
}


# ------------------------------------------------------------------ batching
def pad_batch(seqs: list[list[int]], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    lens = torch.tensor([len(s) for s in seqs], device=device)
    out = torch.full((len(seqs), int(lens.max())), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, device=device)
    return out, lens


def response_logprobs(model, prompts, prompt_lens, resps):
    """Sum log pi(response tokens) per sample. resps: (B, Lr) PAD after EOS."""
    B, Lr = resps.shape
    full = torch.full((B, prompts.shape[1] + Lr), PAD, dtype=torch.long,
                      device=prompts.device)
    full[:, :prompts.shape[1]] = prompts
    resp_mask = resps != PAD
    for b in range(B):
        n = int(resp_mask[b].sum())
        full[b, prompt_lens[b]:prompt_lens[b] + n] = resps[b, :n]
    logits = model(full[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    tgt = full[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (B, L-1)
    # mask: positions belonging to the response
    pos = torch.arange(full.shape[1] - 1, device=prompts.device)[None]
    n_resp = resp_mask.sum(1)
    mask = (pos >= (prompt_lens - 1)[:, None]) & (pos < (prompt_lens - 1 + n_resp)[:, None])
    return (tok_lp * mask).sum(1), mask.sum(1)


# ------------------------------------------------------------------ SFT
def run_sft(model, opt, rng, steps=600, batch=64, decay=0.5):
    """SFT on a geometric mixture over levels (weight decay^level): shallow
    levels dominate, deep levels are seen rarely — so post-SFT pass rates
    decay smoothly with depth instead of cliffing to exactly 0 (mirrors the
    paper's 'brief SFT to ensure non-zero initial pass rate')."""
    w = np.array([decay ** l for l in LEVELS])
    w = w / w.sum()
    model.train()
    for step in range(steps):
        lvls = np.random.choice(LEVELS, size=batch, p=w)
        pairs = [sft_example(int(l), rng) for l in lvls]
        seqs = [p + r for p, r in pairs]
        plens = [len(p) for p, _ in pairs]
        ids, lens = pad_batch(seqs, DEVICE)
        logits = model(ids[:, :-1])
        tgt = ids[:, 1:]
        lp = F.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        pos = torch.arange(ids.shape[1] - 1, device=DEVICE)[None]
        plens_t = torch.tensor(plens, device=DEVICE)
        mask = (pos >= (plens_t - 1)[:, None]) & (pos < (lens - 1)[:, None])
        loss = -(lp * mask).sum() / mask.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            print(f"  sft step {step} loss {loss.item():.3f}", flush=True)


# ------------------------------------------------------------------ eval
def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Chen et al. 2021 unbiased pass@k estimator: 1 - C(n-c,k)/C(n,k)."""
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - c - i) / (n - i)
    return 1.0 - prod


@torch.no_grad()
def evaluate(model, eval_tasks, n_samples=8, batch_cap=256, pass_ks=(1, 8)):
    """Per-level sampled pass rate + unbiased pass@k on fixed held-out mazes.

    Returns {level: mean_pass} plus {"passk": {level: {k: pass@k}}} computed
    per maze from its n_samples rollouts (Chen et al. 2021 estimator).
    """
    model.eval()
    out = {}
    passk = {}
    for level, tasks in eval_tasks.items():
        per_task_c = {id(t): 0 for t in tasks}
        reps = [(t, s) for t in tasks for s in range(n_samples)]
        for i in range(0, len(reps), batch_cap):
            chunk = [t for t, _ in reps[i:i + batch_cap]]
            prompts, plens = pad_batch([t.prompt for t in chunk], DEVICE)
            resp = model.generate(prompts, plens, MOVE_BUDGET[level] + 1, EOS)
            for j, t in enumerate(chunk):
                toks = [int(x) for x in resp[j] if int(x) != PAD]
                per_task_c[id(t)] += verify(t.grid, t.goal, toks)
        cs = np.array(list(per_task_c.values()))
        out[level] = float(cs.sum()) / (len(tasks) * n_samples)
        passk[level] = {
            k: float(np.mean([pass_at_k_unbiased(n_samples, int(c), k) for c in cs]))
            for k in pass_ks if k <= n_samples
        }
    model.train()
    out["passk"] = passk
    return out


# ------------------------------------------------------------------ RL loop
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", choices=list(TEACHERS), default="uniform")
    ap.add_argument("--estimator", choices=list(ESTIMATORS), default="maxrl")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks-per-step", type=int, default=8)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--sft-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--max-seconds", type=int, default=None,
                    help="stop after this much RL wall-clock (matched-compute comparisons)")
    ap.add_argument("--hindsight", action="store_true",
                    help="relabel dead (K=0) groups to the deepest cell reached")
    ap.add_argument("--hindsight-scale", type=float, default=1.0)
    ap.add_argument("--hindsight-dense", action="store_true",
                    help="relabel EVERY failed rollout (depth >= --hindsight-min-depth) "
                         "to its reached cell, not just the group's best")
    ap.add_argument("--hindsight-min-depth", type=int, default=6)
    ap.add_argument("--hindsight-cap", type=int, default=16,
                    help="max relabeled trajectories per step (compute bound)")
    ap.add_argument("--hindsight-to-teacher", action="store_true",
                    help="relabeled successes update the teacher posterior at the "
                         "matching distance level (curriculum rides hindsight gains)")
    ap.add_argument("--save-ckpt", type=str, default=None,
                    help="save the final model state_dict to this path")
    ap.add_argument("--teacher-power", type=float, default=1.0,
                    help="sample levels ∝ utility^power (V6: 4 for ordered levels)")
    ap.add_argument("--d-model", type=int, default=128,
                    help="model width (capacity probe: per-step legality ceiling)")
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--sft-ckpt", type=str, default="sft_warmstart.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    model = TinyTransformer(d_model=args.d_model, n_layers=args.n_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params/1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ---- SFT warmstart (shared across runs with the same seed) ----
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"seed{args.seed}_{args.sft_ckpt}")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded SFT checkpoint {ckpt}", flush=True)
    else:
        sft_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        run_sft(model, sft_opt, rng, steps=args.sft_steps)
        torch.save(model.state_dict(), ckpt)
        print(f"saved SFT checkpoint {ckpt}", flush=True)

    # fixed eval set: 16 mazes per level, seeded independently of training
    eval_rng = random.Random(12345)
    eval_tasks = {l: [sample_task(l, eval_rng) for _ in range(16)] for l in LEVELS}

    teacher_kwargs = {"n_rollouts": args.rollouts, "seed": args.seed + 77}
    if args.teacher == "frontier_alp" and args.teacher_power != 1.0:
        teacher_kwargs["power"] = args.teacher_power
    teacher = TEACHERS[args.teacher](**teacher_kwargs)
    est = ESTIMATORS[args.estimator]

    out_path = args.out or f"log_{args.teacher}_{args.estimator}_s{args.seed}.jsonl"
    log_f = open(out_path, "w")

    ev = evaluate(model, eval_tasks)
    passk0 = ev.pop("passk")
    print(f"post-SFT eval: {ev}", flush=True)
    log_f.write(json.dumps({"step": -1, "eval": ev, "passk": passk0}) + "\n")
    log_f.flush()

    t0 = time.time()
    max_new = MOVE_BUDGET[LEVELS[-1]] + 1
    step = -1
    while True:
        step += 1
        if args.max_seconds is not None:
            if time.time() - t0 >= args.max_seconds:
                break
        elif step >= args.steps:
            break
        levels = [int(x) for x in teacher.sample_levels(args.tasks_per_step)]
        tasks = [sample_task(lv, rng) for lv in levels]
        # one batched generation for all groups: (tasks*rollouts) sequences
        flat_prompts = [t.prompt for t in tasks for _ in range(args.rollouts)]
        prompts, plens = pad_batch(flat_prompts, DEVICE)
        resp = model.generate(prompts, plens, max_new, EOS)

        step_stats = {"dead_groups": 0, "mean_reward": [], "relabeled": 0}
        keep_rows, keep_w = [], []
        hs_prompts, hs_resps, hs_depths = [], [], []  # hindsight-relabeled
        for g, (lv, task) in enumerate(zip(levels, tasks)):
            rows = range(g * args.rollouts, (g + 1) * args.rollouts)
            rewards = np.array([
                float(verify(task.grid, task.goal,
                             [int(x) for x in resp[j] if int(x) != PAD]))
                for j in rows])
            teacher.observe(lv, rewards)
            step_stats["mean_reward"].append(rewards.mean())
            w = est(rewards)
            if not np.any(w != 0):
                step_stats["dead_groups"] += 1
                if args.hindsight_dense:
                    # relabel every rollout whose legal prefix is deep enough:
                    # each becomes a success for the cell it reached
                    for j in rows:
                        if len(hs_prompts) >= args.hindsight_cap:
                            break
                        toks = [int(x) for x in resp[j] if int(x) != PAD]
                        n_ok, pos = simulate_prefix(task.grid, toks)
                        if n_ok >= args.hindsight_min_depth and pos != (1, 1):
                            hs_prompts.append(encode_prompt(task.grid, pos))
                            hs_resps.append(toks[:n_ok] + [EOS])
                            hs_depths.append(n_ok)
                            step_stats["relabeled"] += 1
                elif args.hindsight:
                    # relabel: goal <- deepest cell legally reached in group
                    best_n, best_pos, best_j = 0, None, None
                    for j in rows:
                        toks = [int(x) for x in resp[j] if int(x) != PAD]
                        n_ok, pos = simulate_prefix(task.grid, toks)
                        if n_ok > best_n and pos != (1, 1):
                            best_n, best_pos, best_j = n_ok, pos, j
                    if best_j is not None and best_n >= 4:
                        toks = [int(x) for x in resp[best_j] if int(x) != PAD]
                        hs_prompts.append(encode_prompt(task.grid, best_pos))
                        hs_resps.append(toks[:best_n] + [EOS])
                        hs_depths.append(best_n)
                        step_stats["relabeled"] += 1
                continue
            keep_rows.extend(rows)
            keep_w.extend(w)

        if keep_rows or hs_prompts:
            opt.zero_grad()
            if keep_rows:
                rows_t = torch.tensor(keep_rows, device=DEVICE)
                w_t = torch.tensor(np.array(keep_w), device=DEVICE, dtype=torch.float32)
                # micro-batch the backward pass to bound memory
                mb = 128
                for i in range(0, len(keep_rows), mb):
                    sel = rows_t[i:i + mb]
                    lp, _ = response_logprobs(model, prompts[sel], plens[sel], resp[sel])
                    loss = -(w_t[i:i + mb] * lp).sum() / args.tasks_per_step
                    loss.backward()
            if hs_prompts:
                # each relabeled trajectory acts as a K=1 MaxRL group:
                # w_succ = 1 - 1/N, scaled
                hp, hlens = pad_batch(hs_prompts, DEVICE)
                max_r = max(len(r) for r in hs_resps)
                hr = torch.full((len(hs_resps), max_r), PAD, dtype=torch.long,
                                device=DEVICE)
                for b, rr in enumerate(hs_resps):
                    hr[b, :len(rr)] = torch.tensor(rr, device=DEVICE)
                w_hs = args.hindsight_scale * (1.0 - 1.0 / args.rollouts)
                lp, _ = response_logprobs(model, hp, hlens, hr)
                loss = -(w_hs * lp).sum() / args.tasks_per_step
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if args.hindsight_to_teacher and hs_depths:
            # relabeled successes nudge the matching level's posterior so the
            # curriculum advances with hindsight gains instead of waiting for
            # natural successes.  NOTE: deliberately optimistic evidence — the
            # model reached SOME cell at distance d, not a requested one; the
            # posterior decay corrects any overshoot within a few groups.
            from maze_env import LEVEL_DIST
            for d in hs_depths:
                lv_match = min(LEVELS, key=lambda l: abs(LEVEL_DIST[l] - d))
                teacher.observe(lv_match, np.array([1.0]))

        if step % args.eval_every == 0 or step == args.steps - 1:
            ev = evaluate(model, eval_tasks)
            passk = ev.pop("passk")
            rec = {"step": step, "eval": ev, "passk": passk,
                   "teacher_p_hat": teacher.p_hat().round(3).tolist(),
                   "teacher_dist": teacher.distribution().round(3).tolist(),
                   "dead_groups": step_stats["dead_groups"],
                   "relabeled": step_stats["relabeled"],
                   "train_mean_reward": float(np.mean(step_stats["mean_reward"])),
                   "elapsed": time.time() - t0}
            log_f.write(json.dumps(rec) + "\n")
            log_f.flush()
            mean_ev = np.mean(list(ev.values()))
            mean_p8 = np.mean([v.get(8, 0.0) for v in passk.values()])
            print(f"step {step:4d} mean_eval={mean_ev:.3f} mean_pass@8={mean_p8:.3f} "
                  f"levels={dict((k, round(v, 2)) for k, v in ev.items())} "
                  f"dead={step_stats['dead_groups']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # final eval (time-budget runs stop between eval intervals)
    ev = evaluate(model, eval_tasks)
    passk = ev.pop("passk")
    rec = {"step": step, "eval": ev, "passk": passk, "final": True,
           "teacher_p_hat": teacher.p_hat().round(3).tolist(),
           "elapsed": time.time() - t0}
    log_f.write(json.dumps(rec) + "\n")
    print(f"FINAL step {step} mean_eval={np.mean(list(ev.values())):.3f} "
          f"mean_pass@8={np.mean([v.get(8, 0.0) for v in passk.values()]):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if args.save_ckpt:
        torch.save(model.state_dict(), args.save_ckpt)
        print(f"saved checkpoint to {args.save_ckpt}", flush=True)
    log_f.close()


if __name__ == "__main__":
    main()
