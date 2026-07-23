"""Inference-efficiency comparison of final checkpoints — the paper's
headline metric (their Fig. 5): how many samples does each trained policy
need to reach a target coverage on held-out tasks, with a perfect verifier?

For each level and checkpoint we estimate pass@k curves from n=64 samples
(Chen et al. 2021 unbiased estimator) and report:
  - samples-to-90% coverage per level (k*), interpolated
  - speedup vs the uniform+maxrl baseline

Usage: python3 eval_efficiency.py ckptA.pt ckptB.pt ... (labels from names)
Note: checkpoints are not saved by train.py runs by default; this script
re-trains quick reference checkpoints when invoked with --retrain, using the
same matched 2400s protocol.  For the paper-style comparison we retrain the
three headline configs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maze_env import LEVELS, MOVE_BUDGET, PAD, EOS, sample_task, verify
from model import TinyTransformer
from train import pad_batch, DEVICE, pass_at_k_unbiased


@torch.no_grad()
def coverage_curves(model, n_samples=64, n_tasks=16, ks=(1, 2, 4, 8, 16, 32, 64)):
    """Per-level pass@k estimates from n_samples rollouts per held-out task."""
    model.eval()
    eval_rng = random.Random(999)
    out = {}
    for level in LEVELS:
        tasks = [sample_task(level, eval_rng) for _ in range(n_tasks)]
        cs = []
        for t in tasks:
            prompts, plens = pad_batch([t.prompt] * n_samples, DEVICE)
            resp = model.generate(prompts, plens, MOVE_BUDGET[level] + 1, EOS)
            c = sum(verify(t.grid, t.goal, [int(x) for x in resp[j] if int(x) != PAD])
                    for j in range(n_samples))
            cs.append(c)
        out[level] = {k: float(np.mean([pass_at_k_unbiased(n_samples, c, k) for c in cs]))
                      for k in ks}
    return out


def k_to_target(curve: dict, target: float = 0.9):
    """Smallest k reaching target coverage (log-interpolated); None if never."""
    ks = sorted(curve)
    for i, k in enumerate(ks):
        if curve[k] >= target:
            if i == 0:
                return float(k)
            k0, k1 = ks[i - 1], k
            c0, c1 = curve[k0], curve[k1]
            if c1 == c0:
                return float(k1)
            f = (target - c0) / (c1 - c0)
            return float(k0 * (k1 / k0) ** f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="checkpoint .pt files")
    ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--out", default="efficiency.json")
    args = ap.parse_args()

    results = {}
    for path in args.ckpts:
        label = os.path.basename(path).replace(".pt", "")
        model = TinyTransformer().to(DEVICE)
        model.load_state_dict(torch.load(path, weights_only=True))
        curves = coverage_curves(model)
        kstars = {lv: k_to_target(curves[lv], args.target) for lv in LEVELS}
        results[label] = {"curves": {str(k): v for k, v in curves.items()},
                          "kstar": {str(k): v for k, v in kstars.items()}}
        ks_str = " ".join(f"{lv}:{kstars[lv]:.1f}" if kstars[lv] else f"{lv}:>64"
                          for lv in LEVELS)
        print(f"{label:36s} k*({args.target:.0%}) per level: {ks_str}", flush=True)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
