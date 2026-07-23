"""Summarize sweep JSONL logs: final/AUC mean eval, frontier depth, dead groups.

Usage: python3 analyze.py sweep_*.jsonl
"""

from __future__ import annotations

import glob
import json
import sys

import numpy as np


def load(path):
    recs = [json.loads(l) for l in open(path)]
    return [r for r in recs if "eval" in r]


def frontier(ev: dict, thresh: float = 0.5) -> int:
    """Deepest level with pass rate > thresh (-1 if none)."""
    best = -1
    for k in sorted(ev, key=int):
        if ev[k] > thresh:
            best = int(k)
    return best


def summarize(path):
    recs = load(path)
    if len(recs) < 2:
        return None
    rl = [r for r in recs if r["step"] >= 0]
    steps = np.array([r["step"] for r in rl])
    mean_ev = np.array([np.mean(list(r["eval"].values())) for r in rl])
    fr = np.array([frontier(r["eval"]) for r in rl])
    dead = np.array([r.get("dead_groups", np.nan) for r in rl])
    auc = float(np.trapz(mean_ev, steps) / max(steps[-1] - steps[0], 1))
    # retention: worst level-0/1 pass over the last half of training,
    # relative to the post-SFT baseline (forgetting indicator)
    base = next((r for r in recs if r["step"] < 0), None)
    retention = None
    if base is not None:
        half = rl[len(rl) // 2:]
        easy_min = min(min(r["eval"].get("0", r["eval"].get(0, 1.0)),
                           r["eval"].get("1", r["eval"].get(1, 1.0))) for r in half)
        easy_base = min(base["eval"].get("0", base["eval"].get(0, 1.0)),
                        base["eval"].get("1", base["eval"].get(1, 1.0)))
        retention = float(easy_min - easy_base)
    out = {
        "final_mean": float(mean_ev[-1]),
        "best_mean": float(mean_ev.max()),
        "auc": auc,
        "final_frontier": int(fr[-1]),
        "best_frontier": int(fr.max()),
        "avg_dead_groups": float(np.nanmean(dead)),
        "final_eval": rl[-1]["eval"],
        "steps_run": int(steps[-1]),
        "easy_retention": retention,
    }
    # pass@8 coverage (may be absent in older logs)
    last_pk = rl[-1].get("passk")
    if last_pk:
        out["final_pass8"] = float(np.mean([v.get("8", v.get(8, 0.0))
                                            for v in last_pk.values()]))
    return out


def main():
    paths = sys.argv[1:] or sorted(glob.glob("sweep_*.jsonl"))
    print(f"{'config':34s} {'final':>6s} {'best':>6s} {'auc':>6s} "
          f"{'frontier':>8s} {'dead/step':>9s} {'steps':>6s} {'pass@8':>7s}")
    for p in paths:
        s = summarize(p)
        if s is None:
            print(f"{p:34s}  (incomplete)")
            continue
        name = p.replace("sweep_", "").replace("matched_", "").replace("_s0.jsonl", "")
        p8 = f"{s['final_pass8']:7.3f}" if "final_pass8" in s else "      -"
        print(f"{name:34s} {s['final_mean']:6.3f} {s['best_mean']:6.3f} "
              f"{s['auc']:6.3f} {s['best_frontier']:8d} {s['avg_dead_groups']:9.2f} "
              f"{s['steps_run']:6d} {p8}")
    print("\nper-level final pass rates:")
    for p in paths:
        s = summarize(p)
        if s is None:
            continue
        name = p.replace("sweep_", "").replace("matched_", "").replace("_s0.jsonl", "")
        lv = " ".join(f"{v:.2f}" for _, v in sorted(s["final_eval"].items(), key=lambda kv: int(kv[0])))
        print(f"  {name:32s} [{lv}]")


if __name__ == "__main__":
    main()
