#!/usr/bin/env python3
"""Decide which generated references deserve GPU time, before any is spent.

Run over the 150-clip library this refuses 56 of them, and says which fix each needs:

    144/150  reachable on the G1
    150/150  free of self-collision
     24 carry their behaviour, 51 do not, 75 unchecked
     94/150  worth a rollout  (63%)

      50  rephrase -- the clip does not contain the behaviour its prompt named
       6  retarget -- the pose is unreachable on this robot

Those two prescriptions are opposite and must not be merged. `step_over` clips are perfectly
reachable and simply do not contain a step-over, so retargeting one produces a walk;
`crouch_walk` clips contain exactly the requested crouch and are unreachable because a single
joint has no range, so rephrasing cannot help. Fifteen step-overs were generated, screened,
converted, rolled out, rendered and graded before anyone asked the first question.

Usage::

    python scripts/research/gate_references_for_rollout.py \\
        --taxonomy /data/.../taxonomy.json --motions /data/.../motions_4s
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402


def prescription(verdict) -> str:
    if verdict.worth_a_rollout:
        return "rollout"
    if not verdict.embodiment_feasible:
        return "retarget (unreachable)"
    if not verdict.self_collision_free:
        return "retarget (self-collision)"
    return "rephrase (behaviour absent)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    specs = json.loads(args.taxonomy.read_text())["specs"]
    rows = []
    for index, spec in enumerate(specs):
        matches = sorted(args.motions.glob(f"{index:03d}_*.csv"))
        if not matches:
            continue
        verdict = screen_reference(
            np.loadtxt(matches[0], delimiter=","),
            f"{index:03d}", spec["body_mode"], fps=args.fps,
        )
        rows.append({
            "index": index, "body_mode": spec["body_mode"],
            "worth_a_rollout": verdict.worth_a_rollout,
            "embodiment_feasible": verdict.embodiment_feasible,
            "self_collision_free": verdict.self_collision_free,
            "reference_semantic_valid": verdict.reference_semantic_valid,
            "saturated_cell_fraction": round(verdict.saturated_cell_fraction, 4),
            "worst_joint": verdict.worst_joint,
            "worst_joint_fraction": round(verdict.worst_joint_fraction, 3),
            "prescription": prescription(verdict),
            "diagnosis": verdict.diagnosis,
        })

    total = len(rows)
    if not total:
        print(f"no generated clips found under {args.motions}")
        return 1

    worth = sum(1 for r in rows if r["worth_a_rollout"])
    print(f"REFERENCE GATE over {total} generated clips\n")
    print(f"  {sum(1 for r in rows if r['embodiment_feasible']):3d}/{total}  reachable on the G1")
    print(f"  {sum(1 for r in rows if r['self_collision_free']):3d}/{total}  free of self-collision")
    print(f"  {sum(1 for r in rows if r['reference_semantic_valid'] is True):3d} carry their "
          f"behaviour, {sum(1 for r in rows if r['reference_semantic_valid'] is False)} do not, "
          f"{sum(1 for r in rows if r['reference_semantic_valid'] is None)} unchecked")
    print(f"  {worth:3d}/{total}  worth a rollout  ({worth/total:.0%})\n")

    refused = Counter(r["prescription"] for r in rows if not r["worth_a_rollout"])
    if refused:
        print("refused, by what would fix it:")
        for name, count in refused.most_common():
            print(f"  {count:3d}  {name}")
        print()

    by_mode: dict[str, list] = {}
    for row in rows:
        by_mode.setdefault(row["body_mode"], []).append(row)
    print(f"{'body mode':16s}{'n':>4s}{'reach':>7s}{'sem+':>6s}{'sem-':>6s}{'rollout':>9s}")
    for mode, group in sorted(by_mode.items()):
        print(f"{mode:16s}{len(group):4d}"
              f"{sum(1 for r in group if r['embodiment_feasible']):7d}"
              f"{sum(1 for r in group if r['reference_semantic_valid'] is True):6d}"
              f"{sum(1 for r in group if r['reference_semantic_valid'] is False):6d}"
              f"{sum(1 for r in group if r['worth_a_rollout']):9d}")

    print("\nA clip passing this gate may still be untrackable; only a rollout knows.")
    print("What the gate guarantees is that the rollout is worth running.")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
