#!/usr/bin/env python3
"""Grade every generated reference against the behaviour its prompt asked for.

Behavioural fidelity used to be measurable only after a rollout, which meant it was measured
on whatever survived the acceptance gate -- 37 episodes across seven modes -- and cost GPU
minutes per clip. Forward kinematics on the reference answers the same question in about a
second, so the whole library can be graded, including the clips that were never rolled out
and the ones that were rejected.

That matters because the question is about the *generator*, and the acceptance gate is a
filter on the controller. A mode whose clips are mostly rejected is invisible in the
rollout-based number however badly it is phrased.

This is a screen for behavioural intent, never a substitute for physics: a reference that
ducks may still be untrackable, and only a rollout knows. It is the fast half of a loop whose
slow half stays where it is.

Usage::

    python scripts/research/screen_reference_behaviours.py \\
        --taxonomy /data/.../taxonomy/taxonomy.json --motions /data/.../taxonomy/motions_4s
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.behaviour_predicates import (  # noqa: E402
    PredicateError,
    check_behaviour,
)
from gear_sonic.dataset_generation.reference_payload import payload_from_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    specs = json.loads(args.taxonomy.read_text())["specs"]
    rows, missing, unassessable = [], 0, 0

    for index, spec in enumerate(specs):
        matches = sorted(args.motions.glob(f"{index:03d}_*.csv"))
        if not matches:
            missing += 1
            continue
        try:
            payload = payload_from_csv(matches[0], fps=args.fps)
            result = check_behaviour(spec["body_mode"], payload)
        except (PredicateError, ValueError, OSError) as error:
            unassessable += 1
            rows.append({"index": index, "body_mode": spec["body_mode"],
                         "satisfied": None, "reason": str(error)})
            continue
        if result is None:
            rows.append({"index": index, "body_mode": spec["body_mode"],
                         "satisfied": None, "reason": "no predicate for this mode"})
            continue
        rows.append({
            "index": index, "body_mode": spec["body_mode"],
            "satisfied": bool(result.satisfied), "reason": result.reason,
            "measurements": {k: round(v, 4) for k, v in result.measurements.items()},
        })

    print(f"{len(specs)} prompts; {missing} with no generated clip; "
          f"{unassessable} that could not be assessed\n")

    by_mode: dict[str, list] = {}
    for row in rows:
        by_mode.setdefault(row["body_mode"], []).append(row)

    print(f"{'body mode':16s}{'graded':>8s}{'satisfied':>11s}{'rate':>7s}   verdict")
    checked_total = satisfied_total = 0
    for mode, group in sorted(by_mode.items()):
        graded = [r for r in group if r["satisfied"] is not None]
        if not graded:
            print(f"{mode:16s}{'-':>8s}{'-':>11s}{'-':>7s}   no predicate")
            continue
        good = sum(1 for r in graded if r["satisfied"])
        rate = good / len(graded)
        checked_total += len(graded)
        satisfied_total += good
        verdict = (
            "absent" if rate == 0 else "present" if rate >= 0.8
            else "intermittent"
        )
        print(f"{mode:16s}{len(graded):8d}{good:11d}{rate:7.0%}   {verdict}")

    if checked_total:
        print(f"\n{satisfied_total}/{checked_total} references carry the behaviour their "
              f"prompt asked for ({satisfied_total / checked_total:.0%})")
    print("\nThis grades the generator, not the controller. A reference that carries the")
    print("behaviour may still be untrackable, and only a rollout knows.")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
