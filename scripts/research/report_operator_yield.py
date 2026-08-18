#!/usr/bin/env python
"""Operator yield over *valid* nominals, with the invalid ones named rather than dropped.

Three counts get conflated easily and mean different things:

* **nominals screened** — how many reference rollouts were graded, which must be on an
  obstacle-free scene (``--scene plane``); a furnished one cannot separate an untrackable clip
  from one that does not fit the room
* **valid nominals** — how many SONIC accepts, and so can test an operator at all
* **operator yield** — of those, how many the operator survives

Reporting an operator's yield over *all* nominals blames it for clips the controller could not hold
to begin with, which is what happened to the arm tuck: motion x001's nominal is rejected at 277.6 N
external on ``left_knee_link``, so its adapted clip was never a test of the operator. Reporting
yield over valid nominals while silently dropping the invalid ones overstates coverage instead. So
both numbers are printed, and every exclusion is named.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)


def grade(directory: Path) -> tuple[str, dict]:
    """Verdict and diagnostics for one rollout directory, or ("missing", {})."""
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return "missing", {}
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    if payload is None:
        return "unevaluable", {}
    outcome = classify_episode(directory.name, payload)
    return outcome.outcome, {
        "reasons": list(outcome.rejection_reasons),
        "drift_mps": round(float((outcome.diagnostics or {}).get("drift_rate_mps", 0.0)), 4),
        "decomposition": (outcome.diagnostics or {}).get("contact_decomposition", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--motions", nargs="+", required=True, help="e.g. x000 x001 x002 x003")
    ap.add_argument("--operators", nargs="+", default=["crouch", "tuck"])
    ap.add_argument("--nominal-suffix", default="nominal")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    report: dict = {"motions": {}, "operators": {}}
    valid, invalid = [], []
    for motion in args.motions:
        verdict, detail = grade(args.rollouts / f"{motion}_{args.nominal_suffix}")
        report["motions"][motion] = {"nominal": verdict, **detail}
        (valid if verdict == "accepted" else invalid).append((motion, verdict, detail))

    print(f"nominals screened: {len(args.motions)}   valid: {len(valid)}")
    for motion, verdict, detail in invalid:
        bodies = (detail.get("decomposition") or {}).get("external_contact_bodies") or []
        force = (detail.get("decomposition") or {}).get("max_external_contact_force_n", 0.0)
        print(
            f"  excluded {motion}: nominal {verdict}"
            + (f" — {force:.1f} N on {', '.join(bodies)}" if bodies else "")
            + (f" — {', '.join(detail.get('reasons', []))}" if detail.get("reasons") else "")
        )
    if not valid:
        print("\nno valid nominals, so no operator can be scored here")
        return 1

    print()
    for operator in args.operators:
        rows, accepted = [], 0
        for motion, _, _ in valid:
            verdict, detail = grade(args.rollouts / f"{motion}_{operator}")
            rows.append((motion, verdict, detail))
            accepted += verdict == "accepted"
        scored = [r for r in rows if r[1] in ("accepted", "rejected")]
        print(
            f"{operator}: {accepted} of {len(scored)} valid nominals"
            + (
                f"  ({len(rows) - len(scored)} not yet rolled out)"
                if len(scored) != len(rows)
                else ""
            )
        )
        for motion, verdict, detail in rows:
            note = ""
            if verdict == "rejected":
                reasons = detail.get("reasons", [])
                dec = detail.get("decomposition") or {}
                ext = dec.get("max_external_contact_force_n", 0.0)
                # A rejection with no external contact is a pure tracking failure, which is a
                # different finding from a collision and must not be reported as one.
                note = (
                    f"  {', '.join(reasons)}"
                    f"{'' if ext > 1.0 else ' (no external contact — tracking, not collision)'}"
                    f"  drift {detail.get('drift_mps', 0.0):.3f} m/s"
                )
            elif verdict == "accepted":
                note = f"  drift {detail.get('drift_mps', 0.0):.3f} m/s"
            print(f"    {motion} {verdict}{note}")
        report["operators"][operator] = {
            "valid_nominals": len(scored),
            "accepted": accepted,
            "cells": {m: {"outcome": v, **d} for m, v, d in rows},
        }
        print()

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
