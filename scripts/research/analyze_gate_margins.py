#!/usr/bin/env python3
"""Report how close the corpus actually comes to each acceptance threshold.

The plan flags the acceptance thresholds as PROVISIONAL and schedules a reviewed
calibration pilot. Before spending review effort, it is worth knowing which thresholds the
data is anywhere near: a threshold sitting in a wide empty gap between the worst passing
episode and the best failing one produces identical outcomes anywhere in that gap, so its
exact value is not what limits the corpus. A threshold with episodes crowding both sides is
where human judgement actually decides something.

For each gate this prints the passing range, the failing range, and the margin -- the
distance from the threshold to the nearest observation on each side. Gates that never fire
are reported too, because an untested gate is a different kind of unknown from a
well-separated one, and it should not be mistaken for a calibrated one.

Usage::

    python scripts/research/analyze_gate_margins.py --root /data/.../groot-wbc-kimodo-m0 \\
        [--json gate_margins.json]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)


def classify(passing: list[float], failing: list[float]) -> str:
    """Say what the data supports about this gate, in words that stay honest."""
    if not failing:
        return "never fired"
    if not passing:
        return "always fired"
    gap = min(failing) - max(passing)
    if gap <= 0:
        # Passing and failing values overlap, so the gate is not a threshold on this
        # quantity alone -- something else decides, or the threshold is genuinely marginal.
        return "OVERLAPPING"

    # Scale the gap against the passing range, not against the full span. Failing values
    # have a long tail -- one 9686 N self-contact episode -- and dividing by that tail
    # makes a 343 N gap above a 317 N passing maximum look marginal when it is anything
    # but. "Is there room between the worst accepted episode and the best rejected one,
    # in units the accepted episodes themselves occupy?" is the question that matters.
    scale = max(passing)
    if scale <= 0:
        # Every accepted episode reads zero (contact gates), so any nonzero failure is a
        # clean separation.
        return "well separated"
    return "well separated" if gap / scale > 0.25 else "MARGINAL"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    paths = sorted(args.root.rglob("*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectories under {args.root}")

    passing: dict[str, list[float]] = defaultdict(list)
    failing: dict[str, list[float]] = defaultdict(list)
    episodes = accepted = 0

    for path in paths:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
        except Exception as error:  # noqa: BLE001 - a corrupt file must not end the sweep
            print(f"  skip {path.parent.parent.name}: {type(error).__name__}: {error}")
            continue
        report = evaluate_locomotion_trajectory(payload)
        episodes += 1
        accepted += bool(report.accepted)
        for gate in report.gates:
            (passing if gate.passed else failing)[gate.name].append(float(gate.value))

    print(f"{episodes} rollouts, {accepted} accepted ({accepted / episodes:.0%})\n")
    header = f"{'gate':32s} {'verdict':>15} {'worst pass':>11} {'best fail':>11} {'gap':>10}"
    print(header)
    print("-" * len(header))

    records = []
    for name in sorted(set(passing) | set(failing)):
        good, bad = passing.get(name, []), failing.get(name, [])
        verdict = classify(good, bad)
        gap = (min(bad) - max(good)) if (good and bad) else math.nan
        records.append(
            {
                "gate": name,
                "verdict": verdict,
                "passing_n": len(good),
                "failing_n": len(bad),
                "passing_min": min(good) if good else None,
                "passing_max": max(good) if good else None,
                "failing_min": min(bad) if bad else None,
                "failing_max": max(bad) if bad else None,
                "gap": None if math.isnan(gap) else gap,
            }
        )
        worst = f"{max(good):.3f}" if good else "-"
        best = f"{min(bad):.3f}" if bad else "-"
        shown = "-" if math.isnan(gap) else f"{gap:.3f}"
        print(f"{name:32s} {verdict:>15} {worst:>11} {best:>11} {shown:>10}")

    marginal = [r["gate"] for r in records if r["verdict"] in ("MARGINAL", "OVERLAPPING")]
    untested = [r["gate"] for r in records if r["verdict"] == "never fired"]
    print()
    if marginal:
        print(f"Needs human judgement (episodes crowd the threshold): {', '.join(marginal)}")
    else:
        print("No gate is marginal on this corpus: every threshold sits in an empty gap,")
        print("so its exact value does not change any outcome here. That is not the same")
        print("as calibrated -- it means a more diverse corpus is what will test them.")
    if untested:
        print(f"Never fired, so untested rather than calibrated: {', '.join(untested)}")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {"episodes": episodes, "accepted": accepted, "gates": records},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
