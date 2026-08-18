#!/usr/bin/env python
"""How faithfully does the controller execute what the generator asked for, by behaviour?

A generator-plus-controller stack has a retargeting gap: the reference asks for one thing and the
robot does another. That gap is what a retargeter exists to minimise, and it is almost never
reported per behaviour -- yet it is the number that says which behaviours a stack can actually
deliver. Aggregating it costs nothing: every graded episode already records how far the executed
root drifted from its reference, how far below the reference height the robot sank, and how much
further it tilted.

Reported as a distribution rather than a mean, because these are small samples and a mean over four
episodes invites a confidence the data does not support.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import re
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

BEHAVIOURS = (
    ("crouch", r"crouch|duck|lower|stoop"),
    ("side/narrow", r"sideways|side[_ ]step|narrow|squeeze|shuffle"),
    ("arms", r"arm|reach|carry|hold"),
    ("turn", r"turn|curv"),
    ("stop/start", r"stop|stand[_ ]still|begins"),
    ("walk", r"walk"),
)


def behaviour_of(name: str) -> str:
    for tag, pattern in BEHAVIOURS:
        if re.search(pattern, name):
            return tag
    return "other"


def spread(values: list[float]) -> str:
    """Median and range. A mean over four episodes claims more than the sample supports."""
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{statistics.median(values):.3f}  [{min(values):.3f}, {max(values):.3f}]"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    by_behaviour = defaultdict(lambda: {"drift": [], "sink": [], "tilt": [], "outcomes": []})
    scanned = 0
    for manifest in args.rollouts.rglob("success_manifest.json"):
        try:
            data = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            continue
        motion = (data.get("capture_context") or {}).get("motion") or {}
        source = motion.get("path") or manifest.parent.name
        cell = manifest.parent
        paths = sorted(cell.glob("trajectories/*.trajectory.pkl"))
        if not paths:
            continue
        try:
            with open(paths[0], "rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
        except (SegmentError, Exception):  # noqa: BLE001
            continue
        if payload is None:
            continue
        scanned += 1
        outcome = classify_episode(cell.name, payload)
        diagnostics = outcome.diagnostics or {}
        tag = behaviour_of(Path(source).stem + " " + cell.name)
        bucket = by_behaviour[tag]
        bucket["drift"].append(abs(float(diagnostics.get("drift_rate_mps", 0.0))))
        bucket["sink"].append(float(diagnostics.get("max_root_height_below_reference_m", 0.0)))
        bucket["tilt"].append(float(diagnostics.get("max_root_tilt_above_reference_rad", 0.0)))
        bucket["outcomes"].append(outcome.outcome)

    print(f"{scanned} evaluable episodes\n")
    print(
        f"{'behaviour':>13s}{'n':>5s}{'accepted':>10s}"
        f"{'drift m/s':>26s}{'sink below ref m':>26s}"
    )
    rows = {}
    for tag, _ in BEHAVIOURS + (("other", ""),):
        bucket = by_behaviour.get(tag)
        if not bucket or not bucket["outcomes"]:
            continue
        n = len(bucket["outcomes"])
        accepted = sum(1 for o in bucket["outcomes"] if o == "accepted")
        print(
            f"{tag:>13s}{n:5d}{accepted:>7d}/{n:<2d}"
            f"{spread(bucket['drift']):>26s}{spread(bucket['sink']):>26s}"
        )
        rows[tag] = {
            "episodes": n,
            "accepted": accepted,
            "drift_median": round(statistics.median(bucket["drift"]), 4),
            "sink_median": round(statistics.median(bucket["sink"]), 4),
            "tilt_median": round(statistics.median(bucket["tilt"]), 4),
        }

    print("\nmedian and [min, max]. Drift is how far the executed root left its reference; sink is")
    print("how far below the reference height the robot settled. Both are retargeting gaps: the")
    print("reference asked for one thing and the controller delivered another.")
    if args.json:
        args.json.write_text(
            json.dumps({"episodes": scanned, "by_behaviour": rows}, indent=2) + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
