#!/usr/bin/env python
"""Does the motion generator produce clips the tracking controller can actually execute?

A dataset built on generated motion has two failure surfaces that are easy to conflate. The
generator can emit a clip the robot's *body* cannot hold -- joints past their limits, a root height
no leg length reaches -- and it can emit a kinematically fine clip that the *controller* cannot
track. The first is caught cheaply on CPU; the second costs a rollout. Reporting one acceptance
rate hides which is which, and hides that a permissive kinematic screen makes the corpus look
healthier than it is.

This joins the generator's own screen to SONIC's verdicts on the same clips, matching on prompt
text because the two pipelines number their outputs differently. The matched subset is small and is
reported as such: the point is the ratio between the two surfaces, not a headline rate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)


def clip_key(name: str) -> str:
    """The clip stem the two pipelines share.

    The generator writes ``045_a_person_crouches_..._s0.csv``; the rollout batch writes
    ``clutter_045_a_person_crouches_..._s0_s0.pkl`` -- same index, one added prefix and one added
    seed suffix. Stripping the prefix and *all* trailing seed groups recovers a common stem, which
    matched 24 of 24 on the batch checked.

    An earlier version truncated to the first 44 characters of prompt text instead, on the
    assumption that the indices differed. They do not, and the truncation collapsed 150 distinct
    clips into 56 keys -- silently merging different motions and making any rate computed from the
    join meaningless.
    """
    stem = Path(name).stem
    stem = re.sub(r"^(clutter|placed|batch)_", "", stem)
    # Every trailing seed group, not one: the generator writes a single _s0 and the rollout writes
    # _s0_s0, so stripping one from each leaves the two sides one suffix apart and nothing joins.
    stem = re.sub(r"(_s\d+)+$", "", stem)
    return stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--rollouts", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    screen = {}
    for record in json.loads(args.screen.read_text()).get("records", []):
        screen[clip_key(record["csv"])] = record

    tracked, unmatched = {}, 0
    for manifest in args.rollouts.rglob("success_manifest.json"):
        try:
            data = json.loads(manifest.read_text())
        except Exception:  # noqa: BLE001
            continue
        motion = (data.get("capture_context") or {}).get("motion") or {}
        path = motion.get("path")
        if not path:
            continue
        key = clip_key(path)
        if key not in screen:
            unmatched += 1
            continue
        cell = manifest.parent
        pkl = sorted(cell.glob("trajectories/*.trajectory.pkl"))
        if not pkl:
            continue
        try:
            with open(pkl[0], "rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
        except SegmentError:
            # A capture that splits into many short segments is the robot falling and resetting
            # repeatedly. No gate can grade it, and treating that as "no data" would quietly drop
            # the worst tracking failures out of the yield.
            tracked.setdefault(key, []).append("reset_loop")
            continue
        except Exception:  # noqa: BLE001
            tracked.setdefault(key, []).append("unreadable")
            continue
        if payload is None:
            tracked.setdefault(key, []).append("unevaluable")
            continue
        tracked.setdefault(key, []).append(classify_episode(cell.name, payload).outcome)

    rows = []
    for key, outcomes in sorted(tracked.items()):
        record = screen[key]
        # A clip counts as tracked if any rollout of it was accepted; several were rolled out more
        # than once under different scenes, and one acceptance proves the controller can hold it.
        rows.append(
            {
                "clip": key,
                "kinematically_feasible": bool(record.get("passed")),
                "saturated_frame_fraction": record.get("saturated_frame_fraction"),
                "root_height_min_m": record.get("root_height_min_m"),
                "is_crouch": bool(record.get("is_crouch")),
                "rollouts": len(outcomes),
                "tracked": "accepted" in outcomes,
                "outcomes": dict(Counter(outcomes)),
            }
        )

    total_screened = len(screen)
    feasible = sum(1 for r in screen.values() if r.get("passed"))
    matched = len(rows)
    matched_feasible = [r for r in rows if r["kinematically_feasible"]]
    tracked_count = sum(1 for r in matched_feasible if r["tracked"])

    print(
        f"generator screen:      {total_screened} clips, {feasible} kinematically feasible "
        f"({feasible / total_screened:.0%})"
    )
    print(f"rolled out and joined: {matched} clips ({unmatched} rollouts had no screen record)")
    if matched_feasible:
        print(
            f"of those, SONIC tracked: {tracked_count} of {len(matched_feasible)} "
            f"({tracked_count / len(matched_feasible):.0%})"
        )
        print()
        print(
            "The two surfaces are different sizes. A kinematic screen that passes "
            f"{feasible / total_screened:.0%} of clips"
        )
        print("says little about whether the controller can hold them, which is the number that")
        print("decides how much generated motion a dataset actually yields.")

    crouches = [r for r in matched_feasible if r["is_crouch"]]
    if crouches:
        ok = sum(1 for r in crouches if r["tracked"])
        print(f"\ncrouch-labelled clips in the matched set: {len(crouches)}, tracked {ok}")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "screened": total_screened,
                    "kinematically_feasible": feasible,
                    "matched": matched,
                    "unmatched_rollouts": unmatched,
                    "tracked_of_feasible": tracked_count,
                    "matched_feasible": len(matched_feasible),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
