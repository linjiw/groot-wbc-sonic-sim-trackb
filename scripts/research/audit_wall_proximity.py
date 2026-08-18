#!/usr/bin/env python
"""Flag rollouts whose verdict may be about the room rather than the motion.

A rejection is only evidence about a motion if the motion had room to execute. One nominal here was
recorded as untrackable when it had in fact been pushed back by a wall 0.21 m away, carrying a force
of exactly (0.0, 277.6, 0.0) N -- purely lateral, straight into the wall. The same room had already
caused a family to fail on a wall rather than on its obstacle.

This measures, for every graded rollout, how close the root came to a wall face, and flags anything
tighter than a threshold. It is cheap, reads only what physics loaded, and should run before any
yield number is quoted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Metres. Below this the root is close enough that a limb can reach the wall, so the verdict is
#: not safely attributable to the motion. The observed failure sat at 0.21 m.
SUSPECT_GAP_M = 0.35
#: Walls are inset from the floor's edge by half their thickness.
WALL_INSET_M = 0.05


def room_size(scene_dir: Path, scene_id: str) -> tuple[float, float] | None:
    """Floor extent, read from the USDA physics loads rather than from any builder variable."""
    path = scene_dir / f"{scene_id}.usda"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    floor = text[text.index('Plane "Floor"') :]
    width = re.search(r"double width = ([\d.]+)", floor)
    length = re.search(r"double length = ([\d.]+)", floor)
    if not (width and length):
        return None
    return float(width.group(1)), float(length.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rollouts", type=Path, required=True, help="searched recursively")
    ap.add_argument(
        "--scenes", type=Path, default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
    )
    ap.add_argument("--threshold", type=float, default=SUSPECT_GAP_M)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.rollouts.rglob("*.trajectory.pkl")):
        cell = path.parents[1].name
        log = path.parents[2] / "logs" / f"{cell}.runner.log"
        if not log.exists():
            continue
        match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
        if not match:
            continue
        room = room_size(args.scenes, match.group(1))
        if room is None:
            continue
        with open(path, "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        if payload is None:
            continue
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)[:, :2]
        gap = min(
            float(room[0] / 2 - WALL_INSET_M - np.abs(root[:, 0]).max()),
            float(room[1] / 2 - WALL_INSET_M - np.abs(root[:, 1]).max()),
        )
        rows.append(
            {
                "cell": cell,
                "scene": match.group(1),
                "wall_gap_m": round(gap, 4),
                "outcome": classify_episode(cell, payload).outcome,
            }
        )

    rows.sort(key=lambda r: r["wall_gap_m"])
    suspect = [r for r in rows if r["wall_gap_m"] < args.threshold]
    print(f"{'wall gap':>10s}{'outcome':>10s}  cell / scene")
    for row in rows[: max(10, len(suspect))]:
        flag = "   <-- SUSPECT" if row["wall_gap_m"] < args.threshold else ""
        print(
            f"{row['wall_gap_m']:8.2f} m{row['outcome']:>10s}  "
            f"{row['cell']} / {row['scene']}{flag}"
        )
    print(f"\naudited {len(rows)} cells; under {args.threshold} m: {len(suspect)}")
    if suspect:
        print(
            "A rejection among these is not evidence about the motion. Re-run it in a room "
            "sized to the path, or in an empty screening scene, before quoting any yield."
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 1 if suspect else 0


if __name__ == "__main__":
    raise SystemExit(main())
