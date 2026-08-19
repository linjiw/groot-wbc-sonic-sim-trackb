#!/usr/bin/env python3
"""Write the per-frame distance-to-constraint record beside every rolled-out cell.

The corpus records, for each episode, whether the body came close enough to the obstacle to matter
and by how much at its worst. It does not record the approach. A learner given only the worst frame
can be taught *that* a behaviour was required and never *when* to begin it, which is the timing gap
the plan names as the single most-wanted missing field.

Nothing here is a new measurement. Each series is the un-reduced form of a number the gate already
computes, from the same function, so a record and the verdict it explains cannot disagree.

The obstacle box comes from the scene file **named in the cell's own runtime manifest**, not from
the family name or the builder's variables. That indirection is the point: two families were once
built, eight cells rolled out and every cell accepted, because the obstacle sat in the reference
frame while the rollout offset the motion by -2 m -- and the builder agreed with itself the whole
time.

Usage::

    python scripts/research/export_constraint_distance.py \\
        /data/.../wsA/families --json /data/.../wsA/constraint_distance.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import build_counterfactual_family as cf  # noqa: E402

from gear_sonic.dataset_generation.constraint_distance import (  # noqa: E402
    constraint_distance_from_payload,
)
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload  # noqa: E402

#: Rate of the physics record, as distinct from the 30 fps reference clips.
RECORD_FPS = 50.0


def scene_of(cell: Path) -> Path | None:
    """The scene file this cell was actually rolled out against, from its runtime manifest."""
    manifest = cell / "success_manifest.json"
    if not manifest.exists():
        return None
    try:
        context = json.loads(manifest.read_text())["capture_context"]["scene"]
    except (KeyError, json.JSONDecodeError):
        return None
    path = Path(context.get("resolved") or context.get("path", ""))
    return path if path.exists() else None


def export_cell(cell: Path, *, fps: float = RECORD_FPS) -> dict | None:
    """Write one cell's record and return its summary row, or None if it cannot be built."""
    found = sorted(cell.glob("trajectories/*.trajectory.pkl"))
    scene = scene_of(cell)
    if not found or scene is None:
        return None
    try:
        payload, _ = best_evaluable_payload(pickle.load(open(found[0], "rb")))
    except Exception:  # noqa: BLE001
        return None
    if payload is None:
        return None

    box = cf.rendered_shelf_box(scene)
    record = constraint_distance_from_payload(payload, box, fps=fps)
    out = cell / "constraint_distance.npz"
    np.savez_compressed(out, **record.as_arrays())

    return {
        "cell": f"{cell.parent.name}/{cell.name}",
        "scene": scene.name,
        "frames": record.frames,
        "route_length_m": round(record.route_length_m, 4),
        "station_frame": record.station_frame,
        "bottleneck_frame": record.bottleneck_frame,
        "min_clearance_m": round(float(record.body_clearance_m.min()), 5),
        "binding_body": record.binding_body[record.bottleneck_frame],
        "first_overlap_frame": record.first_overlap_frame,
        "approach_m": round(float(record.remaining_to_station_m[0]), 3),
        "npz": str(out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory of family directories")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=RECORD_FPS)
    args = parser.parse_args()

    rows = []
    skipped = []
    for family in sorted(p for p in args.root.iterdir() if p.is_dir()):
        for cell in sorted(p for p in family.iterdir() if p.is_dir()):
            row = export_cell(cell, fps=args.fps)
            if row is None:
                skipped.append(f"{family.name}/{cell.name}")
                continue
            rows.append(row)

    header = f"{'cell':44s}{'frames':>7s}{'station':>8s}{'bottle':>7s}{'clear m':>9s}{'overlap':>8s}  binding"
    print(header)
    for row in rows:
        overlap = "-" if row["first_overlap_frame"] is None else str(row["first_overlap_frame"])
        print(
            f"{row['cell']:44s}{row['frames']:7d}{row['station_frame']:8d}"
            f"{row['bottleneck_frame']:7d}{row['min_clearance_m']:9.4f}{overlap:>8s}  "
            f"{row['binding_body']}"
        )

    print(f"\n{len(rows)} cells written, {len(skipped)} skipped")
    if skipped:
        # Named, not counted. A cell without a manifest is not a cell with no obstacle.
        print("skipped (no trajectory, no manifest, or no scene on disk):")
        for name in skipped:
            print(f"  {name}")
    if args.json:
        args.json.write_text(json.dumps({"cells": rows, "skipped": skipped}, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
