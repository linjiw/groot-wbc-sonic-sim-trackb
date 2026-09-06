#!/usr/bin/env python3
"""Post-outcome forensic audit; never changes or refits the frozen selector."""

import argparse
from collections import Counter
import json
from pathlib import Path

from motion2scene_timing_diagnostic import artifact, checked, write_new


def audit(result_path, output):
    result = json.loads(result_path.read_text())
    rows = []
    for row in result["rows"]:
        if row["mode"] != "reactive":
            continue
        ref = row["sensor"]
        sensor = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        hits = [
            (o["time_s"], r["hit"])
            for o in sensor["observations"]
            for r in o["rays"]
            if r["hit"] and 1.15 <= r["hit"]["position"][2] <= 1.5
        ]
        rows.append(
            {
                "cell_id": row["cell_id"],
                "condition": row["condition"],
                "seed": row["seed"],
                "sensor": ref,
                "first_occupied_hit": hits[0] if hits else None,
                "qualifying_hit_identity_counts": dict(Counter(h["path"] for _, h in hits)),
                "switches": sensor["switches"],
                "switch_contract_pass": row["switch_contract_pass"],
            }
        )
    write_new(
        output,
        {
            "result": artifact(result_path),
            "implementation": artifact(Path(__file__)),
            "analysis_role": "post-outcome failure diagnosis; hit identity never fed to selector",
            "rows": rows,
            "finding": (
                "All four absent/raised controls first trigger on WallEast. A hit in a height band "
                "does not establish free space underneath an obstacle."
            ),
            "next_test": (
                "Distinguish overhead occupancy from a wall with lower/upper free-space observations; "
                "enforce legal transition timing; freeze fresh tests before execution."
            ),
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit(args.result, args.output)
