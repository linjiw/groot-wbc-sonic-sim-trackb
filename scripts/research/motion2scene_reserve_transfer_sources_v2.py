#!/usr/bin/env python3
# ruff: noqa: E501 -- JSON protocol descriptions.
"""Inventory new source seeds with a conservative raw-text negative prefilter."""

import json
import os
from pathlib import Path
import re
import time

from motion2scene_fresh_sources import DATA, SIBLING, large_artifact, seed_mentions
from motion2scene_timing_diagnostic import ROOT, artifact, write_new

IDS = list(range(9460001, 9460009))


def scan(raw, ids):
    value = json.loads(raw)
    pattern = rb"(?<![0-9])(?:" + b"|".join(str(i).encode() for i in ids) + rb")(?![0-9])"
    # Integers cannot be encoded without their digits. Implied seed ranges and
    # escaped keys force the original full scan; this shortcut cannot waive them.
    if not re.search(pattern, raw) and b"seed_base" not in raw and b"\\" not in raw:
        return set()
    return seed_mentions(value, ids)


def main():
    start = time.monotonic()
    rows = []
    collisions = []
    unparsed = []
    for root in (ROOT, SIBLING, DATA):
        for directory, dirs, files in os.walk(root):
            dirs[:] = [
                d
                for d in dirs
                if d not in {".git", "__pycache__", "node_modules"} and not d.startswith(".venv")
            ]
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                if time.monotonic() - start > 180:
                    raise TimeoutError("180 s inventory ceiling")
                path = Path(directory) / name
                raw = path.read_bytes()
                ref = large_artifact(path)
                rows.append(ref)
                try:
                    found = scan(raw, IDS)
                except (ValueError, UnicodeError) as e:
                    unparsed.append({**ref, "error": str(e)})
                    continue
                if found:
                    collisions.append({**ref, "seeds": sorted(found)})
    out = DATA / "m2s-final-transfer-reservation-v2"
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "inventory.json",
        {
            "roots": [str(ROOT), str(SIBLING), str(DATA)],
            "files": rows,
            "collisions": collisions,
            "unparsed": unparsed,
            "seconds": time.monotonic() - start,
            "scope": "local JSON only; conservative integer/range scan with equivalent negative prefilter",
        },
    )
    if collisions or unparsed:
        raise ValueError("freshness unresolved; no reservation")
    write_new(
        out / "reservation.json",
        {
            "source_ids": IDS,
            "split": "final_transfer_test_only",
            "inventory": artifact(out / "inventory.json"),
            "driver": artifact(Path(__file__)),
            "prior_refused_inventory": artifact(
                DATA / "m2s-final-transfer-reservation-v1/inventory.json"
            ),
            "generation_started": False,
            "training_eligible": False,
            "scope": "No integer/range mentions in inventoried local JSON; unrecorded or external history is outside this claim",
            "grouping": "all derivatives, scene variants, physics repeats and sensor traces remain in this test-only ancestor group",
            "candidate_funnel": "eight assigned candidates, no replacement for qualification refusals",
            "next": "hash-pin acquisition and same-switching-contract qualification before generating; never tune on transfer outcomes",
        },
    )
    print(json.dumps({"source_ids": IDS, "files": len(rows), "seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
