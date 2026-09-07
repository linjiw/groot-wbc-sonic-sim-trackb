#!/usr/bin/env python3
"""Bounded source reservation with byte filtering and iterative metadata traversal."""

import hashlib
import json
import os
from pathlib import Path
import time

from motion2scene_fresh_sources import DATA, SIBLING
from motion2scene_timing_diagnostic import ROOT, artifact, write_new

IDS = list(range(9480001, 9480009))


def mentions(value, seeds):
    """Same integer and implied-range semantics; avoid recursive scalar-set creation."""
    seeds = set(seeds)
    hits = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            base, count, prompts = (
                item.get("seed_base"),
                item.get("seeds_per_prompt"),
                item.get("prompts", 1),
            )
            if all(type(x) is int for x in (base, count, prompts)) and count > 0 and prompts > 0:
                for seed in seeds:
                    index = (
                        0
                        if item.get("seed_design") == "shared_across_prompts"
                        else min((seed - base) // 1000, prompts - 1)
                    )
                    if index >= 0 and base + 1000 * index <= seed < base + 1000 * index + count:
                        hits.add(seed)
            children = item.values()
        elif isinstance(item, list):
            children = item
        else:
            if type(item) is int and item in seeds:
                hits.add(item)
            continue
        for child in children:
            if isinstance(child, (dict, list)):
                stack.append(child)
            elif type(child) is int and child in seeds:
                hits.add(child)
    return hits


def scan(raw, seeds):
    # ASCII integer spellings cannot hide in valid UTF-8 JSON. Unicode key escapes
    # and UTF-16/32 (NUL bytes) force parsing so implied ranges cannot evade checking.
    possible = any(str(seed).encode() in raw for seed in seeds)
    if not possible and b"seed_base" not in raw and b"\\u" not in raw and b"\x00" not in raw:
        return set()
    return mentions(json.loads(raw), seeds)


def main():
    start = time.monotonic()
    rows, collisions, unparsed = [], [], []
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
                ref = {
                    "path": str(path.absolute()),
                    "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
                }
                rows.append(ref)
                try:
                    found = scan(raw, IDS)
                except (ValueError, UnicodeError) as error:
                    unparsed.append({**ref, "error": str(error)})
                    continue
                if found:
                    collisions.append({**ref, "seeds": sorted(found)})
    out = DATA / "m2s-final-transfer-reservation-v4"
    out.mkdir(exist_ok=False)
    write_new(
        out / "inventory.json",
        {
            "roots": [str(p) for p in (ROOT, SIBLING, DATA)],
            "files": rows,
            "collisions": collisions,
            "unparsed": unparsed,
            "seconds": time.monotonic() - start,
            "scope": "local JSON metadata at scan time; byte-negative proof is not a whole-file syntax audit",
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
            "generation_started": False,
            "training_eligible": False,
            "scope": "no integer/implied-range mentions in inventoried local JSON; no external or unrecorded-history claim",
            "grouping": "all derivatives, scenes, physics repeats and traces stay with their ancestor",
            "candidate_funnel": "eight candidates; qualification refusals retained without replacement",
            "next": "freeze same-0.30-s-switch qualification and acquisition manifest before generation",
        },
    )
    print(json.dumps({"source_ids": IDS, "files": len(rows), "seconds": time.monotonic() - start}))


if __name__ == "__main__":
    main()
