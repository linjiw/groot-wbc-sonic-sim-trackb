#!/usr/bin/env python3
# ruff: noqa: E501 -- Complete JSON protocol descriptions.
"""Reserve new source identities using the existing conservative local seed inventory."""

import json
from pathlib import Path

import motion2scene_fresh_sources as inventory_tools
from motion2scene_timing_diagnostic import ROOT, artifact, write_new


def main():
    source_ids = list(range(46001, 46009))
    # Explicit reuse of the general inventory with a new candidate set; original
    # 430xx registration, inputs and implementation stay unchanged.
    inventory_tools.SOURCE_IDS = source_ids
    inventory = inventory_tools.inventory()
    out = ROOT.parent / "research-data/groot-wbc/m2s-final-transfer-reservation-v1"
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "inventory.json", inventory)
    if inventory["collisions"] or inventory["unparsed"]:
        raise ValueError("source freshness unresolved; reservation refused")
    write_new(
        out / "reservation.json",
        {
            "source_ids": source_ids,
            "split": "final_transfer_test_only",
            "inventory": artifact(out / "inventory.json"),
            "driver": artifact(Path(__file__)),
            "inventory_implementation": artifact(Path(inventory_tools.__file__)),
            "generation_started": False,
            "training_eligible": False,
            "scope": "No seed occurrence in inventoried local JSON records/ranges; not a claim about unrecorded or external data",
            "grouping": "All motion derivatives, scene variants, physics seeds and traces stay with their source ancestor",
            "candidate_funnel": "Eight assigned candidates; no replacement after qualification refusals",
            "next": "Separate hash-pinned acquisition and same-switching-contract qualification manifest before generation/physics; do not inspect transfer outcomes to tune methods",
        },
    )
    print(json.dumps({"reserved_source_ids": source_ids, "generation_started": False}))


if __name__ == "__main__":
    main()
