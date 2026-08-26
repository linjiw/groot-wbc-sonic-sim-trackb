#!/usr/bin/env python3
"""Build the E4 index view from Phase-1 rows plus E2-verified variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_episode_fields, base_episodes = read(args.baseline / "episodes.csv")
    base_family_fields, base_families = read(args.baseline / "families.csv")
    e2_episode_fields, e2_episodes = read(args.e2 / "episodes.csv")
    e2_family_fields, e2_families = read(args.e2 / "families.csv")
    if base_episode_fields != e2_episode_fields:
        raise SystemExit("E2 episode schema drifted from the additive baseline")
    verified = {
        row["family_id"]
        for row in e2_families
        if row["status"] == "verified" and row["evidence_valid"] == "1"
    }
    added_episodes = [row for row in e2_episodes if row["family_id"] in verified]
    added_families = [row for row in e2_families if row["family_id"] in verified]
    episode_ids = {row["episode_id"] for row in base_episodes}
    family_ids = {row["family_id"] for row in base_families}
    if episode_ids.intersection(row["episode_id"] for row in added_episodes):
        raise SystemExit("E2 episode IDs overlap the Phase-1 baseline")
    if family_ids.intersection(row["family_id"] for row in added_families):
        raise SystemExit("E2 family IDs overlap the Phase-1 baseline")

    family_fields = base_family_fields + [
        field for field in e2_family_fields if field not in base_family_fields
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "episodes.csv", base_episode_fields, base_episodes + added_episodes)
    write(args.out / "families.csv", family_fields, base_families + added_families)
    shutil.copyfile(args.baseline / "motions.csv", args.out / "motions.csv")
    provenance = {
        "schema_version": "lfh_e4_index_v1",
        "baseline": {
            "path": str(args.baseline),
            "episodes_sha256": sha256(args.baseline / "episodes.csv"),
            "families_sha256": sha256(args.baseline / "families.csv"),
            "motions_sha256": sha256(args.baseline / "motions.csv"),
        },
        "e2": {
            "path": str(args.e2),
            "episodes_sha256": sha256(args.e2 / "episodes.csv"),
            "families_sha256": sha256(args.e2 / "families.csv"),
        },
        "inclusion": "E2 rows whose variant is verified with evidence_valid=1",
        "added_verified_variants": sorted(verified),
        "added_episodes": len(added_episodes),
        "generated_e3_families": 0,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"PASS: {len(base_episodes)} + {len(added_episodes)} episodes; "
        f"{len(base_families)} + {len(added_families)} variants"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
