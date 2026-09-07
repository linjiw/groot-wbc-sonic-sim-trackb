#!/usr/bin/env python3
"""Conservatively charge a batch by its latest activity, including delayed execution."""

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_timing_diagnostic import artifact, write_new


def review(data, remaining=20, now=None):
    now = now or datetime.datetime.now(datetime.timezone.utc)
    rows = []
    for path in sorted(data.glob("m2s-*/*run_record*.json")):
        r = json.loads(path.read_text())
        cost = float(r.get("budget", {}).get("actual_contended_gpu_hours", 0))
        if not cost:
            continue
        if r.get("analysis_only"):
            raise ValueError("analysis derivation must not enter physics spending ledger")
        timestamps = [r.get("started_at"), r.get("ended_at")]
        for c in r.get("cells", {}).values():
            timestamps += [c.get("started_at"), c.get("ended_at")]
        latest = max(datetime.datetime.fromisoformat(t) for t in timestamps if t)
        age = (now - latest).total_seconds() / 86400
        rows.append(
            {
                "record": artifact(path),
                "hours": cost,
                "latest_activity": latest.isoformat(),
                "age_days": age,
            }
        )
    day = sum(r["hours"] for r in rows if r["age_days"] <= 1)
    week = sum(r["hours"] for r in rows if r["age_days"] <= 7)
    reserve = remaining * 375 / 3600
    return {
        "rows": rows,
        "daily_hours": day,
        "weekly_hours": week,
        "reserved_hours": reserve,
        "admitted": day + reserve <= 8 and week + reserve <= 24,
        "checked_at": now.isoformat(),
        "rule": (
            "charge all actual batch cost at latest activity; conservative across "
            "rolling windows, never preflight date alone"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--next-blocks", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.next_blocks <= 30
    master = json.loads((args.out / "master.json").read_text())
    for _ in range(args.next_blocks):
        next_block = next(
            (b for b in master["blocks"] if not (Path(b["directory"]) / "admission.json").exists()),
            None,
        )
        if next_block is None:
            return
        folder = Path(next_block["directory"])
        record_path = folder / "run_record.json"
        r = json.loads(record_path.read_text()) if record_path.exists() else {"cells": {}}
        if any(c["status"] not in ("completed", "not_started") for c in r["cells"].values()):
            raise RuntimeError("active or failed cells require reconciliation before resuming")
        remaining = 20 - sum(c["status"] == "completed" for c in r["cells"].values())
        receipt = review(args.out.parent, remaining)
        receipt["implementation"] = artifact(Path(__file__))
        path = folder / f'budget_review_{len(list(folder.glob("budget_review_*.json"))):03d}.json'
        write_new(path, receipt)
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in ("daily_hours", "weekly_hours", "reserved_hours", "admitted")
                }
            ),
            flush=True,
        )
        if not args.execute or not receipt["admitted"]:
            return
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("motion2scene_independent_layouts.py")),
                "run",
                "--max-blocks",
                "1",
                "--out",
                str(args.out),
            ],
            check=True,
        )
        if not (folder / "admission.json").exists():
            return
        if not json.loads((folder / "admission.json").read_text())["admitted"]:
            raise RuntimeError("block audit did not admit inputs; do not launch next block")


if __name__ == "__main__":
    main()
