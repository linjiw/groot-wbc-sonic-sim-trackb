#!/usr/bin/env python3
"""Report real execution outcomes with paired carrier/layout/seed denominators."""

import argparse
import json
from pathlib import Path

from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np
from scipy.stats import binomtest

PAIRS = [
    ("motion2scene", "analytic"),
    ("motion2scene", "uniform"),
    ("analytic", "uniform"),
    ("motion2scene", "no_contrast"),
]


def compare(rows, first, second):
    left = {r["group_id"]: r for r in rows if r["arm"] == first}
    right = {r["group_id"]: r for r in rows if r["arm"] == second}
    assert left.keys() == right.keys()
    counts = {"both_pass": 0, "only_first": 0, "only_second": 0, "both_fail": 0}
    carriers = {}
    for key in left:
        a, b = left[key], right[key]
        assert a["source"] == b["source"]
        name = (
            "both_pass"
            if a["pass"] and b["pass"]
            else "only_first" if a["pass"] else "only_second" if b["pass"] else "both_fail"
        )
        counts[name] += 1
        carriers.setdefault(str(a["source"]), []).append(int(a["pass"]) - int(b["pass"]))
    discordant = counts["only_first"] + counts["only_second"]
    test = binomtest(counts["only_first"], discordant) if discordant else None
    interval = test.proportion_ci() if test else None
    return {
        "first": first,
        "second": second,
        "paired_conditions": len(left),
        "counts": counts,
        "carrier_differences": {s: float(np.mean(v)) for s, v in carriers.items()},
        "carrier_averaged_difference": (
            float(np.mean([np.mean(v) for v in carriers.values()])) if carriers else None
        ),
        "descriptive_discordance": {
            "p_value": test.pvalue if test else None,
            "first_win_fraction_interval": [interval.low, interval.high] if interval else None,
            "limitation": (
                "Binomial independence is not established across repeated conditions within "
                "three carriers; not a population claim"
            ),
        },
    }


def summarize(out):
    master = json.loads((out / "evaluation_master.json").read_text())
    rows, pending, refs = [], [], []
    for b in master["batches"]:
        folder = Path(b["directory"])
        if not (folder / "admission.json").exists():
            pending.append(b)
            continue
        a = json.loads((folder / "admission.json").read_text())
        assert a["admitted"]
        result = json.loads(checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text())
        rows.extend(result["rows"])
        refs.append(artifact(folder / "admission.json"))
    traversal = [r for r in rows if r["suite"] == "traversal"]
    backgrounds = [r for r in rows if r["suite"] != "traversal"]
    per_arm = {}
    for arm in sorted({r["arm"] for r in rows}):
        selected = [r for r in traversal if r["arm"] == arm]
        carriers = {
            str(s): [r for r in selected if r["source"] == s]
            for s in sorted({r["source"] for r in selected})
        }
        per_arm[arm] = {
            "completed": len(selected),
            "pass": sum(r["pass"] for r in selected),
            "carrier_passage": {
                s: sum(r["pass"] for r in rr) / len(rr) for s, rr in carriers.items()
            },
            "d040_requests": sum(r["readout"]["requested_action"] == 1 for r in selected),
            "refusals": sum(r["readout"]["refusal"] for r in selected),
            "successful_walk_refusals": sum(
                r["readout"]["refusal"] and r["pass"] for r in selected
            ),
        }
    comparisons = [compare(traversal, a, b) for a, b in PAIRS] if traversal else []
    result = {
        "master": artifact(out / "evaluation_master.json"),
        "admissions": refs,
        "completed": len(rows),
        "assigned": 540,
        "pending": sum(b["assigned"] for b in pending),
        "traversal_completed": len(traversal),
        "traversal_assigned": 432,
        "background_completed": len(backgrounds),
        "per_arm": per_arm,
        "comparisons": comparisons,
        "rows": rows,
        "complete": not pending,
        "scope": "Actual admitted executions only; partial results are not equivalence or the finished endpoint",
    }
    write_new(out / f"comparison_{len(rows):03d}.json", result)
    print(json.dumps({"complete": not pending, "admitted": len(rows), "assigned": 540}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    summarize(p.parse_args().out)
