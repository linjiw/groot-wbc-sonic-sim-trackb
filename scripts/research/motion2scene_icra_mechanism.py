#!/usr/bin/env python3
"""Read-only corpus and leave-four-out diagnostics for the fixed ICRA study."""

import argparse
import collections
import json
from pathlib import Path

from motion2scene_icra_compare import compare
from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action


def analyze(out, comparison):
    fit = json.loads((out / "fit.json").read_text())
    result = json.loads(comparison.read_text())
    by_group = collections.defaultdict(dict)
    for r in result["rows"]:
        by_group[r["group_id"]][r["arm"]] = r
    ordered = list(by_group)
    inputs = []
    for g in ordered:
        r = by_group[g]["analytic"]
        d = json.loads(checked(Path(r["decision"]["path"]), r["decision"]["sha256"]).read_text())
        inputs.append(d["features"])
    x = torch.tensor(np.array(inputs, np.float32))
    folds = []
    pairs = {p["group_id"]: p for p in fit["pairs"]}
    for fold in fit["folds"]:
        model = fold["model"]
        with np.load(checked(Path(model["path"]), model["sha256"])) as f:
            with torch.no_grad():
                p = (
                    (
                        (x / torch.tensor(f["scale"])) @ torch.tensor(f["weights"])
                        + torch.tensor(f["bias"])
                    )
                    .sigmoid()
                    .numpy()
                )
        choices = select_action(p)
        original = [by_group[g][fold["arm"]]["readout"]["selected_action"] for g in ordered]
        folds.append(
            {
                "arm": fold["arm"],
                "fold": fold["fold"],
                "withheld_ids": fold["withheld_labelled"],
                "withheld_useful": [
                    k for k in fold["withheld_labelled"] if pairs[k]["outcomes"] == [False, True]
                ],
                "request_d040": int(np.sum(choices == 1)),
                "refusals": int(np.sum(choices == -1)),
                "changed_selected_actions": int(np.sum(choices != original)),
                "conditions": len(choices),
                "model": model,
            }
        )
    measured = []
    for group, records in by_group.items():
        actions = collections.defaultdict(list)
        for r in records.values():
            actions[r["readout"]["requested_action"]].append(r)
        for action, rows in actions.items():
            assert (
                len(
                    {
                        (
                            r["pass"],
                            r["maximum_beam_normal_force_n_through_passage"],
                            r["reset_count"],
                        )
                        for r in rows
                    }
                )
                == 1
            )
        outcome = {str(a): rr[0]["pass"] for a, rr in actions.items()}
        measured.append(
            {
                "group": group,
                "source": next(iter(records.values()))["source"],
                "measured_actions": sorted(actions),
                "outcomes": outcome,
                "useful": outcome == {"0": False, "1": True},
                "harmful": outcome == {"0": True, "1": False},
            }
        )
    admission_refs = fit["admissions"]
    acquisition = []
    for ref in admission_refs:
        a = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        r = json.loads(checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text())
        acquisition.extend(r["rows"])
    report = {
        "comparison": artifact(comparison),
        "fit": artifact(out / "fit.json"),
        "fold_readouts": folds,
        "measured_action_coverage": measured,
        "extra_comparisons": [
            compare(result["rows"], "analytic", a) for a in ("scripted_rays", "privileged_geometry")
        ],
        "acquisition_rows": acquisition,
        "scope": (
            "Recorded-input fold sensitivity is not new deployed-policy performance; "
            "unknown command outcomes remain unknown."
        ),
    }
    write_new(out / f"mechanism_{result['completed']:03d}.json", report)
    print(
        json.dumps(
            {
                "folds": len(folds),
                "paired_commands_available": sum(len(r["measured_actions"]) == 2 for r in measured),
                "conditions": len(measured),
                "useful_measured": sum(r["useful"] for r in measured),
                "harmful_measured": sum(r["harmful"] for r in measured),
                "analytic_folds": [r for r in folds if r["arm"] == "analytic"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--comparison", type=Path, required=True)
    a = p.parse_args()
    analyze(a.out, a.comparison)
