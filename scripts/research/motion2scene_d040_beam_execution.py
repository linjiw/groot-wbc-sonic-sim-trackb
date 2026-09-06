#!/usr/bin/env python3
"""Complete the missing intermediate cells of the frozen 41002 beam panel."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-beam-intervention-resource-v3"
SEEDS = (7911, 7912, 7913)


def summarize(rows):
    expected = {
        (s, label, condition)
        for s in SEEDS
        for label in ("neutral", "d040", "d055")
        for condition in ("absent", "present")
    }
    keys = {(r["seed"], r["label"], r["condition"]) for r in rows}
    if len(rows) != len(expected) or keys != expected:
        raise ValueError("requires the complete 18-cell panel without duplicates")
    rates = {
        f"{label}_{condition}": sum(
            r["pass"] for r in rows if r["label"] == label and r["condition"] == condition
        )
        for label in ("neutral", "d040", "d055")
        for condition in ("absent", "present")
    }
    return {
        "pass_counts_out_of_three": rates,
        "predictions": {
            "p1": all(
                r["pass"] and r["tracker_outcome"] == "accepted"
                for r in rows
                if r["label"] == "d040" and r["condition"] == "absent"
            ),
            "p2": rates["d040_present"] >= 1,
        },
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    prior_path = PRIOR / "paired_manifest.json"
    middle_path = DATA / "m2s-ladder-extension-resource-v2/manifest.json"
    prior = json.loads(prior_path.read_text())
    middle = json.loads(middle_path.read_text())["cells"][0]
    controls = json.loads((PRIOR / "controls_result.json").read_text())
    if not controls["controls_pass"]:
        raise ValueError("requires completed physical contact controls")
    manifest = copy.deepcopy(prior)
    cells = []
    for cell in prior["cells"]:
        if cell["label"] != "d055":
            continue
        new = copy.deepcopy(cell)
        for key in ("motion", "reference", "body_mode", "ladder_level", "label"):
            new[key] = copy.deepcopy(middle[key])
        new["cell_id"] = f"beam_{cell['runtime_seed']}_{cell['condition']}_d040"
        new["output"] = str(out / "rollouts" / new["cell_id"])
        cells.append(new)
    manifest.update(
        cells=cells,
        experiment="M2S-frozen-beam-d040-v1",
        analysis_role="selected_carrier_intermediate_development_diagnostic",
        registered_predictions=artifact(ROOT / "docs/motion2scene/BEAM_D040_EXECUTION_V1.md"),
        new_dependencies=prior["new_dependencies"]
        + [
            artifact(Path(__file__)),
            artifact(prior_path),
            artifact(middle_path),
            artifact(PRIOR / "paired_result.json"),
            artifact(PRIOR / "paired_run_record.json"),
            artifact(PRIOR / "controls_result.json"),
            controls["manifest"],
            controls["run_record"],
        ],
    )
    manifest["execution_policy"]["cost_ceiling"].update(rollouts=6, gpu_hours_contended=0.625)
    manifest["execution_policy"][
        "timing_override"
    ] = "User requests ICRA execution gaps; six frozen d040 cells after source pilot."
    manifest["authorization"][
        "note"
    ] = "User explicitly requests work toward the three-tier physical beam validation."
    write_new(out / "manifest.json", manifest)


def analyze(out):
    rows = []
    records = []
    for manifest_path, record_path in (
        (PRIOR / "paired_manifest.json", PRIOR / "paired_run_record.json"),
        (out / "manifest.json", out / "run_record.json"),
    ):
        manifest = json.loads(manifest_path.read_text())
        record = json.loads(record_path.read_text())
        if (
            record["status"] != "completed"
            or record["manifest_sha256"] != artifact(manifest_path)["sha256"]
        ):
            raise ValueError("requires completed hash-matched batches")
        records.append(record)
        for cell in manifest["cells"]:
            payload, forces, row = read_cell(cell, record)
            row.update(score_passage(payload, forces, manifest["beam"]), seed=cell["runtime_seed"])
            rows.append(row)
    old = json.loads((PRIOR / "paired_result.json").read_text())
    for row in old["rows"]:
        new = next(r for r in rows if r["cell_id"] == row["cell_id"])
        for key in (
            "pass",
            "observed_beam_contact",
            "maximum_beam_normal_force_n_through_passage",
            "tracker_outcome",
        ):
            if row[key] != new[key]:
                raise ValueError("historical paired result changed")
    write_new(
        out / "result.json",
        {
            "manifest": artifact(out / "manifest.json"),
            "run_record": artifact(out / "run_record.json"),
            "prior_result": artifact(PRIOR / "paired_result.json"),
            "rows": rows,
            **summarize(rows),
            "new_physics_runs": 6,
            "new_contended_gpu_hours": records[1]["budget"]["actual_contended_gpu_hours"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
        return
    manifest = json.loads((args.out / "manifest.json").read_text())
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    if args.command == "analyze":
        analyze(args.out)
        return
    command = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(args.out / "run_record.json"),
    ]
    if args.command == "preflight":
        command.append("--dry-run")
    rc = subprocess.call(command)
    if rc:
        raise SystemExit(rc)
    if args.command == "run" and not (args.out / "result.json").exists():
        analyze(args.out)


if __name__ == "__main__":
    main()
