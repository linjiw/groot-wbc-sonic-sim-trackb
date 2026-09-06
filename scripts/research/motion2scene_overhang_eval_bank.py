#!/usr/bin/env python3
"""Freeze evaluation-bank repair and audit actual loaded references."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import joblib
from motion2scene_overhang_interface import analyze as passage_analyze
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_reference_bank import audit_root_route

DATA = ROOT.parent / "research-data/groot-wbc"
PRIOR = DATA / "m2s-overhang-interface-v1"
PREFIX = "gear_sonic.dataset_generation.hallucination.motion2scene_overhang_eval_execution"


def prepare(out):
    manifest = json.loads((PRIOR / "manifest.json").read_text())
    result = json.loads((PRIOR / "result.json").read_text())
    if result["predictions"] != {"p1": False, "p2": False, "p3": True, "p4": True}:
        raise ValueError("unexpected prior outcome")
    manifest = copy.deepcopy(manifest)
    manifest.update(
        experiment="M2S-overhang-evaluation-bank-v2",
        registered_predictions=artifact(ROOT / "docs/motion2scene/OVERHANG_EVALUATION_BANK_V2.md"),
    )
    manifest["new_dependencies"] += [
        artifact(Path(__file__)),
        artifact(PRIOR / "manifest.json"),
        artifact(PRIOR / "result.json"),
        artifact(PRIOR / "run_record.json"),
        artifact(ROOT / "gear_sonic/utils/motion_lib/motion_lib_base.py"),
        *[
            artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{n}.py")
            for n in ("motion2scene_overhang_eval_execution", "motion2scene_reference_bank")
        ],
    ]
    for c in manifest["cells"]:
        c["cell_id"] = c["cell_id"].replace("overhang_", "evalbank_")
        c["output"] = str(out / "rollouts" / c["cell_id"])
        c["hydra_overrides"] = [
            s.replace(
                "gear_sonic.dataset_generation.hallucination.motion2scene_overhang_execution.OverhangExecutionEnvCfg",
                f"{PREFIX}.EvaluationBankEnvCfg",
            ).replace(
                "gear_sonic.dataset_generation.hallucination.motion2scene_overhang_execution.OverhangRecorderCfg",
                f"{PREFIX}.EvaluationBankRecorderCfg",
            )
            for s in c["hydra_overrides"]
        ]
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "manifest.json", manifest)


def analyze(out):
    if not (out / "result.json").exists():
        passage_analyze(out)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = []
    first = None
    for c in manifest["cells"]:
        folder = Path(c["output"]) / "trajectories"
        path = folder / "loaded_reference_bank.npz"
        metadata = folder / "loaded_reference_bank.json"
        with np.load(path) as bank:
            roots = bank["root_xyz"]
            dofs = bank["joint_pos"]
            if roots.shape != (2, 199, 3) or dofs.shape != (2, 199, 29) or float(bank["fps"]) != 50:
                raise ValueError("invalid recorded reference bank")
            errors = []
            for i, ref in enumerate([c["motion"], manifest["alternate_motion"]]):
                checked(Path(ref["path"]), ref["sha256"])
                entry = next(iter(joblib.load(ref["path"]).values()))
                errors.append(
                    audit_root_route(roots[i], entry["root_trans_offset"], float(entry["fps"]))
                )
            if first is None:
                first = (roots.copy(), dofs.copy())
            difference = max(
                float(np.max(np.abs(roots - first[0]))), float(np.max(np.abs(dofs - first[1])))
            )
        meta = json.loads(metadata.read_text())
        if (
            meta["source_route_max_error_m"] != errors
            or meta["alternate_loader"] != "load_motions_for_evaluation"
        ):
            raise ValueError("runtime bank audit mismatch")
        rows.append(
            {
                "cell_id": c["cell_id"],
                "bank": artifact(path),
                "metadata": artifact(metadata),
                "source_route_errors_m": errors,
                "max_difference_from_first_bank": difference,
            }
        )
    write_new(
        out / "bank_audit.json",
        {
            "result": artifact(out / "result.json"),
            "rows": rows,
            "p5": len(rows) == 14 and all(r["max_difference_from_first_bank"] == 0 for r in rows),
            "scope": "Source-root XY interpolation and identical realized root/DOF banks across all cells",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.out)
        return
    m = json.loads((a.out / "manifest.json").read_text())
    for r in m["new_dependencies"]:
        checked(Path(r["path"]), r["sha256"])
    if a.command == "analyze":
        analyze(a.out)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(a.out / "manifest.json"),
        "--run-record",
        str(a.out / "run_record.json"),
    ]
    if a.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if a.command == "run":
        analyze(a.out)


if __name__ == "__main__":
    main()
