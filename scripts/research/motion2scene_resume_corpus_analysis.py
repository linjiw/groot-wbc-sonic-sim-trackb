#!/usr/bin/env python3
"""Resume analysis-only assembly after the frozen union omitted its proposal copy."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_complete_comparative_corpus import FIRST
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new


def analyze(out):
    union = out / "analysis_union"
    m = json.loads((union / "manifest.json").read_text())
    record = json.loads((union / "run_record.json").read_text())
    assert m["analysis_only"] and m["execution_policy"]["not_authorized"]
    assert record["analysis_only"] and len(record["cells"]) == len(m["cells"]) == 76
    assert record["manifest_sha256"] == artifact(union / "manifest.json")["sha256"]
    for ref in record["source_records"]:
        checked(Path(ref["path"]), ref["sha256"])
    proposal = m["proposal_assignment"]
    source = checked(Path(proposal["path"]), proposal["sha256"])
    assert source.resolve() == (FIRST / "proposals.json").resolve()
    write_new(
        out / "analysis_assembly_repair.json",
        {
            "failure": "frozen analysis assembly omitted proposals.json; no result or fits were written",
            "failure_log": artifact(Path("/tmp/m2s-corpus-analysis.log")),
            "repair": "copy the identical original frozen proposal bytes, then rerun unchanged analysis",
            "implementation": artifact(Path(__file__)),
            "proposal": proposal,
            "union_manifest": artifact(union / "manifest.json"),
            "scope": "CPU-only; no change to registered labels, fitting IDs, physics or prior artifacts",
        },
    )
    with (union / "proposals.json").open("xb") as handle:
        handle.write(source.read_bytes())
    from motion2scene_comparative_acquisition import analyze as analyze_slice

    analyze_slice(union)
    audit_driver = ROOT / "scripts/research/audit_motion2scene_comparison_capture.py"
    for command in ("register", "analyze"):
        subprocess.run(
            [sys.executable, str(audit_driver), command, "--out", str(union)], check=True
        )
    result = json.loads((union / "result.json").read_text())
    audit = json.loads((union / "capture_audit.json").read_text())
    returns = []
    for row in result["rows"]:
        if row["action"] != 1:
            continue
        sensors = json.loads(
            checked(Path(row["sensor"]["path"]), row["sensor"]["sha256"]).read_text()
        )
        exits = [s for s in sensors["switches"] if s["to"] == 0]
        returns.append(
            {
                "cell_id": row["cell_id"],
                "entry_executed": row["command_executed"],
                "return_logged": bool(exits),
                "return_times_s": [s["time_s"] for s in exits],
                "reset_count": row["reset_count"],
                "pass": row["pass"],
                "scope": (
                    "logged command return; no claim that a reset-containing capture "
                    "is one uninterrupted encounter"
                ),
            }
        )
    admitted = (
        len(result["pairs"]) == 38
        and all(p["valid"] for p in result["pairs"])
        and all(audit["predictions"].values())
    )
    write_new(
        out / "admission.json",
        {
            "registration": artifact(out / "registration.json"),
            "result": artifact(union / "result.json"),
            "capture_audit": artifact(union / "capture_audit.json"),
            "all_inputs_admitted": admitted,
            "pairs": result["pairs"],
            "command_returns": returns,
            "complete_scene_pairs": len(result["pairs"]),
            "physics_predictions": {
                "all_pairs_match": all(p["valid"] for p in result["pairs"]),
                "all_capture_checks": all(audit["predictions"].values()),
            },
            "promotion": "complete matched development corpus only; explicit fitting subset in registration",
            "held_out_evaluation": False,
        },
    )
    print(
        json.dumps(
            {
                "admitted": admitted,
                "pairs": len(result["pairs"]),
                "returns_logged": sum(r["return_logged"] for r in returns),
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    analyze(parser.parse_args().out)
