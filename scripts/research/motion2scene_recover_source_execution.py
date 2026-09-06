#!/usr/bin/env python3
"""Recover complete saved source-pilot data without repeating a physics cell."""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

from motion2scene_source_execution import read_cell
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

sys.path.insert(0, str(ROOT / "scripts/research/hallucination"))
from run_approved_manifest import score, verify_cell, verify_manifest_artifacts  # noqa: E402


def recover(source, out):
    manifest_path = source / "absent_manifest.json"
    record_path = source / "absent_run_record.json"
    manifest = json.loads(manifest_path.read_text())
    record = json.loads(record_path.read_text())
    if record["manifest_sha256"] != artifact(manifest_path)["sha256"]:
        raise ValueError("original manifest/record mismatch")
    if record["status"] != "infrastructure_failure":
        raise ValueError("recovery requires the recorded infrastructure failure")
    verify_manifest_artifacts(manifest)
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    for cell in manifest["cells"]:
        verify_cell(cell)
    failed = [k for k, v in record["cells"].items() if v["status"] == "infrastructure_failure"]
    if failed != ["source_41005_absent_d055"]:
        raise ValueError("this recovery is restricted to the documented interrupted cell")
    if record["cells"][failed[0]]["error"] != "RuntimeError: driver exited -13":
        raise ValueError("unexpected failure mechanism")
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "recovery_registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/SOURCE_EXECUTION_RECOVERY_V1.md"),
            "driver": artifact(Path(__file__)),
            "original_manifest": artifact(manifest_path),
            "original_record": artifact(record_path),
            "rerun_existing_cells": False,
        },
    )
    recovered = copy.deepcopy(record)
    validations = []
    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        if cell_id not in record["cells"]:
            continue
        original = record["cells"][cell_id]
        scientific = score(cell)
        if original["status"] == "completed" and scientific != original["scientific"]:
            raise ValueError("completed scientific result changed")
        recovered["cells"][cell_id] = {
            **original,
            "status": "completed",
            "scientific": scientific,
            "artifact_recovery": original["status"] != "completed",
        }
        _, _, validation = read_cell(cell, recovered)
        validations.append(validation)
    # Copy bytes: original science and all existing raw output paths are retained.
    for name in (
        "absent_manifest.json",
        "present_template.json",
        "proposals.json",
        "registration.json",
    ):
        shutil.copyfile(source / name, out / name)
    recovered.update(
        status="preflight_passed",
        ended_at=None,
        manifest=str(out / "absent_manifest.json"),
        recovery_registration=artifact(out / "recovery_registration.json"),
    )
    write_new(out / "recovery_validation.json", {"rows": validations, "new_physics_runs": 0})
    write_new(out / "absent_run_record.json", recovered)
    print(f"Recovered {len(validations)} saved cells; no physics rerun; {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recover(args.source, args.out)


if __name__ == "__main__":
    main()
