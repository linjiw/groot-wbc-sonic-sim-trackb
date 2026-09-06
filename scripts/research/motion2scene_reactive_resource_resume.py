#!/usr/bin/env python3
"""Preserve completed interface cells and resume only unstarted work at 7500 MiB."""

import argparse
import copy
import json
from pathlib import Path
import sys

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new

sys.path.insert(0, str(ROOT / "scripts/research/hallucination"))
from run_approved_manifest import score, verify_cell, verify_manifest_artifacts  # noqa: E402


def prepare(prior, out):
    manifest_path, record_path = prior / "manifest.json", prior / "run_record.json"
    manifest, record = json.loads(manifest_path.read_text()), json.loads(record_path.read_text())
    if (
        record["status"] != "yielded_gpu_contention"
        or record["manifest_sha256"] != artifact(manifest_path)["sha256"]
    ):
        raise ValueError("requires a hash-matched resource yield")
    verify_manifest_artifacts(manifest)
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    for cell in manifest["cells"]:
        verify_cell(cell)
        previous = record["cells"].get(cell["cell_id"], {})
        if previous.get("status") == "completed":
            if score(cell) != previous["scientific"]:
                raise ValueError("historical result changed")
        elif previous.get("status") not in (None, "not_started"):
            raise ValueError("may only continue unstarted cells")
    out.mkdir(parents=True, exist_ok=False)
    manifest, record = copy.deepcopy(manifest), copy.deepcopy(record)
    manifest["new_dependencies"] += [
        artifact(manifest_path),
        artifact(record_path),
        artifact(Path(__file__)),
        artifact(ROOT / "docs/motion2scene/REACTIVE_INTERFACE_RESOURCE_V1.md"),
    ]
    manifest["execution_policy"]["runtime"]["free_gpu_mib_required"] = 7500
    manifest["stop_conditions"][0] = "refuse hash mismatches; yield below 7500 MiB free GPU"
    manifest["resource_revision"] = artifact(
        ROOT / "docs/motion2scene/REACTIVE_INTERFACE_RESOURCE_V1.md"
    )
    for cell in manifest["cells"]:
        if record["cells"].get(cell["cell_id"], {}).get("status") != "completed":
            cell["output"] = str(out / "rollouts" / cell["cell_id"])
    write_new(out / "manifest.json", manifest)
    record.update(
        manifest=str(out / "manifest.json"),
        manifest_sha256=artifact(out / "manifest.json")["sha256"],
        status="preflight_passed",
        ended_at=None,
        prior_record=artifact(record_path),
    )
    write_new(out / "run_record.json", record)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    prepare(args.prior, args.out)
