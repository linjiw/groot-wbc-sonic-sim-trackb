#!/usr/bin/env python3
"""Register typed-ray repair after preserving v2 callback failures."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_reactive_interface import analyze
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new


def prepare(prior, out):
    manifest_path, record_path = prior / "manifest.json", prior / "run_record.json"
    manifest, record = json.loads(manifest_path.read_text()), json.loads(record_path.read_text())
    if (
        record["status"] != "infrastructure_failure"
        or record["manifest_sha256"] != artifact(manifest_path)["sha256"]
    ):
        raise ValueError("requires retained v2 interruption")
    logs = sorted(prior.glob("rollouts/*/rollout.log"))
    errors = {str(p): p.read_text().count("object has no attribute 'get'") for p in logs}
    if not sum(errors.values()):
        raise ValueError("expected typed-hit callback failure not found")
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "prior_failure.json",
        {
            "manifest": artifact(manifest_path),
            "run_record": artifact(record_path),
            "status": "sensor_infrastructure_invalid_then_stopped",
            "logs": [artifact(p) for p in logs],
            "callback_error_counts": errors,
            "completed_runtime_cells": sum(
                v["status"] == "completed" for v in record["cells"].values()
            ),
            "interrupted_cells": [
                k for k, v in record["cells"].items() if v["status"] == "infrastructure_failure"
            ],
            "requested_cells": len(manifest["cells"]),
            "contended_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "science_note": (
                "normal runtime completion did not establish valid sensing; no v2 sensing pass admitted"
            ),
        },
    )
    manifest = copy.deepcopy(manifest)
    manifest["experiment"] = "M2S-reactive-interface-typed-v3"
    manifest["registered_predictions"] = artifact(
        ROOT / "docs/motion2scene/REACTIVE_INTERFACE_TYPED_V3.md"
    )
    manifest["new_dependencies"] += [
        artifact(manifest_path),
        artifact(record_path),
        artifact(Path(__file__)),
        artifact(out / "prior_failure.json"),
        *[artifact(p) for p in logs],
        *[
            artifact(ROOT / "gear_sonic/dataset_generation/hallucination" / f"{n}.py")
            for n in ("motion2scene_ray_observer", "motion2scene_reactive_typed_execution")
        ],
        artifact(
            Path(
                "/home/linjiw/isaaclab-install/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx-107.3.26+107.3.3.lx64.r.cp311.u353/omni/physx/bindings/_physx.pyi"
            )
        ),
    ]
    manifest["typed_hit_repair"] = artifact(out / "prior_failure.json")
    for cell in manifest["cells"]:
        cell["cell_id"] = cell["cell_id"].replace("query_", "typed_")
        cell["output"] = str(out / "rollouts" / cell["cell_id"])
        cell["hydra_overrides"] = [
            s.replace(
                "motion2scene_reactive_query_execution.QueryEnabledExecutionEnvCfg",
                "motion2scene_reactive_typed_execution.TypedRayExecutionEnvCfg",
            )
            for s in cell["hydra_overrides"]
        ]
    write_new(out / "manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    p.add_argument("--prior", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.prior, args.out)
        return
    manifest = json.loads((args.out / "manifest.json").read_text())
    for ref in manifest["new_dependencies"]:
        checked(Path(ref["path"]), ref["sha256"])
    if args.command == "analyze":
        analyze(args.out)
        return
    cmd = [
        sys.executable,
        str(ROOT / "scripts/research/hallucination/run_approved_manifest.py"),
        "--manifest",
        str(args.out / "manifest.json"),
        "--run-record",
        str(args.out / "run_record.json"),
    ]
    if args.command == "preflight":
        cmd.append("--dry-run")
    subprocess.run(cmd, check=True)
    if args.command == "run":
        analyze(args.out)


if __name__ == "__main__":
    main()
