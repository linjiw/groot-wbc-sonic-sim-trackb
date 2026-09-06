#!/usr/bin/env python3
"""Register the one-setting scene-query repair, retaining the whole v1 batch."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

from motion2scene_reactive_interface import analyze
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new


def prepare(prior, out):
    manifest_path = prior / "manifest.json"
    record_path = prior / "run_record.json"
    result_path = prior / "result.json"
    manifest = json.loads(manifest_path.read_text())
    record = json.loads(record_path.read_text())
    if (
        record["status"] != "completed"
        or record["manifest_sha256"] != artifact(manifest_path)["sha256"]
    ):
        raise ValueError("requires immutable completed v1")
    checked(result_path, artifact(result_path)["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(manifest)
    manifest["experiment"] = "M2S-reactive-interface-query-v2"
    manifest["registered_predictions"] = artifact(
        ROOT / "docs/motion2scene/REACTIVE_INTERFACE_QUERY_V2.md"
    )
    manifest["new_dependencies"] += [
        artifact(Path(__file__)),
        artifact(manifest_path),
        artifact(record_path),
        artifact(result_path),
        artifact(
            ROOT
            / "gear_sonic/dataset_generation/hallucination/motion2scene_reactive_query_execution.py"
        ),
        artifact(
            Path(
                "/home/linjiw/isaaclab-install/IsaacLab/source/isaaclab/isaaclab/sim/simulation_cfg.py"
            )
        ),
    ]
    manifest["prior_attempt"] = artifact(result_path)
    manifest["runtime_delta"] = {"sim.enable_scene_query_support": True}
    for cell in manifest["cells"]:
        cell["cell_id"] = cell["cell_id"].replace("interface_", "query_")
        cell["output"] = str(out / "rollouts" / cell["cell_id"])
        cell["hydra_overrides"] = [
            s.replace(
                "motion2scene_reactive_execution.ReactiveExecutionEnvCfg",
                "motion2scene_reactive_query_execution.QueryEnabledExecutionEnvCfg",
            )
            for s in cell["hydra_overrides"]
        ]
    write_new(out / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "preflight", "run", "analyze"))
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.prior, args.out)
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
    subprocess.run(command, check=True)
    if args.command == "run":
        analyze(args.out)


if __name__ == "__main__":
    main()
