#!/usr/bin/env python3
"""Register and attempt conditional continuous-placement certificates for two beams."""

import argparse
import json
from pathlib import Path
import time

from motion2scene_inverse_learning import DOMAIN, inputs
from motion2scene_margin_learning import load
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import numpy_perturbed
import numpy as np

from gear_sonic.dataset_generation.hallucination.placement_certificate import certify_placement


def prepare(args):
    result = json.loads(args.result.read_text())
    if not result["analysis_complete"]:
        raise ValueError("requires completed margin analysis")
    ref = result["manifest"]
    checked(Path(ref["path"]), ref["sha256"])
    load(Path(ref["path"]))
    candidates = []
    for arm in ("margin_no_kl", "constraints_no_kl"):
        row = next(r for r in result["rows"] if r["arm"] == arm and r["seed"] == 8121)
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        slack = np.minimum(
            np.array(row["worst_target_clearance_m"]) - 0.01,
            -0.01 - np.array(row["worst_upright_clearance_m"]),
        )
        index = int(np.argmax(slack))
        candidates.append(
            {
                "arm": arm,
                "seed": 8121,
                "proposal_index": index,
                "scene": raw["scenes"][index].tolist(),
                "raw": ref,
                "sampled_margin_slack_m": float(slack[index]),
            }
        )
    args.out.mkdir(parents=True, exist_ok=False)
    write_new(
        args.out / "manifest.json",
        {
            "schema_version": "motion2scene_placement_certificate_v1",
            "source_result": artifact(args.result),
            "source_manifest": result["manifest"],
            "protocol": artifact(args.protocol),
            "implementations": [
                artifact(Path(__file__)),
                artifact(ROOT / "scripts/research/motion2scene_margin_learning.py"),
                artifact(
                    ROOT / "gear_sonic/dataset_generation/hallucination/placement_certificate.py"
                ),
            ],
            "candidates": candidates,
            "lower": [-0.02, -0.02, -0.01, -0.02],
            "upper": [0.02, 0.02, 0.01, 0.02],
            "max_queries_per_candidate": 4095,
            "assumed_absolute_numerical_error_m": 1e-8,
            "margin_m": 0.01,
            "horizontal_radius_m": float(np.hypot(DOMAIN.depth / 2, DOMAIN.width / 2)),
            "max_seconds_per_candidate": 120,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )
    print(json.dumps({"manifest": str(args.out / "manifest.json")}), flush=True)


def run(args):
    manifest = json.loads(args.manifest.read_text())
    for ref in [
        manifest["source_result"],
        manifest["source_manifest"],
        manifest["protocol"],
        *manifest["implementations"],
        *[c["raw"] for c in manifest["candidates"]],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    _, source, states, route = load(Path(manifest["source_manifest"]["path"]))
    _, _, _, _, yaw, _, _ = inputs(states, route, source["records"])
    mask = np.array([r["label"] == "d055" for r in source["records"]])
    root = args.manifest.parent / "runs"
    root.mkdir(exist_ok=False)
    rows = []
    for candidate in manifest["candidates"]:
        started = time.monotonic()
        queries = []

        def query(offset):
            if time.monotonic() - started > manifest["max_seconds_per_candidate"]:
                raise TimeoutError("certificate time budget")
            values = numpy_perturbed(
                np.array([candidate["scene"]]), offset[None], states, route, yaw.item()
            )[0, 0]
            queries.append(
                {"offset": offset.tolist(), "per_recording_clearance_m": values.tolist()}
            )
            return float(values[mask].min()), float(values[~mask].max())

        try:
            certificate = certify_placement(
                query,
                manifest["lower"],
                manifest["upper"],
                horizontal_radius=manifest["horizontal_radius_m"],
                margin=manifest["margin_m"],
                query_error=manifest["assumed_absolute_numerical_error_m"],
                max_queries=manifest["max_queries_per_candidate"],
            )
        except TimeoutError:
            certificate = {"status": "unresolved_timeout", "queries": len(queries)}
        path = root / f'{candidate["arm"]}.json'
        write_new(
            path,
            {
                "candidate": candidate,
                "certificate": certificate,
                "queries": queries,
                "elapsed_s": time.monotonic() - started,
            },
        )
        row = {
            **candidate,
            "status": certificate["status"],
            "queries": certificate["queries"],
            "elapsed_s": time.monotonic() - started,
            "artifact": artifact(path),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_new(
        args.manifest.parent / "result.json",
        {
            "manifest": artifact(args.manifest),
            "rows": rows,
            "analysis_complete": True,
            "numerical_error_bound_formally_validated": False,
            "continuous_time_certified": False,
            "imported_robot_geometry_certified": False,
            "training_eligible": False,
            "execution_eligible": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
