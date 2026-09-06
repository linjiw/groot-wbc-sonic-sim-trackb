#!/usr/bin/env python3
"""Test d040 against the frozen seed-matched neutral/d055 repeatability executions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import pickle
import sys

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

SEEDS = (7901, 7902, 7903)


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    result_path = args.source / "result.json"
    result = json.loads(result_path.read_text())
    if not result["analysis_complete"] or not result["summary"]["strict_development_repeatability"]:
        raise ValueError("extension requires the complete strict development pair")
    manifest = json.loads(
        checked(args.source / "manifest.json", result["manifest"]["sha256"]).read_text()
    )
    candidates_path = args.candidates
    candidates = json.loads(candidates_path.read_text())
    ladder = next(item for item in candidates["ladders"] if item["generation_seed"] == 41002)
    level = next(item for item in ladder["levels"] if item["label"] == "d040")
    if (
        level["q0_status"] != "passed"
        or level["q1_status"] != "passed"
        or level["semantic_assessment"]["semantic_status"] != "route_aligned"
    ):
        raise ValueError("intermediate reference failed admission")
    motion = checked(Path(level["sonic_motion"]), level["sonic_motion_sha256"])
    csv = checked(Path(level["csv"]), level["csv_sha256"])
    provenance = motion.with_suffix(".pkl.manifest.json")
    qpos = np.loadtxt(csv, delimiter=",")
    neutral_ref = manifest["cells"][0]["reference"]
    neutral = np.loadtxt(checked(Path(neutral_ref["path"]), neutral_ref["sha256"]), delimiter=",")
    if not np.array_equal(qpos[:, :2], neutral[:, :2]) or not np.array_equal(
        qpos[:, 3:7], neutral[:, 3:7]
    ):
        raise ValueError("reference route/orientation mismatch")
    cells = []
    for seed in SEEDS:
        cell = copy.deepcopy(
            next(
                cell
                for cell in manifest["cells"]
                if cell["runtime_seed"] == seed and cell["label"] == "neutral"
            )
        )
        cell_id = f"m2s_ladder_41002_{seed}__d040"
        cell.update(
            cell_id=cell_id,
            label="d040",
            body_mode="controlled_crouch",
            ladder_level=1,
            motion={
                **artifact(motion),
                "conversion_provenance": str(provenance),
                "conversion_provenance_sha256": artifact(provenance)["sha256"],
                "scene_start_xyz": [0.0, 0.0, 0.0],
            },
            reference={**artifact(csv), "whole_body_top_m": level["whole_body_top_m"]},
            output=str(args.output / "rollouts" / cell_id),
        )
        cells.append(cell)
    manifest.update(
        experiment="M2S-E1-original-clock-ladder-extension-v1",
        purpose="Intermediate d040 level against frozen seed-matched comparators",
        cells=cells,
        registered_predictions=artifact(ROOT / "docs/motion2scene/LADDER_EXTENSION_V1.md"),
        diagnostic_implementation=artifact(Path(__file__)),
        source_result=artifact(result_path),
        source_manifest=artifact(args.source / "manifest.json"),
        candidates=artifact(candidates_path),
        eligibility={
            "rule": "strict development pair retained; intermediate reference Q0/Q1/S3 passes",
            "reference_gate": artifact(result_path),
        },
    )
    manifest["execution_policy"]["cost_ceiling"].update(rollouts=3, gpu_hours_contended=0.3125)
    manifest["execution_policy"][
        "timing_override"
    ] = "Three new intermediate-level executions registered after complete pair repeatability."
    write_new(args.output / "manifest.json", manifest)


def analyze(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in [
        manifest["diagnostic_implementation"],
        manifest["registered_predictions"],
        manifest["source_result"],
        manifest["source_manifest"],
        *manifest["analysis_dependencies"].values(),
        *manifest["motion_analysis_implementation"].values(),
    ]:
        checked(Path(item["path"]), item["sha256"])
    source = json.loads(Path(manifest["source_result"]["path"]).read_text())
    source_manifest = json.loads(Path(manifest["source_manifest"]["path"]).read_text())
    run_path = args.output / "run_record.json"
    run = json.loads(run_path.read_text())
    if (
        run["status"] != "completed"
        or run["manifest_sha256"] != artifact(manifest_path)["sha256"]
        or len(run["cells"]) != 3
    ):
        raise ValueError("requires the complete three-cell batch")
    module = Path(manifest["motion_analysis_implementation"]["route_retention"]["path"])
    sys.path[:0] = [str(ROOT), str(module.parents[2])]
    from motion2scene.motion.paired_semantics import (
        PairedSemanticPolicy,
        assess_controller_retention,
        assess_paired_reduction,
    )
    from motion2scene.motion.route_retention import assess_route_retention, compare_routes

    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    stations = np.linspace(0, 1, 101)
    event = (stations >= 0.45) & (stations <= 0.55)
    policy = PairedSemanticPolicy(minimum_effect=0.05)
    rows = []
    for cell in manifest["cells"]:
        scientific = run["cells"][cell["cell_id"]]["scientific"]
        inputs = scientific["artifacts"]
        path = checked(Path(inputs["trajectory"]), inputs["trajectory_sha256"])
        with path.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        ref = cell["reference"]
        qpos = np.loadtxt(checked(Path(ref["path"]), ref["sha256"]), delimiter=",")
        route = compare_routes(
            qpos[:, :2],
            np.asarray(payload["root_pos_w"])[:, :2],
            expected_route="straight",
            reference_fps=30,
            achieved_fps=float(payload["fps"]),
        )
        decision = assess_route_retention(route)
        top = extract_envelope(extract_keypoints(payload), cell["cell_id"], fractions=stations).up_m
        seed = cell["runtime_seed"]
        matched = {row["label"]: row for row in source["rows"] if row["runtime_seed"] == seed}
        neutral_ref = next(
            c["reference"]
            for c in source_manifest["cells"]
            if c["runtime_seed"] == seed and c["label"] == "neutral"
        )
        reference = assess_paired_reduction(
            stations,
            np.asarray(ref["whole_body_top_m"]),
            np.asarray(neutral_ref["whole_body_top_m"]),
            route_valid=True,
            policy=policy,
        )
        achieved = assess_paired_reduction(
            stations,
            top,
            np.asarray(matched["neutral"]["whole_body_top_m"]),
            route_valid=decision.retained and matched["neutral"]["route_retained"],
            policy=policy,
        )
        retained = assess_controller_retention(reference, achieved, policy=policy)
        gaps = [
            float(np.min((np.asarray(matched["neutral"]["whole_body_top_m"]) - top)[event])),
            float(np.min((top - np.asarray(matched["d055"]["whole_body_top_m"]))[event])),
        ]
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "runtime_seed": seed,
                "label": "d040",
                "trajectory": artifact(path),
                "tracker_accepted": scientific["outcome"] == "accepted",
                "route_retained": decision.retained,
                "route_metrics": route.to_dict(),
                "route_decision": decision.to_dict(),
                "diagnostics": scientific["diagnostics"],
                "rejection_reasons": scientific["rejection_reasons"],
                "reference_semantic": reference.to_dict(),
                "achieved_semantic": achieved.to_dict(),
                "controller_retention": retained.to_dict(),
                "whole_body_top_m": top.tolist(),
                "paired_behavior_retained": scientific["outcome"] == "accepted"
                and matched["neutral"]["tracker_accepted"]
                and retained.semantic_status.value == "controller_retained",
                "minimum_adjacent_gap_m": gaps,
                "ordered_central_profiles": min(gaps) > 0,
            }
        )
    predictions = {
        "p1_all_tracker_and_route_pass": all(
            r["tracker_accepted"] and r["route_retained"] for r in rows
        ),
        "p2_all_behavior_retained": all(r["paired_behavior_retained"] for r in rows),
        "p3_all_central_profiles_ordered": all(r["ordered_central_profiles"] for r in rows),
    }
    result = {
        "schema_version": "motion2scene_ladder_extension_result_v1",
        "analysis_complete": True,
        "manifest": artifact(manifest_path),
        "run_record": artifact(run_path),
        "driver": artifact(Path(__file__)),
        "source_result": manifest["source_result"],
        "budget": run["budget"],
        "rows": rows,
        "predictions": predictions,
        "three_level_development_repeatability": all(predictions.values()),
        "q4_admitted_ladders": 0,
        "training_eligible": False,
    }
    write_new(args.output / "result.json", result)
    print(json.dumps(predictions, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/linjiw/research-data/groot-wbc/m2s-repeatability-v1"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path(
            "/home/linjiw/research-data/groot-wbc/cg-wbc-v2-shared-seed-confirmatory/e1_controlled_duck_ladders_v1/candidates.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"prepare": prepare, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
