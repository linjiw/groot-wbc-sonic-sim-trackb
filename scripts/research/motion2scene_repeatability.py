#!/usr/bin/env python3
"""Prepare and analyze the original-clock 41002 paired repeatability experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

SEEDS = (7901, 7902, 7903)
LEVELS = ("neutral", "d055")


def summarize(rows):
    expected = {(seed, level) for seed in SEEDS for level in LEVELS}
    indexed = {(row["runtime_seed"], row["label"]): row for row in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("requires exactly the six registered cells")
    if any(row["status"] not in ("completed", "skipped_dependency") for row in rows):
        raise ValueError("requires a complete batch, not interim outcomes")
    counts = {}
    for level in LEVELS:
        group = [indexed[seed, level] for seed in SEEDS]
        counts[level] = {
            "registered": len(SEEDS),
            "completed": sum(row["status"] == "completed" for row in group),
            "tracker_and_route_pass": sum(
                row.get("tracker_accepted") is True and row.get("route_retained") is True
                for row in group
            ),
            "paired_behavior_retained": (
                None
                if level == "neutral"
                else sum(row.get("paired_behavior_retained") is True for row in group)
            ),
            "unique_achieved_state_fingerprints": len(
                {row["state_sha256"] for row in group if "state_sha256" in row}
            ),
        }
    predictions = {
        "p1_all_neutrals_tracker_and_route_pass": counts["neutral"]["tracker_and_route_pass"] == 3,
        "p2_all_crouches_tracker_and_route_pass": counts["d055"]["tracker_and_route_pass"] == 3,
        "p3_all_pairs_behavior_retained": counts["d055"]["paired_behavior_retained"] == 3,
    }
    return {
        "counts": counts,
        "predictions": predictions,
        "strict_development_repeatability": all(predictions.values()),
        "independent_carriers": 1,
        "physics_repeats": 3,
        "q4_admitted_ladders": 0,
        "training_eligible": False,
    }


def prepare(args):
    if args.output.exists():
        raise FileExistsError("refusing to overwrite repeatability artifacts")
    result_path = args.source / "result.json"
    result = json.loads(result_path.read_text())
    template_path = checked(args.source / "manifest.json", result["manifest"]["sha256"])
    template = json.loads(template_path.read_text())
    matches = [
        row
        for row in result["rows"]
        if row["generation_seed"] == 41002 and row["condition"] == "original"
    ]
    if (
        not result["analysis_complete"]
        or len(matches) != 2
        or not all(row["tracker_accepted"] and row["route_retained"] for row in matches)
    ):
        raise ValueError("source pair must pass its complete diagnostic")
    if not next(row for row in matches if row["label"] == "d055")["paired_behavior_retained"]:
        raise ValueError("source crouch did not retain paired behavior")
    source_cells = {
        cell["label"]: cell
        for cell in template["cells"]
        if cell["generation_seed"] == 41002 and cell["condition"] == "original"
    }
    cells = []
    for seed in SEEDS:
        for label in LEVELS:
            cell = copy.deepcopy(source_cells[label])
            checked(Path(cell["reference"]["path"]), cell["reference"]["sha256"])
            cell_id = f"m2s_repeat_41002_{seed}__{label}"
            cell.update(
                cell_id=cell_id,
                runtime_seed=seed,
                hydra_overrides=[f"++seed={seed}"],
                output=str(args.output / "rollouts" / cell_id),
            )
            if label != "neutral":
                cell["depends_on_acceptance_of"] = f"m2s_repeat_41002_{seed}__neutral"
            cells.append(cell)
    template.update(
        {
            "experiment": "M2S-E1-original-clock-repeatability-v1",
            "purpose": "Three new physics seeds for the selected 41002 pair; development only",
            "analysis_role": "selected_carrier_development_repeatability",
            "cells": cells,
            "registered_predictions": artifact(ROOT / "docs/motion2scene/REPEATABILITY_V1.md"),
            "diagnostic_implementation": artifact(Path(__file__)),
            "source_result": artifact(result_path),
            "eligibility": {
                "rule": "41002 original pair selected post-outcome from the complete timing diagnostic",
                "reference_gate": artifact(result_path),
            },
        }
    )
    template["execution_policy"]["cost_ceiling"].update(rollouts=6, gpu_hours_contended=0.625)
    template["execution_policy"][
        "timing_override"
    ] = "New development repeatability seeds registered before launch; no Q4 admission."
    template["analysis_dependencies"] = {
        "io_helpers": artifact(ROOT / "scripts/research/motion2scene_timing_diagnostic.py"),
        **{
            name: artifact(ROOT / "gear_sonic/dataset_generation" / name)
            for name in (
                "hallucination/keypoints.py",
                "hallucination/motion_envelope.py",
                "swept_volume.py",
                "trajectory_segments.py",
                "trajectory_acceptance.py",
            )
        },
    }
    write_new(args.output / "manifest.json", template)
    print(f"Prepared {len(cells)} registered cells")


def analyze(args):
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for item in [
        manifest["diagnostic_implementation"],
        manifest["registered_predictions"],
        *manifest["analysis_dependencies"].values(),
        *manifest["motion_analysis_implementation"].values(),
    ]:
        checked(Path(item["path"]), item["sha256"])
    run_path = args.output / "run_record.json"
    run = json.loads(run_path.read_text())
    if run["status"] != "completed" or run["manifest_sha256"] != artifact(manifest_path)["sha256"]:
        raise ValueError("requires the complete matching run record")
    # Import the same standalone modules whose recorded files were checked above.
    route_module = Path(manifest["motion_analysis_implementation"]["route_retention"]["path"])
    sys.path[:0] = [str(ROOT), str(route_module.parents[2])]
    from motion2scene.motion.paired_semantics import (
        PairedSemanticPolicy,
        assess_controller_retention,
        assess_paired_reduction,
    )
    from motion2scene.motion.route_retention import assess_route_retention, compare_routes

    from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints
    from gear_sonic.dataset_generation.hallucination.motion_envelope import extract_envelope
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    rows, envelopes = [], {}
    stations = np.linspace(0, 1, 101)
    for cell in manifest["cells"]:
        record = run["cells"][cell["cell_id"]]
        row = {key: cell[key] for key in ("cell_id", "generation_seed", "runtime_seed", "label")}
        row.update(
            status=record["status"],
            tracker_accepted=None,
            route_retained=None,
            paired_behavior_retained=None,
        )
        if record["status"] == "completed":
            scientific = record["scientific"]
            source = scientific["artifacts"]
            path = checked(Path(source["trajectory"]), source["trajectory_sha256"])
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
            top = extract_envelope(
                extract_keypoints(payload), cell["cell_id"], fractions=stations
            ).up_m
            envelopes[cell["cell_id"]] = top
            state = np.concatenate(
                [payload["root_pos_w"], payload["root_quat_w"], payload["dof_pos"]], axis=1
            )
            fingerprint = (
                "sha256:" + hashlib.sha256(np.asarray(state, dtype="<f8").tobytes()).hexdigest()
            )
            row.update(
                tracker_accepted=scientific["outcome"] == "accepted",
                route_retained=decision.retained,
                route_decision=decision.to_dict(),
                route_metrics=route.to_dict(),
                diagnostics=scientific["diagnostics"],
                rejection_reasons=scientific["rejection_reasons"],
                trajectory=artifact(path),
                state_sha256=fingerprint,
                whole_body_top_m=top.tolist(),
            )
        rows.append(row)
    by_id = {row["cell_id"]: row for row in rows}
    cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    policy = PairedSemanticPolicy(minimum_effect=0.05)
    for row in rows:
        if row["label"] == "neutral" or row["status"] != "completed":
            continue
        cell_id = row["cell_id"]
        neutral_id = cells[cell_id]["depends_on_acceptance_of"]
        reference = assess_paired_reduction(
            stations,
            np.asarray(cells[cell_id]["reference"]["whole_body_top_m"]),
            np.asarray(cells[neutral_id]["reference"]["whole_body_top_m"]),
            route_valid=True,
            policy=policy,
        )
        achieved = assess_paired_reduction(
            stations,
            envelopes[cell_id],
            envelopes[neutral_id],
            route_valid=row["route_retained"] and by_id[neutral_id]["route_retained"],
            policy=policy,
        )
        retention = assess_controller_retention(reference, achieved, policy=policy)
        row.update(
            reference_semantic=reference.to_dict(),
            achieved_semantic=achieved.to_dict(),
            controller_retention=retention.to_dict(),
            paired_behavior_retained=bool(
                row["tracker_accepted"]
                and by_id[neutral_id]["tracker_accepted"]
                and retention.semantic_status.value == "controller_retained"
            ),
        )
    result = {
        "schema_version": "motion2scene_repeatability_result_v1",
        "analysis_complete": True,
        "manifest": artifact(manifest_path),
        "run_record": artifact(run_path),
        "driver": artifact(Path(__file__)),
        "summary": summarize(rows),
        "budget": run["budget"],
        "rows": rows,
    }
    write_new(args.output / "result.json", result)
    print(json.dumps(result["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/linjiw/research-data/groot-wbc/m2s-timing-diagnostic-v1"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"prepare": prepare, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
