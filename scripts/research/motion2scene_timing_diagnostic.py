#!/usr/bin/env python3
"""Prepare/analyze a paired Motion2Scene timing diagnostic without promoting motion labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (41001, 41002, 41003)
CONDITIONS = ("original", "shared_clock")
LEVELS = ("neutral", "d055")


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def checked(path: Path, expected: str) -> Path:
    if sha(path) != expected:
        raise ValueError(f"hash mismatch: {path}")
    return path


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha(path)}


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def summarize(rows: list[dict]) -> dict:
    """Require the complete registered factorial; retain dependency skips in denominators."""
    expected = {
        (seed, condition, level) for seed in SEEDS for condition in CONDITIONS for level in LEVELS
    }
    indexed = {(r["generation_seed"], r["condition"], r["label"]): r for r in rows}
    if len(rows) != len(expected) or set(indexed) != expected:
        raise ValueError("analysis requires exactly the twelve registered cells")
    if any(r["status"] not in {"completed", "skipped_dependency"} for r in rows):
        raise ValueError("analysis requires a finished batch; no interim adjudication")
    counts = {}
    for condition in CONDITIONS:
        for level in LEVELS:
            selected = [indexed[seed, condition, level] for seed in SEEDS]
            counts[f"{condition}/{level}"] = {
                "registered": len(SEEDS),
                "completed": sum(r["status"] == "completed" for r in selected),
                "tracker_accepted": sum(r.get("tracker_accepted") is True for r in selected),
                "relative_route_pass": sum(r.get("route_retained") is True for r in selected),
                "paired_behavior_retained": sum(
                    r.get("paired_behavior_retained") is True for r in selected
                ),
            }
    deltas = []
    for seed in SEEDS:
        for level in LEVELS:
            original = indexed[seed, "original", level]
            slower = indexed[seed, "shared_clock", level]
            measured = all(r["status"] == "completed" for r in (original, slower))
            deltas.append(
                {
                    "generation_seed": seed,
                    "label": level,
                    "measured": measured,
                    "endpoint_error_delta_m": (
                        slower["diagnostics"]["endpoint_error_m"]
                        - original["diagnostics"]["endpoint_error_m"]
                        if measured
                        else None
                    ),
                    "tracker_acceptance_delta": (
                        int(slower["tracker_accepted"]) - int(original["tracker_accepted"])
                        if measured
                        else None
                    ),
                }
            )
    duck_deltas = [
        r["endpoint_error_delta_m"] for r in deltas if r["label"] == "d055" and r["measured"]
    ]
    return {
        "counts": counts,
        "paired_deltas_slower_minus_original": deltas,
        "predictions": {
            "p1_crouch_endpoint_improves_in_at_least_two_carriers": sum(d < 0 for d in duck_deltas)
            >= 2,
            "p2_more_tracker_accepted_crouches": counts["shared_clock/d055"]["tracker_accepted"]
            > counts["original/d055"]["tracker_accepted"],
            "p3_at_least_two_shared_clock_neutrals_accepted": counts["shared_clock/neutral"][
                "tracker_accepted"
            ]
            >= 2,
        },
    }


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError("refusing to overwrite diagnostic artifacts")
    template_path = (
        args.research_repo / "experiments/registrations/E1_CONTROLLED_DUCK_Q3_V1_MANIFEST.json"
    )
    template = json.loads(template_path.read_text())
    inputs = {
        "original": args.data_root / "e1_controlled_duck_ladders_v1/candidates.json",
        "shared_clock": args.data_root / "e1_shared_clock_duck_v1/candidates.json",
    }
    candidates = {condition: json.loads(path.read_text()) for condition, path in inputs.items()}
    selected = {
        condition: {item["generation_seed"]: item for item in source["ladders"]}
        for condition, source in candidates.items()
    }
    cells = []
    for seed in SEEDS:
        # Alternate order to reduce a fixed condition/order confound.
        conditions = CONDITIONS if seed % 2 else CONDITIONS[::-1]
        originals = {item["label"]: item for item in selected["original"][seed]["levels"]}
        for condition in conditions:
            ladder = selected[condition][seed]
            levels = {item["label"]: item for item in ladder["levels"]}
            if condition == "shared_clock" and not ladder["root_route_and_clock_matched"]:
                raise ValueError("shared-clock references must match within carrier")
            for label in LEVELS:
                ref = levels[label]
                if ref["q0_status"] != "passed" or ref["q1_status"] != "passed":
                    raise ValueError("reference failed Q0/Q1")
                if (
                    label != "neutral"
                    and ref["semantic_assessment"]["semantic_status"] != "route_aligned"
                ):
                    raise ValueError("crouch reference must pass S3")
                if (
                    condition == "shared_clock"
                    and ref["source_csv_sha256"] != originals[label]["csv_sha256"]
                ):
                    raise ValueError("timing comparison has different source motions")
                csv = checked(Path(ref["csv"]), ref["csv_sha256"])
                motion = checked(Path(ref["sonic_motion"]), ref["sonic_motion_sha256"])
                cell_id = f"m2s_timing_{seed}_{condition}__{label}"
                provenance = args.output / "provenance" / f"{cell_id}.json"
                write_new(
                    provenance,
                    {
                        "schema_version": "motion2scene_sonic_conversion_provenance_v1",
                        "scene_start_xyz": [0.0, 0.0, 0.0],
                        "scene_yaw": 0.0,
                        "canonicalize_horizontal_origin": True,
                        "source_fps": 30.0,
                        "input": artifact(csv),
                        "output": artifact(motion),
                        "candidate_manifest": artifact(inputs[condition]),
                    },
                )
                cell = copy.deepcopy(template["cells"][0])
                cell.update(
                    {
                        "cell_id": cell_id,
                        "generation_seed": seed,
                        "condition": condition,
                        "label": label,
                        "base_carrier_id": ladder["base_carrier_id"],
                        "ladder_group_id": ladder["ladder_group_id"],
                        "ladder_level": ref["ladder_level"],
                        "body_mode": "walk" if label == "neutral" else "controlled_crouch",
                        "runtime_seed": 7900,
                        "hydra_overrides": ["++seed=7900"],
                        "motion": {
                            **artifact(motion),
                            "conversion_provenance": str(provenance),
                            "conversion_provenance_sha256": sha(provenance),
                            "scene_start_xyz": [0.0, 0.0, 0.0],
                        },
                        "reference": {**artifact(csv), "whole_body_top_m": ref["whole_body_top_m"]},
                        "output": str(args.output / "rollouts" / cell_id),
                    }
                )
                if label != "neutral":
                    cell["depends_on_acceptance_of"] = f"m2s_timing_{seed}_{condition}__neutral"
                cells.append(cell)
    template.update(
        {
            "experiment": "M2S-E1-paired-timing-diagnostic-v1",
            "purpose": "Development-only paired timing intervention; no Q3/Q4 or training admission",
            "analysis_role": "exploratory_controller_diagnostic",
            "cells": cells,
            "registered_predictions": artifact(args.registration),
            "eligibility": {
                "rule": "first three carrier IDs; Q0/Q1 and adapted S3; paired original/shared-clock sources",
                "sources": {c: artifact(p) for c, p in inputs.items()},
            },
            "diagnostic_implementation": artifact(Path(__file__)),
            "motion_analysis_implementation": {
                name: artifact(args.research_repo / "src/motion2scene/motion" / f"{name}.py")
                for name in ("paired_semantics", "route_semantics", "route_retention")
            },
            "promotion": {"q4_admitted_ladders": 0, "training_eligible": False},
        }
    )
    template["authorization"]["note"] = (
        "User requested continued performance research toward learned scene generation; "
        "separate diagnostic lineage."
    )
    policy = template["execution_policy"]
    policy["cost_ceiling"].update({"rollouts": 12, "gpu_hours_contended": 1.25})
    policy["runtime"].update({"free_gpu_mib_required": 9000, "hang_timeout_seconds": 375})
    policy["timing_override"] = (
        "Paired diagnostic registered before launch; previous gated Q3 pilot remains unlaunched."
    )
    template["stop_conditions"] = [
        "refuse hash mismatches; yield below 9000 MiB free GPU memory",
        "serial execution; 375 second per-cell timeout; stop on infrastructure failure",
        "retain rejections and dependency skips; no scientific retries",
        "no historical relabeling, motion-bank promotion, or learning eligibility",
    ]
    write_new(args.output / "manifest.json", template)
    print(f"Prepared {len(cells)} cells in {args.output}")


def analyze(args: argparse.Namespace) -> None:
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    checked(Path(__file__), manifest["diagnostic_implementation"]["sha256"])
    for item in manifest["motion_analysis_implementation"].values():
        checked(Path(item["path"]), item["sha256"])
    run_path = args.output / "run_record.json"
    run = json.loads(run_path.read_text())
    if run["status"] != "completed" or run["manifest_sha256"] != sha(manifest_path):
        raise ValueError("requires the complete hash-bound batch")
    sys.path[:0] = [str(ROOT), str(args.research_repo / "src")]
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
        row = {key: cell[key] for key in ("cell_id", "generation_seed", "condition", "label")}
        row.update(
            {
                "status": record["status"],
                "tracker_accepted": None,
                "route_retained": None,
                "paired_behavior_retained": None,
            }
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
            envelopes[cell["cell_id"]] = extract_envelope(
                extract_keypoints(payload), cell["cell_id"], fractions=stations
            ).up_m
            row.update(
                {
                    "tracker_accepted": scientific["outcome"] == "accepted",
                    "route_retained": decision.retained,
                    "route_decision": decision.to_dict(),
                    "route_metrics": route.to_dict(),
                    "diagnostics": scientific["diagnostics"],
                    "rejection_reasons": scientific["rejection_reasons"],
                    "trajectory": artifact(path),
                }
            )
        rows.append(row)
    by_id = {row["cell_id"]: row for row in rows}
    by_cell = {cell["cell_id"]: cell for cell in manifest["cells"]}
    semantic_policy = PairedSemanticPolicy(minimum_effect=0.05)
    for row in rows:
        if row["label"] == "neutral" or row["status"] != "completed":
            continue
        cell_id = row["cell_id"]
        neutral_id = by_cell[cell_id]["depends_on_acceptance_of"]
        reference = assess_paired_reduction(
            stations,
            np.asarray(by_cell[cell_id]["reference"]["whole_body_top_m"]),
            np.asarray(by_cell[neutral_id]["reference"]["whole_body_top_m"]),
            route_valid=True,
            policy=semantic_policy,
        )
        achieved = assess_paired_reduction(
            stations,
            envelopes[cell_id],
            envelopes[neutral_id],
            route_valid=row["route_retained"] and by_id[neutral_id]["route_retained"],
            policy=semantic_policy,
        )
        retention = assess_controller_retention(reference, achieved, policy=semantic_policy)
        row.update(
            {
                "reference_semantic": reference.to_dict(),
                "achieved_semantic": achieved.to_dict(),
                "controller_retention": retention.to_dict(),
                "paired_behavior_retained": bool(
                    row["tracker_accepted"]
                    and by_id[neutral_id]["tracker_accepted"]
                    and retention.semantic_status.value == "controller_retained"
                ),
            }
        )
    result = {
        "schema_version": "motion2scene_paired_timing_diagnostic_result_v1",
        "analysis_complete": True,
        "analysis_role": manifest["analysis_role"],
        "manifest": artifact(manifest_path),
        "run_record": artifact(run_path),
        "driver": artifact(Path(__file__)),
        "independent_unit": "generation_seed",
        "independent_units": len(SEEDS),
        "physics_seed": 7900,
        "q4_admitted_ladders": 0,
        "training_eligible": False,
        "budget": run["budget"],
        "summary": summarize(rows),
        "rows": rows,
    }
    write_new(args.output / "result.json", result)
    print(json.dumps(result["summary"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "analyze"))
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--registration", type=Path, default=ROOT / "docs/motion2scene/TIMING_DIAGNOSTIC_V1.md"
    )
    args = parser.parse_args()
    args.output = args.output.resolve()
    {"prepare": prepare, "analyze": analyze}[args.mode](args)


if __name__ == "__main__":
    main()
