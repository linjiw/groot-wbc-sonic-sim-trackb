#!/usr/bin/env python3
"""Audit every held-out route across the existing calibration grid without regrading it."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import sys

from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-repo", type=Path, required=True)
    parser.add_argument("--heldout-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(args.research_repo / "src")]
    from motion2scene.motion.route_retention import assess_route_retention, compare_routes
    from motion2scene.motion.route_semantics import DEFAULT_ROUTE_POLICY

    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    result_path = args.heldout_root / "q3/relative_retention_result_v2.json"
    run_path = args.heldout_root / "q3/run_record.json"
    result = json.loads(result_path.read_text())
    run = json.loads(checked(run_path, result["run_record_sha256"]).read_text())
    if not result["analysis_complete"] or result["completed"] != result["registered"]:
        raise ValueError("requires complete held-out evidence")
    rows = []
    for row in result["rows"]:
        csv = checked(Path(row["csv"]), row["csv_sha256"])
        reference = np.loadtxt(csv, delimiter=",")[:, :2]
        source = run["cells"][row["cell_id"]]["scientific"]["artifacts"]
        path = checked(Path(source["trajectory"]), row["trajectory_sha256"])
        with path.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        achieved = np.asarray(payload["root_pos_w"])[:, :2]
        grid = []
        # The grid was used in the earlier neutral-control calibration; report every cell.
        for smoothing in (0.25, 0.50, 0.75, 1.00):
            for spacing in (0.25, 0.50, 0.75):
                policy = replace(
                    DEFAULT_ROUTE_POLICY,
                    centerline_smoothing_s=smoothing,
                    centerline_station_spacing_m=spacing,
                )
                metrics = compare_routes(
                    reference,
                    achieved,
                    expected_route="straight",
                    reference_fps=30,
                    achieved_fps=float(payload["fps"]),
                    policy=policy,
                )
                decision = assess_route_retention(metrics)
                grid.append(
                    {
                        "smoothing_seconds": smoothing,
                        "spacing_metres": spacing,
                        "metrics": metrics.to_dict(),
                        "descriptive_decision": decision.to_dict(),
                    }
                )
        frozen = next(
            cell
            for cell in grid
            if cell["smoothing_seconds"] == 0.50 and cell["spacing_metres"] == 0.50
        )
        if frozen["descriptive_decision"]["retained"] != row["route_retained"]:
            raise ValueError("frozen decision did not reproduce")
        for name, value in row["retention"].items():
            if isinstance(value, (int, float)) and not np.isclose(
                frozen["metrics"][name], value, atol=1e-10, rtol=1e-10
            ):
                raise ValueError(f"frozen metric did not reproduce: {name}")
        rows.append(
            {
                "generation_seed": row["generation_seed"],
                "tracker_survived": row["tracker_survived"],
                "frozen_route_retained": row["route_retained"],
                "reference": artifact(csv),
                "trajectory": artifact(path),
                "grid": grid,
                "heading_error_range_rad": [
                    min(c["metrics"]["signed_heading_error_rad"] for c in grid),
                    max(c["metrics"]["signed_heading_error_rad"] for c in grid),
                ],
                "descriptive_grid_passes": sum(c["descriptive_decision"]["retained"] for c in grid),
                "grid_cells": len(grid),
            }
        )
    write_new(
        args.out,
        {
            "schema_version": "motion2scene_heldout_route_sensitivity_v1",
            "analysis_role": "post_outcome_instrument_diagnostic_without_regrading",
            "sources": {
                "heldout_result": artifact(result_path),
                "run_record": artifact(run_path),
                "driver": artifact(Path(__file__)),
                "route_semantics": artifact(
                    args.research_repo / "src/motion2scene/motion/route_semantics.py"
                ),
                "route_retention": artifact(
                    args.research_repo / "src/motion2scene/motion/route_retention.py"
                ),
            },
            "selection_rule": "report all grid cells; no policy selected or admission changed",
            "original_retained_survivors": result["retained_survivors"],
            "q4_admitted_ladders": 0,
            "training_eligible": False,
            "rows": rows,
        },
    )
    for row in rows:
        print(
            row["generation_seed"],
            row["heading_error_range_rad"],
            row["descriptive_grid_passes"],
            "/12",
        )


if __name__ == "__main__":
    main()
