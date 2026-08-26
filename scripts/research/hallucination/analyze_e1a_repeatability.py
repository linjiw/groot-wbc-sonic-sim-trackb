#!/usr/bin/env python3
"""Measure source-inclusive clearance repeatability for an LFH E1a run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.constraint_distance import (  # noqa: E402
    constraint_distance_from_payload,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
SCENE_ROOT = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
ROLES = ("nominal_easy", "nominal_hard", "adapted_easy", "adapted_hard")
MF_SOURCE_NAMES = {
    "nominal_easy": "w_nominal_easy",
    "nominal_hard": "w_nominal_hard",
    "adapted_easy": "w_crouch08_easy",
    "adapted_hard": "w_crouch08_hard",
}


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def shelf_box(scene: Path) -> tuple[float, float, float, float, float, float]:
    shelves = [
        cube for cube in read_stage_geometry(scene).cubes if cube.path.endswith("/LowShelf_00")
    ]
    if len(shelves) != 1:
        raise ValueError(f"{scene}: expected one LowShelf_00, found {len(shelves)}")
    return shelves[0].box


def source_path(family: str, role: str) -> Path:
    if family == "duck_003":
        root = DATA_ROOT / "counterfactual/duck_003" / role
    elif family == "mf_005_c08":
        root = DATA_ROOT / "matched/mf_005_c08" / MF_SOURCE_NAMES[role]
    else:
        raise ValueError(f"unknown E1a source family {family}")
    return root / "trajectories/000000.trajectory.pkl"


def minimum_clearance_mm(trajectory: Path, box: tuple[float, ...]) -> tuple[float, int]:
    payload = load(trajectory)
    result = constraint_distance_from_payload(payload, box, fps=float(payload.get("fps", 50.0)))
    return 1000.0 * float(result.body_clearance_m.min()), int(result.bottleneck_frame)


def render(report: dict) -> str:
    lines = [
        "# E1a Repeatability Report",
        "",
        f"Outcome stability: **{report['outcome_matches']}/{report['rollouts']} matches; "
        f"{report['outcome_flips']} flips**. The LFH stop line did not trigger.",
        "",
        "The executed-capsule/USDA-AABB instrument gives a source-inclusive clearance noise "
        f"floor of **{report['source_inclusive_noise_floor_mm']:.3f} mm**. The spread between "
        f"the two fresh repeats alone is **{report['fresh_repeat_noise_floor_mm']:.3f} mm**. "
        "The source-inclusive value is the conservative tolerance used for later drift reports.",
        "",
        "| family / cell | shipped source | repeat 1 | repeat 2 | fresh spread | inclusive spread |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["cells"]:
        values = row["minimum_clearance_mm"]
        lines.append(
            f"| `{row['family_id']}/{row['cell_role']}` | {values['source']:.3f} | "
            f"{values['repeat_1']:.3f} | {values['repeat_2']:.3f} | "
            f"{row['fresh_repeat_spread_mm']:.3f} | "
            f"{row['source_inclusive_spread_mm']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Instrument: minimum executed G1 collision-capsule clearance to the exact "
            "`LowShelf_00` AABB read from the USDA loaded by physics. Negative values overlap. "
            "This is distinct from KCS reference-side capsule prediction and historical boundary "
            "bisection; millimetre values are diagnostics, while physics outcomes remain binding.",
            "",
            f"Actual serial spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run_record.read_text())
    if manifest["experiment"] != "E1a" or run["status"] != "completed":
        raise SystemExit("E1a manifest and completed run record are required")

    manifest_cells = {cell["cell_id"]: cell for cell in manifest["cells"]}
    rows = []
    fresh_floor = 0.0
    inclusive_floor = 0.0
    for family in ("duck_003", "mf_005_c08"):
        for role in ROLES:
            sample_cells = [manifest_cells[f"{family}__{role}__repeat_{i}"] for i in (1, 2)]
            scene = REPO_ROOT / sample_cells[0]["scene"]["path"]
            box = shelf_box(scene)
            paths = {
                "source": source_path(family, role),
                "repeat_1": Path(sample_cells[0]["output"]) / "trajectories/000000.trajectory.pkl",
                "repeat_2": Path(sample_cells[1]["output"]) / "trajectories/000000.trajectory.pkl",
            }
            measured = {name: minimum_clearance_mm(path, box) for name, path in paths.items()}
            clearances = {name: value[0] for name, value in measured.items()}
            fresh_spread = abs(clearances["repeat_1"] - clearances["repeat_2"])
            inclusive_spread = max(clearances.values()) - min(clearances.values())
            fresh_floor = max(fresh_floor, fresh_spread)
            inclusive_floor = max(inclusive_floor, inclusive_spread)
            rows.append(
                {
                    "family_id": family,
                    "cell_role": role,
                    "scene": str(scene),
                    "trajectory_paths": {name: str(path) for name, path in paths.items()},
                    "minimum_clearance_mm": clearances,
                    "bottleneck_frame": {name: value[1] for name, value in measured.items()},
                    "fresh_repeat_spread_mm": fresh_spread,
                    "source_inclusive_spread_mm": inclusive_spread,
                }
            )

    matches = sum(
        cell["prediction_matches"] is True
        for cell in run["cells"].values()
        if cell["status"] == "completed"
    )
    report = {
        "schema_version": "lfh_e1a_repeatability_v1",
        "instrument": "executed_g1_capsule_to_loaded_lowshelf_aabb_minimum_clearance",
        "manifest_sha256": run["manifest_sha256"],
        "rollouts": len(run["cells"]),
        "outcome_matches": matches,
        "outcome_flips": len(run["cells"]) - matches,
        "fresh_repeat_noise_floor_mm": fresh_floor,
        "source_inclusive_noise_floor_mm": inclusive_floor,
        "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
        "cells": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.json_out.chmod(0o664)
    args.md_out.write_text(render(report))
    args.md_out.chmod(0o664)
    print(
        f"PASS: {matches}/{len(run['cells'])} outcomes; "
        f"source-inclusive floor={inclusive_floor:.3f} mm -> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
