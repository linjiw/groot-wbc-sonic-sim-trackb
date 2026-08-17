#!/usr/bin/env python3
"""Build the evaluation-scene package: rooms that may grade a policy.

The training packages cannot. Their clearance was fitted to one executed trajectory, so
they measure whether a policy reproduces that path rather than whether it can navigate;
``eval_scene_gate`` rejects them, and applied to the corpus it left only the two
hand-authored rooms eligible -- not enough for held-out-scene evaluation.

These rooms clear the *reference* path by the measured body half-width plus an allowance
for policy drift, and each is certified to admit routes other than the reference before it
is written. A scene that cannot be certified is reported and skipped, never shipped.

Usage::

    python scripts/research/build_eval_scenes.py \\
        --csv-dir /data/.../taxonomy/motions \\
        --out gear_sonic/data/assets/scenes/g1_eval --count 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.clutter_scene_builder import render_scene_usda  # noqa: E402
from gear_sonic.dataset_generation.eval_scene_builder import (  # noqa: E402
    MEASURED_BODY_HALF_WIDTH_M,
    POLICY_DRIFT_ALLOWANCE_M,
    EvalSceneError,
    build_eval_scene,
)
from gear_sonic.dataset_generation.route_placement import canonical_path_xy  # noqa: E402

ROBOT_CLEARANCE_HEIGHT_M = 1.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pieces", type=int, default=16)
    parser.add_argument("--body-half-width", type=float, default=MEASURED_BODY_HALF_WIDTH_M)
    parser.add_argument("--drift-allowance", type=float, default=POLICY_DRIFT_ALLOWANCE_M)
    args = parser.parse_args()

    csvs = sorted(args.csv_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.csv_dir}")

    args.out.mkdir(parents=True, exist_ok=True)
    scenes: list[dict] = []
    skipped: list[dict] = []

    for index, csv_path in enumerate(csvs):
        if len(scenes) >= args.count:
            break
        scene_id = f"eval_{csv_path.stem[:44]}"
        path = canonical_path_xy(np.loadtxt(csv_path, delimiter=","))
        try:
            spec, report = build_eval_scene(
                path,
                scene_id=scene_id,
                seed=args.seed + index,
                target_pieces=args.pieces,
                body_half_width_m=args.body_half_width,
                drift_allowance_m=args.drift_allowance,
            )
        except EvalSceneError as error:
            print(f"  SKIP {scene_id}: {error}")
            skipped.append({"scene_id": scene_id, "reason": str(error)})
            continue

        scene_file = args.out / f"{scene_id}.usda"
        scene_file.write_text(render_scene_usda(spec), encoding="utf-8")
        digest = hashlib.sha256(scene_file.read_bytes()).hexdigest()
        (min_x, min_y), (max_x, max_y) = spec.walkable_bounds()
        placed = spec.path_xy

        scenes.append(
            {
                "scene_id": scene_id,
                "scene_family": "eval_reference_clearance",
                "file": scene_file.name,
                "sha256": f"sha256:{digest}",
                "size_bytes": scene_file.stat().st_size,
                "split_group": f"{scene_id}_eval_v1",
                "license_spdx": "Apache-2.0",
                "redistribution_allowed": True,
                # Deliberately NOT "repo_authored"; the gate keys on provenance and this
                # package must not fake its way past it. Eligibility is established by the
                # certified route count, which the gate checks separately.
                "provenance": "repo_generated_reference_clearance_geometry",
                "default_prim": "/World",
                "up_axis": "Z",
                "meters_per_unit": 1.0,
                "support_floor_prim": "/World/Structure/Floor",
                "support_z_m": 0.0,
                "walkable_bounds_xy": {"min": [min_x, min_y], "max": [max_x, max_y]},
                "route_xy": [
                    [float(x), float(y)] for x, y in placed[:: max(1, len(placed) // 8)]
                ],
                "route_clearance_radius_m": report.body_half_width_m,
                "robot_clearance_height_m": ROBOT_CLEARANCE_HEIGHT_M,
                "minimum_collision_prims": 4 + len(spec.pieces),
                # No source_motion key: the geometry is budgeted from the reference path,
                # and nothing about any executed trajectory enters it.
                "eval_certification": {
                    "clearance_m": report.clearance_m,
                    "body_half_width_m": report.body_half_width_m,
                    "drift_allowance_m": report.drift_allowance_m,
                    "independent_routes": report.independent_routes,
                    "required_routes": report.required_routes,
                    "min_route_clearance_m": report.min_route_clearance_m,
                },
                "generation": {
                    "seed": args.seed + index,
                    "clearance_m": report.clearance_m,
                    "scene_start_xy": spec.metrics["scene_start_xy"],
                    "placed_pieces": report.placed_pieces,
                    "clutter_occupancy": report.occupancy,
                },
            }
        )
        print(
            f"  {scene_id[:46]:46s} {report.placed_pieces:2d} pieces  "
            f"occ {report.occupancy:5.1%}  routes {report.independent_routes}  "
            f"clearance {report.clearance_m:.3f} m"
        )

    if not scenes:
        raise SystemExit("no scene could be certified for evaluation use")

    manifest = {
        "schema_version": 1,
        "package_id": "g1_eval_scenes",
        "description": (
            "Evaluation scenes. Clearance is budgeted from the reference path -- the "
            "measured body half-width plus an allowance for policy drift -- so no executed "
            "trajectory enters the geometry. Each scene is certified to admit routes other "
            "than the reference."
        ),
        "license": {
            "spdx": "Apache-2.0",
            "source": "../../../../../LICENSE",
            "redistribution_allowed": True,
        },
        "provenance": {
            "kind": "repo_generated_reference_clearance_geometry",
            "description": (
                "Generated from axis-aligned USD Cube furniture proxies and a Plane support "
                "floor, cleared against the commanded path rather than an executed one."
            ),
        },
        "eval_policy": {
            "body_half_width_m": args.body_half_width,
            "drift_allowance_m": args.drift_allowance,
            "rationale": (
                "Body half-width is the measured maximum swept half-width over 48 episodes "
                "(0.434-0.664 m). The drift allowance exceeds the largest measured "
                "reference-to-executed deviation (0.357 m over 40 diverse episodes), because "
                "an evaluation must admit policies worse than the one that was measured."
            ),
        },
        "scenes": scenes,
        "skipped": skipped,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {len(scenes)} certified eval scene(s), skipped {len(skipped)} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
