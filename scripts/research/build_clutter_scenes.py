#!/usr/bin/env python3
"""Generate cluttered indoor scenes around known-good Kimodo motion paths.

Inverts the placement problem. Instead of fitting a motion into a fixed scene --
which is placement-limited and yields episodes crossing mostly empty floor --
this takes the motion's path as given and builds the room and its furniture
around it. Every solid clears the corridor, so the episode is traversable by
construction, and the furniture is sampled *near* the corridor so the robot
actually threads through clutter.

Emits a scene package (USDA files + manifest.json) in the same primitive subset
that ``scene_asset_preflight`` validates, so generated scenes pass the identical
gate as the hand-authored ones and can be checked with::

    python scripts/research/check_g1_dataset_scenes.py --package <out-dir>

Usage::

    python scripts/research/build_clutter_scenes.py \\
        --csv-dir /path/with/kimodo_*.csv \\
        --out gear_sonic/data/assets/scenes/g1_clutter \\
        --motions 01_single_text_prompt 03_full_body_keyframes \\
        --seeds 0 1 2
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

from gear_sonic.dataset_generation.clutter_scene_builder import (  # noqa: E402
    build_clutter_scene,
    render_scene_usda,
)
from gear_sonic.dataset_generation.route_placement import (  # noqa: E402
    canonical_path_xy,
)

# Measured, not assumed. Across 48 episodes carrying per-body pose, the swept half-width
# reaches 0.434-0.664 m (median 0.500). The previous value of 0.45 m was exceeded on 98% of
# them: it described the pelvis, not the robot, and the arms are what a room has to clear.
BODY_RADIUS_M = 0.664

# What the room must actually leave free is the swept half-width *plus* how far the executed
# path drifts from the reference the furniture was placed against. Measured per frame over a
# 40-episode diverse batch, that sum ranged 0.482-0.819 m (median 0.582). The default below
# covers the observed maximum with a little room; the previous 0.75 m (0.45 body + 0.30
# margin) left only 0.086 m of real margin once the body radius is measured honestly, and
# three of those 40 episodes needed more than it provided.
#
# Note the two terms are not independent -- the widest swept volume and the largest path
# deviation rarely coincide -- which is why this is 0.85 rather than the 0.96 m that naively
# adding the two maxima would suggest.
DEFAULT_ROUTE_CLEARANCE_M = 0.85
ROBOT_CLEARANCE_HEIGHT_M = 1.9


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--motions", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--pieces", type=int, default=22)
    parser.add_argument(
        "--clearance-m",
        type=float,
        default=DEFAULT_ROUTE_CLEARANCE_M,
        help="minimum distance from any solid to the path (measured swept half-width + path drift)",
    )
    parser.add_argument("--max-distance-m", type=float, default=3.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = []
    for motion in args.motions:
        csv_path = args.csv_dir / f"kimodo_{motion}.csv"
        if not csv_path.is_file():
            print(f"ERROR: missing {csv_path}", file=sys.stderr)
            return 2
        path = canonical_path_xy(np.loadtxt(csv_path, delimiter=","))
        for seed in args.seeds:
            scene_id = f"clutter_{motion}_s{seed}"
            spec = build_clutter_scene(
                path,
                scene_id=scene_id,
                seed=seed,
                clearance_m=args.clearance_m,
                target_pieces=args.pieces,
                max_distance_from_path_m=args.max_distance_m,
            )
            usda = render_scene_usda(spec)
            scene_file = args.out / f"{scene_id}.usda"
            scene_file.write_text(usda, encoding="utf-8")
            digest = hashlib.sha256(scene_file.read_bytes()).hexdigest()

            # The motion was recentred on the room origin; record the offset so the
            # rollout can place the same motion with a matching scene-start.
            offset = spec.metrics["scene_start_xy"]
            placed = spec.path_xy
            (min_x, min_y), (max_x, max_y) = spec.walkable_bounds()
            scenes.append(
                {
                    "scene_id": scene_id,
                    "scene_family": "household_clutter",
                    "file": scene_file.name,
                    "sha256": f"sha256:{digest}",
                    "size_bytes": scene_file.stat().st_size,
                    "split_group": spec.split_group,
                    "license_spdx": "Apache-2.0",
                    "redistribution_allowed": True,
                    "provenance": "repo_generated_primitive_geometry",
                    "default_prim": "/World",
                    "up_axis": "Z",
                    "meters_per_unit": 1.0,
                    "support_floor_prim": "/World/Structure/Floor",
                    "support_z_m": 0.0,
                    "walkable_bounds_xy": {"min": [min_x, min_y], "max": [max_x, max_y]},
                    "route_xy": [[float(x), float(y)] for x, y in placed[:: max(1, len(placed) // 8)]],
                    "route_clearance_radius_m": BODY_RADIUS_M,
                    "robot_clearance_height_m": ROBOT_CLEARANCE_HEIGHT_M,
                    "minimum_collision_prims": 4 + len(spec.pieces),
                    "source_motion": motion,
                    "generation": {
                        "seed": seed,
                        "clearance_m": args.clearance_m,
                        "scene_start_xy": offset,
                        **{
                            key: spec.metrics[key]
                            for key in (
                                "placed_pieces",
                                "min_distance_to_path_m",
                                "median_distance_to_path_m",
                                "clutter_occupancy",
                                "path_length_m",
                            )
                        },
                    },
                }
            )
            print(
                f"{scene_id}: {spec.metrics['placed_pieces']}/{args.pieces} pieces, "
                f"room {spec.room_size_xy[0]:.1f}x{spec.room_size_xy[1]:.1f} m, "
                f"nearest {spec.metrics['min_distance_to_path_m']:.2f} m, "
                f"occupancy {spec.metrics['clutter_occupancy']:.1%}"
            )

    manifest = {
        "schema_version": 1,
        "package_id": "g1_clutter_generated_scenes",
        "description": (
            "Procedurally generated cluttered indoor scenes built around known-good "
            "Kimodo motion paths. Every solid clears the executed corridor."
        ),
        "license": {
            "spdx": "Apache-2.0",
            "source": "../../../../../LICENSE",
            "redistribution_allowed": True,
        },
        "provenance": {
            "kind": "repo_generated_primitive_geometry",
            "description": (
                "Generated from axis-aligned USD Cube furniture proxies and a Plane support "
                "floor by gear_sonic.dataset_generation.clutter_scene_builder; no imported "
                "meshes, materials, textures, or third-party scene content."
            ),
            "third_party_assets": [],
        },
        "scenes": scenes,
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {len(scenes)} scenes + manifest to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
