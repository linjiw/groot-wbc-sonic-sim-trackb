#!/usr/bin/env python3
"""Plan scene placements for Kimodo G1 reference motions and emit a rollout manifest.

Replaces "drop every motion at the scene's route start with a fixed yaw", which
made acceptance placement-limited rather than controller-limited: laterally
curving references walked into racks regardless of tracking quality.

For each motion the planner searches yaw and translation for placements whose whole
reference path keeps the scene's declared body-clearance radius plus a tracking
margin, and stays inside the walkable rectangle. The searched transform is exactly
the one ``kimodo_motion_adapter.transform_qpos_to_scene`` applies.

Usage::

    python scripts/research/plan_kimodo_placements.py \\
        --csv-dir /path/with/kimodo_*.csv \\
        --out placements.json \\
        --scenes factory_aisle household_room \\
        --per-scene 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.route_placement import (  # noqa: E402
    DEFAULT_TRACKING_MARGIN_M,
    canonical_path_xy,
    plan_placements,
    summarize_placements,
)
from gear_sonic.dataset_generation.scene_asset_preflight import (  # noqa: E402
    load_scene_obstacle_map,
)

#: Kimodo resamples a 30 Hz source to the SONIC 50 Hz control rate.
SOURCE_FPS = 30.0
CONTROL_FPS = 50.0


def resampled_frame_count(source_frames: int) -> int:
    """Frames the SONIC motion library produces from a 30 Hz clip at 50 Hz."""
    if source_frames < 2:
        return source_frames
    return int(np.floor((source_frames - 1) * CONTROL_FPS / SOURCE_FPS)) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["factory_aisle", "household_room"])
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--pattern", default="kimodo_*.csv")
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--tracking-margin-m", type=float, default=DEFAULT_TRACKING_MARGIN_M)
    parser.add_argument("--translation-step-m", type=float, default=0.25)
    parser.add_argument("--yaw-steps", type=int, default=24)
    args = parser.parse_args()

    maps = {scene: load_scene_obstacle_map(scene) for scene in args.scenes}
    csv_paths = sorted(args.csv_dir.glob(args.pattern))
    if not csv_paths:
        print(f"ERROR: no CSVs matching {args.pattern} in {args.csv_dir}", file=sys.stderr)
        return 2

    entries: list[dict] = []
    for csv_path in csv_paths:
        motion_id = csv_path.stem.removeprefix("kimodo_")
        if motion_id in args.exclude:
            print(f"skip {motion_id} (excluded)")
            continue
        qpos = np.loadtxt(csv_path, delimiter=",")
        path = canonical_path_xy(qpos)
        frames = resampled_frame_count(len(qpos))
        for scene, obstacle_map in maps.items():
            placements = plan_placements(
                path,
                obstacle_map,
                max_results=args.per_scene,
                tracking_margin_m=args.tracking_margin_m,
                translation_step_m=args.translation_step_m,
                yaw_steps=args.yaw_steps,
            )
            summary = summarize_placements(placements)
            print(
                f"{motion_id:32s} {scene:16s} placements={summary['count']} "
                f"clearance={summary.get('min_clearance_m', 0):.2f}-"
                f"{summary.get('max_clearance_m', 0):.2f} m"
            )
            for index, placement in enumerate(placements):
                entries.append(
                    {
                        "config_id": f"{motion_id}__{scene}__p{index}",
                        "motion_id": motion_id,
                        "scene_id": scene,
                        "csv": str(csv_path.resolve()),
                        "source_frames": int(len(qpos)),
                        "resampled_frames": frames,
                        # eval_agent_trl records max_render_steps - 1 frames, so this
                        # captures exactly one motion pass with no clock reset.
                        "max_render_steps": frames + 1,
                        "placement": placement.to_dict(),
                    }
                )

    payload = {
        "schema_version": 1,
        "tracking_margin_m": args.tracking_margin_m,
        "translation_step_m": args.translation_step_m,
        "yaw_steps": args.yaw_steps,
        "scenes": {
            scene: {
                "route_clearance_radius_m": obstacle_map.route_clearance_radius_m,
                "required_clearance_m": obstacle_map.route_clearance_radius_m
                + args.tracking_margin_m,
                "obstacle_count": len(obstacle_map.obstacles),
            }
            for scene, obstacle_map in maps.items()
        },
        "configs": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nplanned {len(entries)} rollout configs -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
