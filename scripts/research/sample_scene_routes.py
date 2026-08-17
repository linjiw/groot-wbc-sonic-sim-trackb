#!/usr/bin/env python3
"""Plan routes through existing scenes and emit Kimodo root constraints.

This is the scene-first generation direction. Every episode so far was built the other way
-- the room around a motion the robot had already walked -- which guarantees traversability
but makes the action statistically independent of the geometry given the motion. Planning
the route through a fixed scene restores geometry -> action causality, and the generation
direction is recorded per route so the two subsets stay separable at analysis time.

Reads the released ``.usda`` for obstacles, so the routes are planned against the same
geometry a downloader would get.

Usage::

    python scripts/research/sample_scene_routes.py \\
        --package gear_sonic/data/assets/scenes/g1_clutter \\
        --out /data/.../routes --per-scene 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.scene_build_sheet import (  # noqa: E402
    STRUCTURAL_ROLES,
    parse_scene_usda,
)
from gear_sonic.dataset_generation.scene_route_sampler import (  # noqa: E402
    DEFAULT_BODY_HALF_WIDTH_M,
    DEFAULT_MARGIN_M,
    Obstacle,
    RouteSamplingError,
    sample_routes,
)


def scene_inputs(usda_text: str) -> tuple[list[Obstacle], tuple[float, float]]:
    """Obstacles and room size, read from the released scene file."""
    prims = parse_scene_usda(usda_text)
    floors = [p for p in prims if p.role == "support_floor"]
    if not floors:
        raise RouteSamplingError("scene has no support_floor prim")
    room = (float(floors[0].size[0]), float(floors[0].size[1]))
    obstacles = [
        Obstacle(rect=prim.rect, z_base=prim.z_base)
        for prim in prims
        if prim.role not in STRUCTURAL_ROLES
    ]
    return obstacles, room


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=0.8, help="m/s along the route")
    parser.add_argument("--body-half-width", type=float, default=DEFAULT_BODY_HALF_WIDTH_M)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_M)
    parser.add_argument("--min-length", type=float, default=2.0)
    args = parser.parse_args()

    manifest = json.loads((args.package / "manifest.json").read_text(encoding="utf-8"))
    scenes = manifest["scenes"]
    scenes = list(scenes.values()) if isinstance(scenes, dict) else scenes

    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    unroutable: list[dict] = []

    for index, entry in enumerate(scenes):
        scene_id = entry.get("scene_id", f"scene_{index}")
        usda = args.package / entry["file"]
        if not usda.exists():
            print(f"  skip {scene_id}: missing {usda.name}")
            continue
        obstacles, room = scene_inputs(usda.read_text(encoding="utf-8"))
        try:
            routes = sample_routes(
                obstacles,
                room,
                count=args.per_scene,
                seed=args.seed + index,
                body_half_width_m=args.body_half_width,
                margin_m=args.margin,
                speed_mps=args.speed,
                min_length_m=args.min_length,
            )
        except RouteSamplingError as error:
            # A scene too cluttered to walk is a fact about the scene, worth recording
            # rather than dropping -- it is exactly the density ceiling we want to know.
            print(f"  UNROUTABLE {scene_id}: {error}")
            unroutable.append({"scene_id": scene_id, "reason": str(error)})
            continue

        for number, route in enumerate(routes):
            name = f"{scene_id}__route{number}"
            payload = {
                "route_id": name,
                "scene_id": scene_id,
                "package_id": manifest.get("package_id", args.package.name),
                "generation_direction": "scene_first",
                "path_xy": route.path_xy.tolist(),
                "trajectory_xy": route.trajectory_xy.tolist(),
                "kimodo_root2d": route.to_kimodo_root2d().tolist(),
                "frames": route.frames,
                "duration_s": route.duration_s,
                "speed_mps": route.speed_mps,
                "path_length_m": route.path_length_m,
                "min_clearance_m": route.min_clearance_m,
                "required_clearance_m": route.required_clearance_m,
                "seed": route.seed,
            }
            (args.out / f"{name}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            records.append(
                {k: payload[k] for k in
                 ("route_id", "scene_id", "frames", "path_length_m", "min_clearance_m")}
            )
        print(
            f"  {scene_id:34s} {len(routes)} route(s)  "
            f"lengths {min(r.path_length_m for r in routes):.2f}-"
            f"{max(r.path_length_m for r in routes):.2f} m"
        )

    summary = {
        "routes": len(records),
        "scenes_routed": len({r["scene_id"] for r in records}),
        "scenes_unroutable": unroutable,
        "generation_direction": "scene_first",
        "speed_mps": args.speed,
        "required_clearance_m": args.body_half_width + args.margin,
        "records": records,
    }
    (args.out / "routes.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"\n{len(records)} route(s) across {summary['scenes_routed']} scene(s); "
        f"{len(unroutable)} scene(s) unroutable -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
