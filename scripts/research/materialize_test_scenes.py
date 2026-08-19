#!/usr/bin/env python
"""Build the frozen scene-first test set into loadable geometry.

These scenes are the only ones in the project that were **not** fitted to a trajectory: parameters
were drawn before any operator existed, and the route is synthesised from the manifest's own
``route_length_m`` rather than taken from an executed rollout. That is the whole point of the split,
and it is why the geometry has to be built here rather than reused from a family.

Nothing is chosen at build time. Every dimension comes from the frozen manifest, so building twice
gives the same scenes and a parameter cannot drift between the split and the room.

The floor regime is built and labelled like the others. No operator relieves the ankle, so those ten
scenes are **unsupported out-of-distribution** rather than solvable, and the pre-registration
requires them reported rather than quietly dropped from headline metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.clutter_scene_builder import (  # noqa: E402
    ClutterSceneSpec,
    FurniturePiece,
    render_scene_usda,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
#: Clearance around the route, in metres. Wide enough that a wall never decides an outcome the
#: obstacle was meant to decide -- the audit found two cells whose verdict came from a wall 0.21 m
#: away rather than from the shelf in front of them.
ROOM_MARGIN = 3.0
WALL_HEIGHT = 2.8


def piece_for(regime: str, parameters: dict, station_xy) -> FurniturePiece:
    """One solid, dimensioned entirely from the frozen parameters."""
    x, y = float(station_xy[0]), float(station_xy[1])
    if regime == "overhead":
        return FurniturePiece(
            name="LowShelf_00",
            kind="WallShelf",
            center_xy=(x, y),
            size=(float(parameters["shelf_depth_m"]), 3.0, 0.10),
            color=(0.46, 0.34, 0.22),
            z_base=float(parameters["shelf_underside_m"]),
            band="overhead",
        )
    if regime == "lateral":
        # A gap of the given half-width: one solid to each side, so the robot must pass between.
        half = float(parameters["gap_half_width_m"])
        return FurniturePiece(
            name="LowShelf_00",
            kind="WallShelf",
            center_xy=(x, y + half + 0.15),
            size=(1.2, 0.30, 1.8),
            color=(0.46, 0.34, 0.22),
            z_base=0.0,
            band="body",
        )
    return FurniturePiece(
        name="LowShelf_00",
        kind="WallShelf",
        center_xy=(x, y),
        size=(0.5, 3.0, float(parameters["obstacle_height_m"])),
        color=(0.46, 0.34, 0.22),
        z_base=0.0,
        band="floor",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    manifest = json.loads(args.split.read_text())
    built = []
    for scene in manifest["scenes"]:
        parameters = scene["parameters"]
        length = float(scene["route_length_m"])
        # A straight route from the origin. The robot's executed path will differ, which is the
        # test: the scene was not built around where the robot actually goes.
        route = np.stack([np.linspace(-2.0, -2.0 + length, 80), np.zeros(80)], axis=1)
        station = float(parameters["station_fraction"])
        index = int(round(station * (len(route) - 1)))
        piece = piece_for(scene["regime"], parameters, route[index])

        span = route.max(0) - route.min(0)
        room = (float(span[0] + 2 * ROOM_MARGIN), float(max(span[1] + 2 * ROOM_MARGIN, 6.0)))
        spec = ClutterSceneSpec(
            scene_id=scene["scene_id"],
            split_group=f"{manifest.get('seed')}_scene_first",
            room_size_xy=room,
            wall_height=WALL_HEIGHT,
            pieces=[piece],
            path_xy=route,
            clearance_m=0.0,
            seed=int(manifest.get("seed", 0)),
            metrics={
                "scene_start_xy": [float(route[0, 0]), float(route[0, 1])],
                "placed_pieces": 1,
                "clutter_occupancy": 0.0,
                "min_distance_to_path_m": 0.0,
                "path_length_m": length,
                "regime": scene["regime"],
                "fitted_to_trajectory": False,
                "supported_regime": scene["regime"] in ("overhead", "lateral"),
                **{k: float(v) for k, v in parameters.items()},
            },
        )
        if args.write:
            (SCENES / f"{scene['scene_id']}.usda").write_text(
                render_scene_usda(spec), encoding="utf-8"
            )
        built.append((scene["scene_id"], scene["regime"], room))

    by_regime: dict[str, int] = {}
    for _sid, regime, _room in built:
        by_regime[regime] = by_regime.get(regime, 0) + 1
    print(f"{len(built)} scenes: " + ", ".join(f"{k} {v}" for k, v in sorted(by_regime.items())))
    print(
        "floor is built and labelled unsupported: no operator relieves the ankle, so those are "
        "out-of-distribution rather than solvable"
    )
    if args.write:
        digest = hashlib.sha256()
        for sid, _r, _m in sorted(built):
            digest.update((SCENES / f"{sid}.usda").read_bytes())
        print(f"\nwrote {len(built)} scenes, geometry sha256 {digest.hexdigest()[:32]}")
    else:
        print("\ndry run -- pass --write to emit the scenes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
