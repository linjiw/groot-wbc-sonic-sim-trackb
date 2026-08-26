#!/usr/bin/env python3
"""Materialize a multi-obstacle LFH scene without adding a second causal constraint.

The binding obstacle comes from an already certified trajectory-conditioned atom. Context
obstacles are placed in the route frame, then rejected unless both executed motions retain the
configured keep-out margin. This demonstrates 3D scene composition; it does not promote lateral,
floor, or oblique context faces into verified critical support.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.archetypes import (  # noqa: E402
    AuthoredPrimitive,
    get_archetype,
)
from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    ConstraintSpec,
    sha256_file,
)
from gear_sonic.dataset_generation.hallucination.instantiate import render_scene_usda  # noqa: E402
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    load_executed_payload,
    validate_pair,
    write_report,
)

CONTEXT = (
    # name, route progress, lateral offset, vertical center, along/across/vertical size,
    # face normal pointing toward the route, display colour
    (
        "ContextLeftPillar",
        0.22,
        1.05,
        0.65,
        (0.40, 0.40, 1.30),
        (0.0, -1.0, 0.0),
        (0.55, 0.42, 0.27),
    ),
    (
        "ContextRightRack",
        0.43,
        -1.15,
        0.75,
        (0.35, 0.55, 1.50),
        (0.0, 1.0, 0.0),
        (0.32, 0.39, 0.34),
    ),
    (
        "ContextLeftCrate",
        0.72,
        0.95,
        0.28,
        (0.55, 0.55, 0.56),
        (0.0, -1.0, 0.0),
        (0.52, 0.34, 0.20),
    ),
    (
        "ContextRightHighDuct",
        0.83,
        -0.95,
        2.00,
        (0.70, 0.45, 0.35),
        (0.0, 0.0, -1.0),
        (0.42, 0.50, 0.54),
    ),
)


def _axes(spec: ConstraintSpec, along: float, across: float, vertical: float):
    return (along, across, vertical) if spec.route_axis == "x" else (across, along, vertical)


def _context_primitives(spec: ConstraintSpec) -> tuple[tuple[AuthoredPrimitive, ...], list[dict]]:
    payloads = (load_executed_payload(spec.orig), load_executed_payload(spec.edit))
    roots = np.concatenate(tuple(np.asarray(payload["root_pos_w"])[:, :2] for payload in payloads))
    along_axis = 0 if spec.route_axis == "x" else 1
    across_axis = 1 - along_axis
    along_min = float(roots[:, along_axis].min())
    along_max = float(roots[:, along_axis].max())
    across_center = float(np.median(roots[:, across_axis]))
    station = spec.binding_station_xy_m
    primitives = []
    records = []
    for name, progress, lateral, z, size, normal, color in CONTEXT:
        world_xy = [0.0, 0.0]
        world_xy[along_axis] = along_min + progress * (along_max - along_min)
        world_xy[across_axis] = across_center + lateral
        local = (world_xy[0] - station[0], world_xy[1] - station[1], z)
        primitive = AuthoredPrimitive(
            name,
            local,
            _axes(spec, *size),
            color,
            "constraint_context",
        )
        primitives.append(primitive)
        records.append(
            {
                "name": name,
                "route_progress": progress,
                # The along component is route-progress dependent and the across component is
                # measured from the station, not from the route's median.  Recording
                # [0, lateral, z] made every reconstruction of world = station + offset wrong by
                # up to 1.2 m along route.  The authored geometry was always correct; only this
                # record was not.
                "route_frame_offset_m": [
                    world_xy[along_axis] - station[along_axis],
                    world_xy[across_axis] - station[across_axis],
                    z,
                ],
                "across_offset_from_route_median_m": lateral,
                "world_center_m": [world_xy[0], world_xy[1], z],
                "size_route_frame_m": list(size),
                "face_normal_route_frame": list(normal),
                "support_status": "context_only_not_a_verified_critical_axis",
            }
        )
    return tuple(primitives), records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archetype", default="hanging_panel")
    parser.add_argument("--seed", type=int, default=33101)
    args = parser.parse_args()

    args.spec = args.spec.resolve()
    args.out_package = args.out_package.resolve()
    args.out = args.out.resolve()
    spec = ConstraintSpec.load(args.spec)
    if spec.route_axis not in {"x", "y"}:
        raise SystemExit("context-rich v1 requires a certified axis-aligned route")
    args.out_package.mkdir(parents=True, exist_ok=True)
    context, records = _context_primitives(spec)
    scenes = {}
    for cell, coordinate in (("easy", spec.easy_coordinate_m), ("hard", spec.hard_coordinate_m)):
        binding = get_archetype(args.archetype, spec.axis_type).build(spec, coordinate, args.seed)
        scene_id = f"{spec.spec_id}__{args.archetype}_context4__s{args.seed:08d}__{cell}"
        path = args.out_package / f"{scene_id}.usda"
        path.write_text(render_scene_usda(spec, scene_id, binding + context), encoding="utf-8")
        path.chmod(0o664)
        scenes[cell] = path

    keepout = validate_pair(spec, scenes["easy"], scenes["hard"])
    keepout_path = args.out_package / f"{spec.spec_id}__context4__s{args.seed:08d}.keepout.json"
    write_report(keepout, keepout_path)
    result = {
        "schema_version": "lfh_context_scene_v1",
        "purpose": "multi-obstacle 3D presentation and context-survival physics pilot",
        "scientific_semantics": {
            "binding_count": 1,
            "context_count": len(context),
            "context_is_not_critical_support": True,
            "physics_verdict_is_not_predicted": True,
        },
        "spec": str(args.spec),
        "spec_sha256": sha256_file(args.spec),
        "archetype": args.archetype,
        "seed": args.seed,
        "context": records,
        "scenes": {cell: str(path.relative_to(REPO_ROOT)) for cell, path in scenes.items()},
        "scene_sha256": {cell: sha256_file(path) for cell, path in scenes.items()},
        "keepout": str(keepout_path.relative_to(REPO_ROOT)),
        "keepout_sha256": sha256_file(keepout_path),
        "keepout_ok": keepout["ok"],
        "keepout_min_clearance_mm": {
            cell: keepout["reports"][cell]["keepout_min_clearance_mm"] for cell in ("easy", "hard")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if keepout["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
