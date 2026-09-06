#!/usr/bin/env python3
"""Benchmark proxy/native geometry and bound a declared 200 Hz body interpolant."""

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import time

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_temporal_geometry import (
    clearance_lower_bounds,
    interpolate_poses,
    interval_displacement,
)
from gear_sonic.dataset_generation.swept_volume import (
    G1_COLLISION_CAPSULES,
    CollisionCapsule,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import find_reset_boundaries

DATA = ROOT.parent / "research-data/groot-wbc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    paths = {
        "repeat": DATA / "m2s-repeatability-v1/result.json",
        "middle": DATA / "m2s-ladder-extension-resource-v2/result.json",
        "native": DATA / "m2s-native-beam-audit-v1/result.json",
        "teacher": DATA / "m2s-beam-teacher-v1/result.json",
    }
    records = {k: json.loads(p.read_text()) for k, p in paths.items()}
    ref = records["native"]["geometry"]
    geometry = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    for layer in geometry["layers"]:
        checked(Path(layer["path"]), layer["sha256"])
    variants = {"proxy": G1_COLLISION_CAPSULES, "native_inner": {}, "native_outer": {}}
    for shape in geometry["shapes"]:
        cap = CollisionCapsule(tuple(shape["start"]), tuple(shape["end"]), shape["radius"])
        variants["native_outer"].setdefault(shape["owner"], []).append(cap)
        if shape["role"] == "native_primitive_subset":
            variants["native_inner"].setdefault(shape["owner"], []).append(cap)
    beam = records["teacher"]["selected_candidate"]
    height = next(
        p["beam_underside_m"] for p in records["teacher"]["proposals"] if p["quantile"] == 0.5
    )
    offsets = list(product((-0.02, 0, 0.02), (-0.02, 0, 0.02), (-0.01, 0, 0.01), (-0.02, 0, 0.02)))
    sources = records["repeat"]["rows"] + records["middle"]["rows"]
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/TEMPORAL_NATIVE_AUDIT_V1.md"),
            "inputs": {k: artifact(p) for k, p in paths.items()},
            "geometry": ref,
            "dependencies": [
                artifact(Path(__file__)),
                artifact(
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_temporal_geometry.py"
                ),
                artifact(ROOT / "gear_sonic/dataset_generation/capsule_box_exact.py"),
                artifact(ROOT / "gear_sonic/dataset_generation/swept_volume.py"),
            ],
            "trajectories": [s["trajectory"] for s in sources],
            "offsets": offsets,
            "beam": beam,
            "underside_m": height,
            "subdivisions": [1, 4],
            "gpu_seconds": 0,
        },
    )
    start = time.monotonic()
    rows, baseline_errors = [], []
    old = {r["cell_id"]: r for r in records["native"]["rows"]}
    for source in sources:
        ref = source["trajectory"]
        with checked(Path(ref["path"]), ref["sha256"]).open("rb") as handle:
            payload = pickle.load(handle)
        if payload["quat_format"] != "wxyz" or payload["fps"] != 50:
            raise ValueError("requires original 50 Hz wxyz recordings")
        end = next(
            iter(find_reset_boundaries(payload["motion_time_s"])), len(payload["body_pos_w"])
        )
        for subdivisions in (1, 4):
            tick = time.monotonic()
            p, q = interpolate_poses(
                payload["body_pos_w"][:end], payload["body_quat_w"][:end], subdivisions
            )
            interpolation_seconds = time.monotonic() - tick
            for kind, shapes in variants.items():
                if time.monotonic() - start > 300:
                    raise TimeoutError("temporal audit budget exhausted")
                tick = time.monotonic()
                a, b, radii, _ = body_capsules_world(p, q, payload["body_names"], capsules=shapes)
                displacement = interval_displacement(p, q, payload["body_names"], shapes)
                geometry_seconds = time.monotonic() - tick
                minima, lower, witnesses = [], [], []
                tick = time.monotonic()
                for dx, dy, dz, dyaw in offsets:
                    yaw = beam["yaw_rad"] + dyaw
                    rotation = np.array(
                        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
                    )
                    center = np.array(
                        [
                            beam["center_xy_m"][0] + dx,
                            beam["center_xy_m"][1] + dy,
                            height + 0.05 + dz,
                        ]
                    )
                    half = np.array([beam["length_m"], beam["width_m"], 0.1]) / 2
                    c = capsule_box_clearance(
                        (a - center) @ rotation, (b - center) @ rotation, radii, -half, half
                    )
                    minima.append(float(c.min()))
                    witnesses.append(np.unravel_index(np.argmin(c), c.shape))
                    lower.append(clearance_lower_bounds(c, displacement))
                query_seconds = time.monotonic() - tick
                minima, lower = np.array(minima), np.array(lower)
                identifier = f"{source['cell_id']}_{kind}_{50 * subdivisions}hz"
                raw = args.out / f"{identifier}.npz"
                np.savez_compressed(
                    raw, sampled_minima=minima, interval_lower_bounds=lower, witnesses=witnesses
                )
                row = {
                    "cell_id": source["cell_id"],
                    "label": source["label"],
                    "geometry": kind,
                    "hz": 50 * subdivisions,
                    "poses": len(p),
                    "shapes": len(radii),
                    "interpolation_seconds": interpolation_seconds,
                    "geometry_seconds": geometry_seconds,
                    "query_seconds": query_seconds,
                    "whole_motion_queries": len(offsets),
                    "sampled_minimum_m": float(minima.min()),
                    "sampled_weakest_interference_m": float(minima.max()),
                    "interval_minimum_lower_bound_m": float(lower.min()),
                    "sampled_clearance_10mm_offsets": int((minima >= 0.01).sum()),
                    "sampled_interference_10mm_offsets": int((minima <= -0.01).sum()),
                    "interval_clearance_10mm_offsets": int((lower.min(axis=(1, 2)) >= 0.01).sum()),
                    "interval_clearance_positive_offsets": int((lower.min(axis=(1, 2)) > 0).sum()),
                    "raw": artifact(raw),
                }
                rows.append(row)
                role = "native_outer" if source["label"] == "d055" else "native_inner"
                if subdivisions == 1 and kind == role and source["cell_id"] in old:
                    baseline_errors.append(
                        float(np.abs(minima - old[source["cell_id"]]["offset_clearances_m"]).max())
                    )
        print(f"audited {source['cell_id']}", flush=True)
    targets = [
        r
        for r in rows
        if r["label"] == "d055" and r["geometry"] == "native_outer" and r["hz"] == 200
    ]
    upright = [
        r
        for r in rows
        if r["label"] == "neutral" and r["geometry"] == "native_inner" and r["hz"] == 200
    ]
    if len(targets) != 3 or len(upright) != 3 or len(baseline_errors) != 6:
        raise ValueError("unexpected audit denominator")
    write_new(
        args.out / "result.json",
        {
            "registration": artifact(args.out / "registration.json"),
            "rows": rows,
            "baseline_maximum_error_m": max(baseline_errors),
            "baseline_agreement_within_1um": max(baseline_errors) <= 1e-6,
            "predictions": {
                "p1": all(r["sampled_clearance_10mm_offsets"] == 81 for r in targets),
                "p2": all(r["interval_clearance_10mm_offsets"] == 81 for r in targets),
                "p3": all(r["sampled_interference_10mm_offsets"] == 81 for r in upright),
            },
            "whole_motion_queries": sum(r["whole_motion_queries"] for r in rows),
            "elapsed_seconds": time.monotonic() - start,
            "gpu_seconds": 0,
            "new_physics_runs": 0,
            "actual_continuous_dynamics_certified": False,
        },
    )


if __name__ == "__main__":
    main()
