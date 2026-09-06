#!/usr/bin/env python3
"""Native authored geometry and conditional time bounds for 384 immutable placements."""

from itertools import product
import json
from pathlib import Path
import time

from motion2scene_fresh_sources import load_cases
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
import torch

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_carriers import shared_origin_pair
from gear_sonic.dataset_generation.hallucination.motion2scene_temporal_geometry import (
    clearance_lower_bounds,
    interpolate_poses,
    interval_displacement,
)
from gear_sonic.dataset_generation.reference_payload import payload_from_reference
from gear_sonic.dataset_generation.swept_volume import (
    G1_COLLISION_CAPSULES,
    CollisionCapsule,
    body_capsules_world,
)

HALF = np.array([0.05, 0.6, 0.05])


def values(a, b, radii, center, rotation):
    a = (a - center) @ rotation
    b = (b - center) @ rotation
    radius = np.broadcast_to(radii, a.shape[:-1])
    reach = radius[..., None] + 0.02
    keep = (np.minimum(a, b) - reach <= HALF).all(-1) & (np.maximum(a, b) + reach >= -HALF).all(-1)
    c = np.full(a.shape[:-1], 0.02)
    if keep.any():
        c[keep] = np.minimum(
            0.02, capsule_box_clearance(a[keep], b[keep], radius[keep], -HALF, HALF)
        )
    return c


def main():
    torch.set_num_threads(2)
    data = ROOT.parent / "research-data/groot-wbc"
    source = data / "m2s-fresh-source-v1"
    out = data / "m2s-fresh-native-temporal-v1"
    original = json.loads((source / "result.json").read_text())
    jobs = [r for r in original["rows"] if r["arm"] == "pattern"]
    if len(jobs) != 48 or any(r["valid"] != 8 for r in jobs):
        raise ValueError("requires frozen full denominator")
    native = data / "m2s-native-beam-audit-v1/geometry.json"
    geometry = json.loads(native.read_text())
    binding = data / "m2s-native-frame-binding-v1/result.json"
    if not json.loads(binding.read_text())["prediction_all_pass"]:
        raise ValueError("frame binding gate failed")
    refs = [
        artifact(source / "result.json"),
        original["registration"],
        artifact(native),
        artifact(binding),
        *geometry["layers"],
        *[r["raw"] for r in jobs],
    ]
    dependencies = [
        Path(__file__),
        ROOT / "docs/motion2scene/FRESH_NATIVE_TEMPORAL_V1.md",
        *[
            ROOT / "gear_sonic/dataset_generation" / name
            for name in [
                "capsule_box_exact.py",
                "reference_payload.py",
                "swept_volume.py",
                "hallucination/motion2scene_temporal_geometry.py",
                "hallucination/motion2scene_carriers.py",
            ]
        ],
        ROOT / "scripts/research/motion2scene_fresh_sources.py",
    ]
    refs += [artifact(p) for p in dependencies]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "nominal_outputs": 384,
            "sample_hz": [30, 120],
            "allowance_m": 1e-5,
            "scope": "nominal fixed placements only; no fitting",
            "cpu_seconds_ceiling": 1200,
            "gpu_hours": 0,
        },
    )
    start = time.monotonic()
    reg, _, cases = load_cases(Path(original["registration"]["path"]))
    cached = {}
    preprocessing_seconds = time.monotonic() - start
    native_outer = {}
    native_inner = {}
    for s in geometry["shapes"]:
        cap = CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        native_outer.setdefault(s["owner"], []).append(cap)
        if s["role"] == "native_primitive_subset":
            native_inner.setdefault(s["owner"], []).append(cap)
    prep = time.monotonic()
    for key, case in cases.items():
        qpos = shared_origin_pair(
            *[
                np.loadtxt(case["metadata"][name]["path"], delimiter=",")
                for name in ["parent", "target"]
            ]
        )
        cached[key] = {}
        for level, q in enumerate(qpos):
            payload = payload_from_reference(q, fps=30, mjcf_path=reg["mjcf"]["path"])
            for subdiv in (1, 4):
                p, quat = interpolate_poses(payload["body_pos_w"], payload["body_quat_w"], subdiv)
                for kind, shapes in [
                    ("proxy", G1_COLLISION_CAPSULES),
                    ("native", native_inner if level == 0 else native_outer),
                ]:
                    a, b, r, owners = body_capsules_world(
                        p, quat, payload["body_names"], capsules=shapes
                    )
                    bounds = interval_displacement(p, quat, payload["body_names"], shapes)
                    cached[key][(level, subdiv, kind)] = (a, b, r, owners, bounds)
    fk_seconds = time.monotonic() - prep
    rows = []
    cost = {f"{kind}_{30*sub}hz": 0.0 for kind in ["proxy", "native"] for sub in [1, 4]}
    queries = {k: 0 for k in cost}
    for job in jobs:
        case = cases[job["case_id"]]
        cache = cached[job["case_id"]]
        arc = np.r_[0, np.linalg.norm(np.diff(case["route"], axis=0), axis=-1).cumsum()]
        arc /= arc[-1]
        yaw = case["yaw"].item()
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        with np.load(job["raw"]["path"]) as raw:
            scenes = raw["scenes"]
            zero = np.flatnonzero((raw["offsets"] == 0).all(1))
            saved = raw["clearances"][:, zero[0]]
            if len(zero) != 1 or not np.array_equal(scenes, job["scenes"]):
                raise ValueError("scene changed")
        for index, (station, height) in enumerate(scenes):
            if time.monotonic() - start > 1200:
                raise TimeoutError("CPU audit cap")
            center = np.array(
                [np.interp(station, arc, case["route"][:, j]) for j in range(2)] + [height + 0.05]
            )
            metrics = {}
            raw_values = {}
            for level, subdiv, kind in product((0, 1), (1, 4), ("proxy", "native")):
                a, b, r, owners, bounds = cache[(level, subdiv, kind)]
                tick = time.monotonic()
                clearance = values(a, b, r, center, rotation)
                lower = clearance_lower_bounds(clearance, bounds, numerical_allowance_m=1e-5)
                name = f"{kind}_{30*subdiv}hz"
                cost[name] += time.monotonic() - tick
                queries[name] += 1
                suffix = f"{name}_{level}"
                raw_values[suffix] = clearance
                raw_values[suffix + "_lower"] = lower
                frame, shape = np.unravel_index(clearance.argmin(), clearance.shape)
                metrics[suffix] = {
                    "minimum_m": float(clearance.min()),
                    "interval_lower_m": float(lower.min()),
                    "witness_frame": int(frame),
                    "witness_owner": owners[shape],
                }
            old = np.minimum(0.02, saved[index])
            reproduction = max(
                abs(metrics[f"proxy_30hz_{level}"]["minimum_m"] - old[level]) for level in (0, 1)
            )
            passes = {
                f"native_{hz}hz": metrics[f"native_{hz}hz_1"]["minimum_m"] >= 0.01
                and metrics[f"native_{hz}hz_0"]["minimum_m"] <= -0.01
                for hz in (30, 120)
            }
            continuous = (
                metrics["native_120hz_1"]["interval_lower_m"] >= 0.01
                and metrics["native_120hz_0"]["minimum_m"] <= -0.01
            )
            path = out / f"{job['seed']}_{job['case_id']}_{index}.npz"
            np.savez_compressed(path, **raw_values)
            rows.append(
                {
                    "source": job["carrier_seed"],
                    "case_id": job["case_id"],
                    "seed": job["seed"],
                    "output_index": index,
                    "station_height": [float(station), float(height)],
                    "center_m": center.tolist(),
                    "metrics": metrics,
                    "sampled_pass": passes,
                    "continuous_conditional_pass": continuous,
                    "proxy_reproduction_error_m": float(reproduction),
                    "raw": artifact(path),
                }
            )
        print(f"audited {len(rows)}/384", flush=True)
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "predictions": {
                "p1_sampled_all": all(all(r["sampled_pass"].values()) for r in rows),
                "p2_continuous_all": all(r["continuous_conditional_pass"] for r in rows),
                "p3_proxy_reproduction": all(r["proxy_reproduction_error_m"] <= 1e-8 for r in rows),
            },
            "sampled_counts": {
                str(hz): sum(r["sampled_pass"][f"native_{hz}hz"] for r in rows) for hz in (30, 120)
            },
            "conditional_continuous_count": sum(r["continuous_conditional_pass"] for r in rows),
            "requested": 384,
            "query_seconds": cost,
            "whole_motion_queries": queries,
            "preprocessing_seconds": preprocessing_seconds,
            "fk_geometry_cache_seconds": fk_seconds,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_hours": 0,
            "scope": (
                "Nominal authored native enclosures and declared rigid-body interpolant; "
                "not cooked-mesh CCD or controller execution. No training or selection on 430xx."
            ),
        },
    )


if __name__ == "__main__":
    main()
