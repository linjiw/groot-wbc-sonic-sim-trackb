#!/usr/bin/env python3
"""Independent uncapped target distances; preserve original capped-bound failure."""

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
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule, body_capsules_world


def main():
    torch.set_num_threads(2)
    data = ROOT.parent / "research-data/groot-wbc"
    prior = data / "m2s-fresh-native-temporal-v1"
    out = data / "m2s-fresh-native-uncapped-v2"
    result = json.loads((prior / "result.json").read_text())
    oldreg = json.loads((prior / "registration.json").read_text())
    if result["predictions"] != {
        "p1_sampled_all": True,
        "p2_continuous_all": False,
        "p3_proxy_reproduction": True,
    }:
        raise ValueError("unexpected prior result")
    refs = [
        artifact(prior / "result.json"),
        artifact(prior / "registration.json"),
        *oldreg["references"],
        artifact(Path(__file__)),
        artifact(ROOT / "docs/motion2scene/FRESH_NATIVE_UNCAPPED_V2.md"),
    ]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "requested_outputs": 384,
            "allowance_m": 1e-5,
            "cpu_ceiling_seconds": 300,
            "gpu_hours": 0,
        },
    )
    start = time.monotonic()
    reg, _, cases = load_cases(data / "m2s-fresh-source-v1/registration.json")
    geometry = json.loads((data / "m2s-native-beam-audit-v1/geometry.json").read_text())
    shapes = {}
    for s in geometry["shapes"]:
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
    cached = {}
    for name, case in cases.items():
        qpos = shared_origin_pair(
            *[np.loadtxt(case["metadata"][k]["path"], delimiter=",") for k in ["parent", "target"]]
        )[1]
        payload = payload_from_reference(qpos, fps=30, mjcf_path=reg["mjcf"]["path"])
        p, q = interpolate_poses(payload["body_pos_w"], payload["body_quat_w"], 4)
        a, b, r, owners = body_capsules_world(p, q, payload["body_names"], capsules=shapes)
        cached[name] = (a, b, r, owners, interval_displacement(p, q, payload["body_names"], shapes))
    setup = time.monotonic() - start
    seconds = 0
    rows = []
    for row in result["rows"]:
        if time.monotonic() - start > 300:
            raise TimeoutError("CPU ceiling")
        a, b, r, owners, bounds = cached[row["case_id"]]
        yaw = cases[row["case_id"]]["yaw"].item()
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        center = np.array(row["center_m"])
        half = np.array([0.05, 0.6, 0.05])
        tick = time.monotonic()
        values = capsule_box_clearance(
            (a - center) @ rotation, (b - center) @ rotation, r, -half, half
        )
        lower = clearance_lower_bounds(values, bounds, numerical_allowance_m=1e-5)
        seconds += time.monotonic() - tick
        checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
        with np.load(row["raw"]["path"]) as raw:
            error = float(np.abs(np.minimum(0.02, values) - raw["native_120hz_1"]).max())
        frame, shape = np.unravel_index(lower.argmin(), lower.shape)
        path = out / f"{row['seed']}_{row['case_id']}_{row['output_index']}.npz"
        np.savez_compressed(path, clearances_m=values, interval_lower_m=lower)
        rows.append(
            {
                "source": row["source"],
                "case_id": row["case_id"],
                "seed": row["seed"],
                "output_index": row["output_index"],
                "sampled_target_minimum_m": float(values.min()),
                "interval_target_minimum_m": float(lower.min()),
                "neutral_native_minimum_m": row["metrics"]["native_120hz_0"]["minimum_m"],
                "capped_array_reproduction_error_m": error,
                "pass": bool(
                    lower.min() >= 0.01 and row["metrics"]["native_120hz_0"]["minimum_m"] <= -0.01
                ),
                "witness_interval": int(frame),
                "witness_owner": owners[shape],
                "raw": artifact(path),
            }
        )
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "prior_result": artifact(prior / "result.json"),
            "rows": rows,
            "requested": len(rows),
            "passed": sum(r["pass"] for r in rows),
            "predictions": {
                "u1_reproduction": all(
                    r["capped_array_reproduction_error_m"] <= 1e-8 for r in rows
                ),
                "u2_all_bound": len(rows) == 384 and all(r["pass"] for r in rows),
            },
            "setup_seconds": setup,
            "query_seconds": seconds,
            "elapsed_seconds": time.monotonic() - start,
            "gpu_hours": 0,
            "scope": (
                "Nominal authored native envelope and declared 120 Hz body interpolant; "
                "uncapped bound verification only; no fitting or scene changes"
            ),
        },
    )
    print(
        json.dumps(
            {
                "passed": sum(r["pass"] for r in rows),
                "worst_bound_m": min(r["interval_target_minimum_m"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
