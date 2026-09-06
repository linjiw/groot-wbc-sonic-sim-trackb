#!/usr/bin/env python3
"""Audit all evaluation-bank runs using recorded native outer-envelope support."""

import argparse
import json
from pathlib import Path
import pickle
import time

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_native_crossing import (
    downstream_support,
    stable_finish,
)
from gear_sonic.dataset_generation.swept_volume import CollisionCapsule

DATA = ROOT.parent / "research-data/groot-wbc"


def audit(out):
    source = DATA / "m2s-overhang-eval-bank-v2/result.json"
    result = json.loads(source.read_text())
    native = DATA / "m2s-native-beam-audit-v1/result.json"
    ref = json.loads(native.read_text())["geometry"]
    geometry = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    manifest_ref = result["manifest"]
    manifest = json.loads(checked(Path(manifest_ref["path"]), manifest_ref["sha256"]).read_text())
    for ref in geometry["layers"]:
        checked(Path(ref["path"]), ref["sha256"])
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "source": artifact(source),
            "geometry": artifact(Path(json.loads(native.read_text())["geometry"]["path"])),
            "dependencies": [
                artifact(Path(__file__)),
                artifact(
                    ROOT
                    / "gear_sonic/dataset_generation/hallucination/motion2scene_native_crossing.py"
                ),
                artifact(ROOT / "gear_sonic/dataset_generation/swept_volume.py"),
            ],
            "protocol": artifact(ROOT / "docs/motion2scene/NATIVE_CROSSING_AUDIT_V1.md"),
            "gpu_hours": 0,
            "cpu_seconds_ceiling": 300,
        },
    )
    shapes = {}
    for s in geometry["shapes"]:
        shapes.setdefault(s["owner"], []).append(
            CollisionCapsule(tuple(s["start"]), tuple(s["end"]), s["radius"])
        )
    rows = []
    start = time.monotonic()
    for row in result["rows"]:
        if time.monotonic() - start > 300:
            raise TimeoutError("CPU ceiling")
        ref = row["trajectory"]
        with checked(Path(ref["path"]), ref["sha256"]).open("rb") as f:
            payload = pickle.load(f)
        if payload["quat_format"] != "wxyz" or payload["fps"] != 50:
            raise ValueError("pose contract")
        end = row["first_episode_frames"]
        support = downstream_support(
            payload["body_pos_w"][:end],
            payload["body_quat_w"][:end],
            payload["body_names"],
            shapes,
            manifest["beam"],
        )
        upright = (np.asarray(payload["root_pos_w"])[:end, 2] >= 0.5) & (
            -np.asarray(payload["projected_gravity_b"])[:end, 2] >= 0.5
        )
        finish = stable_finish(support, upright, manifest["beam"]["length_m"] / 2 + 0.1)
        horizon = finish or end
        ref = row["physics_contacts"]
        checked(Path(ref["path"]), ref["sha256"])
        with np.load(ref["path"]) as raw:
            force = raw["force_w"]
            dt = float(raw["physics_dt"])
            if force.shape != (796, 30, 3) or dt != 0.005:
                raise ValueError("force contract")
            peak = float(np.linalg.norm(force[: horizon * 4], axis=-1).max())
        passed = finish is not None and peak <= 1
        raw_path = out / (row["cell_id"] + ".npz")
        np.savez_compressed(raw_path, downstream_support_m=support, upright=upright)
        previous = row["passage_finish_frame_exclusive"]
        rows.append(
            {
                "cell_id": row["cell_id"],
                "condition": row["condition"],
                "mode": row["mode"],
                "seed": row["seed"],
                "previous_pass": row["pass"],
                "native_outer_pass": passed,
                "body_origin_finish_frame": previous,
                "native_outer_finish_frame": finish,
                "finish_delay_s": (
                    (finish - previous) / 50
                    if finish is not None and previous is not None
                    else None
                ),
                "peak_normal_force_n_through_native_horizon": peak,
                "max_downstream_support_m": float(support.max()),
                "trajectory": row["trajectory"],
                "physics_contacts": row["physics_contacts"],
                "support": artifact(raw_path),
            }
        )
    for ref in geometry["layers"]:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "source_result": artifact(source),
            "rows": rows,
            "prediction_unchanged": all(r["previous_pass"] == r["native_outer_pass"] for r in rows),
            "passed": sum(r["native_outer_pass"] for r in rows),
            "requested": len(rows),
            "native_outer_shapes": sum(map(len, shapes.values())),
            "cpu_seconds": time.monotonic() - start,
            "gpu_hours": 0,
            "scope": (
                "Authored native outer shapes at recorded 50 Hz poses; "
                "200 Hz measured contact. Not cooked-mesh CCD."
            ),
        },
    )
    print(
        json.dumps(
            {
                "passed": sum(r["native_outer_pass"] for r in rows),
                "requested": len(rows),
                "changed": sum(r["previous_pass"] != r["native_outer_pass"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit(a.out)
