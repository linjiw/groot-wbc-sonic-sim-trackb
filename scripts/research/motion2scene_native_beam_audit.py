#!/usr/bin/env python3
"""Audit frozen beam margins on recorded poses with cached native geometry roles."""

import argparse
from itertools import product
import json
from pathlib import Path
import pickle
import time

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
from pxr import Usd

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_native_geometry import (
    extract_native_geometry,
)
from gear_sonic.dataset_generation.swept_volume import body_capsules_world
from gear_sonic.dataset_generation.trajectory_segments import find_reset_boundaries

DATA = ROOT.parent / "research-data/groot-wbc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    repeat_path = DATA / "m2s-repeatability-v1/result.json"
    beam_path = DATA / "m2s-beam-teacher-v1/result.json"
    repeat, teacher = (json.loads(p.read_text()) for p in (repeat_path, beam_path))
    beam = teacher["selected_candidate"]
    height = next(p["beam_underside_m"] for p in teacher["proposals"] if p["quantile"] == 0.5)
    offsets = list(product((-0.02, 0, 0.02), (-0.02, 0, 0.02), (-0.01, 0, 0.01), (-0.02, 0, 0.02)))
    write_new(
        args.out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/NATIVE_BEAM_AUDIT_V1.md"),
            "driver": artifact(Path(__file__)),
            "helper": artifact(
                ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_native_geometry.py"
            ),
            "geometry": artifact(ROOT / "gear_sonic/dataset_generation/capsule_box_exact.py"),
            "pose_geometry": artifact(ROOT / "gear_sonic/dataset_generation/swept_volume.py"),
            "repeatability": artifact(repeat_path),
            "beam": artifact(beam_path),
            "asset": artifact(args.asset),
            "offsets": offsets,
            "cap_seconds": 180,
            "gpu_seconds": 0,
            "new_execution_runs": 0,
        },
    )
    start = time.monotonic()
    stage = Usd.Stage.Open(str(args.asset))
    inner, outer, shapes = extract_native_geometry(stage)
    write_new(
        args.out / "geometry.json",
        {
            "shapes": shapes,
            "layers": [
                artifact(Path(layer.realPath))
                for layer in stage.GetUsedLayers()
                if layer.realPath and Path(layer.realPath).is_file()
            ],
        },
    )
    rows = []
    half = np.array([beam["length_m"], beam["width_m"], 0.1]) / 2
    for source in repeat["rows"]:
        if time.monotonic() - start > 180:
            raise TimeoutError("native audit cap")
        ref = source["trajectory"]
        with checked(Path(ref["path"]), ref["sha256"]).open("rb") as handle:
            payload = pickle.load(handle)
        end = next(
            iter(find_reset_boundaries(payload["motion_time_s"])), len(payload["body_pos_w"])
        )
        target = source["label"] == "d055"
        starts, ends, radii, owners = body_capsules_world(
            payload["body_pos_w"][:end],
            payload["body_quat_w"][:end],
            payload["body_names"],
            capsules=outer if target else inner,
        )
        minima, witnesses = [], []
        for dx, dy, dz, dyaw in offsets:
            yaw = beam["yaw_rad"] + dyaw
            rotation = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            center = np.array(
                [beam["center_xy_m"][0] + dx, beam["center_xy_m"][1] + dy, height + 0.05 + dz]
            )
            clearances = capsule_box_clearance(
                (starts - center) @ rotation, (ends - center) @ rotation, radii, -half, half
            )
            frame, shape = np.unravel_index(np.argmin(clearances), clearances.shape)
            minima.append(float(clearances[frame, shape]))
            witnesses.append(
                {"frame": int(frame), "shape_index": int(shape), "owner": owners[shape]}
            )
        passing = np.array(minima) >= 0.01 if target else np.array(minima) <= -0.01
        index = int(np.argmin(minima) if target else np.argmax(minima))
        row = {
            "cell_id": source["cell_id"],
            "label": source["label"],
            "trajectory": ref,
            "frames": end,
            "shapes": len(radii),
            "geometry_role": "outer_union" if target else "native_primitive_subset",
            "whole_motion_queries": len(offsets),
            "passing_offsets": int(passing.sum()),
            "all_offsets_pass": bool(passing.all()),
            "worst_clearance_m": minima[index],
            "worst_offset": offsets[index],
            "worst_witness": witnesses[index],
            "offset_clearances_m": minima,
            "witnesses": witnesses,
        }
        rows.append(row)
        print(
            json.dumps({k: row[k] for k in ("cell_id", "passing_offsets", "worst_clearance_m")}),
            flush=True,
        )
    write_new(
        args.out / "result.json",
        {
            "registration": artifact(args.out / "registration.json"),
            "geometry": artifact(args.out / "geometry.json"),
            "rows": rows,
            "all_six_cached_geometry_margins_retained": all(r["all_offsets_pass"] for r in rows),
            "whole_motion_queries": sum(r["whole_motion_queries"] for r in rows),
            "elapsed_seconds": time.monotonic() - start,
            "gpu_seconds": 0,
            "new_execution_runs": 0,
            "runtime_shape_agreement_verified": False,
        },
    )


if __name__ == "__main__":
    main()
