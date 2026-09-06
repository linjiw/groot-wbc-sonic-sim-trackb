#!/usr/bin/env python3
"""Audit recorded delay ages and packet/state binding before supervised fitting."""

import json
import math
from pathlib import Path

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)


def main():
    data = ROOT.parent / "research-data/groot-wbc"
    source = data / "m2s-overhang-variation-v1/result.json"
    result = json.loads(source.read_text())
    out = data / "m2s-learning-input-audit-v2"
    out.mkdir(parents=True, exist_ok=False)
    refs = [
        artifact(source),
        artifact(Path(__file__)),
        artifact(
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_reset_capture.py"
        ),
        artifact(data / "m2s-learning-input-audit-v1/registration.json"),
        artifact(data / "m2s-learning-input-audit-v1/failure.json"),
    ]
    for row in result["rows"]:
        refs += [row["sensor"], row["trajectory"]]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "predictions": {
                "age": (
                    "all delivered packets have age ceil(requested_delay*50)/50 "
                    "from capture indices and elapsed timestamps"
                ),
                "state_binding": (
                    "packet upper-ray origin matches recorded root at next recorder index within 1e-6 m; "
                    "audit all available 198 aligned frames per cell"
                ),
            },
            "scope": "CPU measurement contract, not policy fitting; last packet has no next recorded state",
        },
    )
    rows = []
    for row in result["rows"]:
        s = json.loads(Path(row["sensor"]["path"]).read_text())
        p = load_reset_capture(Path(row["trajectory"]["path"]))
        ages = []
        expected = math.ceil(row["requested_delay_s"] * 50 - 1e-9) / 50
        origin_errors = []
        phase_errors = []
        for i, o in enumerate(s["observations"]):
            packet = o["delivered"]
            if packet is not None:
                index_age = (i - packet["capture_frame"]) / 50
                timestamp_age = i / 50 - packet["capture_elapsed_s"]
                ages.append([index_age, timestamp_age])
            if i + 1 < len(p["root_pos_w"]):
                root = np.asarray(p["root_pos_w"])[i + 1]
                w, x, y, z = np.asarray(p["root_quat_w"])[i + 1]
                yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
                origin = root + np.array([0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4])
                origin_errors.append(float(np.max(np.abs(origin - o["origin"]))))
                phase_errors.append(abs(float(p["motion_time_s"][i + 1]) - o["time_s"]))
        a = np.array(ages)
        rows.append(
            {
                "cell_id": row["cell_id"],
                "requested_delay_s": row["requested_delay_s"],
                "realized_delay_s": expected,
                "delivered_packets": len(ages),
                "missing_warmup_packets": len(s["observations"]) - len(ages),
                "index_age_max_error_s": float(abs(a[:, 0] - expected).max()),
                "timestamp_age_max_error_s": float(abs(a[:, 1] - expected).max()),
                "aligned_pose_frames": len(origin_errors),
                "origin_max_error_m": max(origin_errors),
                "reference_phase_max_difference_s": max(phase_errors),
            }
        )
    predictions = {
        "age": all(
            r["index_age_max_error_s"] < 1e-12 and r["timestamp_age_max_error_s"] < 1e-12
            for r in rows
        ),
        "state_binding": all(r["origin_max_error_m"] <= 1e-6 for r in rows),
    }
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "predictions": predictions,
            "robot_data_fits": 0,
            "scope": (
                "Age uses monotonic capture time, not resettable reference phase; "
                "next recorder index is the same observed robot state, "
                "not authorization to use a future state"
            ),
        },
    )
    print(json.dumps(predictions))


if __name__ == "__main__":
    main()
