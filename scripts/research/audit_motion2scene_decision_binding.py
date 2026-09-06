#!/usr/bin/env python3
"""Bind the prespecified first 0.20 s decision to its recorded physical state."""

import json
import math
from pathlib import Path

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    decision_features,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_reset_capture import (
    load_reset_capture,
)


def main():
    data = ROOT.parent / "research-data/groot-wbc"
    source = data / "m2s-overhang-variation-v1/result.json"
    result = json.loads(source.read_text())
    out = data / "m2s-decision-binding-v1"
    out.mkdir(parents=True, exist_ok=False)
    refs = [
        artifact(source),
        artifact(Path(__file__)),
        artifact(data / "m2s-learning-input-audit-v3/result.json"),
        artifact(
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_reset_capture.py"
        ),
        artifact(
            ROOT / "gear_sonic/dataset_generation/hallucination/motion2scene_outcome_learner.py"
        ),
    ]
    for row in result["rows"]:
        refs += [row["sensor"], row["trajectory"]]
    for ref in refs:
        checked(Path(ref["path"]), ref["sha256"])
    write_new(
        out / "registration.json",
        {
            "references": refs,
            "decision_time_s": 0.2,
            "prediction": (
                "first decision packet matches same-index post-physics pose "
                "within 1e-6 m in all 42 cells"
            ),
            "scope": (
                "first decision only, fixed before input audits; "
                "whole-capture reset/terminal binding failed and remains separate"
            ),
            "training_eligible": False,
        },
    )
    rows = []
    keys = ("projected_gravity_b", "root_lin_vel_w", "root_ang_vel_w", "dof_pos", "dof_vel")
    for row in result["rows"]:
        obs = json.loads(Path(row["sensor"]["path"]).read_text())["observations"]
        p = load_reset_capture(Path(row["trajectory"]["path"]))
        i = next(i for i, o in enumerate(obs) if o["time_s"] == 0.2)
        o = obs[i]
        w, x, y, z = p["root_quat_w"][i]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        origin = p["root_pos_w"][i] + np.array([0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4])
        error = float(np.max(np.abs(origin - o["origin"])))
        delivered = o["delivered"]
        feature_shape = None
        if delivered is not None:
            age = (i - delivered["capture_frame"]) / 50
            features = decision_features(delivered, {k: p[k][i] for k in keys}, 0.2, 0, age)
            feature_shape = list(features.shape)
        rows.append(
            {
                "cell_id": row["cell_id"],
                "packet_index": i,
                "recorded_state_index": i,
                "reference_phase_before_command_s": float(p["motion_time_s"][i]),
                "decision_phase_after_command_update_s": 0.2,
                "origin_error_m": error,
                "sensor_available": delivered is not None,
                "feature_shape": feature_shape,
            }
        )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "prediction": all(r["origin_error_m"] <= 1e-6 for r in rows),
            "training_eligible": False,
            "robot_data_fits": 0,
        },
    )
    print(
        json.dumps(
            {
                "prediction": all(r["origin_error_m"] <= 1e-6 for r in rows),
                "max_origin_error_m": max(r["origin_error_m"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
