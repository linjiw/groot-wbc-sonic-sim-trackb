#!/usr/bin/env python3
"""Independently verify directly captured learning inputs and loaded skill banks."""

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["register", "analyze"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    registration = out / "capture_audit_registration.json"
    if args.command == "register":
        data = ROOT.parent / "research-data/groot-wbc"
        old = json.loads((data / "m2s-action-label-completion-v1/result.json").read_text())
        bank = old["rows"][0]["bank"]
        write_new(
            registration,
            {
                "manifest": artifact(out / "manifest.json"),
                "bank": bank,
                "implementation": artifact(Path(__file__)),
                "predictions": [
                    "direct features recompute exactly",
                    "rays originate from direct pose within 1e-6 m",
                    "same-index recorder pose/state matches direct callback within 1e-7",
                    "every loaded bank matches previous bound neutral/d040 bank bitwise",
                ],
                "scope": "first 0.30 s callback only, no whole-capture index shift",
            },
        )
        return
    reg = json.loads(registration.read_text())
    for key in ("manifest", "bank", "implementation"):
        checked(Path(reg[key]["path"]), reg[key]["sha256"])
    manifest = json.loads(Path(reg["manifest"]["path"]).read_text())
    result = json.loads((out / "result.json").read_text())
    rows = []
    for c, r in zip(manifest["cells"], result["rows"]):
        assert c["cell_id"] == r["cell_id"]
        d = json.loads(checked(Path(r["decision"]["path"]), r["decision"]["sha256"]).read_text())
        p = load_reset_capture(checked(Path(r["trajectory"]["path"]), r["trajectory"]["sha256"]))
        i = d["capture_frame"]
        assert list(p["dof_joint_names"]) == d["joint_names"]
        feature = decision_features(
            d["packet"], d["state"], d["phase_s"], d["active_before"], d["observation_age_s"]
        )
        w, x, y, z = d["root_quat_w"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        origin = np.array(d["root_pos_w"]) + [0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4]
        state_errors = {
            k: float(np.max(abs(np.array(v) - p[k][i])))
            for k, v in {
                **d["state"],
                "root_pos_w": d["root_pos_w"],
                "root_quat_w": d["root_quat_w"],
            }.items()
        }
        with np.load(Path(c["output"]) / "trajectories/loaded_reference_bank.npz") as bank, np.load(
            reg["bank"]["path"]
        ) as old:
            same = all(np.array_equal(bank[k], old[k]) for k in ("root_xyz", "joint_pos", "fps"))
        rows.append(
            {
                "cell_id": c["cell_id"],
                "feature_exact": bool(
                    np.array_equal(feature, np.array(d["features"], dtype=np.float32))
                ),
                "origin_error_m": float(np.max(abs(origin - d["packet"]["origin"]))),
                "state_errors": state_errors,
                "bank_exact": same,
            }
        )
    predicates = {
        "features_exact": all(r["feature_exact"] for r in rows),
        "ray_origin_bound": all(r["origin_error_m"] <= 1e-6 for r in rows),
        "recorder_state_bound": all(max(r["state_errors"].values()) <= 1e-7 for r in rows),
        "reference_bank_exact": all(r["bank_exact"] for r in rows),
    }
    write_new(
        out / "capture_audit.json",
        {
            "registration": artifact(registration),
            "result": artifact(out / "result.json"),
            "rows": rows,
            "predictions": predicates,
        },
    )
    print(json.dumps(predicates))


if __name__ == "__main__":
    main()
