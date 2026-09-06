#!/usr/bin/env python3
"""Check reference-FK versus recorded Isaac body frames before native batch screening."""

import json
from pathlib import Path
import time

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.reference_payload import payload_from_reference
from gear_sonic.dataset_generation.trajectory_export import (
    convert_trajectory_joint_order_to_mujoco,
    load_validated_trajectory,
)


def main():
    data = ROOT.parent / "research-data/groot-wbc"
    out = data / "m2s-native-frame-binding-v1"
    source = data / "m2s-overhang-eval-bank-v2/result.json"
    result = json.loads(source.read_text())
    original = data / "m2s-fresh-source-v1/registration.json"
    mjcf = json.loads(original.read_text())["mjcf"]
    checked(Path(mjcf["path"]), mjcf["sha256"])
    native = data / "m2s-native-beam-audit-v1/geometry.json"
    geometry = json.loads(native.read_text())
    owners = sorted({s["owner"] for s in geometry["shapes"]})
    out.mkdir(parents=True, exist_ok=False)
    write_new(
        out / "registration.json",
        {
            "protocol": artifact(ROOT / "docs/motion2scene/NATIVE_FRAME_BINDING_V1.md"),
            "source_result": artifact(source),
            "original_registration": artifact(original),
            "mjcf": mjcf,
            "native_geometry": artifact(native),
            "dependencies": [
                artifact(Path(__file__)),
                artifact(ROOT / "gear_sonic/dataset_generation/reference_payload.py"),
                artifact(ROOT / "gear_sonic/dataset_generation/trajectory_export.py"),
            ],
            "trajectories": [r["trajectory"] for r in result["rows"]],
            "cpu_ceiling_seconds": 120,
            "gpu_hours": 0,
        },
    )
    start = time.monotonic()
    rows = []
    for row in result["rows"]:
        if time.monotonic() - start > 120:
            raise TimeoutError("CPU ceiling")
        ref = row["trajectory"]
        payload = load_validated_trajectory(checked(Path(ref["path"]), ref["sha256"]))
        normalized, _ = convert_trajectory_joint_order_to_mujoco(payload)
        qpos = np.concatenate(
            [normalized["root_pos_w"], normalized["root_quat_w"], normalized["dof_pos"]], axis=1
        )
        fk = payload_from_reference(qpos, fps=50, mjcf_path=mjcf["path"])
        measured_ids = [list(payload["body_names"]).index(o) for o in owners]
        fk_ids = [list(fk["body_names"]).index(o) for o in owners]
        p = np.asarray(payload["body_pos_w"])[:, measured_ids]
        q = np.asarray(payload["body_quat_w"], dtype=np.float64)[:, measured_ids]
        q /= np.linalg.norm(q, axis=-1, keepdims=True)
        position = np.linalg.norm(p - fk["body_pos_w"][:, fk_ids], axis=-1)
        angle = 2 * np.arccos(np.clip(np.abs((q * fk["body_quat_w"][:, fk_ids]).sum(-1)), 0, 1))
        radii = np.array(
            [
                max(
                    np.linalg.norm(s[k])
                    for s in geometry["shapes"]
                    if s["owner"] == o
                    for k in ["start", "end"]
                )
                for o in owners
            ]
        )
        endpoint_bound = position + 2 * radii * np.sin(angle / 2)
        raw = out / (row["cell_id"] + ".npz")
        np.savez_compressed(
            raw,
            position_error_m=position,
            orientation_error_rad=angle,
            endpoint_displacement_bound_m=endpoint_bound,
        )
        per_owner = {
            o: {
                "position_m": float(position[:, i].max()),
                "orientation_deg": float(np.degrees(angle[:, i]).max()),
                "endpoint_bound_m": float(endpoint_bound[:, i].max()),
            }
            for i, o in enumerate(owners)
        }
        rows.append(
            {
                "cell_id": row["cell_id"],
                "frames": len(qpos),
                "owners": len(owners),
                "raw": artifact(raw),
                "maximum_position_error_m": float(position.max()),
                "maximum_orientation_error_deg": float(np.degrees(angle).max()),
                "maximum_endpoint_displacement_bound_m": float(endpoint_bound.max()),
                "per_owner": per_owner,
                "pass": bool(position.max() <= 0.001 and np.degrees(angle).max() <= 0.1),
            }
        )
    write_new(
        out / "result.json",
        {
            "registration": artifact(out / "registration.json"),
            "rows": rows,
            "prediction_all_pass": all(r["pass"] for r in rows),
            "frames": sum(r["frames"] for r in rows),
            "cpu_seconds": time.monotonic() - start,
            "gpu_hours": 0,
            "scope": (
                "Measured-qpos replay into reference FK; frame agreement, "
                "not executed-reference tracking or collision safety"
            ),
        },
    )
    print(
        json.dumps(
            {
                "all_pass": all(r["pass"] for r in rows),
                "position_m": max(r["maximum_position_error_m"] for r in rows),
                "orientation_deg": max(r["maximum_orientation_error_deg"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
