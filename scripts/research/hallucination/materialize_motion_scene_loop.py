#!/usr/bin/env python3
"""Materialize a fresh Kimodo motion and its LFH local-crouch counterfactual twin.

This is the file boundary between motion generation and LFH.  It does not author a scene or
predict a physics verdict: it verifies the generated artifact, applies the existing deterministic
operator, and records the route-relative 3D frame in which later obstacle proposals are expressed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    active_frames,
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402


def route_frame(qpos: np.ndarray, station_fraction: float) -> dict[str, object]:
    root_xy = np.asarray(qpos[:, :2], dtype=np.float64)
    progress = route_progress(root_xy)
    index = int(np.argmin(np.abs(progress - station_fraction)))
    before = max(0, index - 2)
    after = min(len(root_xy) - 1, index + 2)
    tangent = root_xy[after] - root_xy[before]
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-9:
        raise ValueError("route tangent is degenerate at the LFH station")
    tangent /= norm
    lateral = np.asarray((-tangent[1], tangent[0]), dtype=np.float64)
    return {
        "station_fraction": station_fraction,
        "reference_frame": index,
        "station_xyz_m": [float(root_xy[index, 0]), float(root_xy[index, 1]), 0.0],
        "route_tangent_xyz": [float(tangent[0]), float(tangent[1]), 0.0],
        "route_lateral_xyz": [float(lateral[0]), float(lateral[1]), 0.0],
        "world_vertical_xyz": [0.0, 0.0, 1.0],
        "candidate_face_normals": {
            "overhead_down": [0.0, 0.0, -1.0],
            "lateral_left": [float(lateral[0]), float(lateral[1]), 0.0],
            "lateral_right": [float(-lateral[0]), float(-lateral[1]), 0.0],
            "floor_up": [0.0, 0.0, 1.0],
        },
    }


def save_motion(path: Path, key: str, qpos: np.ndarray) -> None:
    entry = qpos_to_sonic_motion_entry(qpos, source_fps=30.0)
    save_sonic_motion_file(path, motion_key=key, motion_entry=entry)
    path.chmod(0o664)


def gate_dict(gate) -> dict[str, object]:
    return {
        "worth_a_rollout": gate.worth_a_rollout,
        "diagnosis": gate.diagnosis,
        "embodiment_feasible": gate.embodiment_feasible,
        "self_collision_free": gate.self_collision_free,
        "reference_semantic_valid": gate.reference_semantic_valid,
        "saturated_cell_fraction": gate.saturated_cell_fraction,
    }


def write_provenance(
    path: Path,
    *,
    key: str,
    csv_path: Path,
    generation_record: Path,
    role: str,
    operator: dict[str, object] | None,
) -> Path:
    output = path.with_suffix(path.suffix + ".manifest.json")
    payload = {
        "schema_version": "lfh_motion_scene_loop_motion_v1",
        "motion_key": key,
        "motion_role": role,
        "source_fps": 30.0,
        "scene_start_xyz": [0.0, 0.0, 0.0],
        "scene_yaw": 0.0,
        "generation_record": {
            "path": str(generation_record),
            "sha256": sha256_file(generation_record),
        },
        "input": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "operator": operator,
        "output": {"path": str(path), "sha256": sha256_file(path)},
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    output.chmod(0o664)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-csv", type=Path, required=True)
    parser.add_argument("--generation-record", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--station-fraction", type=float, default=0.55)
    parser.add_argument("--target-drop-m", type=float, default=0.08)
    parser.add_argument("--window-fraction", type=float, default=0.30)
    args = parser.parse_args()

    generation = json.loads(args.generation_record.read_text())
    if generation.get("csv") != args.generated_csv.name:
        raise SystemExit("generation record does not name the supplied CSV")
    qpos = np.loadtxt(args.generated_csv, delimiter=",")
    nominal_gate = screen_reference(qpos, "lfh_fresh_curve_nominal", "walk")
    if not nominal_gate.worth_a_rollout:
        raise SystemExit(f"generated nominal refused: {nominal_gate.diagnosis}")

    adapted, operator_report = local_crouch(
        qpos,
        args.station_fraction,
        target_drop_m=args.target_drop_m,
        window=args.window_fraction,
    )
    adapted_gate = screen_reference(adapted, "lfh_fresh_curve_crouch", "walk")
    if not adapted_gate.worth_a_rollout or not operator_report.root_path_preserved:
        raise SystemExit(f"adapted motion refused: {adapted_gate.diagnosis}")

    args.out.mkdir(parents=True, exist_ok=True)
    nominal_csv = args.out / "nominal.csv"
    adapted_csv = args.out / "adapted.csv"
    np.savetxt(nominal_csv, qpos, delimiter=",", fmt="%.10f")
    np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
    nominal_motion = args.out / "nominal.pkl"
    adapted_motion = args.out / "adapted.pkl"
    save_motion(nominal_motion, "lfh_fresh_curve__nominal", qpos)
    save_motion(adapted_motion, "lfh_fresh_curve__adapted", adapted)

    operator = {
        "name": "local_crouch",
        "station_fraction": args.station_fraction,
        "target_drop_m": args.target_drop_m,
        "window_fraction": args.window_fraction,
        "report": asdict(operator_report),
    }
    nominal_manifest = write_provenance(
        nominal_motion,
        key="lfh_fresh_curve__nominal",
        csv_path=nominal_csv,
        generation_record=args.generation_record,
        role="nominal",
        operator=None,
    )
    adapted_manifest = write_provenance(
        adapted_motion,
        key="lfh_fresh_curve__adapted",
        csv_path=adapted_csv,
        generation_record=args.generation_record,
        role="adapted",
        operator=operator,
    )
    active = np.flatnonzero(active_frames(qpos, adapted))
    report = {
        "schema_version": "lfh_motion_scene_loop_pair_v1",
        "status": "cpu_motion_pair_ready_for_empty_scene_physics",
        "generation": {
            "prompt": generation["prompt"],
            "seed": generation["seed"],
            "model": generation["model"],
            "record": str(args.generation_record),
            "record_sha256": sha256_file(args.generation_record),
        },
        "route_frame": route_frame(qpos, args.station_fraction),
        "supported_constraint_axes": ["overhead"],
        "unsupported_until_executed_separation_exists": ["lateral", "floor", "oblique"],
        "operator": operator,
        "active_reference_frames": [int(active[0]), int(active[-1])],
        "gates": {"nominal": gate_dict(nominal_gate), "adapted": gate_dict(adapted_gate)},
        "artifacts": {
            "nominal_csv": str(nominal_csv),
            "nominal_motion": str(nominal_motion),
            "nominal_motion_sha256": sha256_file(nominal_motion),
            "nominal_manifest": str(nominal_manifest),
            "nominal_manifest_sha256": sha256_file(nominal_manifest),
            "adapted_csv": str(adapted_csv),
            "adapted_motion": str(adapted_motion),
            "adapted_motion_sha256": sha256_file(adapted_motion),
            "adapted_manifest": str(adapted_manifest),
            "adapted_manifest_sha256": sha256_file(adapted_manifest),
        },
    }
    output = args.out / "motion_pair.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    output.chmod(0o664)
    print(
        f"PASS: fresh Kimodo pair; drop={1000 * operator_report.silhouette_drop_m:.1f} mm -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
