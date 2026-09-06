#!/usr/bin/env python3
"""Propose finite beam tokens from full world-frame reference and executed capsule geometry.

This is a deterministic analytic teacher diagnostic. It creates no physics verdict or
learning-eligible label. The searched roof interval is checked against the finite beam
at every point of the explicitly recorded discrete placement-jitter grid.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
import pickle
import sys

from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

sys.path.insert(0, str(ROOT))
from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance  # noqa: E402
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from gear_sonic.dataset_generation.swept_volume import body_capsules_world  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload  # noqa: E402

MARGIN = 0.01
JITTER_Z = 0.01
THICKNESS = 0.10


def capsules(payload):
    starts, ends, radii, owners = body_capsules_world(
        np.asarray(payload["body_pos_w"]), np.asarray(payload["body_quat_w"]), payload["body_names"]
    )
    return {"starts": starts, "ends": ends, "radii": radii, "owners": owners}


def local_capsules(state, center_xy, yaw, length, width):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    origin = np.array([*center_xy, 0.0])
    starts = (state["starts"] - origin) @ rotation
    ends = (state["ends"] - origin) @ rotation
    radii = np.broadcast_to(state["radii"], starts.shape[:-1])
    # Certified broad-phase exclusion: a segment AABB farther than radius+margin
    # in either horizontal coordinate cannot violate that clearance margin.
    reach = radii[..., None] + MARGIN + JITTER_Z
    half = np.array([length, width]) / 2
    keep = np.all(np.minimum(starts[..., :2], ends[..., :2]) - reach <= half, axis=-1) & np.all(
        np.maximum(starts[..., :2], ends[..., :2]) + reach >= -half, axis=-1
    )
    return starts[keep], ends[keep], radii[keep]


def clearance(local, height, length, width, *, roof=False):
    starts, ends, radii = local
    if len(starts) == 0:
        # Broad-phase absence is sufficient for margin tests, not a measured minimum.
        return MARGIN + JITTER_Z
    return min(
        MARGIN + JITTER_Z,
        float(
            np.min(
                capsule_box_clearance(
                    starts,
                    ends,
                    radii,
                    [-length / 2, -width / 2, height],
                    [length / 2, width / 2, 3.0 if roof else height + THICKNESS],
                )
            )
        ),
    )


def boundary(local, threshold, length, width):
    """Roof clearance increases monotonically as its underside rises."""
    low, high = 0.8, 1.8
    if clearance(local, low, length, width, roof=True) >= threshold:
        return low
    if clearance(local, high, length, width, roof=True) < threshold:
        return None
    for _ in range(20):
        midpoint = (low + high) / 2
        if clearance(local, midpoint, length, width, roof=True) >= threshold:
            high = midpoint
        else:
            low = midpoint
    # Return an enclosing bracket; use high for target clearance and low for strike.
    return (low, high)


def interval(states, center, yaw, length, width):
    values = {}
    for name, state in states.items():
        local = local_capsules(state, center, yaw, length, width)
        threshold = -(MARGIN + JITTER_Z) if state["label"] == "neutral" else MARGIN + JITTER_Z
        bound = boundary(local, threshold, length, width)
        if bound is None or not isinstance(bound, tuple):
            return {"nonempty": False, "reason": "roof boundary not bracketed", "source": name}
        values[name] = bound[0] if state["label"] == "neutral" else bound[1]
    low = max(value for name, value in values.items() if states[name]["label"] == "d055")
    high = min(value for name, value in values.items() if states[name]["label"] == "neutral")
    return {
        "lower_m": low,
        "upper_m": high,
        "width_m": high - low,
        "nonempty": high > low,
        "boundaries_m": values,
    }


def jitter_audit(states, candidate, height):
    rows = []
    for dx, dy, dz, dyaw in product(
        (-0.02, 0.0, 0.02), (-0.02, 0.0, 0.02), (-JITTER_Z, 0.0, JITTER_Z), (-0.02, 0.0, 0.02)
    ):
        center = np.asarray(candidate["center_xy_m"]) + [dx, dy]
        values = {}
        for name, state in states.items():
            local = local_capsules(
                state,
                center,
                candidate["yaw_rad"] + dyaw,
                candidate["length_m"],
                candidate["width_m"],
            )
            values[name] = clearance(
                local, height + dz, candidate["length_m"], candidate["width_m"]
            )
        target = min(value for name, value in values.items() if states[name]["label"] == "d055")
        weaker = max(value for name, value in values.items() if states[name]["label"] == "neutral")
        rows.append(
            {
                "offset_xyzh": [dx, dy, dz, dyaw],
                "target_min_clearance_m": target,
                "weaker_worst_clearance_m": weaker,
                "passed": target >= MARGIN and weaker <= -MARGIN,
                "per_source_clearance_m": values,
            }
        )
    return {
        "tested_placements": len(rows),
        "passed_placements": sum(row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeatability", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = json.loads(args.repeatability.read_text())
    if not result["analysis_complete"] or not result["summary"]["strict_development_repeatability"]:
        raise ValueError("requires a complete strictly repeated development pair")
    manifest_path = checked(Path(result["manifest"]["path"]), result["manifest"]["sha256"])
    manifest = json.loads(manifest_path.read_text())
    states, sources = {}, {
        "repeatability": artifact(args.repeatability),
        "manifest": artifact(manifest_path),
    }
    route = None
    for cell in manifest["cells"]:
        label = cell["label"]
        if f"reference_{label}" in states:
            continue
        ref = cell["reference"]
        csv = checked(Path(ref["path"]), ref["sha256"])
        provenance = json.loads(
            checked(
                Path(cell["motion"]["conversion_provenance"]),
                cell["motion"]["conversion_provenance_sha256"],
            ).read_text()
        )
        if provenance["scene_start_xyz"] != [0.0, 0.0, 0.0] or provenance.get("scene_yaw", 0) != 0:
            raise ValueError("teacher requires the registered world origin and yaw")
        qpos = np.loadtxt(csv, delimiter=",")
        qpos[:, :2] -= qpos[0, :2]
        payload = payload_from_reference(qpos, fps=30)
        states[f"reference_{label}"] = {**capsules(payload), "label": label}
        sources[f"reference_{label}"] = artifact(csv)
        if label == "neutral":
            route = qpos[:, :2]
    for row in result["rows"]:
        path = checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        with path.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        states[row["cell_id"]] = {**capsules(payload), "label": row["label"]}
        sources[row["cell_id"]] = artifact(path)
    if any(
        np.max(np.maximum(s["starts"][..., 2], s["ends"][..., 2]) + s["radii"]) >= 3.0
        for s in states.values()
    ):
        raise ValueError("roof top must exceed every capsule")
    chord = route[-1] - route[0]
    yaw = math.atan2(chord[1], chord[0])
    arclength = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(route, axis=0), axis=1))]
    arclength /= arclength[-1]
    candidates = []
    for station, length in product(np.linspace(0.35, 0.65, 9), (0.10, 0.20, 0.30)):
        center = np.array([np.interp(station, arclength, route[:, axis]) for axis in range(2)])
        candidate = {
            "candidate_id": f"beam_{len(candidates):03d}",
            "route_progress": float(station),
            "center_xy_m": center.tolist(),
            "yaw_rad": yaw,
            "length_m": length,
            "width_m": 1.2,
        }
        candidate["interval"] = interval(states, center, yaw, length, 1.2)
        candidates.append(candidate)
    ordered = sorted(
        (c for c in candidates if c["interval"]["nonempty"]),
        key=lambda c: (-c["interval"]["width_m"], c["length_m"], c["candidate_id"]),
    )
    proposals = []
    if ordered:
        candidate = ordered[0]
        for quantile in (0.25, 0.5, 0.75):
            height = candidate["interval"]["lower_m"] + quantile * candidate["interval"]["width_m"]
            proposals.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "quantile": quantile,
                    "beam_underside_m": height,
                    "thickness_m": THICKNESS,
                    "jitter_audit": jitter_audit(states, candidate, height),
                    "training_eligible": False,
                }
            )
    result = {
        "schema_version": "motion2scene_finite_beam_teacher_v1",
        "analysis_role": "analytic_geometry_diagnostic",
        "sources": sources,
        "driver": artifact(Path(__file__)),
        "geometry_implementation": artifact(
            ROOT / "gear_sonic/dataset_generation/capsule_box_exact.py"
        ),
        "capsule_model_implementation": artifact(
            ROOT / "gear_sonic/dataset_generation/swept_volume.py"
        ),
        "source_count": len(states),
        "margin_m": MARGIN,
        "vertical_reserve_m": JITTER_Z,
        "attempted_intervals": len(candidates),
        "nonempty_intervals": len(ordered),
        "candidates": candidates,
        "selection_rule": "widest common roof interval; shorter beam then ID break ties; three height quantiles",
        "selected_candidate": ordered[0] if ordered else None,
        "proposals": proposals,
        "geometry_robust_proposals": sum(p["jitter_audit"]["all_passed"] for p in proposals),
        "qualification": {
            "q4_admitted_ladders": 0,
            "obstacle_present_physics_runs": 0,
            "training_eligible": False,
        },
        "limits": [
            "exact static capsule primitive distance; collision-mesh agreement not established",
            "positive clearance is capped at 20 mm after certified broad-phase exclusion",
            "sampled trajectory frames and 81 discrete jitter placements; no continuous certificate",
            "fixed world frame shared across sources; no achieved-path recentering",
            "finite beam checked after roof proposal; no collision or task-success labels inferred",
        ],
    }
    write_new(args.output / "result.json", result)
    if ordered:
        chosen = next(
            (p for p in proposals if p["quantile"] == 0.5 and p["jitter_audit"]["all_passed"]), None
        )
        if chosen:
            c = ordered[0]
            scene = args.output / "diagnostic_beam.usda"
            scene.write_text(
                '#usda 1.0\n(\n    defaultPrim = "World"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n'
                'def Xform "World" {\n    def Cube "diagnostic_beam" {\n        double size = 1\n'
                f'        double3 xformOp:translate = ({c["center_xy_m"][0]}, '
                f'{c["center_xy_m"][1]}, {chosen["beam_underside_m"] + THICKNESS / 2})\n'
                f'        double xformOp:rotateZ = {math.degrees(c["yaw_rad"])}\n'
                f'        double3 xformOp:scale = ({c["length_m"]}, {c["width_m"]}, {THICKNESS})\n'
                '        uniform token[] xformOpOrder = ["xformOp:translate", '
                '"xformOp:rotateZ", "xformOp:scale"]\n'
                "        color3f[] primvars:displayColor = [(0.65, 0.37, 0.15)]\n    }\n}\n"
            )
            write_new(
                args.output / "scene_manifest.json",
                {
                    "scene": artifact(scene),
                    "teacher_result": artifact(args.output / "result.json"),
                    "status": "visual_diagnostic_only_no_physics_schema",
                    "training_eligible": False,
                },
            )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "attempted_intervals",
                    "nonempty_intervals",
                    "geometry_robust_proposals",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
