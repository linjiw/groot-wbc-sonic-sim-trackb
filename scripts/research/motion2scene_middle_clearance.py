#!/usr/bin/env python3
"""Audit the intermediate motion against the already frozen binary beam proposals."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import pickle

from motion2scene_beam_teacher import capsules, clearance, local_capsules
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

from gear_sonic.dataset_generation.reference_payload import payload_from_reference
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    teacher = json.loads(args.teacher.read_text())
    checked(Path(teacher["driver"]["path"]), teacher["driver"]["sha256"])
    checked(
        Path(teacher["geometry_implementation"]["path"]),
        teacher["geometry_implementation"]["sha256"],
    )
    extension = json.loads(args.extension.read_text())
    if not extension["analysis_complete"]:
        raise ValueError("requires the completed extension")
    manifest = json.loads(
        checked(Path(extension["manifest"]["path"]), extension["manifest"]["sha256"]).read_text()
    )
    states, sources = {}, {"teacher": artifact(args.teacher), "extension": artifact(args.extension)}
    ref = manifest["cells"][0]["reference"]
    csv = checked(Path(ref["path"]), ref["sha256"])
    qpos = np.loadtxt(csv, delimiter=",")
    qpos[:, :2] -= qpos[0, :2]
    states["reference_d040"] = capsules(payload_from_reference(qpos, fps=30))
    sources["reference_d040"] = artifact(csv)
    for row in extension["rows"]:
        path = checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
        with path.open("rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        states[row["cell_id"]] = capsules(payload)
        sources[row["cell_id"]] = artifact(path)
    candidate = teacher["selected_candidate"]
    if candidate is None:
        raise ValueError("no frozen binary beam proposal")
    rows = []
    for proposal in teacher["proposals"]:
        per_source = {}
        for name, state in states.items():
            values = []
            for dx, dy, dz, dyaw in product(
                (-0.02, 0.0, 0.02), (-0.02, 0.0, 0.02), (-0.01, 0.0, 0.01), (-0.02, 0.0, 0.02)
            ):
                local = local_capsules(
                    state,
                    np.asarray(candidate["center_xy_m"]) + [dx, dy],
                    candidate["yaw_rad"] + dyaw,
                    candidate["length_m"],
                    candidate["width_m"],
                )
                values.append(
                    clearance(
                        local,
                        proposal["beam_underside_m"] + dz,
                        candidate["length_m"],
                        candidate["width_m"],
                    )
                )
            per_source[name] = {
                "minimum_m": min(values),
                "maximum_m": max(values),
                "target_clear_placements": sum(value >= 0.01 for value in values),
                "weaker_strike_placements": sum(value <= -0.01 for value in values),
                "positive_separation_placements": sum(value > 0 for value in values),
                "clearance_values_m": values,
            }
        rows.append(
            {
                "quantile": proposal["quantile"],
                "beam_underside_m": proposal["beam_underside_m"],
                "all_sources_strike_with_margin": all(
                    source["maximum_m"] <= -0.01 for source in per_source.values()
                ),
                "all_sources_clear_with_margin": all(
                    source["minimum_m"] >= 0.01 for source in per_source.values()
                ),
                "per_source": per_source,
            }
        )
    result = {
        "schema_version": "motion2scene_intermediate_beam_audit_v1",
        "sources": sources,
        "driver": artifact(Path(__file__)),
        "teacher_implementation": teacher["driver"],
        "geometry_implementation": teacher["geometry_implementation"],
        "reference_payload_implementation": artifact(
            ROOT / "gear_sonic/dataset_generation/reference_payload.py"
        ),
        "candidate_id": candidate["candidate_id"],
        "jitter_points_per_source": 81,
        "jitter_order": "product(dx,dy,dz,dyaw), each ascending; xy=+-20mm,z=+-10mm,yaw=+-0.02rad",
        "rows": rows,
        "training_eligible": False,
        "limits": [
            "frozen binary candidate, no new optimization; discrete static capsule geometry only",
            "intermediate failed strict behavior admission; geometry cannot replace qualification",
        ],
    }
    write_new(args.out, result)
    print(
        json.dumps(
            [
                {
                    key: row[key]
                    for key in (
                        "quantile",
                        "all_sources_strike_with_margin",
                        "all_sources_clear_with_margin",
                    )
                }
                for row in rows
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
