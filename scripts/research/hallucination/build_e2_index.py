#!/usr/bin/env python3
"""Write E2 episodes and variants using the repository's additive LFH index contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.sweepcf_coverage import semantic_keypoint  # noqa: E402
from scripts.research.build_dataset_index import EPISODE_COLUMNS  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-record", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--episodes-out", required=True, type=Path)
    parser.add_argument("--families-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run_record.read_text())
    report = json.loads(args.report.read_text())
    report_cells = {
        cell["cell_id"]: (variant, cell)
        for variant in report["variant_results"]
        for cell in variant["cells"]
    }
    pair_manifests = {
        archetype: json.loads((REPO_ROOT / certificate["pair_manifest"]).read_text())
        for archetype, certificate in manifest["cpu_certificates"].items()
    }
    episode_rows = []

    for cell in manifest["cells"]:
        cell_id = cell["cell_id"]
        archetype = cell["archetype_id"]
        variant, measured = report_cells[cell_id]
        scientific = run["cells"][cell_id]["scientific"]
        diagnostics = scientific["diagnostics"]
        contacts = diagnostics["contact_decomposition"]
        role = cell["cell_role"]
        difficulty = role.rsplit("_", 1)[1]
        pair_scene = pair_manifests[archetype]["scenes"][difficulty]
        variant_id = f"cs_duck_003__{archetype}__s00000017"
        contact = measured["contact"]
        video = Path(cell["output"]) / "renders/000000.mp4"
        row = {column: "" for column in EPISODE_COLUMNS}
        row.update(
            {
                "episode_id": f"{variant_id}/{role}",
                "family_id": variant_id,
                "cell_role": role,
                "motion_role": "adapted" if role.startswith("adapted") else "nominal",
                "operator": "local_crouch",
                "behaviour_class": "crouch",
                "scene_id": cell["scene"]["scene_id"],
                "obstacle_underside_m": round(pair_scene["binding_face_realized_m"], 7),
                "route_reaches_obstacle": int(measured["route_reaches_binding"]),
                "frames_inside_obstacle": measured["frames_inside_binding_footprint"],
                "outcome": scientific["outcome"],
                "rejection_reasons": ";".join(scientific["rejection_reasons"]),
                "min_clearance_mm": round(measured["binding_clearance_mm"], 3),
                "clearance_frame": measured["binding_bottleneck_frame"],
                "closest_body": measured["binding_body"],
                "external_force_n": round(contacts["max_external_contact_force_n"], 1),
                "overhead_force_n": round(contacts["max_overhead_contact_force_n"], 1),
                "lateral_force_n": round(contacts["max_lateral_contact_force_n"], 1),
                "self_force_n": round(contacts["max_self_contact_force_n"], 1),
                "contact_body": ";".join(contact["external_bodies"]),
                "contact_frame": "" if contact["first_frame"] is None else contact["first_frame"],
                "drift_rate_mps": round(diagnostics["drift_rate_mps"], 4),
                "frames": scientific["frames"],
                "trajectory_path": scientific["artifacts"]["trajectory"],
                "scene_path": str(REPO_ROOT / cell["scene"]["path"]),
                "video_ego": str(video),
                "n_videos": int(video.exists()),
                "source_family_id": cell["source_family_id"],
                "variant_id": variant_id,
                "source_identity_status": "manifest",
                "constraint_axis": "overhead",
                "constraint_coordinate_m": round(pair_scene["binding_face_realized_m"], 7),
                "binding_keypoint": semantic_keypoint(measured["binding_body"]),
                "binding_keypoint_margin_mm": round(measured["binding_clearance_mm"], 3),
                "edit_behaviour_class": "crouch",
                "binding_station_offset_mm": round(pair_scene["binding_station_offset_mm"], 6),
            }
        )
        episode_rows.append(row)

    family_rows = []
    for variant in report["variant_results"]:
        archetype = variant["archetype_id"]
        variant_id = f"cs_duck_003__{archetype}__s00000017"
        by_role = {cell["cell_role"]: cell["outcome"] for cell in variant["cells"]}
        family_rows.append(
            {
                "family_id": variant_id,
                "source_family_id": "cf_005_056",
                "variant_id": variant_id,
                "source_identity_status": "manifest",
                "status": "verified" if variant["verified"] else "not_verified",
                "evidence_valid": int(variant["verified"]),
                "evidence_reasons": "" if variant["verified"] else "pre_contact_reference_drift",
                "cells": len(variant["cells"]),
                "nominal_easy": by_role["nominal_easy"],
                "nominal_hard": by_role["nominal_hard"],
                "adapted_easy": by_role["adapted_easy"],
                "adapted_hard": by_role["adapted_hard"],
                "operators": "local_crouch",
                "scenes": ";".join(
                    sorted(
                        {
                            cell["scene"]["scene_id"]
                            for cell in manifest["cells"]
                            if cell["archetype_id"] == archetype
                        }
                    )
                ),
                "constraint_axis": "overhead",
            }
        )

    args.episodes_out.parent.mkdir(parents=True, exist_ok=True)
    with args.episodes_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPISODE_COLUMNS)
        writer.writeheader()
        writer.writerows(episode_rows)
    with args.families_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(family_rows[0]))
        writer.writeheader()
        writer.writerows(family_rows)
    print(f"wrote {len(episode_rows)} E2 episodes and {len(family_rows)} E2 variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
