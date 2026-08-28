#!/usr/bin/env python3
"""Recompute E7 hard support from accepted, zero-contact easy trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from gear_sonic.dataset_generation.hallucination.keypoints import (  # noqa: E402
    extract_keypoints,
)
from gear_sonic.dataset_generation.hallucination.reach import (  # noqa: E402
    overhead_face_reach,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

ENGINEERING_MARGIN_M = 0.018044
MINIMUM_WIDTH_M = 0.020


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory segment")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument(
        "--cpu-report",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e7_transfer_cpu.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_record.read_text())
    if run["status"] != "completed" or run["manifest_sha256"] != sha256_file(args.manifest):
        raise SystemExit("completed E7a run does not match its immutable manifest")
    cpu = json.loads(args.cpu_report.read_text())
    expected_variants = len(cpu["variants_detail"])
    expected_sources = len({row["source_pair_id"] for row in cpu["variants_detail"]})
    variants = []
    for variant in cpu["variants_detail"]:
        prefix = Path(variant["scenes"]["easy"]).stem.removesuffix("__easy")
        cells = {
            role: run["cells"][f"{prefix}__{role}"] for role in ("nominal_easy", "adapted_easy")
        }
        reasons = []
        tracks = {}
        for role, cell in cells.items():
            scientific = cell.get("scientific", {})
            if cell.get("status") != "completed" or scientific.get("outcome") != "accepted":
                reasons.append(f"{role}_not_accepted")
                continue
            external = scientific["diagnostics"]["contact_decomposition"][
                "max_external_contact_force_n"
            ]
            if float(external) != 0.0:
                reasons.append(f"{role}_external_contact")
                continue
            tracks[role] = extract_keypoints(load(Path(scientific["artifacts"]["trajectory"])))

        nominal_reach = None
        adapted_reach = None
        lower = None
        upper = None
        if not reasons:
            station = tuple(float(value) for value in variant["station_xy_m"])
            args_common = (
                station,
                variant["route_axis"],
                float(variant["face_along_route_m"]),
                float(variant["face_across_route_m"]),
            )
            nominal_reach = overhead_face_reach(tracks["nominal_easy"], *args_common)
            adapted_reach = overhead_face_reach(tracks["adapted_easy"], *args_common)
            if nominal_reach.binding_keypoint != "head_torso":
                reasons.append("nominal_binding_anatomy_changed")
            if adapted_reach.binding_keypoint != "head_torso":
                reasons.append("adapted_binding_anatomy_changed")
            lower = adapted_reach.reach_m + ENGINEERING_MARGIN_M
            upper = nominal_reach.reach_m - ENGINEERING_MARGIN_M
            if upper - lower < MINIMUM_WIDTH_M:
                reasons.append("scene_conditioned_engineering_window_below_min")
            hard = float(variant["hard_coordinate_m"])
            if hard < lower or hard > upper:
                reasons.append("pinned_hard_coordinate_outside_scene_conditioned_window")

        variants.append(
            {
                "variant_id": variant["variant_id"],
                "source_pair_id": variant["source_pair_id"],
                "archetype_id": variant["archetype_id"],
                "hard_phase_eligible": not reasons,
                "refusal_reasons": reasons,
                "pinned_hard_coordinate_m": variant["hard_coordinate_m"],
                "scene_conditioned_nominal_reach_m": (
                    None if nominal_reach is None else nominal_reach.reach_m
                ),
                "scene_conditioned_adapted_reach_m": (
                    None if adapted_reach is None else adapted_reach.reach_m
                ),
                "scene_conditioned_support_lower_m": lower,
                "scene_conditioned_support_upper_m": upper,
                "scene_conditioned_engineering_width_mm": (
                    None if lower is None or upper is None else 1000 * (upper - lower)
                ),
            }
        )

    report = {
        "schema_version": "lfh_e7_context_recertification_v1",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "run_record": str(args.run_record),
        "run_record_sha256": sha256_file(args.run_record),
        "engineering_margin_mm_each_side": 1000 * ENGINEERING_MARGIN_M,
        "engineering_margin_confidence_level": None,
        "eligible_variants": sum(row["hard_phase_eligible"] for row in variants),
        "eligible_sources": len(
            {row["source_pair_id"] for row in variants if row["hard_phase_eligible"]}
        ),
        "variants": variants,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: hard eligible={report['eligible_variants']}/{expected_variants} across "
        f"{report['eligible_sources']}/{expected_sources} sources"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
