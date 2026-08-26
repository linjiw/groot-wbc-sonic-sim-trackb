#!/usr/bin/env python3
"""Render a source-balanced critical-support census from executed CAL3 reach trials."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.critical_support import (  # noqa: E402
    CriticalSupport,
    source_balanced_weights,
    widest_per_source,
)


def sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def record_id(trial: dict[str, object]) -> str:
    progress = round(1000000 * float(trial["station_progress"]))
    depth = round(1000 * float(trial["face_along_route_m"]))
    return f"{trial['pair_id']}__p{progress:06d}__d{depth:04d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engineering-margin-mm", type=float, default=18.044)
    parser.add_argument("--minimum-width-mm", type=float, default=20.0)
    parser.add_argument("--easy-offset-mm", type=float, default=50.0)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    pairs = {row["pair_id"]: row for row in calibration["pairs"]}
    margin_m = args.engineering_margin_mm / 1000
    records: list[CriticalSupport] = []
    refusal_counts = {
        "calibration_pair_not_accepted_zero_external": 0,
        "trial_geometry_or_timing_refused": 0,
        "symmetric_engineering_window_below_min": 0,
    }
    for trial in calibration["trials"]:
        pair = pairs[trial["pair_id"]]
        if not (
            pair["nominal_outcome"] == "accepted"
            and pair["adapted_outcome"] == "accepted"
            and pair["zero_external_contact"] is True
        ):
            refusal_counts["calibration_pair_not_accepted_zero_external"] += 1
            continue
        reasons = set(trial["refusal_reasons"])
        if reasons - {"coverage_target_empty"}:
            refusal_counts["trial_geometry_or_timing_refused"] += 1
            continue
        record = CriticalSupport(
            record_id=record_id(trial),
            source_pair_id=str(trial["pair_id"]),
            operator="local_crouch",
            axis_type="overhead",
            binding_keypoint=str(trial["binding_keypoint_orig"]),
            route_progress=float(trial["station_progress"]),
            face_along_route_m=float(trial["face_along_route_m"]),
            face_across_route_m=float(trial["face_across_route_m"]),
            nominal_reach_m=float(trial["orig_reach_m"]),
            adapted_reach_m=float(trial["edit_reach_m"]),
            clear_margin_m=margin_m,
            strike_margin_m=margin_m,
            context_status="accepted_screen_empty_zero_external",
        )
        if record.engineering_width_m * 1000 < args.minimum_width_mm:
            refusal_counts["symmetric_engineering_window_below_min"] += 1
            continue
        records.append(record)

    selected = widest_per_source(records)
    weights = source_balanced_weights(records)
    output = {
        "schema_version": "lfh_critical_support_v1",
        "source_calibration": str(args.calibration),
        "source_calibration_sha256": sha256_file(args.calibration),
        "interpretation": (
            "deterministic trajectory-feasible proposal support; physics outcomes remain labels"
        ),
        "engineering_margin": {
            "value_mm_each_side": args.engineering_margin_mm,
            "source": "E1a maximum observed three-run clearance range across eight overhead cells",
            "confidence_level": None,
        },
        "minimum_engineering_width_mm": args.minimum_width_mm,
        "sources": len({row.source_pair_id for row in records}),
        "records": len(records),
        "refusal_counts": refusal_counts,
        "support": [
            {**record.to_dict(), "source_balanced_weight": weights[record.record_id]}
            for record in records
        ],
        "selected": [
            {
                **record.to_dict(),
                "easy_coordinate_m": record.nominal_reach_m + args.easy_offset_mm / 1000,
                "selection_rule": "maximum symmetric engineering width per source; shorter exposure on ties",
            }
            for record in selected
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: {output['records']} support records across {output['sources']} sources; "
        f"selected={len(output['selected'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
