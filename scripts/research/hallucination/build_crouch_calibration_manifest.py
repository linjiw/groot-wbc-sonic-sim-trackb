#!/usr/bin/env python3
"""Build the CAL3 crouch empty-room calibration proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from scripts.research.hallucination.build_phase2_manifests import (  # noqa: E402
    DATA_ROOT,
    _motion,
    _scene,
    _write,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DATA_ROOT / "lfh_crouch_calibration/candidates.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/CROUCH_CALIBRATION_PROPOSED.json",
    )
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text())
    cells = []
    for pair_number, pair in enumerate(candidates["pairs"], start=1):
        pair_id = pair["pair_id"]
        seed = 31600 + pair_number
        nominal_id = f"{pair_id}__cal_nominal"
        for role in ("nominal", "adapted"):
            artifacts = pair[f"{role}_artifacts"]
            cells.append(
                {
                    "cell_id": f"{pair_id}__cal_{role}",
                    "pair_id": pair_id,
                    "pair_role": role,
                    "motion_index": pair["motion_index"],
                    "response_alpha": 0.0 if role == "nominal" else pair["response_alpha"],
                    "cpu_reference_silhouette_drop_m": (
                        0.0 if role == "nominal" else pair["reference_silhouette_drop_m"]
                    ),
                    "depends_on_acceptance_of": nominal_id if role == "adapted" else None,
                    "runtime_seed": seed,
                    "hydra_overrides": [f"++seed={seed}"],
                    "scene": _scene("screen_empty"),
                    "motion": _motion(Path(artifacts["motion"])),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expectation": {
                        "status": "measurement_no_cell_prediction",
                        "outcome": None,
                        "basis": (
                            "crouch trackability is motion-specific; strict reference checks do "
                            "not license an executed outcome prediction"
                        ),
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(DATA_ROOT / "hallucination/crouch_calibration" / pair_id / role),
                }
            )

    _write(
        args.out,
        "LFH-crouch-executed-calibration-3",
        "Measure four CPU-selected local-crouch pairs in the gradeable empty room.",
        cells,
        [
            "skip an adapted cell if its matched nominal is rejected",
            "continue after completed scientific rejections and retain the full denominator",
            "stop on infrastructure failure and preserve every completed capture",
            "do not author overhead geometry until exact executed finite-face reach clears 20 mm",
        ],
        extra={
            "calibration_candidates": {
                "path": str(args.candidates),
                "sha256": sha256_file(args.candidates),
            },
            "response_contract": {
                "operator": candidates["operator"],
                "target_drop_m": candidates["target_drop_m"],
                "station_fraction": candidates["station_fraction"],
                "window_fraction": candidates["window_fraction"],
                "scene_spend_floor_mm": 20.0,
                "models_are_proposal_only": True,
                "finite_face_search": {
                    "station_grid": "time-mapped active-window start/median/end",
                    "along_route_grid_m": [0.1, 0.2, 0.3, 0.4],
                    "minimum_raw_window_mm": 20.0,
                    "require_binding_keypoint": "head_torso",
                },
            },
        },
    )
    print(f"wrote {len(cells)}-cell CAL3 proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
