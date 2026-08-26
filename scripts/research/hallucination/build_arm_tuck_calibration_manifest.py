#!/usr/bin/env python3
"""Build the LFH arm-tuck delivery-calibration GPU proposal."""

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

V5_MANIFEST = REPO_ROOT / "docs/hallucination/manifests/MINIMAL_EMPTY_ROOM_PROBES_APPROVED_V5.json"
V5_RUN = DATA_ROOT / "hallucination/run_records/EMPTY_ROOM_PROBES_V5_2026-08-20.json"
V5_ANCHORS = {"lfh_089_arm_tuck_left", "lfh_092_arm_tuck_right"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DATA_ROOT / "lfh_arm_tuck_calibration/candidates.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/ARM_TUCK_CALIBRATION_PROPOSED.json",
    )
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text())
    v5_run = json.loads(V5_RUN.read_text())
    if v5_run["status"] != "completed":
        raise SystemExit("the two accepted V5 anchor pairs are required")
    cells = []
    reused = {}
    for pair_number, pair in enumerate(candidates["pairs"], start=1):
        pair_id = pair["pair_id"]
        seed = 31300 + pair_number
        is_anchor = pair_id in V5_ANCHORS
        if is_anchor:
            reused[pair_id] = {
                role: v5_run["cells"][f"{pair_id}__probe_{role}"]["scientific"]["artifacts"]
                for role in ("nominal", "adapted")
            }
        else:
            nominal_id = f"{pair_id}__cal_nominal"
            cells.append(
                {
                    "cell_id": nominal_id,
                    "pair_id": pair_id,
                    "pair_role": "nominal",
                    "motion_index": pair["motion_index"],
                    "side": pair["side"],
                    "response_alpha": 0.0,
                    "depends_on_acceptance_of": None,
                    "runtime_seed": seed,
                    "hydra_overrides": [f"++seed={seed}"],
                    "scene": _scene("screen_empty"),
                    "motion": _motion(Path(pair["nominal_artifacts"]["motion"])),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expectation": {
                        "status": "measurement_no_prediction",
                        "outcome": None,
                        "basis": "strict CPU gate only; no accepted empty-room execution exists",
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(
                        DATA_ROOT / "hallucination/arm_tuck_calibration" / pair_id / "nominal"
                    ),
                }
            )
        for level in pair["levels"]:
            alpha = float(level["alpha"])
            if is_anchor and alpha == 1.0:
                continue
            tag = level["tag"]
            cells.append(
                {
                    "cell_id": f"{pair_id}__cal_{tag}",
                    "pair_id": pair_id,
                    "pair_role": "adapted",
                    "motion_index": pair["motion_index"],
                    "side": pair["side"],
                    "response_alpha": alpha,
                    "cpu_reference_window_m": level["reference_window_m"],
                    "depends_on_acceptance_of": (None if is_anchor else f"{pair_id}__cal_nominal"),
                    "runtime_seed": seed,
                    "hydra_overrides": [f"++seed={seed}"],
                    "scene": _scene("screen_empty"),
                    "motion": _motion(Path(level["artifacts"]["motion"])),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expectation": {
                        "status": "predicted" if is_anchor else "measurement_no_prediction",
                        "outcome": "accepted" if is_anchor else None,
                        "basis": (
                            "same accepted anchor motion with a strictly smaller arm-only edit"
                            if is_anchor
                            else "strict CPU gate only; calibration estimates delivery and trackability"
                        ),
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(DATA_ROOT / "hallucination/arm_tuck_calibration" / pair_id / tag),
                }
            )

    _write(
        args.out,
        "LFH-arm-tuck-delivery-calibration-1",
        "Fit executed multi-level lateral delivery before any E3 scene placement.",
        cells,
        [
            "skip a new motion's adapted levels if its nominal is rejected",
            "continue after completed scientific rejections so the response curve remains honest",
            "stop on infrastructure failure and preserve every completed capture",
            "do not instantiate lateral scenes unless an evidence-backed delivery lower bound clears 20 mm",
        ],
        extra={
            "calibration_candidates": {
                "path": str(args.candidates),
                "sha256": sha256_file(args.candidates),
            },
            "reused_v5_evidence": {
                "manifest": str(V5_MANIFEST.relative_to(REPO_ROOT)),
                "manifest_sha256": sha256_file(V5_MANIFEST),
                "run_record": str(V5_RUN),
                "run_record_sha256": sha256_file(V5_RUN),
                "pairs": reused,
            },
            "response_contract": {
                "levels": candidates["levels"],
                "minimum_fit_support": "three commanded levels and two motions per keypoint/side",
                "alignment": "normalized route progress over the reference active window",
                "physics_role": "trackability and executed response only; never a scene verdict",
            },
        },
    )
    print(f"wrote {len(cells)}-cell calibration proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
