#!/usr/bin/env python3
"""Build the LFH CAL2 strong-command empty-room proposal."""

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

CAL1_CANDIDATES = DATA_ROOT / "lfh_arm_tuck_calibration/candidates.json"
V5_CANDIDATES = DATA_ROOT / "lfh_probe_candidates/candidates.json"
CAL1_RUN = DATA_ROOT / "hallucination/run_records/ARM_TUCK_CALIBRATION_2026-08-20.json"
V5_RUN = DATA_ROOT / "hallucination/run_records/EMPTY_ROOM_PROBES_V5_2026-08-20.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DATA_ROOT / "lfh_arm_tuck_strong_calibration/candidates.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT
        / "docs/hallucination/manifests/ARM_TUCK_STRONG_CALIBRATION_PROPOSED.json",
    )
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text())
    cal1_candidates = {
        pair["pair_id"]: pair for pair in json.loads(CAL1_CANDIDATES.read_text())["pairs"]
    }
    v5_candidates = {
        f"lfh_{pair['motion_index']:03d}_arm_tuck_{pair['side']}": pair
        for pair in json.loads(V5_CANDIDATES.read_text())["candidates"]
    }
    cal1_run = json.loads(CAL1_RUN.read_text())
    v5_run = json.loads(V5_RUN.read_text())
    reused = {}
    cells = []
    for pair_number, pair in enumerate(candidates["pairs"], start=1):
        pair_id = pair["pair_id"]
        index = int(pair["motion_index"])
        seed = 31400 + pair_number
        cal1_nominal_id = f"{pair_id}__cal_nominal"
        if pair_id in cal1_candidates and cal1_nominal_id in cal1_run["cells"]:
            old_csv = cal1_candidates[pair_id]["nominal_artifacts"]["csv"]
            baseline = cal1_run["cells"][cal1_nominal_id]["scientific"]
            source = "CAL1"
        elif pair_id in v5_candidates:
            old_csv = v5_candidates[pair_id]["artifacts"]["nominal_csv"]
            baseline = v5_run["cells"][f"{pair_id}__probe_nominal"]["scientific"]
            source = "V5"
        else:
            old_csv = None
            baseline = None
            source = "CAL2"
        if baseline is not None:
            if baseline["outcome"] != "accepted":
                raise SystemExit(f"{pair_id}: reused nominal is not accepted")
            if sha256_file(Path(old_csv)) != pair["nominal_artifacts"]["csv_sha256"]:
                raise SystemExit(f"{pair_id}: nominal reference drifted from {source}")
            reused[pair_id] = {
                "source": source,
                "reference_csv": old_csv,
                "reference_csv_sha256": sha256_file(Path(old_csv)),
                "execution_artifacts": baseline["artifacts"],
            }
        else:
            nominal_id = f"{pair_id}__cal2_nominal"
            cells.append(
                {
                    "cell_id": nominal_id,
                    "pair_id": pair_id,
                    "pair_role": "nominal",
                    "motion_index": index,
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
                        "basis": "strict CPU gate only; no empty-room execution exists",
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(
                        DATA_ROOT
                        / "hallucination/arm_tuck_strong_calibration"
                        / pair_id
                        / "nominal"
                    ),
                }
            )
        cells.append(
            {
                "cell_id": f"{pair_id}__cal2_strong",
                "pair_id": pair_id,
                "pair_role": "adapted",
                "motion_index": index,
                "side": pair["side"],
                "response_alpha": pair["alpha"],
                "cpu_reference_window_m": pair["reference_window_m"],
                "depends_on_acceptance_of": (
                    None if baseline is not None else f"{pair_id}__cal2_nominal"
                ),
                "runtime_seed": seed,
                "hydra_overrides": [f"++seed={seed}"],
                "scene": _scene("screen_empty"),
                "motion": _motion(Path(pair["adapted_artifacts"]["motion"])),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expectation": {
                    "status": "measurement_no_prediction",
                    "outcome": None,
                    "basis": (
                        "strict CPU reference gate passes, but CAL1 establishes no reliable "
                        "delivery or trackability extrapolation beyond alpha=1"
                    ),
                },
                "verdict_policy": "reference_trackability",
                "output": str(
                    DATA_ROOT / "hallucination/arm_tuck_strong_calibration" / pair_id / "strong"
                ),
            }
        )

    _write(
        args.out,
        "LFH-arm-tuck-strong-delivery-calibration-2",
        "Test whether CPU-safe 95-118 mm arm-tuck commands clear the executed scene-spend floor.",
        cells,
        [
            "skip 094/left strong if its new nominal is rejected",
            "continue after completed scientific rejections and retain the full response denominator",
            "stop on infrastructure failure and preserve every completed capture",
            "do not instantiate a lateral scene unless the corrected matched-level lower bound clears 20 mm",
        ],
        extra={
            "calibration_candidates": {
                "path": str(args.candidates),
                "sha256": sha256_file(args.candidates),
            },
            "reused_nominal_evidence": reused,
            "response_contract": {
                "alpha_semantics": candidates["alpha_semantics"],
                "matched_levels": {"left": 2.0, "right": 2.5},
                "fit": "pool commands/responses by alpha; require two motions at every level",
                "scene_spend_floor_mm": 20.0,
                "models_are_proposal_only": True,
            },
        },
    )
    print(f"wrote {len(cells)}-cell CAL2 proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
