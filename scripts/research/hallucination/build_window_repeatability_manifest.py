#!/usr/bin/env python3
"""Build the LFH-E16 empty-scene window repeatability manifest.

Three E12 pairs, each re-rolled at three simulator seeds for the nominal and its deepest accepted
rung. Everything except ``++seed`` is byte-identical to the E12 cells, so the measured spread is
attributable to the simulator seed alone.
"""

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

SEEDS = (36001, 36002, 36003)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder", type=Path, default=REPO_ROOT / "docs/hallucination/e12_crouch_ladder.json"
    )
    parser.add_argument(
        "--candidates", type=Path, default=DATA_ROOT / "lfh_crouch_ladder/candidates.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E16_WINDOW_REPEATABILITY_PROPOSED.json",
    )
    parser.add_argument("--pairs", type=int, default=3)
    args = parser.parse_args()

    ladder = json.loads(args.ladder.read_text())
    candidates = {
        pair["pair_id"]: pair for pair in json.loads(args.candidates.read_text())["pairs"]
    }
    eligible = [
        motion
        for motion in ladder["motions"]
        if motion["window"] and motion["nominal_outcome"] == "accepted"
    ]
    eligible.sort(key=lambda motion: -motion["window"]["raw_window_mm"])
    chosen = eligible[: args.pairs]
    if not chosen:
        raise SystemExit("no E12 pair carries an executed window")

    cells = []
    for motion in chosen:
        pair_id = motion["pair_id"]
        pair = candidates[pair_id]
        deepest = max(
            (rung for rung in motion["rungs"] if rung["outcome"] == "accepted"),
            key=lambda rung: rung["commanded_drop_mm"],
        )
        rung_artifacts = next(
            rung["artifacts"] for rung in pair["rungs"] if rung["label"] == deepest["label"]
        )
        for seed in SEEDS:
            for role, artifacts in (
                ("nominal", pair["nominal_artifacts"]),
                (deepest["label"], rung_artifacts),
            ):
                cells.append(
                    {
                        "cell_id": f"{pair_id}__{role}__s{seed}",
                        "pair_id": pair_id,
                        "pair_role": "nominal" if role == "nominal" else "adapted",
                        "rung_label": role,
                        "motion_index": pair["motion_index"],
                        "body_mode": pair["body_mode"],
                        "runtime_seed": seed,
                        "hydra_overrides": [f"++seed={seed}"],
                        "scene": _scene("screen_empty"),
                        "motion": _motion(Path(artifacts["motion"])),
                        "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                        "expectation": {
                            "status": "registered_trackability_prediction",
                            "outcome": "accepted",
                            "basis": (
                                "this exact motion accepted in E12; a rejection here is seed "
                                "sensitivity and is retained as a finding, never retried"
                            ),
                        },
                        "verdict_policy": "reference_trackability",
                        "output": str(
                            DATA_ROOT
                            / "hallucination/window_repeatability"
                            / pair_id
                            / f"{role}_s{seed}"
                        ),
                    }
                )

    _write(
        args.out,
        "LFH-E16-window-repeatability",
        "Measure the three-seed range of the executed empty-scene critical window, so the "
        "engineering margin can be derived from the quantity it is applied to.",
        cells,
        [
            "continue after completed scientific rejections and retain the full denominator",
            "stop on infrastructure failure and preserve every completed capture",
            "author no geometry in E16",
            "do not change the engineering margin as a result of this batch; E16 measures only",
        ],
        extra={
            "ladder_source": {"path": str(args.ladder), "sha256": sha256_file(args.ladder)},
            "registered_predictions": (
                "docs/prediction_register.md, LFH-E16 entry, filed 2026-08-26 before spend"
            ),
            "measurement_contract": {
                "statistic": "range of executed window across the three simulator seeds, per pair",
                "comparable_to": "E1a maximum three-run range (18.044 mm), same statistic",
                "instrument": "overhead_face_reach at the commanded station, 0.10 m x 3.0 m face",
                "only_varying_field": "++seed",
            },
        },
    )
    print(f"wrote {len(cells)}-cell E16 proposal -> {args.out}")
    for motion in chosen:
        print(f"  {motion['pair_id']}: raw window {motion['window']['raw_window_mm']:.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
