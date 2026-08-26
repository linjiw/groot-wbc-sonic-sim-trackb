#!/usr/bin/env python3
"""Build the E12 empty-scene amplitude-ladder manifest.

Cells run per motion: the nominal first, then rungs in ascending amplitude, each depending on the
acceptance of the rung below it.  The dependency chain is the stopping rule -- a motion stops
costing rollouts at its first rejection, and the last accepted rung is its delivered amplitude.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates", type=Path, default=DATA_ROOT / "lfh_crouch_ladder/candidates.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E12_CROUCH_LADDER_PROPOSED.json",
    )
    parser.add_argument("--seed-base", type=int, default=34000)
    parser.add_argument(
        "--extra-seed",
        type=int,
        action="append",
        help="repeat a fixed seed for every cell, in addition to the per-pair seed-base seed; "
        "used by LFH-E16b to turn single-seed rungs into a seed cohort",
    )
    parser.add_argument("--pair", action="append", help="restrict to these pair ids")
    parser.add_argument(
        "--only-extra-seeds",
        action="store_true",
        help="omit the per-pair seed-base seed; use when an earlier batch already ran it and the "
        "new cells only need to top the cohort up to the required seed count",
    )
    parser.add_argument(
        "--experiment", default="LFH-E12-crouch-amplitude-ladder", help="manifest experiment name"
    )
    args = parser.parse_args()

    candidates = json.loads(args.candidates.read_text())
    cells = []
    selected = candidates["pairs"]
    if args.pair:
        wanted = set(args.pair)
        selected = [pair for pair in selected if pair["pair_id"] in wanted]
        if len(selected) != len(wanted):
            raise SystemExit(
                f"unknown pair id(s): {sorted(wanted - {p['pair_id'] for p in selected})}"
            )
    for pair_number, pair in enumerate(selected, start=1):
        pair_id = pair["pair_id"]
        base_seed = args.seed_base + pair_number
        base_seeds = [] if args.only_extra_seeds else [base_seed]
        for seed in base_seeds + list(args.extra_seed or []):
            nominal_id = f"{pair_id}__nominal__s{seed}"
            cells.append(
                {
                    "cell_id": nominal_id,
                    "pair_id": pair_id,
                    "pair_role": "nominal",
                    "motion_index": pair["motion_index"],
                    "body_mode": pair["body_mode"],
                    "commanded_drop_m": 0.0,
                    "depends_on_acceptance_of": None,
                    "runtime_seed": seed,
                    "hydra_overrides": [f"++seed={seed}"],
                    "scene": _scene("screen_empty"),
                    "motion": _motion(Path(pair["nominal_artifacts"]["motion"])),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expectation": {
                        "status": "registered_trackability_prediction",
                        "outcome": "accepted",
                        "basis": "strict reference gate passed and the clip is unedited",
                    },
                    "verdict_policy": "reference_trackability",
                    "output": str(
                        DATA_ROOT / "hallucination/crouch_ladder" / pair_id / f"nominal_s{seed}"
                    ),
                }
            )
            # Every rung depends on the *nominal* only, not on the rung below it.  Chaining the rungs
            # would save at most a handful of rollouts (the cohort is fixed-size either way) while
            # making the registered monotonicity prediction untestable: a deeper rung is never rolled
            # after a shallower one is rejected, so a non-monotone motion could not be observed.
            for rung in pair["rungs"]:
                cell_id = f"{pair_id}__{rung['label']}__s{seed}"
                cells.append(
                    {
                        "cell_id": cell_id,
                        "pair_id": pair_id,
                        "pair_role": "adapted",
                        "motion_index": pair["motion_index"],
                        "body_mode": pair["body_mode"],
                        "commanded_drop_m": rung["target_drop_m"],
                        "cpu_reference_silhouette_drop_m": rung["reference_silhouette_drop_m"],
                        "cpu_lateral_coupling_mm": rung["lateral_coupling_mm"],
                        "depends_on_acceptance_of": nominal_id,
                        "runtime_seed": seed,
                        "hydra_overrides": [f"++seed={seed}"],
                        "scene": _scene("screen_empty"),
                        "motion": _motion(Path(rung["artifacts"]["motion"])),
                        "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                        "expectation": {
                            "status": "measurement_no_cell_prediction",
                            "outcome": None,
                            "basis": (
                                "crouch delivery is motion-specific; the ladder exists because no "
                                "reference-side check licenses a per-rung outcome prediction"
                            ),
                        },
                        "verdict_policy": "reference_trackability",
                        "output": str(
                            DATA_ROOT
                            / "hallucination/crouch_ladder"
                            / pair_id
                            / f"{rung['label']}_s{seed}"
                        ),
                    }
                )

    _write(
        args.out,
        args.experiment,
        "Measure per-motion crouch delivery across an amplitude ladder, over a body-mode-diverse "
        "cohort drawn from the whole gated clip pool.",
        cells,
        [
            "skip every rung if the nominal is rejected",
            "roll every rung of an accepted nominal, including rungs above a rejected one, so "
            "the amplitude-acceptance curve is observed rather than assumed monotone",
            "continue after completed scientific rejections and retain the full denominator",
            "stop on infrastructure failure and preserve every completed capture",
            "author no geometry from any of these motions inside E12",
        ],
        extra={
            "ladder_candidates": {
                "path": str(args.candidates),
                "sha256": sha256_file(args.candidates),
            },
            "registered_predictions": (
                "docs/prediction_register.md, LFH-E12 entry, filed 2026-08-26 before spend"
            ),
            "response_contract": {
                "operator": candidates["operator"],
                "station_fraction": candidates["station_fraction"],
                "window_fraction": candidates["window_fraction"],
                "delivered_amplitude_rule": "largest accepted rung per motion",
                "scene_spend_floor_mm": 20.0,
                "models_are_proposal_only": True,
            },
        },
    )
    print(f"wrote {len(cells)}-cell E12 proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
