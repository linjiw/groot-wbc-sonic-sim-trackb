#!/usr/bin/env python3
"""Build the staged single-source E6c finite-exposure ablation manifests."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e6c_exposure_cpu.json"
RECERTIFICATION = REPO_ROOT / "docs/hallucination/e6c_context_recertification.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("easy", "hard"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = json.loads(CPU_REPORT.read_text())
    if not report["all_cpu_preflights_pass"] or report["selected_sources"] != 1:
        raise SystemExit("E6c requires exactly one green exposure-ablation source")
    selected = report["selected"][0]
    if selected["source_pair_id"] != "lfh_086_crouch":
        raise SystemExit("E6c is registered only for source 086")
    recert = None
    if args.phase == "hard":
        if not RECERTIFICATION.exists():
            raise SystemExit("hard phase requires completed E6c context recertification")
        recert = json.loads(RECERTIFICATION.read_text())
        if recert["eligible_sources"] != 1:
            raise SystemExit("E6c hard phase is not eligible")

    difficulty = args.phase
    scene_id = Path(selected["scenes"][difficulty]).stem
    roles = (
        (("nominal_easy", "nominal", "accepted"), ("adapted_easy", "adapted", "accepted"))
        if args.phase == "easy"
        else (
            ("nominal_hard", "nominal", "rejected"),
            ("adapted_hard", "adapted", "accepted"),
        )
    )
    cells = []
    for role, motion_role, expected in roles:
        cells.append(
            {
                "cell_id": f"{scene_id.removesuffix(f'__{difficulty}')}__{role}",
                "source_family_id": selected["source_pair_id"],
                "source_variant_id": selected["source_variant_id"],
                "archetype_id": "shelf_plank",
                "cell_role": role,
                "runtime_seed": selected["seed"],
                "hydra_overrides": [f"++seed={selected['seed']}"],
                "scene": _scene(scene_id),
                "motion": _motion(Path(selected["reference_motions"][motion_role])),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expected_source_outcome": expected,
                "output": str(
                    DATA_ROOT
                    / "hallucination/e6c_crouch_exposure"
                    / selected["source_pair_id"]
                    / role
                ),
            }
        )

    default_name = (
        "E6C_CROUCH_EXPOSURE_EASY_PROPOSED.json"
        if args.phase == "easy"
        else "E6C_CROUCH_EXPOSURE_HARD_PROPOSED.json"
    )
    out = args.out or REPO_ROOT / "docs/hallucination/manifests" / default_name
    stops = [
        "stop on infrastructure failure and preserve completed captures",
        "refuse secondary or unattributed authored-geometry contact",
    ]
    if args.phase == "easy":
        stops.extend(
            [
                "authorize no hard cell from this manifest",
                "stop the source if either easy cell rejects or has external contact",
                "require the pinned hard coordinate inside the scene-conditioned symmetric interval",
            ]
        )
    else:
        stops.extend(
            [
                "require nominal-hard binding-head/torso rejection before drift",
                "require adapted-hard acceptance with zero external contact",
            ]
        )
    extra = {
        "cpu_certificate": {
            "report": str(CPU_REPORT.relative_to(REPO_ROOT)),
            "report_sha256": sha256_file(CPU_REPORT),
            "source": {
                key: selected[key]
                for key in (
                    "source_pair_id",
                    "spec",
                    "spec_sha256",
                    "pair_manifest",
                    "pair_manifest_sha256",
                    "keepout",
                    "keepout_sha256",
                    "engineering_window_mm",
                    "easy_coordinate_m",
                    "hard_coordinate_m",
                    "coverage_target",
                )
            },
        },
        "ablation": {
            "changed_variable": "face_along_route_m",
            "failed_e6_value_m": 0.30,
            "e6c_value_m": 0.10,
            "held_fixed": [
                "motion_pair",
                "operator",
                "route_progress",
                "archetype",
                "DCS target",
                "engineering-margin rule",
            ],
        },
        "scope": {
            "generated_family_enters_claim5": False,
            "phase": args.phase,
            "physics_cells": 2,
            "independent_motion_sources": 1,
        },
    }
    if recert is not None:
        extra["context_recertification"] = {
            "path": str(RECERTIFICATION.relative_to(REPO_ROOT)),
            "sha256": sha256_file(RECERTIFICATION),
        }
    _write(
        out,
        f"E6c-crouch-exposure-{args.phase}",
        "Test whether reducing finite shelf exposure rescues source 086 context trackability.",
        cells,
        stops,
        extra=extra,
    )
    print(f"wrote 2-cell E6c {args.phase} proposal -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
