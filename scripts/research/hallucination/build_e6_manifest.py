#!/usr/bin/env python3
"""Build staged E6 crouch manifests from the CPU certificate and context recertification."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e6_crouch_cpu.json"
RECERTIFICATION = REPO_ROOT / "docs/hallucination/e6a_context_recertification.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("easy", "hard"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = json.loads(CPU_REPORT.read_text())
    if not report["all_cpu_preflights_pass"] or report["selected_sources"] != 3:
        raise SystemExit("E6 requires three green CPU-selected sources")
    recert = None
    if args.phase == "hard":
        if not RECERTIFICATION.exists():
            raise SystemExit("hard phase requires completed E6a context recertification")
        recert = json.loads(RECERTIFICATION.read_text())
        if recert["eligible_sources"] < 2:
            raise SystemExit("E6 primary prediction is no longer attainable")

    cells = []
    selected_rows = []
    for selected in report["selected"]:
        pair_id = selected["source_pair_id"]
        if recert is not None:
            source_recert = next(
                row for row in recert["sources"] if row["source_pair_id"] == pair_id
            )
            if not source_recert["hard_phase_eligible"]:
                continue
        selected_rows.append(selected)
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
        for role, motion_role, expected in roles:
            cells.append(
                {
                    "cell_id": f"{scene_id.removesuffix(f'__{difficulty}')}__{role}",
                    "source_family_id": pair_id,
                    "source_variant_id": selected["source_variant_id"],
                    "archetype_id": "shelf_plank",
                    "cell_role": role,
                    "runtime_seed": selected["seed"],
                    "hydra_overrides": [f"++seed={selected['seed']}"],
                    "scene": _scene(scene_id),
                    "motion": _motion(Path(selected["reference_motions"][motion_role])),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expected_source_outcome": expected,
                    "output": str(DATA_ROOT / "hallucination/e6_crouch_critical" / pair_id / role),
                }
            )

    default_name = (
        "E6A_CROUCH_CONTEXT_PROPOSED.json"
        if args.phase == "easy"
        else "E6B_CROUCH_HARD_PROPOSED.json"
    )
    out = args.out or REPO_ROOT / "docs/hallucination/manifests" / default_name
    stops = [
        "stop on infrastructure failure and preserve every completed capture",
        "continue after completed scientific outcomes so the source denominator remains visible",
        "refuse any cell with secondary or unattributed authored-geometry contact",
    ]
    if args.phase == "easy":
        stops.extend(
            [
                "do not run hard cells from this manifest",
                "require both easy cells accepted with zero external contact before source recertification",
                "refuse a source if its pinned hard coordinate leaves the symmetric scene-conditioned interval",
            ]
        )
    else:
        stops.extend(
            [
                "require nominal-hard rejection to be uniquely attributable to the head/torso binding primitive",
                "require adapted-hard acceptance with no secondary contact",
                "stop when two source-level pattern failures make the >=2/3 primary prediction impossible",
            ]
        )
    extra = {
        "cpu_certificate": {
            "report": str(CPU_REPORT.relative_to(REPO_ROOT)),
            "report_sha256": sha256_file(CPU_REPORT),
            "sources": [
                {
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
                }
                for selected in selected_rows
            ],
        },
        "scope": {
            "generated_family_enters_claim5": False,
            "phase": args.phase,
            "physics_cells": len(cells),
            "independent_motion_sources": len(selected_rows),
        },
    }
    if recert is not None:
        extra["context_recertification"] = {
            "path": str(RECERTIFICATION.relative_to(REPO_ROOT)),
            "sha256": sha256_file(RECERTIFICATION),
        }
    _write(
        out,
        f"E6{'a' if args.phase == 'easy' else 'b'}-crouch-critical",
        (
            "Measure easy-scene context transfer before authorizing hard cells."
            if args.phase == "easy"
            else "Complete physics verification for context-recertified CAL3 critical sources."
        ),
        cells,
        stops,
        extra=extra,
    )
    print(f"wrote {len(cells)}-cell E6 {args.phase} proposal -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
