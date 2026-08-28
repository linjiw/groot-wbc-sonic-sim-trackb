#!/usr/bin/env python3
"""Build staged manifests for the source-086 E7c hanging-panel replacement."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e7c_replacement_cpu.json"
RECERTIFICATION = REPO_ROOT / "docs/hallucination/e7c_context_recertification.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("easy", "hard"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = json.loads(CPU_REPORT.read_text())
    if not report["all_cpu_preflights_pass"] or report["variants"] != 1:
        raise SystemExit("E7c requires one green replacement variant")
    variant = report["variants_detail"][0]
    recert = None
    if args.phase == "hard":
        if not RECERTIFICATION.exists():
            raise SystemExit("E7c hard phase requires context recertification")
        recert = json.loads(RECERTIFICATION.read_text())
        if recert["eligible_variants"] != 1:
            raise SystemExit("E7c replacement is not hard-phase eligible")

    difficulty = args.phase
    scene_id = Path(variant["scenes"][difficulty]).stem
    prefix = scene_id.removesuffix(f"__{difficulty}")
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
                "cell_id": f"{prefix}__{role}",
                "source_family_id": variant["source_pair_id"],
                "source_variant_id": (f"{variant['source_variant_id']}__{variant['archetype_id']}"),
                "archetype_id": variant["archetype_id"],
                "cell_role": role,
                "runtime_seed": variant["seed"],
                "hydra_overrides": [f"++seed={variant['seed']}"],
                "scene": _scene(scene_id),
                "motion": _motion(Path(variant["reference_motions"][motion_role])),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expected_source_outcome": expected,
                "output": str(
                    DATA_ROOT
                    / "hallucination/e7c_replacement"
                    / variant["source_pair_id"]
                    / variant["archetype_id"]
                    / role
                ),
            }
        )

    default_name = (
        "E7C_REPLACEMENT_EASY_PROPOSED.json"
        if args.phase == "easy"
        else "E7C_REPLACEMENT_HARD_PROPOSED.json"
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
                "require both easy cells accepted with zero external contact",
                "require the pinned hard coordinate inside the scene-conditioned interval",
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
            "variant_id": variant["variant_id"],
        },
        "adaptive_followup": {
            "parent_result": "E7 5/6; source-086 door-lintel refused contact-after-drift",
            "changed_variable": "archetype_id",
            "held_fixed": report["held_fixed"],
        },
        "scope": {
            "generated_family_enters_claim5": False,
            "phase": args.phase,
            "physics_cells": 2,
            "independent_motion_sources": 1,
            "source_archetype_variants": 1,
        },
    }
    if recert is not None:
        extra["context_recertification"] = {
            "path": str(RECERTIFICATION.relative_to(REPO_ROOT)),
            "sha256": sha256_file(RECERTIFICATION),
        }
    _write(
        out,
        f"E7c-replacement-{args.phase}",
        "Test one bounded hanging-panel replacement for source 086 without changing its atom.",
        cells,
        stops,
        extra=extra,
    )
    print(f"wrote 2-cell E7c {args.phase} proposal -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
