#!/usr/bin/env python3
"""Build staged E7 archetype-transfer manifests from CPU and context certificates."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e7_transfer_cpu.json"
RECERTIFICATION = REPO_ROOT / "docs/hallucination/e7_context_recertification.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("easy", "hard"), required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    report = json.loads(CPU_REPORT.read_text())
    if not report["all_cpu_preflights_pass"] or report["variants"] != 6:
        raise SystemExit("E7 requires six green CPU-certified variants")
    recert = None
    eligible_ids = {row["variant_id"] for row in report["variants_detail"]}
    if args.phase == "hard":
        if not RECERTIFICATION.exists():
            raise SystemExit("E7 hard phase requires context recertification")
        recert = json.loads(RECERTIFICATION.read_text())
        eligible_ids = {
            row["variant_id"] for row in recert["variants"] if row["hard_phase_eligible"]
        }
        eligible_sources = {
            row["source_pair_id"] for row in recert["variants"] if row["hard_phase_eligible"]
        }
        if len(eligible_ids) < 4 or len(eligible_sources) < 3:
            raise SystemExit("E7 registered primary is no longer attainable")

    cells = []
    selected = []
    difficulty = args.phase
    roles = (
        (("nominal_easy", "nominal", "accepted"), ("adapted_easy", "adapted", "accepted"))
        if args.phase == "easy"
        else (
            ("nominal_hard", "nominal", "rejected"),
            ("adapted_hard", "adapted", "accepted"),
        )
    )
    for variant in report["variants_detail"]:
        if variant["variant_id"] not in eligible_ids:
            continue
        selected.append(variant)
        scene_id = Path(variant["scenes"][difficulty]).stem
        prefix = scene_id.removesuffix(f"__{difficulty}")
        for role, motion_role, expected in roles:
            cells.append(
                {
                    "cell_id": f"{prefix}__{role}",
                    "source_family_id": variant["source_pair_id"],
                    "source_variant_id": (
                        f"{variant['source_variant_id']}__{variant['archetype_id']}"
                    ),
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
                        / "hallucination/e7_archetype_transfer"
                        / variant["source_pair_id"]
                        / variant["archetype_id"]
                        / role
                    ),
                }
            )

    default_name = (
        "E7A_ARCHETYPE_CONTEXT_PROPOSED.json"
        if args.phase == "easy"
        else "E7B_ARCHETYPE_HARD_PROPOSED.json"
    )
    out = args.out or REPO_ROOT / "docs/hallucination/manifests" / default_name
    stops = [
        "stop on infrastructure failure and preserve completed captures",
        "continue after completed scientific outcomes so all six proposed variants remain visible",
        "refuse secondary or unattributed authored-geometry contact",
    ]
    if args.phase == "easy":
        stops.extend(
            [
                "authorize no hard cell from this manifest",
                "require both easy cells accepted with zero external contact before variant recertification",
                "require each pinned hard coordinate inside its scene-conditioned interval",
            ]
        )
    else:
        stops.extend(
            [
                "require nominal-hard binding-head/torso rejection before drift",
                "require adapted-hard acceptance with zero external contact",
                "primary requires at least four of six full variants and one transfer per source",
            ]
        )
    extra = {
        "cpu_certificate": {
            "report": str(CPU_REPORT.relative_to(REPO_ROOT)),
            "report_sha256": sha256_file(CPU_REPORT),
            "variant_ids": [row["variant_id"] for row in selected],
        },
        "scope": {
            "generated_family_enters_claim5": False,
            "phase": args.phase,
            "physics_cells": len(cells),
            "independent_motion_sources": len({row["source_pair_id"] for row in selected}),
            "source_archetype_variants": len(selected),
        },
    }
    if recert is not None:
        extra["context_recertification"] = {
            "path": str(RECERTIFICATION.relative_to(REPO_ROOT)),
            "sha256": sha256_file(RECERTIFICATION),
        }
    _write(
        out,
        f"E7{'a' if args.phase == 'easy' else 'b'}-archetype-transfer",
        (
            "Measure context survival for six source/archetype transfers before hard physics."
            if args.phase == "easy"
            else "Complete hard physics for context-eligible E7 source/archetype transfers."
        ),
        cells,
        stops,
        extra=extra,
    )
    print(f"wrote {len(cells)}-cell E7 {args.phase} proposal -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
