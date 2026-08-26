#!/usr/bin/env python3
"""Build the four-cell E10 multi-obstacle context-survival manifest."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e10_context_rich_cpu.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E10_CONTEXT_RICH_PROPOSED.json",
    )
    parser.add_argument("--cpu-report", type=Path, default=CPU_REPORT)
    parser.add_argument("--experiment", default="E10-context-rich")
    parser.add_argument("--data-subdir", default="e10_context_rich")
    args = parser.parse_args()

    cpu_report = args.cpu_report.resolve()
    report = json.loads(cpu_report.read_text())
    if not report["keepout_ok"] or min(report["keepout_min_clearance_mm"].values()) < 50.0:
        raise SystemExit("E10 requires a green context keep-out certificate")
    motions = {
        "nominal": DATA_ROOT / "lfh_crouch_calibration/lfh_086_crouch/nominal.pkl",
        "adapted": DATA_ROOT / "lfh_crouch_calibration/lfh_086_crouch/adapted.pkl",
    }
    roles = (
        ("nominal_easy", "nominal", "easy", "accepted"),
        ("adapted_easy", "adapted", "easy", "accepted"),
        ("nominal_hard", "nominal", "hard", "rejected"),
        ("adapted_hard", "adapted", "hard", "accepted"),
    )
    cells = []
    for role, motion_role, difficulty, expected in roles:
        scene_id = Path(report["scenes"][difficulty]).stem
        cells.append(
            {
                "cell_id": f"{scene_id.removesuffix('__' + difficulty)}__{role}",
                "source_family_id": "lfh_086_crouch",
                "source_variant_id": "cal3_d0100__hanging_panel_context4",
                "archetype_id": "hanging_panel_context4",
                "cell_role": role,
                "runtime_seed": report["seed"],
                "hydra_overrides": [f"++seed={report['seed']}"],
                "scene": _scene(scene_id),
                "motion": _motion(motions[motion_role]),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expected_source_outcome": expected,
                "output": str(DATA_ROOT / "hallucination" / args.data_subdir / role),
            }
        )

    _write(
        args.out,
        args.experiment,
        "Test whether four route-relative context obstacles preserve the verified one-binding LFH counterfactual.",
        cells,
        [
            "stop on infrastructure failure and preserve completed captures",
            "require the original accepted/accepted/rejected/accepted pattern",
            "refuse any contact on a context obstacle as secondary_contact",
            "do not treat context face normals as verified critical-axis support",
        ],
        extra={
            "cpu_certificate": {
                "report": str(cpu_report.relative_to(REPO_ROOT)),
                "report_sha256": sha256_file(cpu_report),
                "minimum_context_clearance_mm": min(report["keepout_min_clearance_mm"].values()),
            },
            "scene_semantics": report["scientific_semantics"],
            "scope": {
                "generated_family_enters_claim5": False,
                "physics_cells": 4,
                "binding_obstacles": 1,
                "context_obstacles": 4,
            },
        },
    )
    print(f"wrote 4-cell E10 proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
