#!/usr/bin/env python3
"""Build the four-cell E3 lateral-gap physics pilot from its CPU certificate."""

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

CPU_REPORT = REPO_ROOT / "docs/hallucination/e3_lateral_cpu.json"
CANDIDATES = DATA_ROOT / "lfh_arm_tuck_strong_calibration/candidates.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E3_LATERAL_PILOT_PROPOSED.json",
    )
    args = parser.parse_args()

    report = json.loads(CPU_REPORT.read_text())
    if report["funnel"]["selected"] != 1:
        raise SystemExit("E3 pilot requires exactly one CPU-selected lateral pair")
    selected = report["selected"][0]
    if selected["coverage_target"] is None:
        raise SystemExit("E3 pilot must fill an explicit DCS target")
    keepout_path = REPO_ROOT / selected["keepout"]
    keepout = json.loads(keepout_path.read_text())
    if not keepout["ok"] or keepout["binding_geometry_signs"] != {
        "easy": {"orig": "+", "edit": "+"},
        "hard": {"orig": "-", "edit": "+"},
    }:
        raise SystemExit("E3 pilot CPU four-sign certificate is not green")

    pair = next(
        value
        for value in json.loads(CANDIDATES.read_text())["pairs"]
        if value["pair_id"] == selected["motion_id"]
    )
    stem = "cs_lfh_084_arm_tuck_right_cal2__pinch_panels__s00031584"
    roles = {
        "nominal_easy": ("easy", pair["nominal_artifacts"]["motion"], "accepted"),
        "nominal_hard": ("hard", pair["nominal_artifacts"]["motion"], "rejected"),
        "adapted_easy": ("easy", pair["adapted_artifacts"]["motion"], "accepted"),
        "adapted_hard": ("hard", pair["adapted_artifacts"]["motion"], "accepted"),
    }
    cells = []
    for role, (difficulty, motion_path, expected) in roles.items():
        scene_id = f"{stem}__{difficulty}"
        cells.append(
            {
                "cell_id": f"{stem}__{role}",
                "source_family_id": selected["motion_id"],
                "source_variant_id": "cal2_alpha_2p5",
                "archetype_id": "pinch_panels",
                "cell_role": role,
                "runtime_seed": 31584,
                "hydra_overrides": ["++seed=31584"],
                "scene": _scene(scene_id),
                "motion": _motion(Path(motion_path)),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expected_source_outcome": expected,
                "output": str(DATA_ROOT / "hallucination/e3_lateral_pilot" / role),
            }
        )

    _write(
        args.out,
        "E3-lateral-gap-pilot",
        "Test the first temporally active, all-capsule, coverage-targeted arm-tuck scene.",
        cells,
        [
            "stop on infrastructure failure and preserve every completed capture",
            "continue after completed scientific rejections so all four cells are observed",
            "refuse the family unless all four outcomes reproduce accepted/rejected/accepted/accepted",
            "require nominal-hard contact to be uniquely attributable to a binding panel",
            "refuse secondary or unattributed contact in every cell",
        ],
        extra={
            "cpu_certificate": {
                "report": str(CPU_REPORT.relative_to(REPO_ROOT)),
                "report_sha256": sha256_file(CPU_REPORT),
                "spec": selected["spec"],
                "spec_sha256": selected["spec_sha256"],
                "pair_manifest": selected["pair_manifest"],
                "pair_manifest_sha256": selected["pair_manifest_sha256"],
                "keepout": selected["keepout"],
                "keepout_sha256": selected["keepout_sha256"],
                "binding_geometry_clearance_mm": selected["binding_geometry_clearance_mm"],
                "raw_window_mm": selected["raw_window_mm"],
                "target_certified_window_mm": selected["certified_window_mm"],
                "coverage_target": selected["coverage_target"],
            },
            "scope": {
                "generated_family_enters_claim5": False,
                "physics_cells": 4,
                "independent_motion_denominator": 1,
            },
        },
    )
    print(f"wrote {len(cells)}-cell E3 lateral pilot -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
