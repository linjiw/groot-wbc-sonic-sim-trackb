#!/usr/bin/env python3
"""Materialize a bounded source-086 hanging-panel replacement for refused E7 door transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    ConstraintSpec,
    sha256_file,
)
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate  # noqa: E402
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    validate_pair,
    write_report,
)

SPEC = REPO_ROOT / "specs/hallucination/cs_lfh_086_crouch_cal3_d0100.json"
E6C_CPU = REPO_ROOT / "docs/hallucination/e6c_exposure_cpu.json"
ARCHETYPE = "hanging_panel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-package",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e7",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e7c_replacement_cpu.json",
    )
    parser.add_argument("--seed", type=int, default=32301)
    args = parser.parse_args()

    spec = ConstraintSpec.load(SPEC)
    if spec.source_family_id != "lfh_086_crouch" or abs(spec.face_along_route_m - 0.10) > 1e-9:
        raise SystemExit("E7c requires the verified source-086 0.10 m exposure spec")
    result = instantiate(spec, ARCHETYPE, args.seed, args.out_package)
    keepout = validate_pair(spec, result.easy.path, result.hard.path)
    keepout_path = args.out_package / (
        f"{spec.spec_id}__{ARCHETYPE}__s{args.seed:08d}.keepout.json"
    )
    write_report(keepout, keepout_path)
    if not keepout["ok"]:
        raise SystemExit(f"E7c Tier-2 refused: {keepout['refusal_reasons']}")

    pair_manifest = json.loads(result.manifest_path.read_text())
    pair_manifest["cpu_certificate"] = {
        "keepout_report": str(keepout_path.relative_to(REPO_ROOT)),
        "binding_geometry_signs": keepout["binding_geometry_signs"],
        "four_sign_pattern_matches": True,
        "source_station_and_exposure_preserved": True,
    }
    result.manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
    references = json.loads(E6C_CPU.read_text())["selected"][0]["reference_motions"]
    variant = {
        "variant_id": f"{spec.source_family_id}__{ARCHETYPE}",
        "source_pair_id": spec.source_family_id,
        "source_variant_id": spec.source_variant_id,
        "archetype_id": ARCHETYPE,
        "seed": args.seed,
        "route_axis": spec.route_axis,
        "station_xy_m": list(spec.binding_station_xy_m),
        "face_along_route_m": spec.face_along_route_m,
        "face_across_route_m": spec.face_across_route_m,
        "easy_coordinate_m": spec.easy_coordinate_m,
        "hard_coordinate_m": spec.hard_coordinate_m,
        "spec": str(SPEC.relative_to(REPO_ROOT)),
        "spec_sha256": sha256_file(SPEC),
        "pair_manifest": str(result.manifest_path.relative_to(REPO_ROOT)),
        "pair_manifest_sha256": sha256_file(result.manifest_path),
        "keepout": str(keepout_path.relative_to(REPO_ROOT)),
        "keepout_sha256": sha256_file(keepout_path),
        "scenes": {
            "easy": str(result.easy.path.relative_to(REPO_ROOT)),
            "hard": str(result.hard.path.relative_to(REPO_ROOT)),
        },
        "scene_sha256": {"easy": result.easy.sha256, "hard": result.hard.sha256},
        "binding_geometry_clearance_mm": {
            cell: {motion: evidence["clearance_mm"] for motion, evidence in motions.items()}
            for cell, motions in keepout["binding_geometry_pattern"].items()
        },
        "reference_motions": references,
    }
    report = {
        "schema_version": "lfh_e7c_replacement_cpu_v1",
        "physics_executed": False,
        "adaptive_followup_to": "E7 refused lfh_086_crouch__door_lintel",
        "changed_variable": "archetype_id",
        "held_fixed": [
            "source",
            "motion_pair",
            "operator",
            "route_station",
            "finite_exposure",
            "easy_hard_coordinates",
            "engineering_margin_rule",
        ],
        "sources": 1,
        "archetypes": [ARCHETYPE],
        "variants": 1,
        "all_cpu_preflights_pass": True,
        "variants_detail": [variant],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("PASS: materialized source-086 hanging-panel replacement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
