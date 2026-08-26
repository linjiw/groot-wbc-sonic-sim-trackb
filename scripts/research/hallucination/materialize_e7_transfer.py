#!/usr/bin/env python3
"""Materialize and Tier-2 certify the six E7 source/archetype transfers."""

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

ARCHETYPES = ("door_lintel", "ibeam")
SPEC_PATHS = {
    "lfh_086_crouch": REPO_ROOT
    / "specs/hallucination/cs_lfh_086_crouch_cal3_d0100.json",
    "lfh_089_crouch": REPO_ROOT / "specs/hallucination/cs_lfh_089_crouch_cal3.json",
    "lfh_090_crouch": REPO_ROOT / "specs/hallucination/cs_lfh_090_crouch_cal3.json",
}
E6_CPU = REPO_ROOT / "docs/hallucination/e6_crouch_cpu.json"
E6C_CPU = REPO_ROOT / "docs/hallucination/e6c_exposure_cpu.json"


def motion_sources() -> dict[str, dict[str, str]]:
    rows = json.loads(E6_CPU.read_text())["selected"]
    rows.extend(json.loads(E6C_CPU.read_text())["selected"])
    return {row["source_pair_id"]: row["reference_motions"] for row in rows}


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
        default=REPO_ROOT / "docs/hallucination/e7_transfer_cpu.json",
    )
    parser.add_argument("--seed-base", type=int, default=32000)
    args = parser.parse_args()

    references = motion_sources()
    variants = []
    for source_index, (source_id, spec_path) in enumerate(SPEC_PATHS.items()):
        spec = ConstraintSpec.load(spec_path)
        if spec.source_family_id != source_id:
            raise SystemExit(f"{spec_path}: source identity mismatch")
        if source_id == "lfh_086_crouch" and abs(spec.face_along_route_m - 0.10) > 1e-9:
            raise SystemExit("E7 must preserve source 086's verified 0.10 m exposure")
        for archetype_index, archetype in enumerate(ARCHETYPES):
            seed = args.seed_base + 100 * source_index + archetype_index + 1
            result = instantiate(spec, archetype, seed, args.out_package)
            keepout = validate_pair(spec, result.easy.path, result.hard.path)
            keepout_path = args.out_package / (
                f"{spec.spec_id}__{archetype}__s{seed:08d}.keepout.json"
            )
            write_report(keepout, keepout_path)
            if not keepout["ok"]:
                raise SystemExit(
                    f"{source_id}/{archetype}: Tier-2 refused "
                    f"{keepout['refusal_reasons']}"
                )

            pair_manifest = json.loads(result.manifest_path.read_text())
            pair_manifest["cpu_certificate"] = {
                "keepout_report": str(keepout_path.relative_to(REPO_ROOT)),
                "binding_geometry_signs": keepout["binding_geometry_signs"],
                "four_sign_pattern_matches": True,
                "source_station_and_exposure_preserved": True,
            }
            result.manifest_path.write_text(
                json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n"
            )
            variants.append(
                {
                    "variant_id": f"{source_id}__{archetype}",
                    "source_pair_id": source_id,
                    "source_variant_id": spec.source_variant_id,
                    "archetype_id": archetype,
                    "seed": seed,
                    "route_axis": spec.route_axis,
                    "station_xy_m": list(spec.binding_station_xy_m),
                    "face_along_route_m": spec.face_along_route_m,
                    "face_across_route_m": spec.face_across_route_m,
                    "easy_coordinate_m": spec.easy_coordinate_m,
                    "hard_coordinate_m": spec.hard_coordinate_m,
                    "spec": str(spec_path.relative_to(REPO_ROOT)),
                    "spec_sha256": sha256_file(spec_path),
                    "pair_manifest": str(result.manifest_path.relative_to(REPO_ROOT)),
                    "pair_manifest_sha256": sha256_file(result.manifest_path),
                    "keepout": str(keepout_path.relative_to(REPO_ROOT)),
                    "keepout_sha256": sha256_file(keepout_path),
                    "scenes": {
                        "easy": str(result.easy.path.relative_to(REPO_ROOT)),
                        "hard": str(result.hard.path.relative_to(REPO_ROOT)),
                    },
                    "scene_sha256": {
                        "easy": result.easy.sha256,
                        "hard": result.hard.sha256,
                    },
                    "binding_geometry_clearance_mm": {
                        cell: {
                            motion: evidence["clearance_mm"]
                            for motion, evidence in motions.items()
                        }
                        for cell, motions in keepout["binding_geometry_pattern"].items()
                    },
                    "reference_motions": references[source_id],
                }
            )

    report = {
        "schema_version": "lfh_e7_transfer_cpu_v1",
        "physics_executed": False,
        "purpose": (
            "Transfer verified source-specific critical atoms across deterministic archetypes"
        ),
        "sources": len(SPEC_PATHS),
        "archetypes": list(ARCHETYPES),
        "variants": len(variants),
        "all_cpu_preflights_pass": len(variants) == len(SPEC_PATHS) * len(ARCHETYPES),
        "selection_policy": (
            "preserve each source's verified route station, finite exposure, and coordinates"
        ),
        "variants_detail": variants,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: materialized and certified {len(variants)} E7 variants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
