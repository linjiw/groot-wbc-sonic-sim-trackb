#!/usr/bin/env python3
"""Build the one-source LFH E2 variant-transfer manifest from CPU-certified pairs."""

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

SCENE_PACKAGE = REPO_ROOT / "gear_sonic/data/assets/scenes/hallucinated_variants_v1_e1"
ARCHETYPES = ("door_lintel", "hvac_duct", "ibeam")
ROLES = {
    "nominal_easy": ("easy", "nominal_easy.pkl", "accepted"),
    "nominal_hard": ("hard", "nominal_hard.pkl", "rejected"),
    "adapted_easy": ("easy", "adapted_easy.pkl", "accepted"),
    "adapted_hard": ("hard", "adapted_hard.pkl", "accepted"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E2_VARIANT_TRANSFER_PROPOSED.json",
    )
    args = parser.parse_args()

    e1a = json.loads((REPO_ROOT / "docs/hallucination/e1a_repeatability.json").read_text())
    e1b = json.loads((REPO_ROOT / "docs/hallucination/e1b_golden.json").read_text())
    if e1a["outcome_flips"] != 0 or not e1b["golden_physics_green"]:
        raise SystemExit("E2 requires green E1a source stability and E1b binding physics")

    source = DATA_ROOT / "counterfactual/duck_003/motions"
    cells = []
    certificates = {}
    for index, archetype in enumerate(ARCHETYPES):
        seed = 31201 + index
        stem = f"cs_duck_003__{archetype}__s00000017"
        pair_manifest = SCENE_PACKAGE / f"{stem}.manifest.json"
        keepout = SCENE_PACKAGE / f"{stem}.keepout.json"
        pair = json.loads(pair_manifest.read_text())
        certificate = pair.get("cpu_certificate", {})
        if certificate.get("four_sign_pattern_matches") is not True:
            raise SystemExit(f"{archetype}: missing four-sign CPU certificate")
        certificates[archetype] = {
            "pair_manifest": str(pair_manifest.relative_to(REPO_ROOT)),
            "pair_manifest_sha256": sha256_file(pair_manifest),
            "keepout_report": str(keepout.relative_to(REPO_ROOT)),
            "keepout_report_sha256": sha256_file(keepout),
            "binding_geometry_signs": certificate["binding_geometry_signs"],
        }
        for role, (difficulty, motion_name, expected) in ROLES.items():
            scene_id = f"{stem}__{difficulty}"
            cells.append(
                {
                    "cell_id": f"{stem}__{role}",
                    "source_family_id": "cf_005_056",
                    "source_variant_id": "duck_003",
                    "archetype_id": archetype,
                    "cell_role": role,
                    "runtime_seed": seed,
                    "hydra_overrides": [f"++seed={seed}"],
                    "scene": _scene(scene_id),
                    "motion": _motion(source / motion_name),
                    "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                    "expected_source_outcome": expected,
                    "output": str(
                        DATA_ROOT / "hallucination/e2_variant_transfer" / archetype / role
                    ),
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    _write(
        args.out,
        "E2-variant-transfer-duck_003",
        "Test context invariance across three CPU-certified overhead archetypes.",
        cells,
        [
            "stop the batch on infrastructure failure and preserve partial outputs",
            "continue after completed scientific rejections so the 3-variant rate is estimable",
            "refuse a variant unless all four outcomes match and hard/nominal contact is uniquely binding",
            "classify ambiguous attribution as unattributed and non-binding contact as secondary_contact",
            "report clearance drift against E1a; do not turn millimetre drift into a verdict gate",
        ],
        extra={
            "source_denominator": {
                "eligible_families": 1,
                "included": ["cf_005_056"],
                "refused": {
                    "mf_005_c08": (
                        "gradeable empty-room adapted probe rejected for endpoint transport cost"
                    )
                },
            },
            "archetype_selection": {
                "selected": list(ARCHETYPES),
                "rationale": (
                    "door_lintel tests licensed jamb context; hvac_duct and ibeam provide distinct "
                    "overhead context while preserving the exact binding face"
                ),
                "golden_shelf_plank_excluded": True,
            },
            "cpu_certificates": certificates,
            "e1a_source_inclusive_noise_floor_mm": e1a["source_inclusive_noise_floor_mm"],
            "e1b_adapted_drift_warning": True,
        },
    )
    print(f"wrote {len(cells)}-cell proposed E2 manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
