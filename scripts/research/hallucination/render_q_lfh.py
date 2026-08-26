#!/usr/bin/env python3
"""Render the evidence-backed, source-balanced LFH seed proposal distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    ConstraintSpec,
    sha256_file,
)
from gear_sonic.dataset_generation.local_adaptation import route_progress  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

ENGINEERING_MARGIN_M = 0.018044
SPEC_PATHS = {
    "cf_005_056": REPO_ROOT / "specs/hallucination/cs_duck_003.json",
    "lfh_086_crouch": REPO_ROOT / "specs/hallucination/cs_lfh_086_crouch_cal3_d0100.json",
    "lfh_089_crouch": REPO_ROOT / "specs/hallucination/cs_lfh_089_crouch_cal3.json",
    "lfh_090_crouch": REPO_ROOT / "specs/hallucination/cs_lfh_090_crouch_cal3.json",
}


def load_payload(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory")
    return payload


def station_progress(spec: ConstraintSpec) -> float:
    payload = load_payload(Path(spec.orig.artifact_path))
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    station = np.asarray(spec.binding_station_xy_m, dtype=np.float64)
    index = int(np.argmin(np.linalg.norm(root[:, :2] - station, axis=1)))
    return float(route_progress(root[:, :2])[index])


def verified_archetypes() -> tuple[dict[str, list[str]], dict[str, str]]:
    e1b_path = REPO_ROOT / "docs/hallucination/e1b_golden.json"
    e2_path = REPO_ROOT / "docs/hallucination/e2_variant_transfer.json"
    e6_path = REPO_ROOT / "docs/hallucination/e6_crouch_pilot.json"
    e6c_path = REPO_ROOT / "docs/hallucination/e6c_exposure.json"
    e7_path = REPO_ROOT / "docs/hallucination/e7_transfer.json"
    e7c_path = REPO_ROOT / "docs/hallucination/e7c_replacement.json"
    e1b = json.loads(e1b_path.read_text())
    e2 = json.loads(e2_path.read_text())
    e6 = json.loads(e6_path.read_text())
    e6c = json.loads(e6c_path.read_text())
    e7 = json.loads(e7_path.read_text())
    e7c = json.loads(e7c_path.read_text())

    if not e1b["golden_physics_green"]:
        raise SystemExit("cf_005_056 shelf golden is not verified")
    if not e6c["adaptive_followup_verified"]:
        raise SystemExit("source 086 shelf is not verified")
    if not e7c["adaptive_replacement_verified"]:
        raise SystemExit("source 086 replacement is not verified")
    e6_sources = {row["source_pair_id"]: row for row in e6["sources"]}
    e7_variants = {row["variant_id"]: row for row in e7["variants"]}
    result = {
        "cf_005_056": ["shelf_plank"],
        "lfh_086_crouch": ["shelf_plank", "hanging_panel"],
        "lfh_089_crouch": ["shelf_plank"],
        "lfh_090_crouch": ["shelf_plank"],
    }
    result["cf_005_056"].extend(
        row["archetype_id"] for row in e2["variant_results"] if row["verified"]
    )
    for source in ("lfh_089_crouch", "lfh_090_crouch"):
        if not e6_sources[source]["verified"]:
            raise SystemExit(f"{source} shelf is not verified")
    for row in e7_variants.values():
        if row["verified"]:
            result[row["source_pair_id"]].append(row["archetype_id"])
    result = {source: sorted(set(values)) for source, values in result.items()}
    evidence = {
        "e1b_golden": sha256_file(e1b_path),
        "e2_variant_transfer": sha256_file(e2_path),
        "e6_crouch_pilot": sha256_file(e6_path),
        "e6c_exposure": sha256_file(e6c_path),
        "e7_transfer": sha256_file(e7_path),
        "e7c_replacement": sha256_file(e7c_path),
    }
    return result, evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/q_lfh_v1.json",
    )
    args = parser.parse_args()

    archetypes, evidence = verified_archetypes()
    sources = sorted(SPEC_PATHS)
    source_weight = 1.0 / len(sources)
    atoms = []
    for source in sources:
        spec_path = SPEC_PATHS[source]
        spec = ConstraintSpec.load(spec_path)
        lower = spec.reach_edit_m + ENGINEERING_MARGIN_M
        upper = spec.reach_orig_m - ENGINEERING_MARGIN_M
        width = upper - lower
        if width <= 0:
            raise SystemExit(f"{source}: empty engineering support")
        visual_support = archetypes[source]
        visual_weight = source_weight / len(visual_support)
        atoms.append(
            {
                "source_pair_id": source,
                "operator": spec.edit.operator,
                "constraint_axis": spec.axis_type,
                "binding_keypoint": spec.binding_keypoint,
                "route_progress": station_progress(spec),
                "station_xy_m": list(spec.binding_station_xy_m),
                "face_along_route_m": spec.face_along_route_m,
                "face_across_route_m": spec.face_across_route_m,
                "engineering_support_lower_m": lower,
                "engineering_support_upper_m": upper,
                "engineering_width_mm": 1000 * width,
                "hard_coordinate_m": spec.hard_coordinate_m,
                "hard_xi": (spec.hard_coordinate_m - lower) / width,
                "easy_coordinate_m": spec.easy_coordinate_m,
                "easy_xi": (spec.easy_coordinate_m - lower) / width,
                "source_weight": source_weight,
                "verified_archetypes": [
                    {
                        "archetype_id": archetype,
                        "conditional_weight": 1.0 / len(visual_support),
                        "joint_weight": visual_weight,
                        "easy_scene_weight": visual_weight / 2,
                        "hard_scene_weight": visual_weight / 2,
                    }
                    for archetype in visual_support
                ],
                "spec": str(spec_path.relative_to(REPO_ROOT)),
                "spec_sha256": sha256_file(spec_path),
            }
        )

    common = sorted(set.intersection(*(set(values) for values in archetypes.values())))
    output = {
        "schema_version": "q_lfh_v1",
        "status": "evidence_backed_seed_distribution_not_fitted_density",
        "interpretation": (
            "designed proposal mass over verified trajectory-conditioned atoms; not P_env and "
            "not a physics-verdict model"
        ),
        "sampling_policy": {
            "source_balanced": True,
            "source_weight": source_weight,
            "archetype_balanced_within_source": True,
            "easy_hard_scene_mass": [0.5, 0.5],
            "physics_supplies_labels": True,
            "dcs_role": "novelty/reporting only after feasible support",
        },
        "engineering_margin": {
            "value_mm_each_side": 1000 * ENGINEERING_MARGIN_M,
            "confidence_level": None,
            "interpretation": "empirical engineering margin, not calibrated uncertainty",
        },
        "sources": len(sources),
        "verified_source_archetype_pairs": sum(len(values) for values in archetypes.values()),
        "common_verified_archetypes": common,
        "count_only_gate_three_archetypes_per_source": all(
            len(values) >= 3 for values in archetypes.values()
        ),
        "strict_crossed_gate_three_common_archetypes": len(common) >= 3,
        "next_required_evidence": (
            None
            if len(common) >= 3
            else "verify one common third archetype across all four sources before crossed E5"
        ),
        "evidence": evidence,
        "atoms": atoms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        f"PASS: q_LFH seed has {output['sources']} sources, "
        f"{output['verified_source_archetype_pairs']} pairs, common={common}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
