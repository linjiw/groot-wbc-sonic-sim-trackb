#!/usr/bin/env python3
"""Regenerate the Phase-2 E1 CPU evidence without launching physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.archetypes import ARCHETYPES  # noqa: E402
from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    ConstraintSpec,
    ConstraintSpecError,
)
from gear_sonic.dataset_generation.hallucination.extract_spec import extract_spec  # noqa: E402
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate  # noqa: E402
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    validate_pair,
    write_report,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
DEFAULT_SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
DEFAULT_PACKAGE = REPO_ROOT / "gear_sonic/data/assets/scenes/hallucinated_variants_v1_e1"
DEFAULT_SUMMARY = REPO_ROOT / "docs/hallucination/e1_cpu/summary.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duck-family", type=Path, default=DATA_ROOT / "counterfactual/duck_003")
    parser.add_argument("--mf-family", type=Path, default=DATA_ROOT / "matched/mf_005_c08")
    parser.add_argument("--scenes-root", type=Path, default=DEFAULT_SCENES)
    parser.add_argument(
        "--spec", type=Path, default=REPO_ROOT / "specs/hallucination/cs_duck_003.json"
    )
    parser.add_argument("--out-package", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    committed = ConstraintSpec.load(args.spec)
    extracted = extract_spec(args.duck_family, args.scenes_root)
    if extracted.to_dict() != committed.to_dict():
        raise SystemExit("committed duck_003 spec does not match a fresh extraction")

    args.out_package.mkdir(parents=True, exist_ok=True)
    archetypes: list[dict[str, object]] = []
    for archetype_id in sorted(
        name for name, value in ARCHETYPES.items() if value.axis_type == "overhead"
    ):
        result = instantiate(committed, archetype_id, args.seed, args.out_package)
        keepout = validate_pair(committed, result.easy.path, result.hard.path)
        keepout_path = args.out_package / (
            f"{committed.spec_id}__{archetype_id}__s{args.seed:08d}.keepout.json"
        )
        write_report(keepout, keepout_path)
        if not keepout["ok"]:
            raise SystemExit(f"{archetype_id}: CPU preflight refused; see {keepout_path}")
        pair_manifest = json.loads(result.manifest_path.read_text())
        geometry_pattern = keepout["binding_geometry_pattern"]
        pair_manifest["cpu_certificate"] = {
            "keepout_report": str(keepout_path),
            "binding_geometry_signs": keepout["binding_geometry_signs"],
            "binding_geometry_clearance_mm": {
                cell: {motion: entry["clearance_mm"] for motion, entry in motions.items()}
                for cell, motions in geometry_pattern.items()
            },
            "four_sign_pattern_matches": all(
                entry["matches"]
                for motions in geometry_pattern.values()
                for entry in motions.values()
            ),
        }
        result.manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
        result.manifest_path.chmod(0o664)
        reports = keepout["reports"]
        archetypes.append(
            {
                "archetype_id": archetype_id,
                "seed": args.seed,
                "manifest": str(result.manifest_path),
                "keepout_report": str(keepout_path),
                "easy_scene": str(result.easy.path),
                "hard_scene": str(result.hard.path),
                "easy_face_realized_m": result.easy.measurement.coordinate_m,
                "hard_face_realized_m": result.hard.measurement.coordinate_m,
                "max_binding_face_offset_mm": max(
                    result.easy.binding_face_offset_mm, result.hard.binding_face_offset_mm
                ),
                "max_binding_station_offset_mm": max(
                    result.easy.binding_station_offset_mm,
                    result.hard.binding_station_offset_mm,
                ),
                "minimum_context_clearance_mm": min(
                    (
                        report["keepout_min_clearance_mm"]
                        for report in reports.values()
                        if report["keepout_min_clearance_mm"] is not None
                    ),
                    default=None,
                ),
                "cpu_preflight": "pass",
            }
        )

    try:
        extract_spec(args.mf_family, args.scenes_root)
    except ConstraintSpecError as error:
        mf_status = {"status": "refused", "reason": str(error)}
    else:
        raise SystemExit("mf_005_c08 unexpectedly became extractable; audit the source evidence")

    summary = {
        "schema_version": "lfh_e1_cpu_v1",
        "physics_executed": False,
        "duck_spec": {
            "path": str(args.spec),
            "fingerprint": committed.fingerprint(),
            "fresh_extraction_matches": True,
            "golden_faces_m": {
                "easy": committed.easy_coordinate_m,
                "hard": committed.hard_coordinate_m,
            },
            "crossing_frames": {
                "orig": list(committed.crossing_frames_orig),
                "edit": list(committed.crossing_frames_edit),
            },
        },
        "archetype_counts": {
            axis: sum(value.axis_type == axis for value in ARCHETYPES.values())
            for axis in ("overhead", "lateral_gap")
        },
        "overhead_archetypes": archetypes,
        "mf_005_c08": mf_status,
        "all_cpu_preflights_pass": True,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.chmod(0o775)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    args.summary.chmod(0o664)
    print(
        f"PASS: {len(archetypes)} overhead archetypes; "
        f"mf_005_c08 refused honestly -> {args.summary}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
