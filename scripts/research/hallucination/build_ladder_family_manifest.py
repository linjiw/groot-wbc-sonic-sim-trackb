#!/usr/bin/env python3
"""Build the LFH-E17 2x2 physics manifest for a scene authored from a ladder pair.

The four cells are the counterfactual: each motion in each scene. A family completes only on
accepted / accepted / rejected / accepted with the nominal-hard contact uniquely attributed to the
binding face before reference drift.

The simulator seed is pinned to the one at which the adapted rung was observed to accept. LFH-E16
showed that a deepest-accepted rung can straddle the tracking gate, so an unpinned seed would
confound a scene result with a delivery coin flip.
"""

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
    _write,
)

CELLS = (
    ("nominal", "easy", "accepted"),
    ("adapted", "easy", "accepted"),
    ("nominal", "hard", "rejected"),
    ("adapted", "hard", "accepted"),
)


def _scene_entry(path: Path, sha: str) -> dict:
    return {
        "scene_id": path.stem,
        "path": str(path.relative_to(REPO_ROOT)),
        "sha256": sha if sha.startswith("sha256:") else f"sha256:{sha}",
    }


def _cells_for(entry: dict, seed: int) -> list[dict]:
    pair_id = entry["pair_id"]
    archetype = entry["archetype"]
    ladder_dir = DATA_ROOT / "lfh_crouch_ladder" / pair_id
    rung_label = Path(entry["reference_motions"]["adapted"]).parents[1].name
    motions = {"nominal": ladder_dir / "nominal.pkl", "adapted": ladder_dir / f"{rung_label}.pkl"}
    for path in motions.values():
        if not path.exists():
            raise SystemExit(f"missing motion artifact: {path}")
    easy_nominal_id = f"{pair_id}__{archetype}__easy__nominal"
    cells = []
    for role, difficulty, expectation in CELLS:
        cell_id = f"{pair_id}__{archetype}__{difficulty}__{role}"
        cells.append(
            {
                "cell_id": cell_id,
                "pair_id": pair_id,
                "archetype": archetype,
                "pair_role": role,
                "scene_difficulty": difficulty,
                "body_mode": entry["body_mode"],
                "depends_on_acceptance_of": None if cell_id == easy_nominal_id else easy_nominal_id,
                "runtime_seed": seed,
                "hydra_overrides": [f"++seed={seed}"],
                "scene": _scene_entry(
                    REPO_ROOT / entry["scenes"][difficulty], entry["scene_sha256"][difficulty]
                ),
                "motion": _motion(motions[role]),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expectation": {
                    "status": "registered_counterfactual_prediction",
                    "outcome": expectation,
                    "basis": (
                        "same executed window and same face coordinate as the verified "
                        "shelf_plank family; only the archetype realising the face differs"
                    ),
                },
                "verdict_policy": "reference_trackability",
                "output": str(
                    DATA_ROOT
                    / "hallucination/ladder_family"
                    / pair_id
                    / f"{archetype}_{difficulty}_{role}"
                ),
            }
        )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-report",
        type=Path,
        action="append",
        help="may be repeated; each report contributes its own 2x2",
    )
    parser.add_argument("--pair-id", default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E17_LADDER_FAMILY_PROPOSED.json",
    )
    args = parser.parse_args()

    reports = args.scene_report or [REPO_ROOT / "docs/hallucination/e17_ladder_scenes.json"]
    entries = []
    for path in reports:
        payload = json.loads(Path(path).read_text())
        scenes = payload["scenes"]
        entries.append(
            next(scene for scene in scenes if scene["pair_id"] == args.pair_id)
            if args.pair_id
            else scenes[0]
        )

    cells = []
    for entry in entries:
        cells.extend(_cells_for(entry, args.seed))
    _write(
        args.out,
        "LFH-E17-ladder-family",
        "Counterfactual 2x2 per archetype at one fixed critical point, testing whether the "
        "appearance of the binding face is free given the executed window.",
        cells,
        [
            "skip every remaining cell if the easy nominal is rejected",
            "continue after completed scientific rejections and retain the full denominator",
            "stop on infrastructure failure and preserve every completed capture",
            "complete the family only on accepted/accepted/rejected/accepted with unique binding "
            "attribution before reference drift",
        ],
        extra={
            "scene_reports": [
                {"path": str(Path(path)), "sha256": sha256_file(Path(path))} for path in reports
            ],
            "registered_predictions": (
                "docs/prediction_register.md, LFH-E17 entry, filed 2026-08-26 before spend"
            ),
            "seed_policy": {
                "pinned_seed": args.seed,
                "reason": (
                    "the adapted rung straddles the tracking gate across seeds (LFH-E16); an "
                    "unpinned seed would confound the scene result with a delivery coin flip"
                ),
            },
            "geometry": [
                {
                    "archetype": item["archetype"],
                    "hard_coordinate_m": item["hard_coordinate_m"],
                    "easy_coordinate_m": item["easy_coordinate_m"],
                    "engineering_window_mm": item["engineering_window_mm"],
                    "binding_keypoint": item["binding_keypoint"],
                }
                for item in entries
            ],
        },
    )
    print(f"wrote {len(cells)}-cell proposal at seed {args.seed} -> {args.out}")
    for item in entries:
        print(f"  {item['pair_id']} / {item['archetype']} @ {item['hard_coordinate_m']:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
