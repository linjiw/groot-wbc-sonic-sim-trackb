#!/usr/bin/env python3
"""Build the staged empty-scene physics manifest for a fresh Kimodo→LFH motion pair."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        type=Path,
        default=DATA_ROOT / "lfh_3d_loop/pair/motion_pair.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/manifests/E9A_MOTION_SCENE_CALIBRATION_PROPOSED.json",
    )
    args = parser.parse_args()

    pair = json.loads(args.pair.read_text())
    if pair["status"] != "cpu_motion_pair_ready_for_empty_scene_physics":
        raise SystemExit("fresh motion pair has not passed the CPU gate")
    cells = []
    for role in ("nominal", "adapted"):
        cell_id = f"lfh_fresh_curve__empty__{role}"
        cells.append(
            {
                "cell_id": cell_id,
                "pair_id": "lfh_fresh_curve",
                "pair_role": role,
                "depends_on_acceptance_of": ("lfh_fresh_curve__empty__nominal" if role == "adapted" else None),
                "runtime_seed": 45001,
                "hydra_overrides": ["++seed=45001"],
                "scene": _scene("screen_empty"),
                "motion": _motion(Path(pair["artifacts"][f"{role}_motion"])),
                "scene_start_xyz_expected": [0.0, 0.0, 0.0],
                "expectation": {
                    "status": "registered_trackability_prediction",
                    "outcome": "accepted",
                    "basis": "strict reference and self-collision gates passed",
                },
                "verdict_policy": "reference_trackability",
                "output": str(DATA_ROOT / "lfh_3d_loop/empty_scene" / role),
            }
        )

    _write(
        args.out,
        "LFH-E9a-fresh-motion-empty-scene",
        "Execute a newly sampled Kimodo motion and its deterministic LFH crouch twin before any "
        "scene is proposed.",
        cells,
        [
            "skip adapted if nominal is rejected",
            "stop on infrastructure failure and retain completed evidence",
            "do not author geometry unless both motions accept with zero external contact",
            "do not call any geometry axis supported until executed finite-face separation exists",
        ],
        extra={
            "motion_pair": {"path": str(args.pair), "sha256": sha256_file(args.pair)},
            "proposal_contract": {
                "frame": "route-relative 3D frame carried by motion_pair.json",
                "geometry_is_proposal_only": True,
                "physics_supplies_verdict": True,
            },
        },
    )
    print(f"wrote {len(cells)}-cell E9a proposal -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
