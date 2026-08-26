#!/usr/bin/env python3
"""Materialize the CPU-selected crouch cohort for empty-room calibration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    DEFAULT_WINDOW,
    active_frames,
    local_crouch,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from scripts.research.hallucination.prepare_arm_tuck_strong_calibration import (  # noqa: E402
    artifact_set,
)
from scripts.research.hallucination.prepare_probe_candidates import (  # noqa: E402
    DATA_ROOT,
    STATION_FRACTION,
    _one_source,
    _save_motion,
    _write_motion_provenance,
)
from scripts.research.hallucination.screen_crouch_strength import (  # noqa: E402
    CALIBRATION_TARGET_M,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--screen",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/crouch_strength_screen.json",
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "lfh_crouch_calibration")
    args = parser.parse_args()

    screen = json.loads(args.screen.read_text())
    recommendations = screen["calibration_recommendations"]
    if len(recommendations) != 4:
        raise SystemExit("the fixed screen must yield exactly four calibration recommendations")
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for recommendation in recommendations:
        index = int(recommendation["motion_index"])
        source = _one_source(args.source_dir, index)
        if sha256_file(source) != sha256_file(Path(recommendation["source_csv"])):
            raise SystemExit(f"motion {index:03d}: source changed since the CPU screen")
        nominal = np.loadtxt(source, delimiter=",")
        adapted, operator = local_crouch(
            nominal,
            STATION_FRACTION,
            target_drop_m=CALIBRATION_TARGET_M,
            window=DEFAULT_WINDOW,
        )
        gate = screen_reference(adapted, f"lfh_{index:03d}_crouch", "walk")
        if not gate.worth_a_rollout or not operator.root_path_preserved:
            raise SystemExit(f"motion {index:03d}: adapted reference gate changed")
        if (
            abs(1000 * operator.silhouette_drop_m - recommendation["reference_silhouette_drop_mm"])
            > 1e-6
        ):
            raise SystemExit(f"motion {index:03d}: operator output changed since the CPU screen")

        pair_id = f"lfh_{index:03d}_crouch"
        pair_dir = args.out / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        nominal_csv = pair_dir / "nominal.csv"
        adapted_csv = pair_dir / "adapted.csv"
        np.savetxt(nominal_csv, nominal, delimiter=",", fmt="%.10f")
        np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
        nominal_motion = pair_dir / "nominal.pkl"
        adapted_motion = pair_dir / "adapted.pkl"
        nominal_key = f"{pair_id}__cal_nominal"
        adapted_key = f"{pair_id}__cal_adapted"
        _save_motion(nominal_motion, nominal_key, nominal)
        _save_motion(adapted_motion, adapted_key, adapted)
        nominal_provenance = _write_motion_provenance(nominal_motion, nominal_key, nominal_csv)
        adapted_provenance = _write_motion_provenance(adapted_motion, adapted_key, adapted_csv)
        pairs.append(
            {
                "pair_id": pair_id,
                "motion_index": index,
                "response_alpha": 1.0,
                "target_drop_m": CALIBRATION_TARGET_M,
                "reference_silhouette_drop_m": operator.silhouette_drop_m,
                "active_frames": np.flatnonzero(active_frames(nominal, adapted)).tolist(),
                "route": recommendation["route"],
                "source_csv": str(source),
                "source_sha256": sha256_file(source),
                "operator_report": asdict(operator),
                "adapted_gate": {
                    "worth_a_rollout": gate.worth_a_rollout,
                    "diagnosis": gate.diagnosis,
                    "self_collision_free": gate.self_collision_free,
                    "saturated_cell_fraction": gate.saturated_cell_fraction,
                },
                "nominal_artifacts": artifact_set(nominal_csv, nominal_motion, nominal_provenance),
                "adapted_artifacts": artifact_set(adapted_csv, adapted_motion, adapted_provenance),
            }
        )

    payload = {
        "schema_version": "lfh_crouch_calibration_candidates_v1",
        "operator": "local_crouch",
        "station_fraction": STATION_FRACTION,
        "window_fraction": DEFAULT_WINDOW,
        "target_drop_m": CALIBRATION_TARGET_M,
        "screen": str(args.screen.relative_to(REPO_ROOT)),
        "screen_sha256": sha256_file(args.screen),
        "selection_basis": (
            "four highest reference silhouette drops at the historical 80 mm command among "
            "strict-reference passes with route straightness >=0.95"
        ),
        "pairs": pairs,
    }
    manifest = args.out / "candidates.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(pairs)} crouch pairs -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
