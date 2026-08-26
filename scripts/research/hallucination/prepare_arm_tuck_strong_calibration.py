#!/usr/bin/env python3
"""Materialize matched strong-command arm-tuck candidates for CAL2."""

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
    _signed_half_widths,
    active_frames,
    local_arm_tuck,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import (  # noqa: E402
    DATA_ROOT,
    STATION_FRACTION,
    WINDOW_FRACTION,
    _one_source,
    _save_motion,
    _write_motion_provenance,
)

PAIRS = (
    (86, "left", 0.12, 2.0),
    (94, "left", 0.12, 2.0),
    (92, "right", 0.15, 2.5),
    (84, "right", 0.15, 2.5),
)


def artifact_set(csv_path: Path, motion: Path, provenance: Path) -> dict[str, str]:
    return {
        "csv": str(csv_path),
        "csv_sha256": sha256_file(csv_path),
        "motion": str(motion),
        "motion_sha256": sha256_file(motion),
        "motion_manifest": str(provenance),
        "motion_manifest_sha256": sha256_file(provenance),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument(
        "--reference-gate",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/coverage/reference_gate.json",
    )
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "lfh_arm_tuck_strong_calibration")
    args = parser.parse_args()

    gate = {int(row["index"]): row for row in json.loads(args.reference_gate.read_text())}
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for index, side, target, alpha in PAIRS:
        source = _one_source(args.source_dir, index)
        nominal = np.loadtxt(source, delimiter=",")
        source_gate = gate[index]
        if not (
            source_gate["embodiment_feasible"]
            and source_gate["self_collision_free"]
            and source_gate["reference_semantic_valid"] is True
        ):
            raise SystemExit(f"motion {index:03d}: source no longer passes the CPU gate")
        adapted, operator = local_arm_tuck(
            nominal,
            STATION_FRACTION,
            target_reduction_m=target,
            window=WINDOW_FRACTION,
            side=side,
        )
        adapted_gate = screen_reference(adapted, f"strong_{index:03d}_{side}", "walk")
        if not adapted_gate.worth_a_rollout or not operator.root_path_preserved:
            raise SystemExit(f"motion {index:03d}/{side}: strong reference gate failed")

        pair_id = f"lfh_{index:03d}_arm_tuck_{side}"
        pair_dir = args.out / pair_id
        pair_dir.mkdir(parents=True, exist_ok=True)
        nominal_csv = pair_dir / "nominal.csv"
        adapted_csv = pair_dir / "strong.csv"
        np.savetxt(nominal_csv, nominal, delimiter=",", fmt="%.10f")
        np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
        nominal_motion = pair_dir / "nominal.pkl"
        adapted_motion = pair_dir / "strong.pkl"
        _save_motion(nominal_motion, f"{pair_id}__cal2_nominal", nominal)
        _save_motion(adapted_motion, f"{pair_id}__cal2_strong", adapted)
        nominal_provenance = _write_motion_provenance(
            nominal_motion, f"{pair_id}__cal2_nominal", nominal_csv
        )
        adapted_provenance = _write_motion_provenance(
            adapted_motion, f"{pair_id}__cal2_strong", adapted_csv
        )
        mask = active_frames(nominal, adapted)
        nominal_left, nominal_right = _signed_half_widths(nominal, DEFAULT_G1_MJCF)
        adapted_left, adapted_right = _signed_half_widths(adapted, DEFAULT_G1_MJCF)
        nominal_side = nominal_left if side == "left" else nominal_right
        adapted_side = adapted_left if side == "left" else adapted_right
        pairs.append(
            {
                "pair_id": pair_id,
                "motion_index": index,
                "side": side,
                "alpha": alpha,
                "target_reduction_m": target,
                "reference_window_m": float(nominal_side[mask].max() - adapted_side[mask].max()),
                "source_csv": str(source),
                "source_sha256": sha256_file(source),
                "source_gate": source_gate,
                "operator_report": asdict(operator),
                "adapted_gate": {
                    "worth_a_rollout": adapted_gate.worth_a_rollout,
                    "diagnosis": adapted_gate.diagnosis,
                    "self_collision_free": adapted_gate.self_collision_free,
                    "saturated_cell_fraction": adapted_gate.saturated_cell_fraction,
                },
                "nominal_artifacts": artifact_set(nominal_csv, nominal_motion, nominal_provenance),
                "adapted_artifacts": artifact_set(adapted_csv, adapted_motion, adapted_provenance),
            }
        )

    payload = {
        "schema_version": "lfh_arm_tuck_strong_calibration_candidates_v1",
        "operator": "local_arm_tuck",
        "alpha_anchor_target_reduction_m": 0.06,
        "alpha_semantics": "nonnegative scale relative to CAL1's 60 mm target",
        "station_fraction": STATION_FRACTION,
        "window_fraction": WINDOW_FRACTION,
        "selection_basis": (
            "two matched strong-command motions per side; reuse three accepted CAL1/V5 "
            "nominals and add CPU-top-ranked 094/left"
        ),
        "pairs": pairs,
    }
    manifest = args.out / "candidates.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"PASS: {len(pairs)} strong candidates -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
