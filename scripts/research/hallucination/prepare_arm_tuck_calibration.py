#!/usr/bin/env python3
"""Materialize a two-motion, three-level arm-tuck delivery calibration cohort."""

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

MAX_TARGET_REDUCTION_M = 0.06
LEVELS = (1 / 3, 2 / 3, 1.0)
PAIR_SELECTION = {"left": (89, 86), "right": (92, 84)}


def window(nominal: np.ndarray, adapted: np.ndarray, side: str) -> float:
    mask = active_frames(nominal, adapted)
    nominal_left, nominal_right = _signed_half_widths(nominal, DEFAULT_G1_MJCF)
    adapted_left, adapted_right = _signed_half_widths(adapted, DEFAULT_G1_MJCF)
    nominal_side = nominal_left if side == "left" else nominal_right
    adapted_side = adapted_left if side == "left" else adapted_right
    return float(nominal_side[mask].max() - adapted_side[mask].max())


def artifacts(path: Path, provenance: Path) -> dict[str, str]:
    return {
        "csv": str(path.with_suffix(".csv")),
        "csv_sha256": sha256_file(path.with_suffix(".csv")),
        "motion": str(path),
        "motion_sha256": sha256_file(path),
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
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "lfh_arm_tuck_calibration")
    args = parser.parse_args()

    gate = {int(row["index"]): row for row in json.loads(args.reference_gate.read_text())}
    args.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for side, indices in PAIR_SELECTION.items():
        for index in indices:
            source = _one_source(args.source_dir, index)
            nominal = np.loadtxt(source, delimiter=",")
            source_gate = gate[index]
            if not (
                source_gate["embodiment_feasible"]
                and source_gate["self_collision_free"]
                and source_gate["reference_semantic_valid"] is True
            ):
                raise SystemExit(f"motion {index:03d}: source no longer passes the CPU gate")
            pair_id = f"lfh_{index:03d}_arm_tuck_{side}"
            pair_dir = args.out / pair_id
            pair_dir.mkdir(parents=True, exist_ok=True)
            nominal_csv = pair_dir / "nominal.csv"
            np.savetxt(nominal_csv, nominal, delimiter=",", fmt="%.10f")
            nominal_motion = pair_dir / "nominal.pkl"
            _save_motion(nominal_motion, f"{pair_id}__nominal", nominal)
            nominal_provenance = _write_motion_provenance(
                nominal_motion, f"{pair_id}__nominal", nominal_csv
            )
            levels = []
            for alpha in LEVELS:
                adapted, report = local_arm_tuck(
                    nominal,
                    STATION_FRACTION,
                    target_reduction_m=MAX_TARGET_REDUCTION_M * alpha,
                    window=WINDOW_FRACTION,
                    side=side,
                )
                adapted_gate = screen_reference(adapted, f"{pair_id}__a{alpha:.3f}", "walk")
                if not adapted_gate.worth_a_rollout or not report.root_path_preserved:
                    raise SystemExit(f"{pair_id} alpha={alpha}: adapted CPU gate failed")
                tag = f"a{round(100 * alpha):03d}"
                adapted_csv = pair_dir / f"{tag}.csv"
                np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
                adapted_motion = pair_dir / f"{tag}.pkl"
                key = f"{pair_id}__{tag}"
                _save_motion(adapted_motion, key, adapted)
                provenance = _write_motion_provenance(adapted_motion, key, adapted_csv)
                levels.append(
                    {
                        "alpha": alpha,
                        "tag": tag,
                        "target_reduction_m": MAX_TARGET_REDUCTION_M * alpha,
                        "reference_window_m": window(nominal, adapted, side),
                        "operator_report": asdict(report),
                        "reference_gate": {
                            "worth_a_rollout": adapted_gate.worth_a_rollout,
                            "diagnosis": adapted_gate.diagnosis,
                            "self_collision_free": adapted_gate.self_collision_free,
                        },
                        "artifacts": artifacts(adapted_motion, provenance),
                    }
                )
            pairs.append(
                {
                    "pair_id": pair_id,
                    "motion_index": index,
                    "side": side,
                    "source_csv": str(source),
                    "source_sha256": sha256_file(source),
                    "source_gate": source_gate,
                    "nominal_artifacts": artifacts(nominal_motion, nominal_provenance),
                    "levels": levels,
                }
            )

    payload = {
        "schema_version": "lfh_arm_tuck_calibration_candidates_v1",
        "operator": "local_arm_tuck",
        "station_fraction": STATION_FRACTION,
        "window_fraction": WINDOW_FRACTION,
        "maximum_target_reduction_m": MAX_TARGET_REDUCTION_M,
        "levels": list(LEVELS),
        "selection": {
            "left": list(PAIR_SELECTION["left"]),
            "right": list(PAIR_SELECTION["right"]),
            "basis": "top two strict-CPU-gated reference windows per side; includes V5 anchors",
        },
        "pairs": pairs,
    }
    path = args.out / "candidates.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o664)
    print(f"PASS: {len(pairs)} pairs x {len(LEVELS)} levels -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
