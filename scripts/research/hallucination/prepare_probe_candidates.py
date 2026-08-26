#!/usr/bin/env python3
"""Materialize the smallest target-ranked CPU motion cohort for plane probing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import sha256_file  # noqa: E402
from gear_sonic.dataset_generation.kimodo_motion_adapter import (  # noqa: E402
    qpos_to_sonic_motion_entry,
    save_sonic_motion_file,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    DELIVERY_RATIO,
    _signed_half_widths,
    active_frames,
    local_arm_tuck,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
STATION_FRACTION = 0.55
WINDOW_FRACTION = 0.30
TARGET_REDUCTION_M = 0.06


def _one_source(source_dir: Path, index: int) -> Path:
    matches = sorted(source_dir.glob(f"{index:03d}_*.csv"))
    if len(matches) != 1:
        raise ValueError(f"motion {index:03d}: expected one source CSV, found {len(matches)}")
    return matches[0]


def _save_motion(path: Path, key: str, qpos: np.ndarray) -> None:
    entry = qpos_to_sonic_motion_entry(qpos, source_fps=30.0)
    save_sonic_motion_file(path, motion_key=key, motion_entry=entry)
    path.chmod(0o664)


def _write_motion_provenance(path: Path, key: str, source_csv: Path) -> Path:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    payload = {
        "schema_version": "lfh_probe_motion_v1",
        "converter": "gear_sonic.dataset_generation.kimodo_motion_adapter",
        "motion_key": key,
        "source_fps": 30.0,
        "canonicalize_horizontal_origin": True,
        "scene_start_xyz": [0.0, 0.0, 0.0],
        "scene_yaw": 0.0,
        "input": {"path": str(source_csv), "sha256": sha256_file(source_csv)},
        "output": {"path": str(path), "sha256": sha256_file(path)},
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest.chmod(0o664)
    return manifest


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
    parser.add_argument(
        "--trackability",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/coverage/trackability.json",
    )
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "lfh_probe_candidates")
    args = parser.parse_args()

    gate_by_index = {
        int(record["index"]): record for record in json.loads(args.reference_gate.read_text())
    }
    tracked_indices = {
        int(record["clip"][:3])
        for record in json.loads(args.trackability.read_text())["rows"]
        if record["tracked"]
    }
    eligible_indices = sorted(
        index
        for index, record in gate_by_index.items()
        if record["body_mode"] == "stand_to_walk"
        and record["embodiment_feasible"]
        and record["self_collision_free"]
        and record["reference_semantic_valid"] is True
    )
    delivery_ratio = DELIVERY_RATIO[("local_arm_tuck", "chest")]
    screen: list[dict[str, object]] = []
    clips: dict[tuple[int, str], tuple[Path, np.ndarray, np.ndarray, object]] = {}
    for index in eligible_indices:
        source = _one_source(args.source_dir, index)
        nominal = np.loadtxt(source, delimiter=",")
        nominal_left, nominal_right = _signed_half_widths(nominal, DEFAULT_G1_MJCF)
        for side in ("left", "right"):
            adapted, report = local_arm_tuck(
                nominal,
                STATION_FRACTION,
                target_reduction_m=TARGET_REDUCTION_M,
                window=WINDOW_FRACTION,
                side=side,
            )
            mask = active_frames(nominal, adapted)
            adapted_left, adapted_right = _signed_half_widths(adapted, DEFAULT_G1_MJCF)
            nominal_width = nominal_left if side == "left" else nominal_right
            adapted_width = adapted_left if side == "left" else adapted_right
            window_m = float(nominal_width[mask].max() - adapted_width[mask].max())
            screen.append(
                {
                    "motion_index": index,
                    "side": side,
                    "reference_window_m": window_m,
                    "predicted_delivered_window_m": window_m * delivery_ratio,
                    "has_nonempty_trackability_prior": index in tracked_indices,
                    "max_joint_change_rad": report.max_joint_change_rad,
                    "excursion_capped": report.excursion_capped,
                }
            )
            clips[(index, side)] = (source, nominal, adapted, report)

    best_left = max(
        (record for record in screen if record["side"] == "left"),
        key=lambda record: record["reference_window_m"],
    )
    tracked_right = [
        record
        for record in screen
        if record["side"] == "right" and record["has_nonempty_trackability_prior"]
    ]
    best_right = max(
        tracked_right or [record for record in screen if record["side"] == "right"],
        key=lambda record: record["reference_window_m"],
    )
    selected = (best_left, best_right)

    args.out.mkdir(parents=True, exist_ok=True)
    args.out.chmod(0o775)
    records: list[dict[str, object]] = []
    for selection in selected:
        index = int(selection["motion_index"])
        side = str(selection["side"])
        source, nominal, adapted, report = clips[(index, side)]
        source_gate = gate_by_index[index]
        if not (
            source_gate["embodiment_feasible"]
            and source_gate["self_collision_free"]
            and source_gate["reference_semantic_valid"] is True
        ):
            raise SystemExit(f"motion {index:03d}: source no longer passes the strict CPU gate")

        adapted_gate = screen_reference(adapted, f"lfh_{index:03d}_arm_tuck_{side}", "walk")
        if not adapted_gate.worth_a_rollout or not report.root_path_preserved:
            raise SystemExit(f"motion {index:03d}: adapted reference fails the CPU gate")

        window_m = float(selection["reference_window_m"])
        predicted_delivered_m = window_m * delivery_ratio
        if predicted_delivered_m < 0.03:
            raise SystemExit(
                f"motion {index:03d}: predicted delivered window "
                f"{1000 * predicted_delivered_m:.1f} mm is below the 30 mm CPU spend gate"
            )

        stem = f"lfh_{index:03d}_arm_tuck_{side}"
        nominal_csv = args.out / f"{stem}__nominal.csv"
        adapted_csv = args.out / f"{stem}__adapted.csv"
        np.savetxt(nominal_csv, nominal, delimiter=",", fmt="%.10f")
        np.savetxt(adapted_csv, adapted, delimiter=",", fmt="%.10f")
        nominal_csv.chmod(0o664)
        adapted_csv.chmod(0o664)
        nominal_motion = args.out / f"{stem}__nominal.pkl"
        adapted_motion = args.out / f"{stem}__adapted.pkl"
        nominal_key = f"{stem}__nominal"
        adapted_key = f"{stem}__adapted"
        _save_motion(nominal_motion, nominal_key, nominal)
        _save_motion(adapted_motion, adapted_key, adapted)
        nominal_provenance = _write_motion_provenance(nominal_motion, nominal_key, nominal_csv)
        adapted_provenance = _write_motion_provenance(adapted_motion, adapted_key, adapted_csv)

        records.append(
            {
                "motion_index": index,
                "side": side,
                "source_csv": str(source),
                "source_sha256": sha256_file(source),
                "source_gate": source_gate,
                "operator": "local_arm_tuck",
                "operator_parameters": {
                    "station_fraction": STATION_FRACTION,
                    "window_fraction": WINDOW_FRACTION,
                    "target_reduction_m": TARGET_REDUCTION_M,
                },
                "operator_report": asdict(report),
                "adapted_gate": {
                    "worth_a_rollout": adapted_gate.worth_a_rollout,
                    "diagnosis": adapted_gate.diagnosis,
                    "embodiment_feasible": adapted_gate.embodiment_feasible,
                    "self_collision_free": adapted_gate.self_collision_free,
                    "saturated_cell_fraction": adapted_gate.saturated_cell_fraction,
                },
                "reference_window_m": window_m,
                "delivery_ratio_used": delivery_ratio,
                "predicted_delivered_window_m": predicted_delivered_m,
                "artifacts": {
                    "nominal_csv": str(nominal_csv),
                    "nominal_csv_sha256": sha256_file(nominal_csv),
                    "adapted_csv": str(adapted_csv),
                    "adapted_csv_sha256": sha256_file(adapted_csv),
                    "nominal_motion": str(nominal_motion),
                    "nominal_motion_sha256": sha256_file(nominal_motion),
                    "nominal_motion_manifest": str(nominal_provenance),
                    "nominal_motion_manifest_sha256": sha256_file(nominal_provenance),
                    "adapted_motion": str(adapted_motion),
                    "adapted_motion_sha256": sha256_file(adapted_motion),
                    "adapted_motion_manifest": str(adapted_provenance),
                    "adapted_motion_manifest_sha256": sha256_file(adapted_provenance),
                },
            }
        )

    payload = {
        "schema_version": "lfh_probe_candidates_v1",
        "selection_basis": {
            "coverage_targets_sha256": "sha256:"
            + hashlib.sha256(
                (REPO_ROOT / "docs/hallucination/coverage/targets.json").read_bytes()
            ).hexdigest(),
            "target": "top-ranked arms/local_arm_tuck/lateral_gap bins",
            "selection": "best left candidate plus tracked right candidate after CPU window screen",
            "minimum_predicted_delivered_window_m": 0.03,
            "eligible_motion_indices": eligible_indices,
        },
        "screen": sorted(
            screen,
            key=lambda record: (
                -float(record["reference_window_m"]),
                int(record["motion_index"]),
                str(record["side"]),
            ),
        ),
        "candidates": records,
    }
    manifest = args.out / "candidates.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    manifest.chmod(0o664)
    print(f"PASS: {2 * len(records)} exact motion artifacts -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
