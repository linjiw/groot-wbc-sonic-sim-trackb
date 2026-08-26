#!/usr/bin/env python3
"""CPU-screen the existing crouch operator over the gated stand-to-walk pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    DEFAULT_WINDOW,
    active_frames,
    local_crouch,
    route_progress,
)
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import (  # noqa: E402
    DATA_ROOT,
    STATION_FRACTION,
    _one_source,
)

TARGET_DROPS_M = (0.04, 0.06, 0.08, 0.10)
CALIBRATION_TARGET_M = 0.08
MIN_ROUTE_STRAIGHTNESS = 0.95
MAX_RECOMMENDATIONS = 4


def route_summary(qpos: np.ndarray) -> dict[str, object]:
    root = np.asarray(qpos[:, :2], dtype=np.float64)
    path_length = float(np.linalg.norm(np.diff(root, axis=0), axis=1).sum())
    displacement = root[-1] - root[0]
    net_displacement = float(np.linalg.norm(displacement))
    progress = route_progress(root)
    station_frame = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    return {
        "route_axis": "x" if abs(displacement[0]) >= abs(displacement[1]) else "y",
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "straightness": net_displacement / path_length if path_length > 1e-9 else 0.0,
        "station_frame": station_frame,
        "station_xy_m": [float(value) for value in root[station_frame]],
    }


def render(report: dict) -> str:
    lines = [
        "# Crouch Strength CPU Screen",
        "",
        "This fixed screen applies only the existing `local_crouch` operator to the strict "
        "stand-to-walk pool. The capsule silhouette drop is reference-side selection evidence, "
        "not executed delivery or a scene verdict.",
        "",
        "| motion | target | silhouette drop | knee change | capped | CPU gate | straightness |",
        "|---|---:|---:|---:|:---:|:---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['motion_index']:03d}` | {row['target_drop_mm']:.0f} mm | "
            f"{row['reference_silhouette_drop_mm']:.2f} mm | "
            f"{row['max_joint_change_rad']:.3f} rad | "
            f"{'yes' if row['excursion_capped'] else 'no'} | "
            f"{'pass' if row['worth_a_rollout'] else 'refused'} | "
            f"{row['route']['straightness']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"Funnel: **{report['source_motions']} motions -> {report['trials']} settings -> "
            f"{report['passing_trials']} strict-reference passes -> "
            f"{len(report['calibration_recommendations'])} calibration recommendations**.",
            "",
            "Recommendations use the historically verified 80 mm command, require route "
            f"straightness >= {report['selection_contract']['minimum_route_straightness']:.2f}, "
            "and rank by reference silhouette drop. They license only a hash-pinned empty-room "
            "calibration proposal. Scene synthesis remains refused until executed, zero-contact "
            "capsule tracks establish a finite-face window.",
            "",
            "Recommended motion indices: "
            + (
                ", ".join(
                    f"`{row['motion_index']:03d}`" for row in report["calibration_recommendations"]
                )
                or "none"
            )
            + ".",
        ]
    )
    return "\n".join(lines) + "\n"


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
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/crouch_strength_screen.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/REPORT_CROUCH_STRENGTH_SCREEN.md",
    )
    args = parser.parse_args()

    gate_by_index = {
        int(record["index"]): record for record in json.loads(args.reference_gate.read_text())
    }
    eligible = sorted(
        index
        for index, record in gate_by_index.items()
        if record["body_mode"] == "stand_to_walk"
        and record["embodiment_feasible"]
        and record["self_collision_free"]
        and record["reference_semantic_valid"] is True
    )
    rows = []
    for index in eligible:
        source = _one_source(args.source_dir, index)
        nominal = np.loadtxt(source, delimiter=",")
        route = route_summary(nominal)
        for target in TARGET_DROPS_M:
            adapted, operator = local_crouch(
                nominal,
                STATION_FRACTION,
                target_drop_m=target,
                window=DEFAULT_WINDOW,
            )
            active = active_frames(nominal, adapted)
            gate = screen_reference(adapted, f"lfh_{index:03d}_crouch_{target:.2f}", "walk")
            rows.append(
                {
                    "motion_index": index,
                    "source_csv": str(source),
                    "target_drop_mm": 1000 * target,
                    "reference_silhouette_drop_mm": 1000 * operator.silhouette_drop_m,
                    "active_frames": int(active.sum()),
                    "max_joint_change_rad": operator.max_joint_change_rad,
                    "excursion_capped": operator.excursion_capped,
                    "root_path_preserved": operator.root_path_preserved,
                    "worth_a_rollout": gate.worth_a_rollout,
                    "gate_diagnosis": gate.diagnosis,
                    "saturated_cell_fraction": gate.saturated_cell_fraction,
                    "route": route,
                }
            )

    recommendations = sorted(
        (
            row
            for row in rows
            if abs(float(row["target_drop_mm"]) - 1000 * CALIBRATION_TARGET_M) < 1e-9
            and row["worth_a_rollout"]
            and row["root_path_preserved"]
            and float(row["route"]["straightness"]) >= MIN_ROUTE_STRAIGHTNESS
        ),
        key=lambda row: (
            -float(row["reference_silhouette_drop_mm"]),
            float(row["saturated_cell_fraction"]),
            int(row["motion_index"]),
        ),
    )[:MAX_RECOMMENDATIONS]
    report = {
        "schema_version": "lfh_crouch_strength_screen_v1",
        "verdict_scope": "reference-side selection proxy only",
        "source_motions": len(eligible),
        "trials": len(rows),
        "passing_trials": sum(bool(row["worth_a_rollout"]) for row in rows),
        "selection_contract": {
            "station_fraction": STATION_FRACTION,
            "window_fraction": DEFAULT_WINDOW,
            "target_drops_m": list(TARGET_DROPS_M),
            "calibration_target_m": CALIBRATION_TARGET_M,
            "minimum_route_straightness": MIN_ROUTE_STRAIGHTNESS,
            "maximum_recommendations": MAX_RECOMMENDATIONS,
        },
        "calibration_recommendations": recommendations,
        "rows": rows,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"PASS: {report['passing_trials']}/{report['trials']} settings; "
        f"recommendations={[row['motion_index'] for row in recommendations]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
