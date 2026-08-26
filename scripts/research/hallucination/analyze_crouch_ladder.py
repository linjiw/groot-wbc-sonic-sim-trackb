#!/usr/bin/env python3
"""Adjudicate LFH-E12 against its registered predictions.

Reads the immutable run record, recovers each motion's delivered amplitude, computes the executed
critical window for every accepted pair, and reports each registered prediction as confirmed or
falsified. It never re-grades a cell: outcomes are copied from the record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import route_progress  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
ENGINEERING_MARGIN_M = 0.018044
MIN_WINDOW_MM = 20.0
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 3.0
STATION_FRACTION = 0.55


def _tracks(path: Path):
    result = best_evaluable_payload(pickle.load(Path(path).open("rb")))
    return extract_keypoints(result[0] if isinstance(result, tuple) else result)


def _station_and_axis(tracks) -> tuple[tuple[float, float], str]:
    root = tracks.root_pos_w[:, :2]
    span = np.ptp(root, axis=0)
    axis = "x" if span[0] >= span[1] else "y"
    progress = route_progress(np.asarray(root, dtype=np.float64))
    index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    return (float(root[index, 0]), float(root[index, 1])), axis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-record",
        type=Path,
        default=DATA_ROOT / "hallucination/run_records/E12_CROUCH_LADDER_2026-08-26.json",
    )
    parser.add_argument(
        "--candidates", type=Path, default=DATA_ROOT / "lfh_crouch_ladder/candidates.json"
    )
    parser.add_argument(
        "--json-out", type=Path, default=REPO_ROOT / "docs/hallucination/e12_crouch_ladder.json"
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/REPORT_E12_CROUCH_LADDER.md",
    )
    args = parser.parse_args()

    record = json.loads(args.run_record.read_text())
    candidates = json.loads(args.candidates.read_text())
    cells = record["cells"]

    def outcome(cell_id: str) -> str | None:
        cell = cells.get(cell_id, {})
        if cell.get("status") == "skipped_dependency":
            return "skipped"
        return (cell.get("scientific") or {}).get("outcome")

    def trajectory(cell_id: str) -> str | None:
        return ((cells.get(cell_id, {}).get("scientific") or {}).get("artifacts") or {}).get(
            "trajectory"
        )

    motions = []
    for pair in candidates["pairs"]:
        pair_id = pair["pair_id"]
        nominal_outcome = outcome(f"{pair_id}__nominal")
        rungs = []
        for rung in pair["rungs"]:
            cell_id = f"{pair_id}__{rung['label']}"
            rungs.append(
                {
                    "label": rung["label"],
                    "commanded_drop_mm": 1000 * rung["target_drop_m"],
                    "reference_drop_mm": 1000 * rung["reference_silhouette_drop_m"],
                    "lateral_coupling_mm": rung["lateral_coupling_mm"],
                    "outcome": outcome(cell_id),
                    "cell_id": cell_id,
                }
            )
        accepted = [rung for rung in rungs if rung["outcome"] == "accepted"]
        delivered = max((rung["commanded_drop_mm"] for rung in accepted), default=None)
        # Monotonicity is observable because every rung depends on the nominal alone.
        graded = [rung for rung in rungs if rung["outcome"] in {"accepted", "rejected"}]
        graded.sort(key=lambda rung: rung["commanded_drop_mm"])
        non_monotone = any(
            graded[i]["outcome"] == "rejected" and graded[j]["outcome"] == "accepted"
            for i in range(len(graded))
            for j in range(i + 1, len(graded))
        )

        window = None
        if delivered is not None and nominal_outcome == "accepted":
            deepest = max(accepted, key=lambda rung: rung["commanded_drop_mm"])
            nominal_traj = trajectory(f"{pair_id}__nominal")
            adapted_traj = trajectory(deepest["cell_id"])
            if nominal_traj and adapted_traj:
                nominal_tracks = _tracks(Path(nominal_traj))
                adapted_tracks = _tracks(Path(adapted_traj))
                station, axis = _station_and_axis(nominal_tracks)
                try:
                    nominal_face = overhead_face_reach(
                        nominal_tracks,
                        station,
                        axis,
                        FACE_ALONG_M,
                        FACE_ACROSS_M,
                        require_all_groups=False,
                    )
                    adapted_face = overhead_face_reach(
                        adapted_tracks,
                        station,
                        axis,
                        FACE_ALONG_M,
                        FACE_ACROSS_M,
                        require_all_groups=False,
                    )
                except ValueError:
                    nominal_face = adapted_face = None
                if nominal_face is not None and np.isfinite(nominal_face.reach_m):
                    raw = nominal_face.reach_m - adapted_face.reach_m
                    window = {
                        "at_commanded_drop_mm": deepest["commanded_drop_mm"],
                        "nominal_reach_m": float(nominal_face.reach_m),
                        "adapted_reach_m": float(adapted_face.reach_m),
                        "raw_window_mm": 1000 * float(raw),
                        "engineering_window_mm": 1000 * float(raw - 2 * ENGINEERING_MARGIN_M),
                        "binding_keypoint": nominal_face.binding_keypoint,
                        "enters_p_feas": bool(
                            1000 * (raw - 2 * ENGINEERING_MARGIN_M) >= MIN_WINDOW_MM
                        ),
                    }

        motions.append(
            {
                "pair_id": pair_id,
                "motion_index": pair["motion_index"],
                "body_mode": pair["body_mode"],
                "straightness": pair["route"]["straightness"],
                "nominal_outcome": nominal_outcome,
                "rungs": rungs,
                "delivered_amplitude_mm": delivered,
                "non_monotone": non_monotone,
                "window": window,
            }
        )

    sourced = [
        m for m in motions if m["nominal_outcome"] == "accepted" and m["delivered_amplitude_mm"]
    ]
    delivered = [m["delivered_amplitude_mm"] for m in sourced]
    entering = [m for m in sourced if m["window"] and m["window"]["enters_p_feas"]]
    coupled = [
        m
        for m in motions
        if m["delivered_amplitude_mm"] and max(r["lateral_coupling_mm"] for r in m["rungs"]) > 20.0
    ]
    coupled_rejects = [
        m
        for m in motions
        if any(r["lateral_coupling_mm"] > 20.0 and r["outcome"] == "rejected" for r in m["rungs"])
    ]

    verdicts = {
        "P1_at_least_8_new_sources": {
            "predicted": ">= 8 clips accept the nominal and at least one rung",
            "observed": len(sourced),
            "confirmed": len(sourced) >= 8,
        },
        "P2_median_delivered_between_40_and_70": {
            "predicted": "median largest-accepted amplitude in [40, 70] mm",
            "observed_median_mm": float(np.median(delivered)) if delivered else None,
            "confirmed": bool(delivered) and 40.0 <= float(np.median(delivered)) <= 70.0,
        },
        "P3_monotone_within_motion": {
            "predicted": "no motion accepts a deeper rung after rejecting a shallower one",
            "violations": [m["pair_id"] for m in motions if m["non_monotone"]],
            "confirmed": not any(m["non_monotone"] for m in motions),
        },
        "P4_at_least_6_clear_20mm_window": {
            "predicted": ">= 6 accepted pairs clear a 20 mm engineering window",
            "observed": len(entering),
            "confirmed": len(entering) >= 6,
        },
        "P5_lateral_coupling_predicts_rejection": {
            "predicted": "lateral coupling above 20 mm predicts rejection",
            "motions_with_coupling_above_20mm": [m["pair_id"] for m in coupled],
            "of_those_rejected_at_that_rung": [m["pair_id"] for m in coupled_rejects],
            "confirmed": None,
        },
    }

    report = {
        "schema_version": "lfh_e12_crouch_ladder_v1",
        "run_record": str(args.run_record),
        "cohort": len(motions),
        "cells_graded": sum(
            1 for m in motions for r in m["rungs"] if r["outcome"] in {"accepted", "rejected"}
        )
        + sum(1 for m in motions if m["nominal_outcome"] in {"accepted", "rejected"}),
        "nominals_accepted": sum(1 for m in motions if m["nominal_outcome"] == "accepted"),
        "motions_with_a_delivered_amplitude": len(sourced),
        "motions_entering_p_feas": len(entering),
        "engineering_margin_m_each_side": ENGINEERING_MARGIN_M,
        "predictions": verdicts,
        "motions": motions,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    lines = [
        "# LFH-E12 — Crouch Amplitude Ladder",
        "",
        f"Cohort of {len(motions)} motions across "
        f"{len({m['body_mode'] for m in motions})} body modes, drawn from the whole gated clip "
        "pool rather than one body mode. Registered before spend in "
        "`docs/prediction_register.md`.",
        "",
        "| motion | mode | straight | nominal | "
        + " | ".join(r["label"] for r in motions[0]["rungs"])
        + " | delivered | raw window | eng. window |",
        "|---|---|---:|---|" + "---|" * len(motions[0]["rungs"]) + "---:|---:|---:|",
    ]
    for m in motions:
        by_label = {r["label"]: r["outcome"] for r in m["rungs"]}
        cells_text = " | ".join((by_label.get(r["label"]) or "-")[:8] for r in motions[0]["rungs"])
        window = m["window"]
        lines.append(
            f"| `{m['pair_id']}` | {m['body_mode']} | {m['straightness']:.3f} | "
            f"{m['nominal_outcome'] or '-'} | {cells_text} | "
            f"{(str(int(m['delivered_amplitude_mm'])) + ' mm') if m['delivered_amplitude_mm'] else '-'} | "
            f"{window['raw_window_mm']:.1f} mm | {window['engineering_window_mm']:.1f} mm |"
            if window
            else f"| `{m['pair_id']}` | {m['body_mode']} | {m['straightness']:.3f} | "
            f"{m['nominal_outcome'] or '-'} | {cells_text} | "
            f"{(str(int(m['delivered_amplitude_mm'])) + ' mm') if m['delivered_amplitude_mm'] else '-'} | - | - |"
        )
    lines.extend(["", "## Registered predictions", ""])
    for name, verdict in verdicts.items():
        state = (
            "not adjudicated"
            if verdict["confirmed"] is None
            else ("**confirmed**" if verdict["confirmed"] else "**falsified**")
        )
        lines.append(
            f"- `{name}` — {state}. {json.dumps({k: v for k, v in verdict.items() if k != 'confirmed'})}"
        )
    args.md_out.write_text("\n".join(lines) + "\n")

    print(f"cohort {len(motions)}; nominals accepted {report['nominals_accepted']}")
    print(f"motions with a delivered amplitude: {len(sourced)}")
    print(f"motions entering P_feas (>= {MIN_WINDOW_MM:.0f} mm): {len(entering)}")
    for name, verdict in verdicts.items():
        print(f"  {name}: {verdict['confirmed']}")
    print(f"-> {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
