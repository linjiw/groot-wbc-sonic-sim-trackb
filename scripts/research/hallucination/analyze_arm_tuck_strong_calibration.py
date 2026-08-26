#!/usr/bin/env python3
"""Adjudicate CAL2 and extend matched-level arm-tuck delivery models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.delivery import (  # noqa: E402
    append_responses,
    fit_delivery_models,
    read_responses,
)
from scripts.research.hallucination.analyze_empty_room_probes import (  # noqa: E402
    load,
    response_rows,
)

SCENE_FLOOR_MM = 20.0


def render(report: dict) -> str:
    lines = [
        "# LFH Strong Arm-Tuck Calibration",
        "",
        "CAL2 completed its registered funnel: **5 proposed -> 4 rolled out -> 3 accepted + "
        "1 rejected -> 1 dependency skip**. Every accepted strong edit had zero external contact.",
        "",
        "| motion / side | role | alpha | outcome | commanded body window | executed body window |",
        "|---|---|---:|---|---:|---:|",
    ]
    for row in report["cells"]:
        alpha = "" if row["alpha"] is None else f"{row['alpha']:.3f}"
        commanded = (
            ""
            if row["commanded_body_window_mm"] is None
            else f"{row['commanded_body_window_mm']:.2f} mm"
        )
        executed = (
            ""
            if row["executed_body_window_mm"] is None
            else f"{row['executed_body_window_mm']:.2f} mm"
        )
        lines.append(
            f"| `{row['pair_id']}` | {row['role']} | {alpha} | {row['outcome']} | "
            f"{commanded} | {executed} |"
        )
    right = report["strong_gates"]["right"]
    lines.extend(
        [
            "",
            "## Scene-spend decision",
            "",
            "Strong-left is **refused as unidentifiable**: `094/left` nominal rejected for "
            "reference endpoint tracking and external robot contact, so alpha=2 has only one "
            "motion. The accepted `086/left` observation is retained, but it cannot license a "
            "scene or invalidate the supported CAL1 range.",
            "",
            f"Strong-right is fully replicated at alpha=2.5. At a mean wrist command of "
            f"{right['commanded_mm']:.2f} mm, the corrected monotone model estimates "
            f"{right['estimate_mm']:.2f} mm delivery with a conservative lower bound of "
            f"**{right['lower_mm']:.2f} mm**. The 20 mm gate is therefore "
            f"**{'cleared' if right['cpu_proposal_eligible'] else 'not cleared'}** for a "
            "finite-face CPU search; it does not license scene instantiation by itself.",
            "",
            f"The preregistered below-20 prediction is **{report['prediction_status']}**: the "
            "right side is adjudicated and the left side is refused for missing matched evidence. "
            "No lateral scene is instantiated from an unreplicated side or from this scalar "
            "response result alone.",
            "",
            f"The response sidecar now contains **{report['response_rows_total']} unique rows**; "
            f"{report['delivery_models_fit']} supported per-keypoint models fit. Actual serial "
            f"spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h**.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--run-record", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--cal1-run-record", required=True, type=Path)
    parser.add_argument("--v5-run-record", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--md-out", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    run = json.loads(args.run_record.read_text())
    candidates = json.loads(args.candidates.read_text())
    cal1 = json.loads(args.cal1_run_record.read_text())
    v5 = json.loads(args.v5_run_record.read_text())
    if run["status"] != "completed" or len(run["cells"]) != 5:
        raise SystemExit("a completed five-cell CAL2 run record is required")

    response_batch = []
    cells = []
    for pair in candidates["pairs"]:
        pair_id = pair["pair_id"]
        nominal_id = f"{pair_id}__cal2_nominal"
        if nominal_id in run["cells"]:
            nominal_record = run["cells"][nominal_id]
        elif f"{pair_id}__cal_nominal" in cal1["cells"]:
            nominal_record = cal1["cells"][f"{pair_id}__cal_nominal"]
        else:
            nominal_record = v5["cells"][f"{pair_id}__probe_nominal"]
        nominal_scientific = nominal_record.get("scientific")
        if nominal_id in run["cells"]:
            cells.append(
                {
                    "pair_id": pair_id,
                    "role": "nominal",
                    "alpha": None,
                    "outcome": nominal_scientific["outcome"],
                    "rejection_reasons": nominal_scientific["rejection_reasons"],
                    "commanded_body_window_mm": None,
                    "executed_body_window_mm": None,
                }
            )
        strong_id = f"{pair_id}__cal2_strong"
        strong_record = run["cells"][strong_id]
        if strong_record["status"] == "skipped_dependency":
            cells.append(
                {
                    "pair_id": pair_id,
                    "role": "strong",
                    "alpha": pair["alpha"],
                    "outcome": "skipped_dependency",
                    "rejection_reasons": ["nominal_not_trackable"],
                    "commanded_body_window_mm": 1000.0 * pair["reference_window_m"],
                    "executed_body_window_mm": None,
                }
            )
            continue
        scientific = strong_record["scientific"]
        external = scientific["diagnostics"]["contact_decomposition"][
            "max_external_contact_force_n"
        ]
        result = None
        if scientific["outcome"] == "accepted" and nominal_scientific["outcome"] == "accepted":
            if external != 0.0:
                raise SystemExit(f"{pair_id}: accepted strong cell has external contact")
            nominal = load(Path(nominal_scientific["artifacts"]["trajectory"]))
            adapted = load(Path(scientific["artifacts"]["trajectory"]))
            candidate = {
                "artifacts": {
                    "nominal_csv": pair["nominal_artifacts"]["csv"],
                    "adapted_csv": pair["adapted_artifacts"]["csv"],
                },
                "reference_window_m": pair["reference_window_m"],
                "predicted_delivered_window_m": 0.0,
                "reference_window_contract": "selection_proxy_only",
            }
            rows, result = response_rows(
                pair_id,
                pair["side"],
                nominal,
                adapted,
                candidate,
                alpha=float(pair["alpha"]),
            )
            response_batch.extend(rows)
        cells.append(
            {
                "pair_id": pair_id,
                "role": "strong",
                "alpha": pair["alpha"],
                "outcome": scientific["outcome"],
                "rejection_reasons": scientific["rejection_reasons"],
                "commanded_body_window_mm": 1000.0 * pair["reference_window_m"],
                "executed_body_window_mm": (
                    None if result is None else result["executed_body_window_mm"]
                ),
            }
        )

    existing = read_responses(args.responses)
    existing_keys = {(row.motion_id, row.keypoint, row.axis, row.alpha) for row in existing}
    new_rows = [
        row
        for row in response_batch
        if (row.motion_id, row.keypoint, row.axis, row.alpha) not in existing_keys
    ]
    append_responses(args.responses, new_rows)
    all_rows = read_responses(args.responses)
    models, refusals = fit_delivery_models(all_rows)

    gates = {}
    for side in ("left", "right"):
        key = ("local_arm_tuck", f"wrist_{side}", f"lateral_{side}")
        model = models[key]
        strong_alpha = 2.0 if side == "left" else 2.5
        if strong_alpha in model.unsupported_alpha_levels:
            gates[side] = {
                "status": "refused_insufficient_matched_motions",
                "alpha": strong_alpha,
                "cpu_proposal_eligible": False,
            }
            continue
        level_index = model.alpha_levels.index(strong_alpha)
        command = model.commanded_mm[level_index]
        estimate, lower, upper = model.predict(command)
        gates[side] = {
            "status": "adjudicated",
            "alpha": strong_alpha,
            "commanded_mm": command,
            "estimate_mm": estimate,
            "lower_mm": lower,
            "upper_mm": upper,
            "residual_q90_mm": model.residual_q90_mm,
            "cpu_proposal_eligible": lower >= SCENE_FLOOR_MM,
        }

    right_confirmed = not gates["right"]["cpu_proposal_eligible"]
    report = {
        "schema_version": "lfh_arm_tuck_strong_calibration_v1",
        "manifest_sha256": run["manifest_sha256"],
        "funnel": {
            "proposed": len(manifest["cells"]),
            "rolled_out": sum(cell["status"] == "completed" for cell in run["cells"].values()),
            "accepted": sum(
                cell.get("scientific", {}).get("outcome") == "accepted"
                for cell in run["cells"].values()
            ),
            "rejected": sum(
                cell.get("scientific", {}).get("outcome") == "rejected"
                for cell in run["cells"].values()
            ),
            "skipped_dependency": sum(
                cell["status"] == "skipped_dependency" for cell in run["cells"].values()
            ),
        },
        "cells": cells,
        "response_rows_cal2": len(response_batch),
        "response_rows_total": len(all_rows),
        "delivery_models_fit": len(models),
        "delivery_model_refusals": {"|".join(key): value for key, value in refusals.items()},
        "strong_gates": gates,
        "prediction_status": "partially confirmed" if right_confirmed else "partially falsified",
        "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"PASS: funnel={report['funnel']} right_lower={gates['right']['lower_mm']:.2f} "
        f"right_eligible={gates['right']['cpu_proposal_eligible']} "
        f"left={gates['left']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
