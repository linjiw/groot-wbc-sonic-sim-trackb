#!/usr/bin/env python3
"""Run motion-to-scene inference over the whole gated clip pool, and say why each case lands.

The verified corpus is built from near-straight walks. This sweeps every gated clip -- curving,
turning, sideways, backward, carrying -- and records, per clip and per crouch amplitude, whether a
critical scene can be proposed and, when it cannot, which gate refused it.

It also measures the quantity that decides whether the *current* geometry can serve turning
motions at all: the angle between the world route axis the face is built on and the executed
tangent at the binding station. Faces are authored axis-aligned, so that angle is the error in the
face's orientation, and it is the argument for oriented faces rather than an opinion about them.

CPU only. Nothing here is a physics verdict.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.reach import overhead_face_reach  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    local_crouch,
    route_progress,
)  # noqa: E402
from gear_sonic.dataset_generation.reference_gate import screen_reference  # noqa: E402
from gear_sonic.dataset_generation.reference_payload import payload_from_reference  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402
from scripts.research.hallucination.screen_crouch_ladder import (  # noqa: E402
    LADDER_M,
    STATION_FRACTION,
    WINDOW_FRACTION,
)

FACE_ALONG_M = 0.10
FACE_ACROSS_M = 3.0
MIN_WINDOW_MM = 20.0
ENGINEERING_MARGIN_MM = 18.044


def route_descriptors(qpos: np.ndarray) -> dict:
    root = np.asarray(qpos[:, :2], dtype=np.float64)
    steps = np.diff(root, axis=0)
    path_length = float(np.linalg.norm(steps, axis=1).sum())
    net = float(np.linalg.norm(root[-1] - root[0]))
    heading = np.unwrap(np.arctan2(steps[:, 1], steps[:, 0]))
    progress = route_progress(root)
    index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    # Executed tangent at the binding station, from a short central difference.
    low, high = max(0, index - 3), min(len(root) - 1, index + 3)
    tangent = root[high] - root[low]
    tangent_angle = float(np.arctan2(tangent[1], tangent[0]))
    span = np.ptp(root, axis=0)
    route_axis = "x" if span[0] >= span[1] else "y"
    axis_angle = 0.0 if route_axis == "x" else math.pi / 2
    # Smallest angle between the authored (axis-aligned) face normal and the executed tangent.
    misalignment = abs(((tangent_angle - axis_angle) + math.pi / 2) % math.pi - math.pi / 2)
    return {
        "route_axis": route_axis,
        "path_length_m": path_length,
        "net_displacement_m": net,
        "straightness": net / path_length if path_length > 1e-9 else 0.0,
        "total_heading_change_deg": (
            float(np.degrees(abs(heading[-1] - heading[0]))) if len(heading) > 1 else 0.0
        ),
        "turn_sign": (
            "left"
            if len(heading) > 1 and heading[-1] - heading[0] > math.radians(10)
            else (
                "right"
                if len(heading) > 1 and heading[-1] - heading[0] < -math.radians(10)
                else "straight"
            )
        ),
        "station_frame": index,
        "station_xy_m": [float(root[index, 0]), float(root[index, 1])],
        "face_misalignment_deg": float(np.degrees(misalignment)),
    }


def _reach(qpos: np.ndarray, station, axis) -> float | None:
    try:
        face = overhead_face_reach(
            extract_keypoints(payload_from_reference(qpos)),
            station,
            axis,
            FACE_ALONG_M,
            FACE_ACROSS_M,
            require_all_groups=False,
        )
    except ValueError:
        return None
    return float(face.reach_m) if np.isfinite(face.reach_m) else None


def study_clip(index: int, path: Path, body_mode: str, predict) -> dict:
    qpos = np.loadtxt(path, delimiter=",")
    route = route_descriptors(qpos)
    station = tuple(route["station_xy_m"])
    axis = route["route_axis"]
    nominal_gate = screen_reference(qpos, f"case_{index:03d}", "walk")
    nominal_reach = _reach(qpos, station, axis)

    rungs = []
    for target in LADDER_M:
        record = {"target_drop_mm": 1000 * target}
        adapted, report = local_crouch(
            qpos, STATION_FRACTION, target_drop_m=target, window=WINDOW_FRACTION
        )
        gate = screen_reference(adapted, f"case_{index:03d}_rung", "walk")
        record["reference_drop_mm"] = 1000 * report.silhouette_drop_m
        record["excursion_capped"] = bool(report.excursion_capped)
        record["reference_gate"] = bool(gate.worth_a_rollout)
        record["root_path_preserved"] = bool(report.root_path_preserved)
        adapted_reach = _reach(adapted, station, axis)
        if nominal_reach is None or adapted_reach is None:
            record["refusal"] = "face_misses_route"
            rungs.append(record)
            continue
        raw = 1000 * (nominal_reach - adapted_reach)
        predicted = 1000 * (predict(nominal_reach) - predict(adapted_reach))
        record["raw_window_mm"] = raw
        record["predicted_window_mm"] = predicted
        record["engineering_window_mm"] = predicted - 2 * ENGINEERING_MARGIN_MM
        if not record["reference_gate"]:
            record["refusal"] = "adapted_reference_gate"
        elif record["excursion_capped"] or record["reference_drop_mm"] < 0.9 * (1000 * target):
            record["refusal"] = "operator_saturated"
        elif record["engineering_window_mm"] < MIN_WINDOW_MM:
            record["refusal"] = "window_below_min"
        else:
            record["refusal"] = None
        rungs.append(record)

    proposable = [rung for rung in rungs if rung.get("refusal") is None]
    return {
        "motion_index": index,
        "body_mode": body_mode,
        "prompt": path.stem,
        "route": route,
        "nominal_reference_gate": bool(nominal_gate.worth_a_rollout),
        "nominal_reach_m": nominal_reach,
        "rungs": rungs,
        "proposable_rungs": len(proposable),
        "best_engineering_window_mm": max(
            (rung["engineering_window_mm"] for rung in proposable), default=None
        ),
        "decision": "propose" if proposable else "refuse",
        "refusal_reasons": sorted({rung.get("refusal") for rung in rungs if rung.get("refusal")}),
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
    parser.add_argument(
        "--model", type=Path, default=REPO_ROOT / "docs/hallucination/reach_delivery_model.json"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/hallucination/case_study.json"
    )
    args = parser.parse_args()

    model = json.loads(args.model.read_text())["fitted_linear"]
    slope, intercept = model["slope"], model["intercept_m"]

    def predict(value: float) -> float:
        return slope * value + intercept

    gate = {row["index"]: row for row in json.loads(args.reference_gate.read_text())}
    clips = []
    for index, row in sorted(gate.items()):
        if not row.get("worth_a_rollout"):
            continue
        matches = sorted(args.source_dir.glob(f"{index:03d}_*.csv"))
        if len(matches) == 1:
            clips.append((index, matches[0], row.get("body_mode", "")))
    if args.limit:
        clips = clips[: args.limit]

    cases = []
    for position, (index, path, body_mode) in enumerate(clips, start=1):
        case = study_clip(index, path, body_mode, predict)
        cases.append(case)
        print(
            f"[{position}/{len(clips)}] {index:03d} {body_mode:<14} "
            f"{case['route']['turn_sign']:<8} straight={case['route']['straightness']:.3f} "
            f"misalign={case['route']['face_misalignment_deg']:5.1f} deg -> "
            f"{case['decision']} ({case['proposable_rungs']}/4)",
            flush=True,
        )

    report = {
        "schema_version": "lfh_case_study_v1",
        "clips": len(cases),
        "total_proposals_evaluated": sum(len(case["rungs"]) for case in cases),
        "clips_with_a_proposal": sum(1 for case in cases if case["decision"] == "propose"),
        "engineering_margin_mm_each_side": ENGINEERING_MARGIN_MM,
        "minimum_engineering_window_mm": MIN_WINDOW_MM,
        "contract": (
            "CPU proposal support only. A proposable rung licenses empty-scene physics for the "
            "pair; it is not a family and not a verdict."
        ),
        "cases": cases,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\n{len(cases)} clips, {report['total_proposals_evaluated']} proposals -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
