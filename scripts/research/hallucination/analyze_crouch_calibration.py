#!/usr/bin/env python3
"""Adjudicate CAL3 and run its registered finite-overhead-face search."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    sha256_file,
)
from gear_sonic.dataset_generation.hallucination.keypoints import (  # noqa: E402
    extract_keypoints,
)
from gear_sonic.dataset_generation.hallucination.propose import (  # noqa: E402
    CoverageTarget,
)
from gear_sonic.dataset_generation.hallucination.reach import (  # noqa: E402
    overhead_face_reach,
)
from gear_sonic.dataset_generation.hallucination.window import (  # noqa: E402
    WindowError,
    solve_window,
)
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    active_frames,
    route_progress,
)
from gear_sonic.dataset_generation.sweepcf_coverage import (  # noqa: E402
    margin_bucket,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

ALONG_ROUTE_GRID_M = (0.1, 0.2, 0.3, 0.4)
ACROSS_ROUTE_M = 3.0
EMPIRICAL_REPEATABILITY_ENVELOPE_MM = 18.044
MIN_WINDOW_M = 0.02
CONTACT_THRESHOLD_N = 1.0


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def render(report: dict) -> str:
    prediction = "confirmed" if report["registered_prediction_confirmed"] else "falsified"
    lines = [
        "# Crouch Executed-Window Calibration Report",
        "",
        f"The CAL3 cohort prediction is **{prediction}**. "
        f"{report['accepted_pairs']}/4 matched pairs produced accepted nominal and adapted "
        f"executions, and the largest temporally active exact head/torso window was "
        f"{report['maximum_eligible_raw_window_mm']:.2f} mm.",
        "",
        "| pair | nominal | adapted | max eligible raw window | finite-face trials | target matches |",
        "|---|---|---|---:|---:|---:|",
    ]
    for pair in report["pairs"]:
        maximum = pair["maximum_eligible_raw_window_mm"]
        lines.append(
            f"| `{pair['pair_id']}` | {pair['nominal_outcome']} | {pair['adapted_outcome']} | "
            f"{'' if maximum is None else f'{maximum:.2f} mm'} | {pair['trials']} | "
            f"{pair['target_matches']} |"
        )
    lines.extend(
        [
            "",
            f"Funnel: **15 motions -> 60 CPU settings -> 4 calibrated pairs -> 8/8 rollouts -> "
            f"{report['accepted_pairs']} accepted pairs -> {report['finite_face_trials']} "
            f"finite-face trials -> {report['scene_candidates']} target-matched scene candidates**. "
            f"Actual serial spend: **{report['actual_contended_gpu_hours']:.3f} contended GPU-h**.",
            "",
        ]
    )
    if report["scene_candidates"]:
        lines.extend(
            [
                "The SweepCF-DCS v2 audit reopens the `head_torso` × `overhead` × "
                "`1.2_1.3` × `clear_25_50` target. V1 had incorrectly marked it occupied from "
                "calibration `probe` episodes rather than canonical cells of an exact verified "
                "variant. The same immutable CAL3 executions therefore yield CPU-valid proposal "
                "support; they do not yet establish a verified family.",
                "",
                "The 18.044 mm value used here is a conservative empirical repeatability envelope "
                "from E1a, not a calibrated confidence bound. E6 applies it symmetrically to the "
                "nominal-strike and adapted-clear sides before selecting any physics cell.",
            ]
        )
    else:
        lines.extend(
            [
                "No coupling-valid empty DCS target intersects an engineering-margin window. "
                "This is a CPU funnel refusal, not an operator or trackability failure.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-record", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/coverage/targets.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/research/sweepcf_dcs_v2.json",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_record.read_text())
    candidates = json.loads(args.candidates.read_text())
    targets = json.loads(args.targets.read_text())["targets"]
    config = json.loads(args.config.read_text())
    if run["status"] != "completed" or len(run["cells"]) != 8:
        raise SystemExit("a completed eight-cell CAL3 run is required")
    if run["manifest_sha256"] != sha256_file(args.manifest):
        raise SystemExit("run record does not match the supplied immutable manifest")

    pairs = []
    trials = []
    for pair in candidates["pairs"]:
        pair_id = pair["pair_id"]
        nominal_scientific = run["cells"][f"{pair_id}__cal_nominal"]["scientific"]
        adapted_scientific = run["cells"][f"{pair_id}__cal_adapted"]["scientific"]
        pair_trials = []
        zero_external_contact = None
        if (
            nominal_scientific["outcome"] == "accepted"
            and adapted_scientific["outcome"] == "accepted"
        ):
            nominal_payload = load(Path(nominal_scientific["artifacts"]["trajectory"]))
            adapted_payload = load(Path(adapted_scientific["artifacts"]["trajectory"]))
            decompositions = (
                decompose_payload_contacts(nominal_payload),
                decompose_payload_contacts(adapted_payload),
            )
            zero_external_contact = all(
                max(value.max_lateral_contact, value.max_overhead_contact) <= CONTACT_THRESHOLD_N
                for value in decompositions
            )
            nominal = extract_keypoints(nominal_payload)
            adapted = extract_keypoints(adapted_payload)
            nominal_reference = np.loadtxt(pair["nominal_artifacts"]["csv"], delimiter=",")
            adapted_reference = np.loadtxt(pair["adapted_artifacts"]["csv"], delimiter=",")
            active = np.flatnonzero(active_frames(nominal_reference, adapted_reference))
            fraction = (
                active[0] / (len(nominal_reference) - 1),
                active[-1] / (len(nominal_reference) - 1),
            )
            temporal_nominal = (
                int(np.floor(fraction[0] * (nominal.frames - 1))),
                int(np.ceil(fraction[1] * (nominal.frames - 1))),
            )
            temporal_adapted = (
                int(np.floor(fraction[0] * (adapted.frames - 1))),
                int(np.ceil(fraction[1] * (adapted.frames - 1))),
            )
            station_frames = (
                temporal_nominal[0],
                (temporal_nominal[0] + temporal_nominal[1]) // 2,
                temporal_nominal[1],
            )
            progress = route_progress(nominal.root_pos_w[:, :2])
            route_axis = (
                "x" if np.ptp(nominal.root_pos_w[:, 0]) >= np.ptp(nominal.root_pos_w[:, 1]) else "y"
            )
            for frame in station_frames:
                station = tuple(float(value) for value in nominal.root_pos_w[frame, :2])
                for depth in ALONG_ROUTE_GRID_M:
                    orig = overhead_face_reach(nominal, station, route_axis, depth, ACROSS_ROUTE_M)
                    edit = overhead_face_reach(adapted, station, route_axis, depth, ACROSS_ROUTE_M)
                    reasons = []
                    if orig.binding_keypoint != "head_torso":
                        reasons.append("binding_anatomy_mismatch")
                    if edit.binding_keypoint != "head_torso":
                        reasons.append("edited_binding_anatomy_mismatch")
                    if not (temporal_nominal[0] <= orig.critical_frame <= temporal_nominal[1]):
                        reasons.append("orig_binding_outside_edit_interval")
                    if not (temporal_adapted[0] <= edit.critical_frame <= temporal_adapted[1]):
                        reasons.append("edit_binding_outside_edit_interval")
                    if not zero_external_contact:
                        reasons.append("calibration_external_contact")
                    solution = None
                    target = None
                    target_window_m = None
                    try:
                        full_solution = solve_window(
                            orig,
                            edit,
                            delta_clear_m=EMPIRICAL_REPEATABILITY_ENVELOPE_MM / 1000,
                            min_window_m=MIN_WINDOW_M,
                        )
                    except WindowError as error:
                        reasons.append(error.code)
                    else:
                        possible = []
                        for value in targets:
                            if (
                                value["edit_behaviour_class"] != "crouch"
                                or value["operator"] != "local_crouch"
                                or value["constraint_axis"] != "overhead"
                                or value["binding_keypoint"] != orig.binding_keypoint
                            ):
                                continue
                            parsed = CoverageTarget.from_mapping(value)
                            lower = max(full_solution.lower_m, parsed.coordinate_lower_m)
                            upper = min(full_solution.upper_m, parsed.coordinate_upper_m)
                            if upper - lower < MIN_WINDOW_M:
                                continue
                            coordinate = 0.5 * (lower + upper)
                            clearance_mm = 1000 * (coordinate - edit.reach_m)
                            if margin_bucket(clearance_mm, config) != value["margin_bucket"]:
                                continue
                            possible.append((int(value["rank"]), value, lower, upper))
                        if possible:
                            _, target, lower, upper = min(possible, key=lambda item: item[0])
                            target_window_m = upper - lower
                            solution = solve_window(
                                orig,
                                edit,
                                delta_clear_m=EMPIRICAL_REPEATABILITY_ENVELOPE_MM / 1000,
                                min_window_m=MIN_WINDOW_M,
                                hard_coordinate_m=0.5 * (lower + upper),
                            )
                        else:
                            reasons.append("coverage_target_empty")
                    trial = {
                        "pair_id": pair_id,
                        "station_progress": float(progress[frame]),
                        "station_xy_m": list(station),
                        "route_axis": route_axis,
                        "face_along_route_m": depth,
                        "face_across_route_m": ACROSS_ROUTE_M,
                        "raw_window_mm": float(1000 * (orig.reach_m - edit.reach_m)),
                        "orig_reach_m": float(orig.reach_m),
                        "edit_reach_m": float(edit.reach_m),
                        "binding_keypoint_orig": orig.binding_keypoint,
                        "binding_keypoint_edit": edit.binding_keypoint,
                        "critical_frame_orig": orig.critical_frame,
                        "critical_frame_edit": edit.critical_frame,
                        "temporal_interval_orig": list(temporal_nominal),
                        "temporal_interval_edit": list(temporal_adapted),
                        "refusal_reasons": sorted(set(reasons)),
                    }
                    if not reasons and solution is not None:
                        trial.update(
                            {
                                "certified_window_mm": 1000 * target_window_m,
                                "hard_coordinate_m": solution.hard_coordinate_m,
                                "coverage_target": target,
                            }
                        )
                    trials.append(trial)
                    pair_trials.append(trial)

        eligible = [
            trial
            for trial in pair_trials
            if not set(trial["refusal_reasons"]) - {"coverage_target_empty"}
        ]
        pairs.append(
            {
                "pair_id": pair_id,
                "nominal_outcome": nominal_scientific["outcome"],
                "adapted_outcome": adapted_scientific["outcome"],
                "adapted_rejection_reasons": adapted_scientific["rejection_reasons"],
                "zero_external_contact": zero_external_contact,
                "trials": len(pair_trials),
                "target_matches": sum(not trial["refusal_reasons"] for trial in pair_trials),
                "maximum_eligible_raw_window_mm": (
                    float(max(trial["raw_window_mm"] for trial in eligible)) if eligible else None
                ),
                "refusal_counts": dict(
                    sorted(
                        Counter(
                            reason for trial in pair_trials for reason in trial["refusal_reasons"]
                        ).items()
                    )
                ),
            }
        )

    accepted_pairs = sum(
        pair["nominal_outcome"] == "accepted"
        and pair["adapted_outcome"] == "accepted"
        and pair["zero_external_contact"] is True
        for pair in pairs
    )
    eligible_windows = [
        trial["raw_window_mm"]
        for trial in trials
        if not set(trial["refusal_reasons"]) - {"coverage_target_empty"}
    ]
    maximum = float(max(eligible_windows, default=0.0))
    scene_candidates = sum(not trial["refusal_reasons"] for trial in trials)
    report = {
        "schema_version": "lfh_crouch_calibration_v1",
        "manifest_sha256": run["manifest_sha256"],
        "accepted_pairs": accepted_pairs,
        "registered_prediction_confirmed": bool(accepted_pairs >= 2 and maximum >= 20.0),
        "maximum_eligible_raw_window_mm": maximum,
        "finite_face_trials": len(trials),
        "scene_candidates": scene_candidates,
        "coverage_target_filled": False,
        "actual_contended_gpu_hours": run["budget"]["actual_contended_gpu_hours"],
        "search_contract": {
            "station_grid": "time-mapped active-window start/median/end",
            "along_route_grid_m": list(ALONG_ROUTE_GRID_M),
            "across_route_m": ACROSS_ROUTE_M,
            "empirical_repeatability_envelope_mm": EMPIRICAL_REPEATABILITY_ENVELOPE_MM,
            "envelope_interpretation": "conservative engineering margin; no calibrated confidence level",
            "minimum_window_mm": 1000 * MIN_WINDOW_M,
            "required_binding_keypoint": "head_torso",
        },
        "pairs": pairs,
        "trials": trials,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(render(report))
    print(
        f"{'PASS' if report['registered_prediction_confirmed'] else 'REFUSED'}: "
        f"pairs={accepted_pairs}/4 max_raw={maximum:.2f}mm scenes={scene_candidates}"
    )
    return 0 if report["registered_prediction_confirmed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
