#!/usr/bin/env python3
"""Search CAL2 executions for temporally valid, noise-certified lateral-gap scenes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import (  # noqa: E402
    ConstraintSpec,
    MotionEvidence,
    sha256_file,
)
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate  # noqa: E402
from gear_sonic.dataset_generation.hallucination.keypoints import extract_keypoints  # noqa: E402
from gear_sonic.dataset_generation.hallucination.propose import CoverageTarget  # noqa: E402
from gear_sonic.dataset_generation.hallucination.reach import lateral_gap_reach  # noqa: E402
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    validate_pair,
    write_report,
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
    coordinate_bucket,
    margin_bucket,
)
from scripts.research.hallucination.analyze_empty_room_probes import load  # noqa: E402

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
CAL1_RUN = DATA_ROOT / "hallucination/run_records/ARM_TUCK_CALIBRATION_2026-08-20.json"
CAL2_RUN = DATA_ROOT / "hallucination/run_records/ARM_TUCK_STRONG_CALIBRATION_2026-08-20.json"
V5_RUN = DATA_ROOT / "hallucination/run_records/EMPTY_ROOM_PROBES_V5_2026-08-20.json"
CANDIDATES = DATA_ROOT / "lfh_arm_tuck_strong_calibration/candidates.json"
SCREEN_EMPTY = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
NOISE_FLOOR_MM = 18.044
MIN_CERTIFIED_WINDOW_M = 0.020
ALONG_ROUTE_GRID_M = (0.10, 0.20, 0.30, 0.40)
RIGHT_ANATOMY = frozenset({"shoulder_right", "wrist_right"})


def _motion_evidence(
    motion_id: str,
    trajectory: Path,
    run_record: Path,
    cell_id: str,
) -> MotionEvidence:
    return MotionEvidence(
        motion_id,
        str(trajectory),
        sha256_file(trajectory),
        "screen_empty",
        "local_arm_tuck",
        str(run_record),
        sha256_file(run_record),
        cell_id,
    )


def _crossing_interval(tracks, station: tuple[float, float], route_axis: str, depth: float):
    axis = 0 if route_axis == "x" else 1
    frames = np.flatnonzero(np.abs(tracks.root_pos_w[:, axis] - station[axis]) <= depth / 2)
    if not len(frames):
        raise ValueError("executed root never enters the candidate face slab")
    return int(frames[0]), int(frames[-1])


def _render(report: dict) -> str:
    selected = report["selected"]
    lines = [
        "# LFH Lateral-Gap CPU Synthesis",
        "",
        "CAL2's strong-right delivery result licensed this CPU search, not scene physics. The "
        "search uses the full separation of two world-fixed faces, every collision capsule, and "
        "the operator's time-mapped active interval.",
        "",
        f"The fixed grid evaluated **{report['funnel']['trials']} trials**. "
        f"**{report['funnel']['certified']}** survived anatomy, temporal, 20 mm window, and "
        "2x-noise-clearance gates; one best station per motion was eligible for selection.",
        "",
        "| motion | status | station progress | face depth | raw window | certified window | binder |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    by_motion = {entry["motion_id"]: entry for entry in selected}
    for motion in report["motions"]:
        chosen = by_motion.get(motion["motion_id"])
        if chosen is None:
            lines.append(
                f"| `{motion['motion_id']}` | refused |  |  |  |  | "
                f"{motion['summary_reason']} |"
            )
        else:
            lines.append(
                f"| `{chosen['motion_id']}` | CPU-certified | "
                f"{chosen['station_progress']:.4f} | {chosen['face_along_route_m']:.2f} m | "
                f"{chosen['raw_window_mm']:.2f} mm | {chosen['certified_window_mm']:.2f} mm | "
                f"`{chosen['binding_keypoint']}` |"
            )
    lines.extend(
        [
            "",
            "The selected `084/right` pinch-panel pair fills a configured DCS target and "
            "reproduces the required CPU sign pattern "
            "(easy: clear/clear; hard: strike/clear). Its panels extend from each inner face by "
            "the declared across-route extent, closing the earlier skirt path. `092/right` is "
            "refused because its apparent early-route windows bind before the strong edit becomes "
            "active; the central face has no positive all-body window.",
            "",
            "No generated-scene verdict is claimed here. Physics remains the next gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-package",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e3",
    )
    parser.add_argument("--spec-dir", type=Path, default=REPO_ROOT / "specs/hallucination")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e3_lateral_cpu.json",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/REPORT_E3_LATERAL_CPU.md",
    )
    args = parser.parse_args()

    runs = {
        "cal1": json.loads(CAL1_RUN.read_text()),
        "cal2": json.loads(CAL2_RUN.read_text()),
        "v5": json.loads(V5_RUN.read_text()),
    }
    pairs = {
        pair["motion_index"]: pair
        for pair in json.loads(CANDIDATES.read_text())["pairs"]
        if pair["side"] == "right"
    }
    nominal_sources = {
        84: ("cal1", CAL1_RUN, "lfh_084_arm_tuck_right__cal_nominal"),
        92: ("v5", V5_RUN, "lfh_092_arm_tuck_right__probe_nominal"),
    }
    screen = read_stage_geometry(SCREEN_EMPTY)
    room_size = screen.room_size_xy_m
    if room_size is None:
        raise SystemExit("screen_empty has no measurable room size")
    config = json.loads((REPO_ROOT / "configs/research/sweepcf_dcs_v1.json").read_text())
    targets = json.loads((REPO_ROOT / "docs/hallucination/coverage/targets.json").read_text())[
        "targets"
    ]
    trials: list[dict[str, object]] = []
    motion_records = []
    certified = []
    for index in sorted(pairs):
        pair = pairs[index]
        source_key, source_path, nominal_cell = nominal_sources[index]
        strong_cell = f"{pair['pair_id']}__cal2_strong"
        nominal_scientific = runs[source_key]["cells"][nominal_cell]["scientific"]
        strong_scientific = runs["cal2"]["cells"][strong_cell]["scientific"]
        nominal_path = Path(nominal_scientific["artifacts"]["trajectory"])
        strong_path = Path(strong_scientific["artifacts"]["trajectory"])
        nominal = extract_keypoints(load(nominal_path))
        strong = extract_keypoints(load(strong_path))
        nominal_progress = route_progress(nominal.root_pos_w[:, :2])
        route_axis = (
            "x" if np.ptp(nominal.root_pos_w[:, 0]) >= np.ptp(nominal.root_pos_w[:, 1]) else "y"
        )

        nominal_reference = np.loadtxt(pair["nominal_artifacts"]["csv"], delimiter=",")
        strong_reference = np.loadtxt(pair["adapted_artifacts"]["csv"], delimiter=",")
        active = np.flatnonzero(active_frames(nominal_reference, strong_reference))
        temporal_fraction = (
            active[0] / (len(nominal_reference) - 1),
            active[-1] / (len(nominal_reference) - 1),
        )
        temporal_nominal = (
            int(np.floor(temporal_fraction[0] * (nominal.frames - 1))),
            int(np.ceil(temporal_fraction[1] * (nominal.frames - 1))),
        )
        temporal_strong = (
            int(np.floor(temporal_fraction[0] * (strong.frames - 1))),
            int(np.ceil(temporal_fraction[1] * (strong.frames - 1))),
        )
        station_frames = (
            temporal_nominal[0],
            (temporal_nominal[0] + temporal_nominal[1]) // 2,
            temporal_nominal[1],
        )

        motion_trials = []
        for frame in station_frames:
            station_progress = float(nominal_progress[frame])
            station = tuple(float(value) for value in nominal.root_pos_w[frame, :2])
            for depth in ALONG_ROUTE_GRID_M:
                orig = lateral_gap_reach(nominal, station, route_axis, depth, (0.0, 2.0))
                edit = lateral_gap_reach(strong, station, route_axis, depth, (0.0, 2.0))
                reasons = []
                if orig.binding_keypoint not in RIGHT_ANATOMY:
                    reasons.append("binding_anatomy_mismatch")
                if edit.binding_keypoint not in RIGHT_ANATOMY:
                    reasons.append("edited_binding_anatomy_mismatch")
                if not (temporal_nominal[0] <= orig.critical_frame <= temporal_nominal[1]):
                    reasons.append("orig_binding_outside_edit_interval")
                if not (temporal_strong[0] <= edit.critical_frame <= temporal_strong[1]):
                    reasons.append("edit_binding_outside_edit_interval")
                solution = None
                target = None
                target_window_m = None
                try:
                    full_solution = solve_window(
                        orig,
                        edit,
                        delta_clear_m=2 * NOISE_FLOOR_MM / 1000,
                        min_window_m=MIN_CERTIFIED_WINDOW_M,
                        easy_coordinate_m=orig.reach_m + 0.100,
                    )
                except WindowError as error:
                    reasons.append(error.code)
                else:
                    possible_targets = []
                    for value in targets:
                        if (
                            value["edit_behaviour_class"] != "arms"
                            or value["operator"] != "local_arm_tuck"
                            or value["constraint_axis"] != "lateral_gap"
                            or value["binding_keypoint"] != orig.binding_keypoint
                        ):
                            continue
                        parsed = CoverageTarget.from_mapping(value)
                        lower = max(full_solution.lower_m, parsed.coordinate_lower_m)
                        upper = min(full_solution.upper_m, parsed.coordinate_upper_m)
                        if upper - lower < MIN_CERTIFIED_WINDOW_M:
                            continue
                        coordinate = 0.5 * (lower + upper)
                        approximate_face_clearance_mm = 500 * (coordinate - edit.reach_m)
                        if (
                            margin_bucket(approximate_face_clearance_mm, config)
                            != value["margin_bucket"]
                        ):
                            continue
                        possible_targets.append((int(value["rank"]), value, lower, upper))
                    if possible_targets:
                        _, target, lower, upper = min(possible_targets, key=lambda value: value[0])
                        target_window_m = upper - lower
                        solution = solve_window(
                            orig,
                            edit,
                            delta_clear_m=2 * NOISE_FLOOR_MM / 1000,
                            min_window_m=MIN_CERTIFIED_WINDOW_M,
                            hard_coordinate_m=0.5 * (lower + upper),
                            easy_coordinate_m=orig.reach_m + 0.100,
                        )
                    else:
                        reasons.append("coverage_target_empty")
                trial = {
                    "motion_id": pair["pair_id"],
                    "station_progress": station_progress,
                    "station_xy_m": list(station),
                    "route_axis": route_axis,
                    "face_along_route_m": depth,
                    "raw_window_mm": 1000 * (orig.reach_m - edit.reach_m),
                    "binding_keypoint_orig": orig.binding_keypoint,
                    "binding_keypoint_edit": edit.binding_keypoint,
                    "critical_frame_orig": orig.critical_frame,
                    "critical_frame_edit": edit.critical_frame,
                    "temporal_interval_orig": list(temporal_nominal),
                    "temporal_interval_edit": list(temporal_strong),
                    "refusal_reasons": sorted(set(reasons)),
                }
                if not reasons and solution is not None:
                    trial["full_certified_window_mm"] = 1000 * solution.width_m
                    trial["certified_window_mm"] = 1000 * target_window_m
                    trial["hard_coordinate_m"] = solution.hard_coordinate_m
                    trial["coverage_target"] = target
                    certified.append(
                        (
                            int(target["rank"]),
                            target_window_m,
                            index,
                            pair,
                            nominal_path,
                            strong_path,
                            source_path,
                            nominal_cell,
                            strong_cell,
                            nominal,
                            strong,
                            orig,
                            edit,
                            solution,
                            target,
                            trial,
                        )
                    )
                trials.append(trial)
                motion_trials.append(trial)
        refusals = Counter(reason for trial in motion_trials for reason in trial["refusal_reasons"])
        motion_records.append(
            {
                "motion_id": pair["pair_id"],
                "trials": len(motion_trials),
                "certified": sum(not trial["refusal_reasons"] for trial in motion_trials),
                "refusal_counts": dict(sorted(refusals.items())),
                "summary_reason": (
                    "no temporally active, same-side, noise-certified full-gap window"
                ),
            }
        )

    best_by_motion = {}
    for entry in certified:
        index = entry[2]
        if index not in best_by_motion or (entry[0], -entry[1]) < (
            best_by_motion[index][0],
            -best_by_motion[index][1],
        ):
            best_by_motion[index] = entry

    args.out_package.mkdir(parents=True, exist_ok=True)
    args.spec_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    for index, entry in sorted(best_by_motion.items()):
        (
            _,
            _,
            _,
            pair,
            nominal_path,
            strong_path,
            source_path,
            nominal_cell,
            strong_cell,
            nominal,
            strong,
            orig,
            edit,
            solution,
            target,
            trial,
        ) = entry
        spec = ConstraintSpec(
            spec_id=f"cs_{pair['pair_id']}_cal2",
            source_kind="inverse_synthesis",
            source_family_id=pair["pair_id"],
            source_variant_id="cal2_alpha_2p5",
            orig=_motion_evidence(
                f"{pair['pair_id']}_nominal", nominal_path, source_path, nominal_cell
            ),
            edit=_motion_evidence(f"{pair['pair_id']}_strong", strong_path, CAL2_RUN, strong_cell),
            crossing_frames_orig=_crossing_interval(
                nominal,
                tuple(trial["station_xy_m"]),
                str(trial["route_axis"]),
                float(trial["face_along_route_m"]),
            ),
            crossing_frames_edit=_crossing_interval(
                strong,
                tuple(trial["station_xy_m"]),
                str(trial["route_axis"]),
                float(trial["face_along_route_m"]),
            ),
            axis_type="lateral_gap",
            expected_link_group=solution.binding_keypoint,
            reach_orig_m=orig.reach_m,
            reach_edit_m=edit.reach_m,
            window_mm=1000 * (orig.reach_m - edit.reach_m),
            easy_coordinate_m=solution.easy_coordinate_m,
            hard_coordinate_m=solution.hard_coordinate_m,
            binding_station_xy_m=tuple(trial["station_xy_m"]),
            route_axis=str(trial["route_axis"]),
            face_along_route_m=float(trial["face_along_route_m"]),
            face_across_route_m=(room_size[1] if trial["route_axis"] == "x" else room_size[0]),
            face_vertical_m=2.0,
            binding_thickness_m=0.12,
            keepout_delta_mm=50.0,
            room_size_xy_m=room_size,
            wall_height_m=2.8,
            source_scene_easy="screen_empty",
            source_scene_hard="screen_empty",
            binding_keypoint=solution.binding_keypoint,
            critical_frame_orig=solution.critical_frame_orig,
            critical_frame_edit=solution.critical_frame_edit,
            per_keypoint_margins_m=solution.per_keypoint_margins_m,
        )
        spec_path = args.spec_dir / f"{spec.spec_id}.json"
        spec.save(spec_path)
        seed = 31500 + index
        result = instantiate(spec, "pinch_panels", seed, args.out_package)
        keepout = validate_pair(spec, result.easy.path, result.hard.path)
        keepout_path = args.out_package / (
            f"{spec.spec_id}__pinch_panels__s{seed:08d}.keepout.json"
        )
        write_report(keepout, keepout_path)
        if not keepout["ok"]:
            raise SystemExit(f"{spec.spec_id}: Tier-2 preflight refused")
        pair_manifest = json.loads(result.manifest_path.read_text())
        pair_manifest["cpu_certificate"] = {
            "keepout_report": str(keepout_path),
            "binding_geometry_signs": keepout["binding_geometry_signs"],
            "four_sign_pattern_matches": True,
        }
        result.manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")
        hard_edit_clearance = keepout["binding_geometry_pattern"]["hard"]["edit"]["clearance_mm"]
        coordinate = coordinate_bucket("lateral_gap", spec.hard_coordinate_m, config)
        margin = margin_bucket(hard_edit_clearance, config)
        if (
            coordinate != target["constraint_coordinate_bucket"]
            or margin != target["margin_bucket"]
        ):
            raise SystemExit(
                f"{spec.spec_id}: exact Tier-2 geometry missed registered coverage target "
                f"{coordinate}/{margin} != {target['constraint_coordinate_bucket']}/"
                f"{target['margin_bucket']}"
            )
        selected.append(
            {
                "motion_id": pair["pair_id"],
                "station_progress": trial["station_progress"],
                "face_along_route_m": trial["face_along_route_m"],
                "raw_window_mm": spec.window_mm,
                "certified_window_mm": trial["certified_window_mm"],
                "full_certified_window_mm": 1000 * solution.width_m,
                "binding_keypoint": spec.binding_keypoint,
                "hard_coordinate_m": spec.hard_coordinate_m,
                "hard_edit_clearance_mm": hard_edit_clearance,
                "coverage_target": target,
                "spec": str(spec_path.relative_to(REPO_ROOT)),
                "spec_sha256": sha256_file(spec_path),
                "pair_manifest": str(result.manifest_path.relative_to(REPO_ROOT)),
                "pair_manifest_sha256": sha256_file(result.manifest_path),
                "keepout": str(keepout_path.relative_to(REPO_ROOT)),
                "keepout_sha256": sha256_file(keepout_path),
                "scenes": {
                    "easy": str(result.easy.path.relative_to(REPO_ROOT)),
                    "hard": str(result.hard.path.relative_to(REPO_ROOT)),
                },
                "scene_sha256": {
                    "easy": result.easy.sha256,
                    "hard": result.hard.sha256,
                },
                "binding_geometry_clearance_mm": {
                    cell: {motion: value["clearance_mm"] for motion, value in values.items()}
                    for cell, values in keepout["binding_geometry_pattern"].items()
                },
            }
        )

    report = {
        "schema_version": "lfh_e3_lateral_cpu_v1",
        "physics_executed": False,
        "search_contract": {
            "station_grid": "time-mapped executed active-window start/median/end",
            "along_route_grid_m": list(ALONG_ROUTE_GRID_M),
            "vertical_band_m": [0.0, 2.0],
            "coordinate": "full symmetric world-fixed gap width",
            "required_binding_anatomy": sorted(RIGHT_ANATOMY),
            "noise_floor_mm": NOISE_FLOOR_MM,
            "delta_clear_gap_mm": 2 * NOISE_FLOOR_MM,
            "easy_gap_margin_mm": 100.0,
            "minimum_certified_window_mm": 1000 * MIN_CERTIFIED_WINDOW_M,
        },
        "funnel": {
            "motions": len(pairs),
            "trials": len(trials),
            "certified": sum(not trial["refusal_reasons"] for trial in trials),
            "selected": len(selected),
        },
        "motions": motion_records,
        "trials": trials,
        "selected": selected,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    args.md_out.write_text(_render(report))
    print(f"PASS: funnel={report['funnel']} -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
