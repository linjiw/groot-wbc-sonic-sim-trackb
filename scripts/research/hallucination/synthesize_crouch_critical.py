#!/usr/bin/env python3
"""Materialize the E6 canonical crouch support with exact CAL3 executed evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
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
from gear_sonic.dataset_generation.hallucination.keypoints import (  # noqa: E402
    extract_keypoints,
)
from gear_sonic.dataset_generation.hallucination.reach import (  # noqa: E402
    overhead_face_reach,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    validate_pair,
    write_report,
)
from gear_sonic.dataset_generation.hallucination.window import solve_window  # noqa: E402
from gear_sonic.dataset_generation.sweepcf_coverage import (  # noqa: E402
    coordinate_bucket,
    margin_bucket,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
CALIBRATION_RUN = DATA_ROOT / "hallucination/run_records/CROUCH_CALIBRATION_2026-08-20.json"
CANDIDATES = DATA_ROOT / "lfh_crouch_calibration/candidates.json"
SCREEN_EMPTY = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
ENGINEERING_MARGIN_M = 0.018044
MINIMUM_ENGINEERING_WIDTH_M = 0.020


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory segment")
    return payload


def crossing_interval(tracks, station: tuple[float, float], route_axis: str, depth: float):
    axis = 0 if route_axis == "x" else 1
    frames = np.flatnonzero(np.abs(tracks.root_pos_w[:, axis] - station[axis]) <= depth / 2)
    if not len(frames):
        raise ValueError("executed root never enters the selected face slab")
    return int(frames[0]), int(frames[-1])


def motion_evidence(
    motion_id: str,
    trajectory: Path,
    cell_id: str,
) -> MotionEvidence:
    return MotionEvidence(
        motion_id=motion_id,
        artifact_path=str(trajectory),
        artifact_sha256=sha256_file(trajectory),
        scene_id="screen_empty",
        operator="local_crouch",
        adjudication_path=str(CALIBRATION_RUN),
        adjudication_sha256=sha256_file(CALIBRATION_RUN),
        adjudication_cell_id=cell_id,
    )


def matching_trial(calibration: dict, selected: dict) -> dict:
    matches = [
        trial
        for trial in calibration["trials"]
        if trial["pair_id"] == selected["source_pair_id"]
        and abs(float(trial["station_progress"]) - float(selected["route_progress"])) < 1e-9
        and abs(float(trial["face_along_route_m"]) - float(selected["face_along_route_m"])) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(f"{selected['record_id']}: expected one CAL3 trial, found {len(matches)}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--support",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/critical_support_cal3.json",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/crouch_calibration.json",
    )
    parser.add_argument(
        "--out-package",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e6",
    )
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=REPO_ROOT / "specs/hallucination",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e6_crouch_cpu.json",
    )
    parser.add_argument(
        "--record-id",
        action="append",
        help="materialize an explicit support record instead of the default widest-per-source set",
    )
    parser.add_argument("--spec-suffix", default="cal3")
    parser.add_argument("--seed-base", type=int, default=31700)
    args = parser.parse_args()

    support = json.loads(args.support.read_text())
    calibration = json.loads(args.calibration.read_text())
    run = json.loads(CALIBRATION_RUN.read_text())
    candidate_pairs = {
        pair["pair_id"]: pair for pair in json.loads(CANDIDATES.read_text())["pairs"]
    }
    config = json.loads((REPO_ROOT / "configs/research/sweepcf_dcs_v2.json").read_text())
    targets = json.loads((REPO_ROOT / "docs/hallucination/coverage/targets.json").read_text())[
        "targets"
    ]
    screen = read_stage_geometry(SCREEN_EMPTY)
    if screen.room_size_xy_m is None:
        raise SystemExit("screen_empty has no measurable room size")

    args.out_package.mkdir(parents=True, exist_ok=True)
    args.spec_dir.mkdir(parents=True, exist_ok=True)
    if args.record_id:
        by_id = {row["record_id"]: row for row in support["support"]}
        missing = set(args.record_id) - set(by_id)
        if missing:
            raise SystemExit(f"unknown critical-support record(s): {sorted(missing)}")
        support_rows = [
            {
                **by_id[record_id],
                "easy_coordinate_m": by_id[record_id]["nominal_reach_m"] + 0.050,
            }
            for record_id in args.record_id
        ]
    else:
        support_rows = support["selected"]

    selected_output = []
    for selected in support_rows:
        pair_id = selected["source_pair_id"]
        pair = candidate_pairs[pair_id]
        trial = matching_trial(calibration, selected)
        nominal_cell = f"{pair_id}__cal_nominal"
        adapted_cell = f"{pair_id}__cal_adapted"
        nominal_path = Path(run["cells"][nominal_cell]["scientific"]["artifacts"]["trajectory"])
        adapted_path = Path(run["cells"][adapted_cell]["scientific"]["artifacts"]["trajectory"])
        nominal = extract_keypoints(load(nominal_path))
        adapted = extract_keypoints(load(adapted_path))
        station = tuple(float(value) for value in trial["station_xy_m"])
        route_axis = str(trial["route_axis"])
        depth = float(selected["face_along_route_m"])
        across = float(selected["face_across_route_m"])
        orig_reach = overhead_face_reach(nominal, station, route_axis, depth, across)
        edit_reach = overhead_face_reach(adapted, station, route_axis, depth, across)
        solution = solve_window(
            orig_reach,
            edit_reach,
            delta_clear_m=ENGINEERING_MARGIN_M,
            delta_strike_m=ENGINEERING_MARGIN_M,
            min_window_m=MINIMUM_ENGINEERING_WIDTH_M,
            hard_coordinate_m=float(selected["hard_coordinate_m"]),
            easy_coordinate_m=float(selected["easy_coordinate_m"]),
        )
        if solution.binding_keypoint != "head_torso":
            raise SystemExit(f"{pair_id}: selected support no longer binds head_torso")

        spec = ConstraintSpec(
            spec_id=f"cs_{pair_id}_{args.spec_suffix}",
            source_kind="inverse_synthesis",
            source_family_id=pair_id,
            source_variant_id=f"{args.spec_suffix}_alpha_1p0",
            orig=motion_evidence(f"{pair_id}_nominal", nominal_path, nominal_cell),
            edit=motion_evidence(f"{pair_id}_adapted", adapted_path, adapted_cell),
            crossing_frames_orig=crossing_interval(nominal, station, route_axis, depth),
            crossing_frames_edit=crossing_interval(adapted, station, route_axis, depth),
            axis_type="overhead",
            expected_link_group="head_torso",
            reach_orig_m=orig_reach.reach_m,
            reach_edit_m=edit_reach.reach_m,
            window_mm=1000 * (orig_reach.reach_m - edit_reach.reach_m),
            easy_coordinate_m=solution.easy_coordinate_m,
            hard_coordinate_m=solution.hard_coordinate_m,
            binding_station_xy_m=station,
            route_axis=route_axis,
            face_along_route_m=depth,
            face_across_route_m=across,
            face_vertical_m=None,
            binding_thickness_m=0.10,
            keepout_delta_mm=50.0,
            room_size_xy_m=screen.room_size_xy_m,
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
        seed = args.seed_base + int(pair["motion_index"])
        result = instantiate(spec, "shelf_plank", seed, args.out_package)
        keepout = validate_pair(spec, result.easy.path, result.hard.path)
        keepout_path = args.out_package / (f"{spec.spec_id}__shelf_plank__s{seed:08d}.keepout.json")
        write_report(keepout, keepout_path)
        if not keepout["ok"]:
            raise SystemExit(f"{spec.spec_id}: Tier-2 preflight refused")
        pair_manifest = json.loads(result.manifest_path.read_text())
        pair_manifest["cpu_certificate"] = {
            "keepout_report": str(keepout_path.relative_to(REPO_ROOT)),
            "binding_geometry_signs": keepout["binding_geometry_signs"],
            "four_sign_pattern_matches": True,
            "engineering_margin_mm_each_side": 1000 * ENGINEERING_MARGIN_M,
            "engineering_margin_confidence_level": None,
        }
        result.manifest_path.write_text(json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n")

        geometry = keepout["binding_geometry_pattern"]
        hard_edit_clearance = geometry["hard"]["edit"]["clearance_mm"]
        coordinate_name = coordinate_bucket("overhead", spec.hard_coordinate_m, config)
        margin_name = margin_bucket(hard_edit_clearance, config)
        target = next(
            (
                value
                for value in targets
                if value["edit_behaviour_class"] == "crouch"
                and value["operator"] == "local_crouch"
                and value["constraint_axis"] == "overhead"
                and value["binding_keypoint"] == "head_torso"
                and value["constraint_coordinate_bucket"] == coordinate_name
                and value["margin_bucket"] == margin_name
            ),
            None,
        )
        if target is None:
            raise SystemExit(f"{pair_id}: selected geometry does not fill a v2 empty target")
        selected_output.append(
            {
                "source_pair_id": pair_id,
                "source_variant_id": spec.source_variant_id,
                "motion_index": pair["motion_index"],
                "seed": seed,
                "station_progress": selected["route_progress"],
                "station_xy_m": list(station),
                "route_axis": route_axis,
                "face_along_route_m": depth,
                "face_across_route_m": across,
                "raw_window_mm": spec.window_mm,
                "engineering_window_mm": 1000 * solution.width_m,
                "easy_coordinate_m": spec.easy_coordinate_m,
                "hard_coordinate_m": spec.hard_coordinate_m,
                "binding_keypoint": spec.binding_keypoint,
                "critical_frames": {
                    "nominal": spec.critical_frame_orig,
                    "adapted": spec.critical_frame_edit,
                },
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
                    for cell, values in geometry.items()
                },
                "reference_motions": {
                    "nominal": pair["nominal_artifacts"]["motion"],
                    "adapted": pair["adapted_artifacts"]["motion"],
                },
            }
        )

    report = {
        "schema_version": "lfh_e6_crouch_cpu_v1",
        "physics_executed": False,
        "source_support": str(args.support.relative_to(REPO_ROOT)),
        "source_support_sha256": sha256_file(args.support),
        "calibration_run": str(CALIBRATION_RUN),
        "calibration_run_sha256": sha256_file(CALIBRATION_RUN),
        "engineering_margin_mm_each_side": 1000 * ENGINEERING_MARGIN_M,
        "engineering_margin_confidence_level": None,
        "selected_sources": len(selected_output),
        "all_cpu_preflights_pass": len(selected_output) == len(support_rows),
        "selected": selected_output,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: materialized {len(selected_output)} E6 source pairs -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
