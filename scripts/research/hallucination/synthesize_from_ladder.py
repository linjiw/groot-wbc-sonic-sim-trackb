#!/usr/bin/env python3
"""Author critical scenes from any LFH-E12 pair whose executed window clears the floor.

`synthesize_crouch_critical.py` is bound to the CAL3 cohort: it reads the CAL3 run record, the
CAL3 candidate file, and the CAL3 support census. This is the same construction generalised to any
accepted executed pair, so a motion that becomes a source in a later batch can author a scene
without a bespoke script.

The evidence chain is unchanged. Reaches come from executed trajectories, the window is closed
form, keep-out is Tier 2, and physics remains the only verdict.
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
from gear_sonic.dataset_generation.local_adaptation import route_progress  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

DATA_ROOT = Path("/data/robotixx/groot-wbc-kimodo-m0")
SCREEN_EMPTY = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual/screen_empty.usda"
ENGINEERING_MARGIN_M = 0.018044
MINIMUM_ENGINEERING_WIDTH_M = 0.020
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 3.0
STATION_FRACTION = 0.55
EASY_CLEARANCE_M = 0.050


def _payload(path: Path) -> dict:
    with Path(path).open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory segment")
    return payload


def _crossing(tracks, station: tuple[float, float], route_axis: str, depth: float):
    axis = 0 if route_axis == "x" else 1
    frames = np.flatnonzero(np.abs(tracks.root_pos_w[:, axis] - station[axis]) <= depth / 2)
    if not len(frames):
        raise ValueError("executed root never enters the selected face slab")
    return int(frames[0]), int(frames[-1])


def _station_and_axis(tracks) -> tuple[tuple[float, float], str]:
    root = tracks.root_pos_w[:, :2]
    span = np.ptp(root, axis=0)
    axis = "x" if span[0] >= span[1] else "y"
    progress = route_progress(np.asarray(root, dtype=np.float64))
    index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
    return (float(root[index, 0]), float(root[index, 1])), axis


def _evidence(motion_id: str, trajectory: Path, cell_id: str, run_record: Path) -> MotionEvidence:
    return MotionEvidence(
        motion_id=motion_id,
        artifact_path=str(trajectory),
        artifact_sha256=sha256_file(trajectory),
        scene_id="screen_empty",
        operator="local_crouch",
        adjudication_path=str(run_record),
        adjudication_sha256=sha256_file(run_record),
        adjudication_cell_id=cell_id,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ladder", type=Path, default=REPO_ROOT / "docs/hallucination/e12_crouch_ladder.json"
    )
    parser.add_argument(
        "--run-record",
        type=Path,
        default=DATA_ROOT / "hallucination/run_records/E12_CROUCH_LADDER_2026-08-26.json",
    )
    parser.add_argument("--archetype", default="shelf_plank")
    parser.add_argument("--seed-base", type=int, default=37000)
    parser.add_argument("--pair-id", action="append")
    parser.add_argument(
        "--out-package",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual_lfh_e17",
    )
    parser.add_argument("--spec-dir", type=Path, default=REPO_ROOT / "specs/hallucination")
    parser.add_argument(
        "--json-out", type=Path, default=REPO_ROOT / "docs/hallucination/e17_ladder_scenes.json"
    )
    parser.add_argument(
        "--minimum-window-mm", type=float, default=1000 * MINIMUM_ENGINEERING_WIDTH_M
    )
    args = parser.parse_args()

    ladder = json.loads(args.ladder.read_text())
    run = json.loads(args.run_record.read_text())
    screen = read_stage_geometry(SCREEN_EMPTY)
    if screen.room_size_xy_m is None:
        raise SystemExit("screen_empty has no measurable room size")

    eligible = [
        motion
        for motion in ladder["motions"]
        if motion["window"]
        and motion["nominal_outcome"] == "accepted"
        and motion["window"]["engineering_window_mm"] >= args.minimum_window_mm
    ]
    if args.pair_id:
        eligible = [motion for motion in eligible if motion["pair_id"] in set(args.pair_id)]
    if not eligible:
        raise SystemExit("no ladder pair clears the engineering-window floor")

    args.out_package.mkdir(parents=True, exist_ok=True)
    args.spec_dir.mkdir(parents=True, exist_ok=True)
    authored = []
    for motion in eligible:
        pair_id = motion["pair_id"]
        deepest = max(
            (rung for rung in motion["rungs"] if rung["outcome"] == "accepted"),
            key=lambda rung: rung["commanded_drop_mm"],
        )
        nominal_cell = f"{pair_id}__nominal"
        adapted_cell = deepest["cell_id"]
        nominal_path = Path(run["cells"][nominal_cell]["scientific"]["artifacts"]["trajectory"])
        adapted_path = Path(run["cells"][adapted_cell]["scientific"]["artifacts"]["trajectory"])
        nominal = extract_keypoints(_payload(nominal_path))
        adapted = extract_keypoints(_payload(adapted_path))
        station, route_axis = _station_and_axis(nominal)

        orig_reach = overhead_face_reach(
            nominal, station, route_axis, FACE_ALONG_M, FACE_ACROSS_M, require_all_groups=False
        )
        edit_reach = overhead_face_reach(
            adapted, station, route_axis, FACE_ALONG_M, FACE_ACROSS_M, require_all_groups=False
        )
        solution = solve_window(
            orig_reach,
            edit_reach,
            delta_clear_m=ENGINEERING_MARGIN_M,
            delta_strike_m=ENGINEERING_MARGIN_M,
            min_window_m=args.minimum_window_mm / 1000.0,
            easy_coordinate_m=orig_reach.reach_m + EASY_CLEARANCE_M,
        )
        if solution.binding_keypoint != "head_torso":
            raise SystemExit(f"{pair_id}: window binds {solution.binding_keypoint}, not head_torso")

        spec = ConstraintSpec(
            spec_id=f"cs_{pair_id}_ladder",
            source_kind="inverse_synthesis",
            source_family_id=pair_id,
            source_variant_id=f"ladder_{deepest['label']}",
            orig=_evidence(f"{pair_id}_nominal", nominal_path, nominal_cell, args.run_record),
            edit=_evidence(f"{pair_id}_adapted", adapted_path, adapted_cell, args.run_record),
            crossing_frames_orig=_crossing(nominal, station, route_axis, FACE_ALONG_M),
            crossing_frames_edit=_crossing(adapted, station, route_axis, FACE_ALONG_M),
            axis_type="overhead",
            expected_link_group="head_torso",
            reach_orig_m=orig_reach.reach_m,
            reach_edit_m=edit_reach.reach_m,
            window_mm=1000 * (orig_reach.reach_m - edit_reach.reach_m),
            easy_coordinate_m=solution.easy_coordinate_m,
            hard_coordinate_m=solution.hard_coordinate_m,
            binding_station_xy_m=station,
            route_axis=route_axis,
            face_along_route_m=FACE_ALONG_M,
            face_across_route_m=FACE_ACROSS_M,
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

        seed = args.seed_base + int(motion["motion_index"])
        result = instantiate(spec, args.archetype, seed, args.out_package)
        keepout = validate_pair(spec, result.easy.path, result.hard.path)
        keepout_path = args.out_package / (
            f"{spec.spec_id}__{args.archetype}__s{seed:08d}.keepout.json"
        )
        write_report(keepout, keepout_path)
        if not keepout["ok"]:
            raise SystemExit(f"{spec.spec_id}: Tier-2 keep-out refused")

        authored.append(
            {
                "pair_id": pair_id,
                "motion_index": motion["motion_index"],
                "body_mode": motion["body_mode"],
                "delivered_amplitude_mm": motion["delivered_amplitude_mm"],
                "archetype": args.archetype,
                "seed": seed,
                "station_xy_m": list(station),
                "route_axis": route_axis,
                "raw_window_mm": spec.window_mm,
                "engineering_window_mm": 1000 * solution.width_m,
                "easy_coordinate_m": spec.easy_coordinate_m,
                "hard_coordinate_m": spec.hard_coordinate_m,
                "binding_keypoint": spec.binding_keypoint,
                "critical_frames": {
                    "nominal": spec.critical_frame_orig,
                    "adapted": spec.critical_frame_edit,
                },
                "spec": str(spec_path.relative_to(REPO_ROOT)),
                "spec_sha256": sha256_file(spec_path),
                "keepout": str(keepout_path.relative_to(REPO_ROOT)),
                "keepout_min_clearance_mm": keepout.get("min_clearance_mm"),
                "scenes": {
                    "easy": str(result.easy.path.relative_to(REPO_ROOT)),
                    "hard": str(result.hard.path.relative_to(REPO_ROOT)),
                },
                "scene_sha256": {"easy": result.easy.sha256, "hard": result.hard.sha256},
                "reference_motions": {
                    "nominal": str(nominal_path),
                    "adapted": str(adapted_path),
                },
            }
        )

    report = {
        "schema_version": "lfh_ladder_scenes_v1",
        "physics_executed": False,
        "ladder_source": str(args.ladder.relative_to(REPO_ROOT)),
        "ladder_sha256": sha256_file(args.ladder),
        "engineering_margin_mm_each_side": 1000 * ENGINEERING_MARGIN_M,
        "minimum_engineering_window_mm": args.minimum_window_mm,
        "authored": len(authored),
        "scenes": authored,
    }
    args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"PASS: authored {len(authored)} ladder scene pair(s) -> {args.json_out}")
    for entry in authored:
        print(
            f"  {entry['pair_id']} ({entry['body_mode']}): window "
            f"{entry['engineering_window_mm']:.1f} mm, hard {entry['hard_coordinate_m']:.4f} m, "
            f"keepout {entry['keepout_min_clearance_mm']} mm"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
