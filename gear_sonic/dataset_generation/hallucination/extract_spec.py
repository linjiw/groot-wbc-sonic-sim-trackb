"""Extract an LFH ConstraintSpec from a verified source-family artifact."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
from typing import Mapping

import numpy as np

from gear_sonic.dataset_generation.counterfactual_family import DEFAULT_SEARCH_TOLERANCE_M
from gear_sonic.dataset_generation.sweepcf_coverage import semantic_keypoint
from gear_sonic.dataset_generation.swept_volume import body_capsules_world
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

from .constraint_spec import ConstraintSpec, ConstraintSpecError, MotionEvidence, sha256_file
from .keypoints import extract_keypoints
from .reach import overhead_face_reach
from .stage_geometry import StageCube, StageGeometry, read_stage_geometry
from .window import WindowError, solve_window

SOURCE_MEASURE_TOLERANCE_MM = 0.5
EXTENT_MARGIN_M = 0.05


def _manifest(family_dir: Path) -> tuple[Path, dict]:
    candidates = (family_dir / "family.json", family_dir.parent / f"{family_dir.name}.json")
    for path in candidates:
        if path.exists():
            return path, json.loads(path.read_text())
    raise ConstraintSpecError(f"{family_dir}: no family manifest")


def _source_scene(family_dir: Path, role: str) -> str:
    logs = family_dir / "logs"
    candidates = sorted(logs.glob(f"*{role}*.runner.log")) + sorted(logs.glob(f"*{role}*.log"))
    for path in candidates:
        match = re.search(r"^scene=(\S+)", path.read_text(errors="ignore"), flags=re.M)
        if match and match.group(1) != "plane":
            return match.group(1)
    raise ConstraintSpecError(f"{family_dir.name}: no executed {role} source scene")


def _plane_evidence_gap(family_dir: Path, label: str) -> str | None:
    cell = family_dir / f"probe_{label}"
    log = family_dir / "logs" / f"probe_{label}.log"
    trajectories = sorted((cell / "trajectories").glob("*.trajectory.pkl"))
    missing = []
    if not log.exists():
        missing.append("attribution log")
    if not trajectories:
        missing.append("trajectory")
    if missing:
        return f"{label} ({', '.join(missing)})"
    scene = re.search(r"^scene=(\S+)", log.read_text(errors="ignore"), flags=re.M)
    if scene is None or scene.group(1) != "plane":
        actual = "unattributed" if scene is None else scene.group(1)
        return f"{label} (scene={actual}, not plane)"
    return None


def _plane_evidence(family_dir: Path, label: str, motion_id: str) -> tuple[MotionEvidence, dict]:
    cell = family_dir / f"probe_{label}"
    trajectories = sorted((cell / "trajectories").glob("*.trajectory.pkl"))
    trajectory = trajectories[0]
    with trajectory.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ConstraintSpecError(f"{family_dir.name}: {label} plane rollout is unevaluable")
    evidence = MotionEvidence(
        motion_id=motion_id,
        artifact_path=str(trajectory),
        artifact_sha256=sha256_file(trajectory),
        scene_id="plane",
    )
    return evidence, payload


def _legacy_shelf(stage: StageGeometry) -> StageCube:
    matches = [cube for cube in stage.cubes if cube.path.endswith("/LowShelf_00")]
    if len(matches) != 1:
        raise ConstraintSpecError(
            f"{stage.path}: expected one LowShelf_00 source primitive, found {len(matches)}"
        )
    return matches[0]


def _binding_group(family_dir: Path) -> str:
    attribution = family_dir / "attribution.json"
    if not attribution.exists():
        raise ConstraintSpecError(f"{family_dir.name}: no contact-attribution artifact")
    data = json.loads(attribution.read_text())
    cell = (data.get("cells") or {}).get("nominal_hard") or {}
    group = semantic_keypoint(cell.get("first_contact_body"))
    if group == "unknown":
        raise ConstraintSpecError(f"{family_dir.name}: binding anatomy is unattributed")
    return group


def _frame_and_extent(
    payload: Mapping[str, object],
    station_xy: tuple[float, float],
    route_axis: str,
) -> tuple[int, tuple[float, float]]:
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    axis = 0 if route_axis == "x" else 1
    frame = int(np.argmin(np.abs(root[:, axis] - station_xy[axis])))
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"]),
        np.asarray(payload["body_quat_w"]),
        list(payload["body_names"]),
    )
    points = np.concatenate((starts[frame], ends[frame]), axis=0)
    padding = np.concatenate((radii, radii))[:, None]
    lower = (points - padding).min(axis=0)
    upper = (points + padding).max(axis=0)
    along_axis, across_axis = (0, 1) if route_axis == "x" else (1, 0)
    along = 2 * max(
        abs(float(lower[along_axis]) - station_xy[along_axis]),
        abs(float(upper[along_axis]) - station_xy[along_axis]),
    )
    across = 2 * max(
        abs(float(lower[across_axis]) - station_xy[across_axis]),
        abs(float(upper[across_axis]) - station_xy[across_axis]),
    )
    return frame, (along + 2 * EXTENT_MARGIN_M, across + 2 * EXTENT_MARGIN_M)


def extract_spec(family_dir: Path, scenes_root: Path) -> ConstraintSpec:
    """Read source measurements and paired executed plane artifacts without reconstruction."""

    _, meta = _manifest(family_dir)
    if meta.get("counterfactual_established") is False:
        raise ConstraintSpecError(f"{family_dir.name}: source family is not physics verified")
    axis_type = str(meta.get("regime") or meta.get("constraint_axis") or "")
    nominal_motion = str(meta.get("nominal_motion") or "").strip()
    adapted_motion = str(meta.get("adapted_motion") or "").strip()
    gaps: list[str] = []
    if not axis_type:
        gaps.append("explicit constraint axis")
    elif axis_type != "overhead":
        gaps.append(f"supported verified overhead axis (got {axis_type})")
    if not nominal_motion:
        gaps.append("nominal motion identity")
    if not adapted_motion:
        gaps.append("adapted motion identity")
    gaps.extend(
        gap
        for label in ("nominal", "adapted")
        if (gap := _plane_evidence_gap(family_dir, label)) is not None
    )
    if not (family_dir / "attribution.json").exists():
        gaps.append("contact-attribution artifact")
    if gaps:
        raise ConstraintSpecError(
            f"{family_dir.name}: cannot extract ConstraintSpec; missing or unsupported evidence: "
            f"{'; '.join(gaps)}"
        )
    orig, orig_payload = _plane_evidence(family_dir, "nominal", nominal_motion)
    edit, edit_payload = _plane_evidence(family_dir, "adapted", adapted_motion)

    easy_scene = _source_scene(family_dir, "nominal_easy")
    hard_scene = _source_scene(family_dir, "nominal_hard")
    easy_stage = read_stage_geometry(scenes_root / f"{easy_scene}.usda")
    hard_stage = read_stage_geometry(scenes_root / f"{hard_scene}.usda")
    easy_shelf, hard_shelf = _legacy_shelf(easy_stage), _legacy_shelf(hard_stage)
    easy_coordinate = easy_shelf.box[2]
    hard_coordinate = hard_shelf.box[2]
    expected_easy = float(meta["easy_shelf_underside_m"])
    expected_hard = float(meta["hard_shelf_underside_m"])
    if 1000 * abs(easy_coordinate - expected_easy) > SOURCE_MEASURE_TOLERANCE_MM:
        raise ConstraintSpecError(f"{family_dir.name}: easy source face fails measure-back")
    if 1000 * abs(hard_coordinate - expected_hard) > SOURCE_MEASURE_TOLERANCE_MM:
        raise ConstraintSpecError(f"{family_dir.name}: hard source face fails measure-back")
    easy_station = easy_shelf.center_m[:2]
    hard_station = hard_shelf.center_m[:2]
    station_offset = 1000 * max(abs(easy_station[i] - hard_station[i]) for i in range(2))
    if station_offset > SOURCE_MEASURE_TOLERANCE_MM:
        raise ConstraintSpecError(f"{family_dir.name}: source scenes disagree on binding station")

    root = np.asarray(orig_payload["root_pos_w"], dtype=np.float64)
    displacement = np.ptp(root[:, :2], axis=0)
    route_axis = "x" if displacement[0] >= displacement[1] else "y"
    orig_frame, orig_extent = _frame_and_extent(orig_payload, easy_station, route_axis)
    edit_frame, edit_extent = _frame_and_extent(edit_payload, easy_station, route_axis)
    along_axis, across_axis = (0, 1) if route_axis == "x" else (1, 0)
    # Along-route depth sets how long a local edit must remain active. Enlarging the historical
    # footprint catches the adapted motion after its crouch releases and changes the family.
    face_along = easy_shelf.size_m[along_axis]
    face_across = max(easy_shelf.size_m[across_axis], orig_extent[1], edit_extent[1])
    walls = [cube.size_m[2] for cube in easy_stage.cubes if cube.role in ("wall", "room_shell")]
    room_size = easy_stage.room_size_xy_m
    if room_size is None or not walls:
        raise ConstraintSpecError(f"{family_dir.name}: source room bounds are not measurable")

    binding_group = _binding_group(family_dir)
    orig_reach = overhead_face_reach(
        extract_keypoints(orig_payload), easy_station, route_axis, face_along, face_across
    )
    edit_reach = overhead_face_reach(
        extract_keypoints(edit_payload), easy_station, route_axis, face_along, face_across
    )
    try:
        solved = solve_window(
            orig_reach,
            edit_reach,
            hard_coordinate_m=expected_hard,
            easy_coordinate_m=expected_easy,
        )
    except WindowError as error:
        raise ConstraintSpecError(
            f"{family_dir.name}: executed critical-set refusal: {error}"
        ) from error
    stored_window = float(meta["window_m"])
    if abs(solved.width_m - stored_window) > DEFAULT_SEARCH_TOLERANCE_M:
        raise ConstraintSpecError(
            f"{family_dir.name}: recomputed window {solved.width_m:.6f} m disagrees with "
            f"stored boundary-search window {stored_window:.6f} m beyond "
            f"{DEFAULT_SEARCH_TOLERANCE_M:.3f} m source tolerance"
        )
    if solved.binding_keypoint != binding_group:
        raise ConstraintSpecError(
            f"{family_dir.name}: recomputed binding {solved.binding_keypoint} disagrees with "
            f"contact attribution {binding_group}"
        )

    spec = ConstraintSpec(
        spec_id=f"cs_{family_dir.name}",
        source_kind="verified_family",
        source_family_id=str(meta.get("family_id") or family_dir.name),
        source_variant_id=family_dir.name,
        orig=orig,
        edit=edit,
        crossing_frames_orig=(orig_frame, orig_frame),
        crossing_frames_edit=(edit_frame, edit_frame),
        axis_type=axis_type,
        expected_link_group=binding_group,
        reach_orig_m=float(meta["nominal_clears_to_m"]),
        reach_edit_m=float(meta["adapted_clears_to_m"]),
        window_mm=1000 * float(meta["window_m"]),
        easy_coordinate_m=expected_easy,
        hard_coordinate_m=expected_hard,
        binding_station_xy_m=tuple(float(v) for v in easy_station),
        route_axis=route_axis,
        face_along_route_m=float(face_along),
        face_across_route_m=float(face_across),
        face_vertical_m=None,
        binding_thickness_m=float(easy_shelf.size_m[2]),
        keepout_delta_mm=50.0,
        room_size_xy_m=tuple(float(v) for v in room_size),
        wall_height_m=max(walls),
        source_scene_easy=easy_scene,
        source_scene_hard=hard_scene,
        binding_keypoint=solved.binding_keypoint,
        critical_frame_orig=solved.critical_frame_orig,
        critical_frame_edit=solved.critical_frame_edit,
        per_keypoint_margins_m=solved.per_keypoint_margins_m,
    )
    spec.validate()
    return spec
