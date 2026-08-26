"""Tier-2 LFH preflight: measure-back, station, extent, route, and single-cause keepout."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from gear_sonic.dataset_generation.counterfactual_family import swept_clearance_to_box
from gear_sonic.dataset_generation.scene_route_check import check_route_meets_obstacle
from gear_sonic.dataset_generation.swept_volume import (
    body_capsules_world,
    swept_volume_clearance,
)
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

from .constraint_spec import ConstraintSpec, MotionEvidence, sha256_file
from .stage_geometry import StageCube, measure_binding, read_stage_geometry

PLACEMENT_TOLERANCE_MM = 0.5
EXEMPT_ROLES = frozenset({"support_floor", "room_shell"})


@dataclass(frozen=True)
class PreflightViolation:
    reason: str
    detail: str
    prim_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"reason": self.reason, "detail": self.detail, "prim_path": self.prim_path}


@dataclass(frozen=True)
class KeepoutReport:
    scene_path: Path
    cell: str
    ok: bool
    binding_face_offset_mm: float | None
    binding_station_offset_mm: float | None
    keepout_min_clearance_mm: float | None
    route_reaches_binding: bool
    violations: tuple[PreflightViolation, ...]

    @property
    def refusal_reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(violation.reason for violation in self.violations))

    def to_dict(self) -> dict[str, object]:
        return {
            "scene_path": str(self.scene_path),
            "cell": self.cell,
            "ok": self.ok,
            "binding_face_offset_mm": self.binding_face_offset_mm,
            "binding_station_offset_mm": self.binding_station_offset_mm,
            "keepout_min_clearance_mm": self.keepout_min_clearance_mm,
            "route_reaches_binding": self.route_reaches_binding,
            "refusal_reasons": list(self.refusal_reasons),
            "violations": [violation.to_dict() for violation in self.violations],
        }


def load_executed_payload(evidence: MotionEvidence) -> Mapping[str, object]:
    if evidence.adjudication_path is not None:
        adjudication_path = Path(evidence.adjudication_path)
        if not adjudication_path.exists():
            raise ValueError(f"missing adjudication artifact {adjudication_path}")
        actual = sha256_file(adjudication_path)
        if actual != evidence.adjudication_sha256:
            raise ValueError(
                f"adjudication artifact hash mismatch: {actual} != "
                f"{evidence.adjudication_sha256}"
            )
        adjudication = json.loads(adjudication_path.read_text())
        try:
            cell = adjudication["cells"][evidence.adjudication_cell_id]
            scientific = cell["scientific"]
            trajectory = scientific["artifacts"]
            external = scientific["diagnostics"]["contact_decomposition"][
                "max_external_contact_force_n"
            ]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"malformed adjudication for {evidence.adjudication_cell_id}"
            ) from error
        if cell.get("status") != "completed" or scientific.get("outcome") != "accepted":
            raise ValueError(f"{evidence.adjudication_cell_id}: executed evidence was not accepted")
        if float(external) != 0.0:
            raise ValueError(
                f"{evidence.adjudication_cell_id}: empty-room evidence has external contact"
            )
        if (
            trajectory.get("trajectory") != evidence.artifact_path
            or trajectory.get("trajectory_sha256") != evidence.artifact_sha256
        ):
            raise ValueError(
                f"{evidence.adjudication_cell_id}: adjudication trajectory identity mismatch"
            )
    path = Path(evidence.artifact_path)
    if not path.exists():
        raise ValueError(f"missing executed artifact {path}")
    actual = sha256_file(path)
    if actual != evidence.artifact_sha256:
        raise ValueError(f"executed artifact hash mismatch: {actual} != {evidence.artifact_sha256}")
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"executed artifact is unevaluable: {path}")
    return payload


def _capsule_bounds(
    payload: Mapping[str, object], interval: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"]),
        np.asarray(payload["body_quat_w"]),
        list(payload["body_names"]),
    )
    first, last = interval
    if last >= starts.shape[0]:
        raise ValueError(f"crossing interval {interval} exceeds {starts.shape[0]} frames")
    points = np.concatenate((starts[first : last + 1], ends[first : last + 1]), axis=1)
    padding = np.concatenate((radii, radii))[None, :, None]
    return (points - padding).min(axis=(0, 1)), (points + padding).max(axis=(0, 1))


def _binding_span(cubes: tuple[StageCube, ...]) -> tuple[float, ...]:
    boxes = [cube.box for cube in cubes]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        max(box[4] for box in boxes),
        max(box[5] for box in boxes),
    )


def _route_reaches(
    spec: ConstraintSpec,
    binding: tuple[StageCube, ...],
    payloads: tuple[Mapping[str, object], Mapping[str, object]],
) -> bool:
    if spec.axis_type == "overhead":
        span = _binding_span(binding)
        rect = (span[0], span[1], span[3], span[4])
        return all(
            check_route_meets_obstacle(np.asarray(payload["root_pos_w"])[:, :2], rect).passes
            for payload in payloads
        )
    along_axis = 0 if spec.route_axis == "x" else 1
    across_axis = 1 - along_axis
    measurement = measure_binding(
        read_stage_geometry_from_cubes(binding), spec.axis_type, spec.route_axis
    )
    half_along = measurement.along_route_m / 2
    for payload, interval in zip(payloads, (spec.crossing_frames_orig, spec.crossing_frames_edit)):
        root = np.asarray(payload["root_pos_w"])
        first, last = interval
        points = root[first : last + 1, :2]
        if not np.any(
            (np.abs(points[:, along_axis] - measurement.station_xy_m[along_axis]) <= half_along)
            & (
                np.abs(points[:, across_axis] - measurement.station_xy_m[across_axis])
                <= measurement.coordinate_m / 2
            )
        ):
            return False
    return True


def read_stage_geometry_from_cubes(cubes: tuple[StageCube, ...]):
    """Small adapter used only to reuse binding measurement in the lateral route check."""

    from .stage_geometry import StageGeometry

    return StageGeometry(Path("<in-memory>"), 1.0, "Z", cubes, None)


def validate_scene(
    spec: ConstraintSpec,
    scene_path: Path,
    cell: str,
    orig_payload: Mapping[str, object],
    edit_payload: Mapping[str, object],
) -> KeepoutReport:
    """Certify a scene geometrically; physics outcome remains intentionally absent."""

    violations: list[PreflightViolation] = []
    stage = read_stage_geometry(scene_path)
    measurement = measure_binding(stage, spec.axis_type, spec.route_axis)
    commanded = spec.easy_coordinate_m if cell == "easy" else spec.hard_coordinate_m
    face_offset = 1000 * abs(measurement.coordinate_m - commanded)
    station_offset = 1000 * max(
        abs(measurement.station_xy_m[i] - spec.binding_station_xy_m[i]) for i in range(2)
    )
    if face_offset > PLACEMENT_TOLERANCE_MM:
        violations.append(
            PreflightViolation(
                "binding_face_offset",
                f"{face_offset:.3f} mm exceeds {PLACEMENT_TOLERANCE_MM:.1f} mm",
            )
        )
    if station_offset > PLACEMENT_TOLERANCE_MM:
        violations.append(
            PreflightViolation(
                "binding_station_mismatch",
                f"{station_offset:.3f} mm exceeds {PLACEMENT_TOLERANCE_MM:.1f} mm",
            )
        )
    if measurement.along_route_m + 1e-9 < spec.face_along_route_m:
        violations.append(
            PreflightViolation(
                "face_extent",
                f"along-route extent {measurement.along_route_m:.4f} m is below "
                f"required {spec.face_along_route_m:.4f} m",
            )
        )
    if spec.axis_type == "overhead" and (
        measurement.across_route_m + 1e-9 < spec.face_across_route_m
    ):
        violations.append(
            PreflightViolation(
                "face_extent",
                f"across-route extent {measurement.across_route_m:.4f} m is below "
                f"required {spec.face_across_route_m:.4f} m",
            )
        )
    if spec.axis_type == "lateral_gap" and (
        measurement.vertical_m + 1e-9 < float(spec.face_vertical_m)
    ):
        violations.append(
            PreflightViolation(
                "face_extent",
                f"vertical extent {measurement.vertical_m:.4f} m is below "
                f"required {spec.face_vertical_m:.4f} m",
            )
        )

    binding_paths = set(measurement.binding_paths)
    binding_cubes = tuple(cube for cube in stage.cubes if cube.path in binding_paths)
    if spec.axis_type == "lateral_gap":
        cross_axis = 1 if spec.route_axis == "x" else 0
        if any(cube.size_m[cross_axis] + 1e-9 < spec.face_across_route_m for cube in binding_cubes):
            violations.append(
                PreflightViolation(
                    "face_extent",
                    "lateral binding panels do not cover the declared outward anti-skirt extent",
                )
            )
    payloads = (orig_payload, edit_payload)
    route_reaches = _route_reaches(spec, binding_cubes, payloads)
    if not route_reaches:
        violations.append(
            PreflightViolation("route_miss", "one or both executed routes miss the binding face")
        )

    # Across-route coverage prevents a lateral skirt. Along-route depth is intentionally not
    # expanded: it defines the finite activation interval of a local edit and is verified against
    # the spec above. Both executed roots must cross that interval via `_route_reaches`.
    if spec.axis_type == "overhead":
        span = _binding_span(binding_cubes)
        margin = spec.keepout_delta_mm / 1000
        across_axis = 1 if spec.route_axis == "x" else 0
        for label, payload, interval in (
            ("orig", orig_payload, spec.crossing_frames_orig),
            ("edit", edit_payload, spec.crossing_frames_edit),
        ):
            try:
                lower, upper = _capsule_bounds(payload, interval)
            except ValueError as error:
                # A reset-spanning capture is evaluated on one segment, which can be shorter
                # than the stored crossing frames.  That is a refusal, not a crash: an
                # uncaught raise produced no KeepoutReport at all.
                violations.append(PreflightViolation("crossing_out_of_range", str(error)))
                continue
            if (
                lower[across_axis] - margin < span[across_axis] - 1e-9
                or upper[across_axis] + margin > span[across_axis + 3] + 1e-9
            ):
                violations.append(
                    PreflightViolation(
                        "face_extent",
                        f"{label} crossing capsules plus keepout margin exceed across-route footprint",
                    )
                )

    context = tuple(
        cube
        for cube in stage.cubes
        if cube.path not in binding_paths and cube.role not in EXEMPT_ROLES
    )
    keepout_min: float | None = None
    delta_m = spec.keepout_delta_mm / 1000
    for cube in context:
        minimum = min(
            swept_volume_clearance(
                np.asarray(payload["body_pos_w"]),
                np.asarray(payload["body_quat_w"]),
                list(payload["body_names"]),
                [(cube.path, cube.box)],
            ).min_clearance_m
            for payload in payloads
        )
        keepout_min = minimum if keepout_min is None else min(keepout_min, minimum)
        if minimum < delta_m:
            violations.append(
                PreflightViolation(
                    "keepout_violation",
                    f"clearance {minimum * 1000:.1f} mm is below "
                    f"{spec.keepout_delta_mm:.1f} mm",
                    cube.path,
                )
            )
    return KeepoutReport(
        scene_path=scene_path,
        cell=cell,
        ok=not violations,
        binding_face_offset_mm=face_offset,
        binding_station_offset_mm=station_offset,
        keepout_min_clearance_mm=(None if keepout_min is None else keepout_min * 1000),
        route_reaches_binding=route_reaches,
        violations=tuple(violations),
    )


def validate_pair(spec: ConstraintSpec, easy: Path, hard: Path) -> dict[str, Any]:
    """Load and hash-check both executed probes, then validate an easy/hard scene pair."""

    orig = load_executed_payload(spec.orig)
    edit = load_executed_payload(spec.edit)
    reports = {
        "easy": validate_scene(spec, easy, "easy", orig, edit),
        "hard": validate_scene(spec, hard, "hard", orig, edit),
    }
    geometry_pattern: dict[str, dict[str, object]] = {}
    expected = {
        ("easy", "orig"): True,
        ("easy", "edit"): True,
        ("hard", "orig"): False,
        ("hard", "edit"): True,
    }
    for cell, path in (("easy", easy), ("hard", hard)):
        stage = read_stage_geometry(path)
        measurement = measure_binding(stage, spec.axis_type, spec.route_axis)
        binding_paths = set(measurement.binding_paths)
        cubes = [cube for cube in stage.cubes if cube.path in binding_paths]
        geometry_pattern[cell] = {}
        for motion, payload in (("orig", orig), ("edit", edit)):
            clearance = min(
                swept_clearance_to_box(
                    np.asarray(payload["body_pos_w"]),
                    np.asarray(payload["body_quat_w"]),
                    list(payload["body_names"]),
                    cube.box,
                )[0]
                for cube in cubes
            )
            clears = clearance > 0
            geometry_pattern[cell][motion] = {
                "clearance_mm": 1000 * clearance,
                "clears": clears,
                "expected_clears": expected[(cell, motion)],
                "matches": clears == expected[(cell, motion)],
            }
    pattern_ok = all(
        entry["matches"] for cell in geometry_pattern.values() for entry in cell.values()
    )
    geometry_signs = {
        cell: {
            motion: "+" if entry["clearance_mm"] > 0 else "-" for motion, entry in motions.items()
        }
        for cell, motions in geometry_pattern.items()
    }
    return {
        "schema_version": "lfh_keepout_v2",
        "spec_id": spec.spec_id,
        "ok": all(report.ok for report in reports.values()) and pattern_ok,
        "refusal_reasons": [] if pattern_ok else ["binding_pattern_mismatch"],
        "binding_geometry_signs": geometry_signs,
        "binding_geometry_pattern": geometry_pattern,
        "reports": {name: report.to_dict() for name, report in reports.items()},
    }


def write_report(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o775)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    path.chmod(0o664)
