"""Versioned LFH constraint specification.

The spec records measured facts needed to reproduce a binding constraint. It never contains a
physics verdict for a generated scene: geometry proposes, and the rollout scorer remains the only
verdict source.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

SPEC_VERSION = "0.1"
SUPPORTED_AXES = frozenset({"overhead", "lateral_gap"})


class ConstraintSpecError(ValueError):
    """The requested spec cannot be supported without inventing evidence."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConstraintSpecError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ConstraintSpecError(f"{field} must be finite")
    return result


def _margin_table(value: object) -> dict[str, dict[str, float]] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ConstraintSpecError("binding.per_keypoint_margins must be a mapping")
    result: dict[str, dict[str, float]] = {}
    for group, margins in value.items():
        if not isinstance(margins, Mapping):
            raise ConstraintSpecError(f"binding.per_keypoint_margins.{group} must be a mapping")
        result[str(group)] = {
            "orig_m": _finite(margins.get("orig_m"), f"{group}.orig_m"),
            "edit_m": _finite(margins.get("edit_m"), f"{group}.edit_m"),
        }
    return result


@dataclass(frozen=True)
class MotionEvidence:
    """One executed empty-room artifact used by the spec."""

    motion_id: str
    artifact_path: str
    artifact_sha256: str
    scene_id: str = "plane"
    operator: str | None = None
    adjudication_path: str | None = None
    adjudication_sha256: str | None = None
    adjudication_cell_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "MotionEvidence":
        rollout = value.get("empty_room_rollout")
        if not isinstance(rollout, Mapping):
            raise ConstraintSpecError("motion is missing empty_room_rollout evidence")
        adjudication = rollout.get("adjudication")
        if adjudication is not None and not isinstance(adjudication, Mapping):
            raise ConstraintSpecError("empty_room_rollout.adjudication must be a mapping")
        result = cls(
            motion_id=str(value.get("motion_id") or "").strip(),
            artifact_path=str(rollout.get("path") or "").strip(),
            artifact_sha256=str(rollout.get("sha256") or "").strip(),
            scene_id=str(rollout.get("scene_id") or "").strip(),
            operator=str(value.get("operator") or "").strip() or None,
            adjudication_path=(
                str(adjudication.get("path") or "").strip() or None
                if adjudication is not None
                else None
            ),
            adjudication_sha256=(
                str(adjudication.get("sha256") or "").strip() or None
                if adjudication is not None
                else None
            ),
            adjudication_cell_id=(
                str(adjudication.get("cell_id") or "").strip() or None
                if adjudication is not None
                else None
            ),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if not self.motion_id:
            raise ConstraintSpecError("motion_id is required")
        if not self.artifact_path:
            raise ConstraintSpecError(f"{self.motion_id}: empty-room artifact path is required")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.artifact_sha256) is None:
            raise ConstraintSpecError(f"{self.motion_id}: artifact sha256 is required")
        if self.scene_id not in {"plane", "screen_empty"}:
            raise ConstraintSpecError(
                f"{self.motion_id}: unsupported empty-room scene {self.scene_id!r}"
            )
        adjudication = (
            self.adjudication_path,
            self.adjudication_sha256,
            self.adjudication_cell_id,
        )
        if self.scene_id == "screen_empty" and any(value is None for value in adjudication):
            raise ConstraintSpecError(
                f"{self.motion_id}: screen_empty evidence requires a hash-pinned adjudication"
            )
        if any(value is not None for value in adjudication):
            if any(value is None for value in adjudication):
                raise ConstraintSpecError(
                    f"{self.motion_id}: adjudication path, sha256, and cell_id are inseparable"
                )
            if re.fullmatch(r"sha256:[0-9a-f]{64}", self.adjudication_sha256) is None:
                raise ConstraintSpecError(f"{self.motion_id}: adjudication sha256 is required")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "motion_id": self.motion_id,
            "empty_room_rollout": {
                "path": self.artifact_path,
                "sha256": self.artifact_sha256,
                "scene_id": self.scene_id,
            },
        }
        if self.operator:
            result["operator"] = self.operator
        if self.adjudication_path is not None:
            result["empty_room_rollout"]["adjudication"] = {
                "path": self.adjudication_path,
                "sha256": self.adjudication_sha256,
                "cell_id": self.adjudication_cell_id,
            }
        return result


@dataclass(frozen=True)
class ConstraintSpec:
    """A measured single-constraint source, suitable for deterministic instantiation."""

    spec_id: str
    source_kind: str
    source_family_id: str
    source_variant_id: str
    orig: MotionEvidence
    edit: MotionEvidence
    crossing_frames_orig: tuple[int, int]
    crossing_frames_edit: tuple[int, int]
    axis_type: str
    expected_link_group: str
    reach_orig_m: float
    reach_edit_m: float
    window_mm: float
    easy_coordinate_m: float
    hard_coordinate_m: float
    binding_station_xy_m: tuple[float, float]
    route_axis: str
    face_along_route_m: float
    face_across_route_m: float
    face_vertical_m: float | None
    binding_thickness_m: float
    keepout_delta_mm: float
    room_size_xy_m: tuple[float, float]
    wall_height_m: float
    source_scene_easy: str
    source_scene_hard: str
    material_provenance: str = "runtime_default"
    spec_version: str = SPEC_VERSION
    binding_keypoint: str | None = None
    critical_frame_orig: int | None = None
    critical_frame_edit: int | None = None
    per_keypoint_margins_m: Mapping[str, Mapping[str, float]] | None = None

    def validate(self) -> None:
        if self.spec_version != SPEC_VERSION:
            raise ConstraintSpecError(
                f"unsupported spec_version {self.spec_version!r}; expected {SPEC_VERSION!r}"
            )
        for field, value in (
            ("spec_id", self.spec_id),
            ("source_kind", self.source_kind),
            ("source_family_id", self.source_family_id),
            ("source_variant_id", self.source_variant_id),
            ("expected_link_group", self.expected_link_group),
        ):
            if not value or value == "unknown":
                raise ConstraintSpecError(f"{field} must carry explicit evidence")
        for field, value in (
            ("source_scene_easy", self.source_scene_easy),
            ("source_scene_hard", self.source_scene_hard),
        ):
            if not value:
                raise ConstraintSpecError(f"{field} must carry explicit evidence")
        self.orig.validate()
        self.edit.validate()
        if self.axis_type not in SUPPORTED_AXES:
            raise ConstraintSpecError(f"axis_type must be one of {sorted(SUPPORTED_AXES)}")
        if self.route_axis not in ("x", "y"):
            raise ConstraintSpecError("route_axis must be x or y for the axis-aligned v1 writer")
        for label, interval in (
            ("crossing.orig", self.crossing_frames_orig),
            ("crossing.edit", self.crossing_frames_edit),
        ):
            if len(interval) != 2 or interval[0] < 0 or interval[1] < interval[0]:
                raise ConstraintSpecError(f"{label} must be an inclusive non-negative interval")
        if len(self.binding_station_xy_m) != 2:
            raise ConstraintSpecError("binding_station_xy_m must contain two coordinates")
        if len(self.room_size_xy_m) != 2:
            raise ConstraintSpecError("room_size_xy_m must contain two dimensions")
        finite_fields = {
            "reach_orig_m": self.reach_orig_m,
            "reach_edit_m": self.reach_edit_m,
            "window_mm": self.window_mm,
            "easy_coordinate_m": self.easy_coordinate_m,
            "hard_coordinate_m": self.hard_coordinate_m,
            "face_along_route_m": self.face_along_route_m,
            "face_across_route_m": self.face_across_route_m,
            "binding_thickness_m": self.binding_thickness_m,
            "keepout_delta_mm": self.keepout_delta_mm,
            "wall_height_m": self.wall_height_m,
        }
        finite_fields.update(
            {f"binding_station_xy_m[{i}]": v for i, v in enumerate(self.binding_station_xy_m)}
        )
        finite_fields.update({f"room_size_xy_m[{i}]": v for i, v in enumerate(self.room_size_xy_m)})
        for field, value in finite_fields.items():
            _finite(value, field)
        if self.face_vertical_m is not None:
            _finite(self.face_vertical_m, "face_vertical_m")
        for field in (
            "window_mm",
            "face_along_route_m",
            "face_across_route_m",
            "binding_thickness_m",
            "keepout_delta_mm",
            "wall_height_m",
        ):
            if float(getattr(self, field)) <= 0:
                raise ConstraintSpecError(f"{field} must be positive")
        if any(value <= 0 for value in self.room_size_xy_m):
            raise ConstraintSpecError("room_size_xy_m must be positive")
        if self.axis_type == "lateral_gap" and (
            self.face_vertical_m is None or self.face_vertical_m <= 0
        ):
            raise ConstraintSpecError("lateral_gap needs a positive face_vertical_m")
        if self.easy_coordinate_m <= self.hard_coordinate_m:
            raise ConstraintSpecError("easy coordinate must be greater than hard coordinate")
        if self.material_provenance != "runtime_default":
            raise ConstraintSpecError("v1 preserves the source scene's runtime-default material")
        critical_set = (
            self.binding_keypoint,
            self.critical_frame_orig,
            self.critical_frame_edit,
            self.per_keypoint_margins_m,
        )
        if any(value is not None for value in critical_set):
            if any(value is None for value in critical_set):
                raise ConstraintSpecError("binding critical-set fields must be supplied together")
            if self.binding_keypoint != self.expected_link_group:
                raise ConstraintSpecError(
                    "binding_keypoint must agree with the contact-attributed expected_link_group"
                )
            if self.critical_frame_orig < 0 or self.critical_frame_edit < 0:
                raise ConstraintSpecError("binding critical frames must be non-negative")
            if self.binding_keypoint not in self.per_keypoint_margins_m:
                raise ConstraintSpecError("binding keypoint is absent from per-keypoint margins")
            for group, margins in self.per_keypoint_margins_m.items():
                if set(margins) != {"orig_m", "edit_m"}:
                    raise ConstraintSpecError(
                        f"{group}: margins must contain exactly orig_m and edit_m"
                    )
                for motion, value in margins.items():
                    _finite(value, f"binding.per_keypoint_margins.{group}.{motion}")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ConstraintSpec":
        try:
            source = payload["source"]
            motions = payload["motions"]
            crossing = payload["crossing"]
            binding = payload["binding"]
            faces = payload["faces"]
            extent = payload["face_extent"]
            keepout = payload["keepout"]
            room = payload["room"]
            scenes = payload["source_scenes"]
            if not all(
                isinstance(value, Mapping)
                for value in (
                    source,
                    motions,
                    crossing,
                    binding,
                    faces,
                    extent,
                    keepout,
                    room,
                    scenes,
                )
            ):
                raise TypeError("nested spec records must be mappings")
            frames = crossing["frames"]
            if not isinstance(frames, Mapping):
                raise TypeError("crossing.frames must be a mapping")
            result = cls(
                spec_version=str(payload.get("spec_version") or ""),
                spec_id=str(payload.get("spec_id") or ""),
                source_kind=str(source.get("kind") or ""),
                source_family_id=str(source.get("family_id") or ""),
                source_variant_id=str(source.get("variant_id") or ""),
                orig=MotionEvidence.from_mapping(motions["orig"]),
                edit=MotionEvidence.from_mapping(motions["edit"]),
                crossing_frames_orig=tuple(int(v) for v in frames["orig"]),
                crossing_frames_edit=tuple(int(v) for v in frames["edit"]),
                axis_type=str(binding.get("axis_type") or ""),
                expected_link_group=str(binding.get("expected_link_group") or ""),
                reach_orig_m=_finite(binding.get("reach_orig_m"), "binding.reach_orig_m"),
                reach_edit_m=_finite(binding.get("reach_edit_m"), "binding.reach_edit_m"),
                window_mm=_finite(binding.get("window_mm"), "binding.window_mm"),
                easy_coordinate_m=_finite(faces.get("easy_m"), "faces.easy_m"),
                hard_coordinate_m=_finite(faces.get("hard_m"), "faces.hard_m"),
                binding_station_xy_m=tuple(
                    _finite(v, "binding_station_xy_m") for v in payload["binding_station_xy_m"]
                ),
                route_axis=str(payload.get("route_axis") or ""),
                face_along_route_m=_finite(extent.get("along_route_m"), "face_extent"),
                face_across_route_m=_finite(extent.get("across_route_m"), "face_extent"),
                face_vertical_m=(
                    None
                    if extent.get("vertical_m") is None
                    else _finite(extent.get("vertical_m"), "face_extent.vertical_m")
                ),
                binding_thickness_m=_finite(
                    extent.get("binding_thickness_m"), "face_extent.binding_thickness_m"
                ),
                keepout_delta_mm=_finite(keepout.get("delta_mm"), "keepout.delta_mm"),
                room_size_xy_m=tuple(_finite(v, "room.size_xy_m") for v in room["size_xy_m"]),
                wall_height_m=_finite(room.get("wall_height_m"), "room.wall_height_m"),
                source_scene_easy=str(scenes.get("easy") or ""),
                source_scene_hard=str(scenes.get("hard") or ""),
                material_provenance=str(payload.get("material_provenance") or ""),
                binding_keypoint=str(binding.get("keypoint") or "") or None,
                critical_frame_orig=(
                    None
                    if not isinstance(binding.get("critical_frames"), Mapping)
                    else int(binding["critical_frames"]["orig"])
                ),
                critical_frame_edit=(
                    None
                    if not isinstance(binding.get("critical_frames"), Mapping)
                    else int(binding["critical_frames"]["edit"])
                ),
                per_keypoint_margins_m=_margin_table(binding.get("per_keypoint_margins")),
            )
        except (KeyError, TypeError) as error:
            raise ConstraintSpecError(f"malformed constraint spec: {error}") from error
        result.validate()
        return result

    @classmethod
    def load(cls, path: Path) -> "ConstraintSpec":
        return cls.from_dict(json.loads(path.read_text()))

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "spec_version": self.spec_version,
            "spec_id": self.spec_id,
            "source": {
                "kind": self.source_kind,
                "family_id": self.source_family_id,
                "variant_id": self.source_variant_id,
            },
            "motions": {"orig": self.orig.to_dict(), "edit": self.edit.to_dict()},
            "crossing": {
                "frames": {
                    "orig": list(self.crossing_frames_orig),
                    "edit": list(self.crossing_frames_edit),
                }
            },
            "binding": {
                "axis_type": self.axis_type,
                "expected_link_group": self.expected_link_group,
                "reach_orig_m": self.reach_orig_m,
                "reach_edit_m": self.reach_edit_m,
                "window_mm": self.window_mm,
            },
            "faces": {"easy_m": self.easy_coordinate_m, "hard_m": self.hard_coordinate_m},
            "binding_station_xy_m": list(self.binding_station_xy_m),
            "route_axis": self.route_axis,
            "face_extent": {
                "along_route_m": self.face_along_route_m,
                "across_route_m": self.face_across_route_m,
                "vertical_m": self.face_vertical_m,
                "binding_thickness_m": self.binding_thickness_m,
            },
            "keepout": {
                "delta_mm": self.keepout_delta_mm,
                "exempt_roles": ["support_floor", "room_shell"],
            },
            "room": {
                "size_xy_m": list(self.room_size_xy_m),
                "wall_height_m": self.wall_height_m,
            },
            "source_scenes": {"easy": self.source_scene_easy, "hard": self.source_scene_hard},
            "material_provenance": self.material_provenance,
        }
        if self.binding_keypoint is not None:
            payload["binding"]["keypoint"] = self.binding_keypoint
            payload["binding"]["critical_frames"] = {
                "orig": self.critical_frame_orig,
                "edit": self.critical_frame_edit,
            }
            payload["binding"]["per_keypoint_margins"] = {
                group: {motion: float(value) for motion, value in margins.items()}
                for group, margins in self.per_keypoint_margins_m.items()
            }
        return payload

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.parent.chmod(0o775)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        path.chmod(0o664)

    def fingerprint(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"
