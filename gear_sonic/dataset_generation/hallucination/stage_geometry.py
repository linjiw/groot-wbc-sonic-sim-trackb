"""Dependency-free measure-back for the axis-aligned USDA subset LFH authors.

The runtime USD stack is not available in lightweight test environments. This reader therefore
implements only the same primitive subset accepted by ``scene_asset_preflight`` and fails closed
on rotations or general matrices. Unlike the older shelf-specific regex, it composes nested
translate/scale xforms, applies ``metersPerUnit``, and normalizes Y-up fixtures to Z-up metres.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_DEF = re.compile(r'\bdef\s+([A-Za-z_]\w*)\s+"([^"]+)"')


class StageGeometryError(ValueError):
    """A stage uses geometry or transforms outside the certified CPU subset."""


@dataclass(frozen=True)
class StageCube:
    path: str
    role: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    binding_face_id: str | None
    binding_sense: str | None

    @property
    def box(self) -> tuple[float, float, float, float, float, float]:
        half = tuple(value / 2 for value in self.size_m)
        return (
            self.center_m[0] - half[0],
            self.center_m[1] - half[1],
            self.center_m[2] - half[2],
            self.center_m[0] + half[0],
            self.center_m[1] + half[1],
            self.center_m[2] + half[2],
        )


@dataclass(frozen=True)
class StageGeometry:
    path: Path
    meters_per_unit: float
    source_up_axis: str
    cubes: tuple[StageCube, ...]
    room_size_xy_m: tuple[float, float] | None

    def role(self, *roles: str) -> tuple[StageCube, ...]:
        accepted = set(roles)
        return tuple(cube for cube in self.cubes if cube.role in accepted)


@dataclass(frozen=True)
class BindingMeasurement:
    coordinate_m: float
    station_xy_m: tuple[float, float]
    along_route_m: float
    across_route_m: float
    vertical_m: float
    binding_paths: tuple[str, ...]


@dataclass(frozen=True)
class _Record:
    type_name: str
    name: str
    path: str
    start: int
    body_start: int
    body_end: int
    direct_body: str


def _matching_brace(text: str, start: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise StageGeometryError("unclosed USDA prim body")


def _records(text: str) -> list[_Record]:
    raw: list[tuple[re.Match[str], int, int]] = []
    for match in _DEF.finditer(text):
        body_start = text.find("{", match.end())
        if body_start < 0:
            raise StageGeometryError(f"prim {match.group(2)!r} has no body")
        raw.append((match, body_start, _matching_brace(text, body_start)))

    result: list[_Record] = []
    stack: list[_Record] = []
    for index, (match, body_start, body_end) in enumerate(raw):
        while stack and match.start() > stack[-1].body_end:
            stack.pop()
        parent_path = stack[-1].path if stack else ""
        path = f"{parent_path}/{match.group(2)}"
        # Everything nested inside this prim is excised, not just up to the first ``def``.
        # Slicing at the first nested def folded any later sibling block's attributes into this
        # prim's own body and dropped this prim's xform ops authored after its first child --
        # both fail *open*, so a 0.5 m binding-coordinate error passed a 0.5 mm gate.
        pieces: list[str] = []
        cursor = body_start + 1
        for later_match, later_start, later_end in raw[index + 1 :]:
            if later_match.start() >= body_end:
                break
            if later_end >= body_end:
                continue
            if later_match.start() < cursor:
                continue
            pieces.append(text[cursor : later_match.start()])
            cursor = later_end + 1
        pieces.append(text[cursor:body_end])
        direct_body = "".join(pieces)
        record = _Record(
            type_name=match.group(1),
            name=match.group(2),
            path=path,
            start=match.start(),
            body_start=body_start,
            body_end=body_end,
            direct_body=direct_body,
        )
        result.append(record)
        stack.append(record)
    return result


def _scalar(body: str, name: str, default: float | None = None) -> float | None:
    match = re.search(rf"\b(?:double|float)\s+{re.escape(name)}\s*=\s*({_NUMBER})\b", body)
    if match is None:
        return default
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def _vector(
    body: str, name: str, default: tuple[float, float, float]
) -> tuple[float, float, float]:
    match = re.search(
        rf"\b(?:double3|float3)\s+{re.escape(name)}\s*=\s*"
        rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)",
        body,
    )
    if match is None:
        return default
    value = tuple(float(match.group(i)) for i in range(1, 4))
    if not all(math.isfinite(component) for component in value):
        raise StageGeometryError(f"{name} contains a non-finite value")
    return value


def _string(body: str, name: str) -> str | None:
    match = re.search(rf'\b(?:custom\s+)?string\s+{re.escape(name)}\s*=\s*"([^"]*)"', body)
    return match.group(1) if match else None


def _validate_xform_order(body: str, path: str) -> None:
    present = [
        name
        for name in ("xformOp:translate", "xformOp:scale")
        if re.search(rf"\b(?:double3|float3)\s+{re.escape(name)}\s*=", body)
    ]
    match = re.search(r"\bxformOpOrder\s*=\s*\[([^\]]*)\]", body, flags=re.S)
    if not present and match is None:
        return
    if match is None:
        raise StageGeometryError(f"{path}: authored xform ops are absent from xformOpOrder")
    ordered = re.findall(r'"([^"]+)"', match.group(1))
    if ordered != present:
        raise StageGeometryError(f"{path}: unsupported xformOpOrder {ordered}; expected {present}")


def _canonical_axis(vector: tuple[float, float, float], up_axis: str) -> tuple[float, float, float]:
    if up_axis == "Z":
        return vector
    if up_axis == "Y":
        return (vector[0], -vector[2], vector[1])
    raise StageGeometryError(f"unsupported upAxis {up_axis!r}")


def read_stage_geometry(path: Path) -> StageGeometry:
    text = path.read_text(encoding="utf-8")
    unit_match = re.search(rf"\bmetersPerUnit\s*=\s*({_NUMBER})", text)
    units = float(unit_match.group(1)) if unit_match else 0.01
    if not math.isfinite(units) or units <= 0:
        raise StageGeometryError("metersPerUnit must be finite and positive")
    axis_match = re.search(r'\bupAxis\s*=\s*"([A-Za-z])"', text)
    up_axis = axis_match.group(1).upper() if axis_match else "Y"
    if up_axis not in ("Y", "Z"):
        raise StageGeometryError(f"unsupported upAxis {up_axis!r}")

    records = _records(text)
    by_path = {record.path: record for record in records}
    transforms: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
        "": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    }
    cubes: list[StageCube] = []
    room_size: tuple[float, float] | None = None
    for record in records:
        if re.search(r"xformOp:(?:rotate|orient|transform)", record.direct_body):
            raise StageGeometryError(f"{record.path}: non-axis-aligned transform is unsupported")
        _validate_xform_order(record.direct_body, record.path)
        parent = record.path.rsplit("/", 1)[0]
        parent_translation, parent_scale = transforms.get(
            parent, ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
        )
        local_translation = _vector(record.direct_body, "xformOp:translate", (0.0, 0.0, 0.0))
        local_scale = _vector(record.direct_body, "xformOp:scale", (1.0, 1.0, 1.0))
        world_translation_units = tuple(
            parent_translation[i] + parent_scale[i] * local_translation[i] for i in range(3)
        )
        world_scale_units = tuple(parent_scale[i] * local_scale[i] for i in range(3))
        transforms[record.path] = (world_translation_units, world_scale_units)

        if (
            record.type_name == "Plane"
            and _string(record.direct_body, "g1Dataset:role") == "support_floor"
        ):
            width = _scalar(record.direct_body, "width")
            length = _scalar(record.direct_body, "length")
            if width and length:
                room_size = (width * units, length * units)
        if record.type_name != "Cube":
            continue
        size = _scalar(record.direct_body, "size")
        if size is None or size <= 0:
            raise StageGeometryError(f"{record.path}: Cube needs a positive size")
        center_units = _canonical_axis(world_translation_units, up_axis)
        scale_units = _canonical_axis(tuple(abs(v) for v in world_scale_units), up_axis)
        cubes.append(
            StageCube(
                path=record.path,
                role=_string(record.direct_body, "g1Dataset:role") or "unknown",
                center_m=tuple(value * units for value in center_units),
                size_m=tuple(abs(size * value * units) for value in scale_units),
                binding_face_id=_string(record.direct_body, "lfh:bindingFaceId"),
                binding_sense=_string(record.direct_body, "lfh:bindingSense"),
            )
        )
    # Def records are required to be nested correctly; a missing parent record is malformed.
    if any(path.rsplit("/", 1)[0] not in by_path and path.count("/") > 1 for path in by_path):
        raise StageGeometryError("stage contains a prim with an unresolved parent")
    return StageGeometry(path, units, up_axis, tuple(cubes), room_size)


def _span(cubes: Iterable[StageCube]) -> tuple[float, float, float, float, float, float]:
    boxes = [cube.box for cube in cubes]
    if not boxes:
        raise StageGeometryError("stage has no binding cubes")
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        max(box[4] for box in boxes),
        max(box[5] for box in boxes),
    )


def measure_binding(stage: StageGeometry, axis_type: str, route_axis: str) -> BindingMeasurement:
    binding = tuple(cube for cube in stage.cubes if cube.binding_face_id)
    if axis_type == "overhead":
        underside = [cube for cube in binding if cube.binding_sense == "underside"]
        if not underside:
            raise StageGeometryError("overhead stage has no underside binding handle")
        coordinates = [cube.box[2] for cube in underside]
        if max(coordinates) - min(coordinates) > 5e-7:
            raise StageGeometryError("overhead binding handles do not share one coordinate")
        box = _span(underside)
        station = ((box[0] + box[3]) / 2, (box[1] + box[4]) / 2)
        x_extent, y_extent = box[3] - box[0], box[4] - box[1]
        along, across = (x_extent, y_extent) if route_axis == "x" else (y_extent, x_extent)
        return BindingMeasurement(
            coordinates[0],
            station,
            along,
            across,
            box[5] - box[2],
            tuple(cube.path for cube in underside),
        )
    if axis_type == "lateral_gap":
        left = [cube for cube in binding if cube.binding_sense == "left_inner"]
        right = [cube for cube in binding if cube.binding_sense == "right_inner"]
        if len(left) != 1 or len(right) != 1:
            raise StageGeometryError(
                "lateral stage needs exactly one left and right binding handle"
            )
        cross_min_axis, cross_max_axis = (1, 4) if route_axis == "x" else (0, 3)
        left_inner = left[0].box[cross_max_axis]
        right_inner = right[0].box[cross_min_axis]
        gap = right_inner - left_inner
        if gap <= 0:
            raise StageGeometryError("lateral binding faces overlap")
        along_axis = 0 if route_axis == "x" else 1
        station_along = (left[0].center_m[along_axis] + right[0].center_m[along_axis]) / 2
        station_cross = (left_inner + right_inner) / 2
        station = (
            (station_along, station_cross) if route_axis == "x" else (station_cross, station_along)
        )
        along_extent = min(left[0].size_m[along_axis], right[0].size_m[along_axis])
        vertical = min(left[0].size_m[2], right[0].size_m[2])
        return BindingMeasurement(
            gap,
            station,
            along_extent,
            gap,
            vertical,
            (left[0].path, right[0].path),
        )
    raise StageGeometryError(f"unsupported binding axis {axis_type!r}")
