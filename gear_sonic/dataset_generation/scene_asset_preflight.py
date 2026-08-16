"""Dependency-free integrity and geometry gate for G1 dataset scene assets.

The M0 scene package intentionally uses a narrow, deterministic USDA subset:
axis-aligned ``Cube`` obstacles plus one Z-up ``Plane`` support floor.  This
module validates that subset without importing USD, Isaac Sim, or Isaac Lab, so
the same gate can run in lightweight CI and immediately before a simulation
job.  Isaac/USD-native validation remains an additional runtime check, not a
replacement for this fail-closed package gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

SCENE_MANIFEST_SCHEMA_VERSION = 1
_ACCEPTED_PROVENANCE_KINDS = frozenset(
    {"repo_authored_primitive_geometry", "repo_generated_primitive_geometry"}
)
DEFAULT_SCENE_PACKAGE_DIR = (
    Path(__file__).resolve().parents[1] / "data" / "assets" / "scenes" / "g1_dataset"
)

_NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_DEF_RE = re.compile(r'\bdef\s+([A-Za-z_]\w*)\s+"([^"]+)"')
_ASSET_REFERENCE_RE = re.compile(r"@[^@\n]+@")


@dataclass(frozen=True)
class AxisAlignedCube:
    """A parsed cube from the supported USDA fixture subset."""

    path: str
    size: float
    translation: tuple[float, float, float]
    scale: tuple[float, float, float]
    collision_enabled: bool

    @property
    def min_corner(self) -> tuple[float, float, float]:
        half = tuple(abs(self.size * component) / 2.0 for component in self.scale)
        return tuple(self.translation[index] - half[index] for index in range(3))

    @property
    def max_corner(self) -> tuple[float, float, float]:
        half = tuple(abs(self.size * component) / 2.0 for component in self.scale)
        return tuple(self.translation[index] + half[index] for index in range(3))


@dataclass(frozen=True)
class AxisAlignedPlane:
    """Finite authored bounds for a Z-up PhysX support plane."""

    path: str
    width: float
    length: float
    translation: tuple[float, float, float]
    collision_enabled: bool

    @property
    def min_xy(self) -> tuple[float, float]:
        return (
            self.translation[0] - self.width / 2.0,
            self.translation[1] - self.length / 2.0,
        )

    @property
    def max_xy(self) -> tuple[float, float]:
        return (
            self.translation[0] + self.width / 2.0,
            self.translation[1] + self.length / 2.0,
        )


@dataclass(frozen=True)
class ScenePreflightResult:
    """Validation result for one manifest scene entry."""

    scene_id: str
    scene_path: Path | None
    errors: tuple[str, ...]
    notes: tuple[str, ...]
    cube_count: int
    collision_prim_count: int
    declared_route_clearance_m: float | None
    minimum_route_clearance_m: float | None

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "scene_path": str(self.scene_path) if self.scene_path is not None else None,
            "ok": self.ok,
            "cube_count": self.cube_count,
            "collision_prim_count": self.collision_prim_count,
            "declared_route_clearance_m": self.declared_route_clearance_m,
            "minimum_route_clearance_m": self.minimum_route_clearance_m,
            "errors": list(self.errors),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class ScenePackagePreflightReport:
    """Machine-readable result for a complete scene package."""

    manifest_path: Path
    package_id: str | None
    errors: tuple[str, ...]
    notes: tuple[str, ...]
    scenes: tuple[ScenePreflightResult, ...]

    @property
    def ok(self) -> bool:
        return not self.errors and all(scene.ok for scene in self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "package_id": self.package_id,
            "ok": self.ok,
            "errors": list(self.errors),
            "notes": list(self.notes),
            "scenes": [scene.to_dict() for scene in self.scenes],
        }


@dataclass(frozen=True)
class _PrimRecord:
    type_name: str
    name: str
    path: str
    metadata: str
    body: str
    start: int
    body_end: int


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _find_matching_delimiter(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


def _parse_prim_records(text: str, errors: list[str]) -> list[_PrimRecord]:
    raw_records: list[tuple[re.Match[str], int, int]] = []
    for match in _DEF_RE.finditer(text):
        body_start = text.find("{", match.end())
        if body_start < 0:
            errors.append(f"prim {match.group(2)!r} has no body")
            continue
        body_end = _find_matching_delimiter(text, body_start, "{", "}")
        if body_end is None:
            errors.append(f"prim {match.group(2)!r} has an unclosed body")
            continue
        raw_records.append((match, body_start, body_end))

    records: list[_PrimRecord] = []
    parent_stack: list[_PrimRecord] = []
    for match, body_start, body_end in raw_records:
        while parent_stack and match.start() > parent_stack[-1].body_end:
            parent_stack.pop()
        parent_path = parent_stack[-1].path if parent_stack else ""
        path = f"{parent_path}/{match.group(2)}"
        record = _PrimRecord(
            type_name=match.group(1),
            name=match.group(2),
            path=path,
            metadata=text[match.end() : body_start],
            body=text[body_start + 1 : body_end],
            start=match.start(),
            body_end=body_end,
        )
        records.append(record)
        parent_stack.append(record)
    return records


def _parse_scalar(body: str, attribute: str) -> float | None:
    match = re.search(
        rf"\b(?:double|float)\s+{re.escape(attribute)}\s*=\s*({_NUMBER_PATTERN})\b", body
    )
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else None


def _parse_vector(body: str, attribute: str) -> tuple[float, float, float] | None:
    match = re.search(
        rf"\b(?:double3|float3)\s+{re.escape(attribute)}\s*=\s*"
        rf"\(\s*({_NUMBER_PATTERN})\s*,\s*({_NUMBER_PATTERN})\s*,\s*"
        rf"({_NUMBER_PATTERN})\s*\)",
        body,
    )
    if match is None:
        return None
    result = tuple(float(match.group(index)) for index in range(1, 4))
    return result if all(math.isfinite(component) for component in result) else None


def _parse_cube(record: _PrimRecord, errors: list[str]) -> AxisAlignedCube | None:
    size = _parse_scalar(record.body, "size")
    translation = _parse_vector(record.body, "xformOp:translate")
    scale = _parse_vector(record.body, "xformOp:scale")
    if size is None or size <= 0.0:
        errors.append(f"{record.path}: Cube size must be explicit, finite, and positive")
    if translation is None:
        errors.append(f"{record.path}: Cube needs a finite xformOp:translate")
    if scale is None or any(component == 0.0 for component in scale):
        errors.append(f"{record.path}: Cube needs a finite, non-zero xformOp:scale")
    if re.search(r"xformOp:(?:rotate|orient|transform)", record.body):
        errors.append(
            f"{record.path}: rotated/transformed cubes are outside the M0 preflight subset"
        )

    has_collision_api = "PhysicsCollisionAPI" in record.metadata
    collision_match = re.search(
        r"\bbool\s+physics:collisionEnabled\s*=\s*(1|0|true|false)\b", record.body
    )
    collision_enabled = bool(
        has_collision_api
        and collision_match is not None
        and collision_match.group(1).lower() in {"1", "true"}
    )
    if not collision_enabled:
        errors.append(f"{record.path}: Cube is not backed by an enabled PhysicsCollisionAPI")

    if size is None or size <= 0.0 or translation is None or scale is None:
        return None
    if any(component == 0.0 for component in scale):
        return None
    return AxisAlignedCube(
        path=record.path,
        size=size,
        translation=translation,
        scale=scale,
        collision_enabled=collision_enabled,
    )


def _parse_plane(record: _PrimRecord, errors: list[str]) -> AxisAlignedPlane | None:
    width = _parse_scalar(record.body, "width")
    length = _parse_scalar(record.body, "length")
    translation = _parse_vector(record.body, "xformOp:translate") or (0.0, 0.0, 0.0)
    axis = re.search(r'\b(?:uniform\s+)?token\s+axis\s*=\s*"([^"]+)"', record.body)
    if axis is None or axis.group(1) != "Z":
        errors.append(f"{record.path}: support Plane axis must be Z")
    if width is None or width <= 0.0 or length is None or length <= 0.0:
        errors.append(f"{record.path}: Plane width/length must be explicit and positive")
    has_collision_api = "PhysicsCollisionAPI" in record.metadata
    collision_match = re.search(
        r"\bbool\s+physics:collisionEnabled\s*=\s*(1|0|true|false)\b", record.body
    )
    collision_enabled = bool(
        has_collision_api
        and collision_match is not None
        and collision_match.group(1).lower() in {"1", "true"}
    )
    if not collision_enabled:
        errors.append(f"{record.path}: Plane is not backed by an enabled PhysicsCollisionAPI")
    if width is None or width <= 0.0 or length is None or length <= 0.0:
        return None
    return AxisAlignedPlane(
        path=record.path,
        width=width,
        length=length,
        translation=translation,
        collision_enabled=collision_enabled,
    )


def _extract_string_metadata(text: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}\s*=\s*"([^"]+)"', text)
    return match.group(1) if match is not None else None


def _extract_number_metadata(text: str, name: str) -> float | None:
    match = re.search(rf"\b{re.escape(name)}\s*=\s*({_NUMBER_PATTERN})\b", text)
    if match is None:
        return None
    number = float(match.group(1))
    return number if math.isfinite(number) else None


def _point_to_rect_distance(
    point: tuple[float, float], rect: tuple[float, float, float, float]
) -> float:
    x, y = point
    min_x, min_y, max_x, max_y = rect
    delta_x = max(min_x - x, 0.0, x - max_x)
    delta_y = max(min_y - y, 0.0, y - max_y)
    return math.hypot(delta_x, delta_y)


def _point_to_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared == 0.0:
        return math.dist(point, start)
    projection = (
        (point[0] - start[0]) * delta_x + (point[1] - start[1]) * delta_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest = (start[0] + projection * delta_x, start[1] + projection * delta_y)
    return math.dist(point, closest)


def _segment_intersects_rect(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    """Return whether a segment intersects a closed rectangle (Liang-Barsky)."""

    min_x, min_y, max_x, max_y = rect
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    for coefficient, offset in (
        (-delta_x, start[0] - min_x),
        (delta_x, max_x - start[0]),
        (-delta_y, start[1] - min_y),
        (delta_y, max_y - start[1]),
    ):
        if abs(coefficient) <= 1e-15:
            if offset < 0.0:
                return False
            continue
        ratio = offset / coefficient
        if coefficient < 0.0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)
    return lower <= upper


def _segment_to_rect_distance(
    start: tuple[float, float],
    end: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> float:
    if _segment_intersects_rect(start, end, rect):
        return 0.0
    min_x, min_y, max_x, max_y = rect
    corners = ((min_x, min_y), (min_x, max_y), (max_x, min_y), (max_x, max_y))
    return min(
        _point_to_rect_distance(start, rect),
        _point_to_rect_distance(end, rect),
        *(_point_to_segment_distance(corner, start, end) for corner in corners),
    )


def _parse_xy(value: object) -> tuple[float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    x = _finite_number(value[0])
    y = _finite_number(value[1])
    return (x, y) if x is not None and y is not None else None


def _read_route(
    scene_config: Mapping[str, object], errors: list[str]
) -> tuple[list[tuple[float, float]], float | None, float | None]:
    raw_route = scene_config.get("route_xy")
    route: list[tuple[float, float]] = []
    if not isinstance(raw_route, Sequence) or isinstance(raw_route, (str, bytes)):
        errors.append("manifest route_xy must be a sequence of at least two finite XY points")
    else:
        for index, raw_point in enumerate(raw_route):
            point = _parse_xy(raw_point)
            if point is None:
                errors.append(f"manifest route_xy[{index}] must be a finite XY point")
            else:
                route.append(point)
        if len(route) < 2 or len(route) != len(raw_route):
            errors.append("manifest route_xy must contain at least two valid points")

    clearance = _finite_number(scene_config.get("route_clearance_radius_m"))
    if clearance is None or clearance <= 0.0:
        errors.append("manifest route_clearance_radius_m must be positive and finite")
        clearance = None
    robot_height = _finite_number(scene_config.get("robot_clearance_height_m"))
    if robot_height is None or robot_height <= 0.0:
        errors.append("manifest robot_clearance_height_m must be positive and finite")
        robot_height = None
    return route, clearance, robot_height


def _validate_route(
    scene_config: Mapping[str, object],
    cubes: Sequence[AxisAlignedCube],
    floor_path: str,
    support_z: float,
    errors: list[str],
) -> tuple[float | None, float | None]:
    route, clearance, robot_height = _read_route(scene_config, errors)
    raw_bounds = scene_config.get("walkable_bounds_xy")
    min_xy = max_xy = None
    if isinstance(raw_bounds, Mapping):
        min_xy = _parse_xy(raw_bounds.get("min"))
        max_xy = _parse_xy(raw_bounds.get("max"))
    if min_xy is None or max_xy is None or min_xy[0] >= max_xy[0] or min_xy[1] >= max_xy[1]:
        errors.append("manifest walkable_bounds_xy must contain ordered finite min/max points")
        min_xy = max_xy = None
    elif route:
        for index, point in enumerate(route):
            if not (min_xy[0] <= point[0] <= max_xy[0]) or not (min_xy[1] <= point[1] <= max_xy[1]):
                errors.append(f"route_xy[{index}] lies outside walkable_bounds_xy")

    if len(route) < 2 or robot_height is None:
        return clearance, None
    obstacles = []
    for cube in cubes:
        if cube.path == floor_path or not cube.collision_enabled:
            continue
        minimum = cube.min_corner
        maximum = cube.max_corner
        if maximum[2] <= support_z or minimum[2] >= support_z + robot_height:
            continue
        obstacles.append((cube.path, (minimum[0], minimum[1], maximum[0], maximum[1])))

    minimum_clearance = math.inf
    nearest_path: str | None = None
    for start, end in zip(route, route[1:]):
        if start == end:
            errors.append("manifest route_xy must not contain zero-length segments")
        for obstacle_path, rectangle in obstacles:
            distance = _segment_to_rect_distance(start, end, rectangle)
            if distance < minimum_clearance:
                minimum_clearance = distance
                nearest_path = obstacle_path
    if math.isinf(minimum_clearance):
        errors.append("scene has no non-floor collision obstacle for route-clearance validation")
        return clearance, None
    if clearance is not None and minimum_clearance + 1e-9 < clearance:
        errors.append(
            f"route clearance {minimum_clearance:.3f} m to {nearest_path} is below declared "
            f"radius {clearance:.3f} m"
        )
    return clearance, minimum_clearance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_scene_path(package_dir: Path, raw_file: object, errors: list[str]) -> Path | None:
    if not isinstance(raw_file, str) or not raw_file:
        errors.append("manifest file must be a non-empty relative path")
        return None
    if Path(raw_file).suffix.lower() != ".usda":
        errors.append("manifest file must name a .usda scene")
    package_root = package_dir.resolve()
    scene_path = (package_root / raw_file).resolve()
    try:
        scene_path.relative_to(package_root)
    except ValueError:
        errors.append(f"manifest file escapes package directory: {raw_file}")
        return None
    if not scene_path.is_file():
        errors.append(f"scene file does not exist: {scene_path}")
        return scene_path
    return scene_path


def _validate_scene(
    scene_config: Mapping[str, object], package_dir: Path, package_license: str | None
) -> ScenePreflightResult:
    scene_id_value = scene_config.get("scene_id")
    scene_id = scene_id_value if isinstance(scene_id_value, str) and scene_id_value else "<unknown>"
    errors: list[str] = []
    notes: list[str] = []
    if scene_id == "<unknown>":
        errors.append("manifest scene_id must be a non-empty string")
    for field in ("scene_family", "split_group", "provenance"):
        if not isinstance(scene_config.get(field), str) or not scene_config[field]:
            errors.append(f"manifest {field} must be a non-empty string")
    if scene_config.get("license_spdx") != package_license:
        errors.append("scene license_spdx must match the package license")
    if scene_config.get("redistribution_allowed") is not True:
        errors.append("scene redistribution_allowed must be true")

    scene_path = _resolve_scene_path(package_dir, scene_config.get("file"), errors)
    declared_clearance = _finite_number(scene_config.get("route_clearance_radius_m"))
    if scene_path is None or not scene_path.is_file():
        return ScenePreflightResult(
            scene_id=scene_id,
            scene_path=scene_path,
            errors=tuple(errors),
            notes=tuple(notes),
            cube_count=0,
            collision_prim_count=0,
            declared_route_clearance_m=declared_clearance,
            minimum_route_clearance_m=None,
        )

    expected_hash = scene_config.get("sha256")
    actual_hash = _sha256(scene_path)
    if not isinstance(expected_hash, str) or not re.fullmatch(
        r"sha256:[0-9a-fA-F]{64}", expected_hash
    ):
        errors.append("manifest sha256 must use the sha256:<64 hex characters> form")
    elif expected_hash.removeprefix("sha256:").lower() != actual_hash:
        errors.append(f"sha256 mismatch for {scene_path.name}")
    expected_size = scene_config.get("size_bytes")
    actual_size = scene_path.stat().st_size
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 1:
        errors.append("manifest size_bytes must be a positive integer")
    elif expected_size != actual_size:
        errors.append(
            f"size mismatch for {scene_path.name}: expected {expected_size}, found {actual_size}"
        )

    try:
        text = scene_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        errors.append(f"could not read USDA as UTF-8: {exc}")
        return ScenePreflightResult(
            scene_id=scene_id,
            scene_path=scene_path,
            errors=tuple(errors),
            notes=tuple(notes),
            cube_count=0,
            collision_prim_count=0,
            declared_route_clearance_m=declared_clearance,
            minimum_route_clearance_m=None,
        )

    if not text.startswith("#usda 1.0"):
        errors.append("scene must start with '#usda 1.0'")
    asset_references = _ASSET_REFERENCE_RE.findall(text)
    if asset_references:
        errors.append("scene must be self-contained; external USD asset references were found")

    expected_default = scene_config.get("default_prim")
    default_name = _extract_string_metadata(text, "defaultPrim")
    actual_default = f"/{default_name}" if default_name else None
    if actual_default != expected_default:
        errors.append(
            f"defaultPrim mismatch: expected {expected_default!r}, found {actual_default!r}"
        )
    up_axis = _extract_string_metadata(text, "upAxis")
    if up_axis != "Z" or scene_config.get("up_axis") != "Z":
        errors.append(f"upAxis must be Z in both manifest and scene; found {up_axis!r}")
    meters_per_unit = _extract_number_metadata(text, "metersPerUnit")
    expected_meters = _finite_number(scene_config.get("meters_per_unit"))
    if (
        meters_per_unit is None
        or expected_meters is None
        or not math.isclose(meters_per_unit, 1.0, abs_tol=1e-12)
        or not math.isclose(expected_meters, 1.0, abs_tol=1e-12)
    ):
        errors.append(
            "metersPerUnit must be 1.0 in both manifest and scene; "
            f"found scene={meters_per_unit!r}, manifest={expected_meters!r}"
        )

    records = _parse_prim_records(text, errors)
    record_paths = {record.path for record in records}
    if not isinstance(expected_default, str) or expected_default not in record_paths:
        errors.append("manifest default_prim does not resolve to a defined prim")
    if _extract_string_metadata(text, "g1Dataset:sceneId") != scene_id:
        errors.append("scene g1Dataset:sceneId does not match manifest scene_id")
    if _extract_string_metadata(text, "g1Dataset:splitGroup") != scene_config.get("split_group"):
        errors.append("scene g1Dataset:splitGroup does not match manifest split_group")

    cube_records = [record for record in records if record.type_name == "Cube"]
    cubes = [cube for record in cube_records if (cube := _parse_cube(record, errors)) is not None]
    collision_records = [
        record
        for record in records
        if "PhysicsCollisionAPI" in record.metadata
        and re.search(r"\bbool\s+physics:collisionEnabled\s*=\s*(?:1|true)\b", record.body)
    ]
    floor_path = scene_config.get("support_floor_prim")
    unsupported_collisions = [
        record.path
        for record in collision_records
        if record.type_name != "Cube"
        and not (record.type_name == "Plane" and record.path == floor_path)
    ]
    if unsupported_collisions:
        errors.append(
            "collision clearance parser only supports Cube obstacles and the support Plane: "
            + ", ".join(unsupported_collisions)
        )
    minimum_collisions = scene_config.get("minimum_collision_prims")
    if (
        not isinstance(minimum_collisions, int)
        or isinstance(minimum_collisions, bool)
        or minimum_collisions < 1
    ):
        errors.append("manifest minimum_collision_prims must be a positive integer")
    elif len(collision_records) < minimum_collisions:
        errors.append(
            f"scene has {len(collision_records)} collision prims; manifest requires at least {minimum_collisions}"
        )

    support_z = _finite_number(scene_config.get("support_z_m"))
    if not isinstance(floor_path, str) or not floor_path:
        errors.append("manifest support_floor_prim must be a non-empty prim path")
        floor_path = ""
    if support_z is None:
        errors.append("manifest support_z_m must be finite")
        support_z = 0.0
    floor_record = next(
        (record for record in records if record.path == floor_path and record.type_name == "Plane"),
        None,
    )
    floor = _parse_plane(floor_record, errors) if floor_record is not None else None
    if floor is None:
        errors.append("support_floor_prim does not resolve to a parsed collision Plane")
    else:
        if not math.isclose(floor.translation[2], support_z, abs_tol=1e-9):
            errors.append(
                f"support floor is z={floor.translation[2]:.6g}, expected z={support_z:.6g}"
            )
        raw_bounds = scene_config.get("walkable_bounds_xy")
        min_xy = max_xy = None
        if isinstance(raw_bounds, Mapping):
            min_xy = _parse_xy(raw_bounds.get("min"))
            max_xy = _parse_xy(raw_bounds.get("max"))
        if (
            min_xy is not None
            and max_xy is not None
            and not (
                floor.min_xy[0] <= min_xy[0] <= max_xy[0] <= floor.max_xy[0]
                and floor.min_xy[1] <= min_xy[1] <= max_xy[1] <= floor.max_xy[1]
            )
        ):
            errors.append("support floor does not contain walkable_bounds_xy")

    declared_clearance, measured_clearance = _validate_route(
        scene_config, cubes, floor_path, support_z, errors
    )
    if measured_clearance is not None:
        notes.append(f"canonical route minimum obstacle clearance: {measured_clearance:.3f} m")
    if not errors:
        notes.append(
            f"validated {len(cubes)} collision-backed cubes plus support Plane; sha256:{actual_hash}"
        )
    return ScenePreflightResult(
        scene_id=scene_id,
        scene_path=scene_path,
        errors=tuple(errors),
        notes=tuple(notes),
        cube_count=len(cube_records),
        collision_prim_count=len(collision_records),
        declared_route_clearance_m=declared_clearance,
        minimum_route_clearance_m=measured_clearance,
    )


def preflight_scene_package(
    package_dir: str | Path = DEFAULT_SCENE_PACKAGE_DIR,
    *,
    manifest_path: str | Path | None = None,
) -> ScenePackagePreflightReport:
    """Validate package metadata, scene integrity, and canonical clear routes.

    The function never imports USD/Isaac modules and reports malformed input as
    errors instead of raising, making it suitable for both CI and runtime gates.
    """

    package_dir = Path(package_dir)
    manifest = Path(manifest_path) if manifest_path is not None else package_dir / "manifest.json"
    errors: list[str] = []
    notes: list[str] = []
    if not manifest.is_file():
        return ScenePackagePreflightReport(
            manifest_path=manifest,
            package_id=None,
            errors=(f"manifest does not exist: {manifest}",),
            notes=(),
            scenes=(),
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ScenePackagePreflightReport(
            manifest_path=manifest,
            package_id=None,
            errors=(f"could not load manifest: {exc}",),
            notes=(),
            scenes=(),
        )
    if not isinstance(payload, Mapping):
        return ScenePackagePreflightReport(
            manifest_path=manifest,
            package_id=None,
            errors=("manifest root must be a JSON object",),
            notes=(),
            scenes=(),
        )

    package_id_value = payload.get("package_id")
    package_id = package_id_value if isinstance(package_id_value, str) else None
    if not package_id:
        errors.append("manifest package_id must be a non-empty string")
    if payload.get("schema_version") != SCENE_MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest schema_version must be {SCENE_MANIFEST_SCHEMA_VERSION}")

    license_config = payload.get("license")
    package_license: str | None = None
    if not isinstance(license_config, Mapping):
        errors.append("manifest license must be an object")
    else:
        spdx = license_config.get("spdx")
        package_license = spdx if isinstance(spdx, str) and spdx else None
        if package_license is None:
            errors.append("manifest license.spdx must be a non-empty string")
        if license_config.get("redistribution_allowed") is not True:
            errors.append("manifest license.redistribution_allowed must be true")
        license_source = license_config.get("source")
        if not isinstance(license_source, str) or not license_source:
            errors.append("manifest license.source must be a non-empty path")
        elif not (package_dir / license_source).resolve().is_file():
            errors.append(f"manifest license.source does not exist: {license_source}")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("manifest provenance must be an object")
    else:
        # Procedurally generated packages are still repo-owned primitive geometry --
        # same Cube/Plane subset, same licence, no third-party content -- so they are
        # accepted, but under a distinct token so a reviewer can tell them apart.
        if provenance.get("kind") not in _ACCEPTED_PROVENANCE_KINDS:
            errors.append(
                "manifest provenance.kind must be one of "
                f"{sorted(_ACCEPTED_PROVENANCE_KINDS)}"
            )
        if provenance.get("third_party_assets") != []:
            errors.append("M0 package provenance.third_party_assets must be an empty list")

    raw_scenes = payload.get("scenes")
    scene_results: list[ScenePreflightResult] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()
    if not isinstance(raw_scenes, list) or not raw_scenes:
        errors.append("manifest scenes must be a non-empty list")
    else:
        for index, raw_scene in enumerate(raw_scenes):
            if not isinstance(raw_scene, Mapping):
                errors.append(f"manifest scenes[{index}] must be an object")
                continue
            scene_id = raw_scene.get("scene_id")
            scene_file = raw_scene.get("file")
            if isinstance(scene_id, str):
                if scene_id in seen_ids:
                    errors.append(f"duplicate scene_id: {scene_id}")
                seen_ids.add(scene_id)
            if isinstance(scene_file, str):
                if scene_file in seen_files:
                    errors.append(f"duplicate scene file: {scene_file}")
                seen_files.add(scene_file)
            scene_results.append(_validate_scene(raw_scene, package_dir, package_license))

    if scene_results:
        notes.append(f"validated {len(scene_results)} manifest scene entries")
    return ScenePackagePreflightReport(
        manifest_path=manifest,
        package_id=package_id,
        errors=tuple(errors),
        notes=tuple(notes),
        scenes=tuple(scene_results),
    )


__all__ = [
    "AxisAlignedCube",
    "AxisAlignedPlane",
    "DEFAULT_SCENE_PACKAGE_DIR",
    "SCENE_MANIFEST_SCHEMA_VERSION",
    "ScenePackagePreflightReport",
    "ScenePreflightResult",
    "preflight_scene_package",
]


# ---------------------------------------------------------------------------
# Public geometry accessors for route planning
# ---------------------------------------------------------------------------
#
# Route placement needs the same obstacle model the route-clearance gate uses.
# Exposing it here keeps exactly one definition of "what counts as an obstacle"
# instead of letting a planner drift from the gate that will judge it.


@dataclass(frozen=True)
class SceneObstacleMap:
    """Planar obstacle footprints and walkable extent for one validated scene."""

    scene_id: str
    #: ``(prim_path, (min_x, min_y, max_x, max_y))`` for every blocking solid.
    obstacles: tuple[tuple[str, tuple[float, float, float, float]], ...]
    walkable_min_xy: tuple[float, float]
    walkable_max_xy: tuple[float, float]
    support_z: float
    robot_clearance_height_m: float
    route_clearance_radius_m: float
    declared_route_xy: tuple[tuple[float, float], ...]

    def clearance_to_obstacles(
        self, points: Sequence[tuple[float, float]]
    ) -> tuple[float, str | None]:
        """Minimum distance from a polyline to any obstacle footprint."""
        if len(points) < 2:
            raise ValueError("a path needs at least two points")
        minimum = math.inf
        nearest: str | None = None
        for start, end in zip(points, points[1:]):
            for obstacle_path, rectangle in self.obstacles:
                distance = _segment_to_rect_distance(start, end, rectangle)
                if distance < minimum:
                    minimum = distance
                    nearest = obstacle_path
        return minimum, nearest

    def within_walkable_bounds(
        self, points: Sequence[tuple[float, float]], *, margin: float = 0.0
    ) -> bool:
        """Whether every point lies inside the walkable rectangle, minus a margin."""
        min_x = self.walkable_min_xy[0] + margin
        min_y = self.walkable_min_xy[1] + margin
        max_x = self.walkable_max_xy[0] - margin
        max_y = self.walkable_max_xy[1] - margin
        return all(min_x <= x <= max_x and min_y <= y <= max_y for x, y in points)


def load_scene_obstacle_map(
    scene_id: str, package_dir: str | Path = DEFAULT_SCENE_PACKAGE_DIR
) -> SceneObstacleMap:
    """Parse one scene's blocking geometry using the same rules as the route gate.

    A solid blocks if it has collision enabled, is not the support floor, and its
    vertical extent overlaps the band ``[support_z, support_z + robot_height)``.
    """
    package = Path(package_dir)
    manifest = json.loads((package / "manifest.json").read_text(encoding="utf-8"))
    scene_config = next(
        (entry for entry in manifest["scenes"] if entry.get("scene_id") == scene_id), None
    )
    if scene_config is None:
        raise ValueError(f"scene {scene_id!r} is not in {package / 'manifest.json'}")

    errors: list[str] = []
    scene_path = package / str(scene_config["file"])
    text = scene_path.read_text(encoding="utf-8")
    records = _parse_prim_records(text, errors)
    cubes = [
        cube
        for record in records
        if record.type_name == "Cube" and (cube := _parse_cube(record, errors)) is not None
    ]
    if errors:
        raise ValueError(f"scene {scene_id} failed geometry parsing: {errors}")

    support_z = _finite_number(scene_config.get("support_z_m")) or 0.0
    robot_height = _finite_number(scene_config.get("robot_clearance_height_m"))
    clearance = _finite_number(scene_config.get("route_clearance_radius_m"))
    if robot_height is None or clearance is None:
        raise ValueError(f"scene {scene_id} manifest lacks route clearance metadata")
    floor_path = str(scene_config.get("support_floor_prim", ""))

    obstacles: list[tuple[str, tuple[float, float, float, float]]] = []
    for cube in cubes:
        if cube.path == floor_path or not cube.collision_enabled:
            continue
        minimum = cube.min_corner
        maximum = cube.max_corner
        if maximum[2] <= support_z or minimum[2] >= support_z + robot_height:
            continue
        obstacles.append((cube.path, (minimum[0], minimum[1], maximum[0], maximum[1])))
    if not obstacles:
        raise ValueError(f"scene {scene_id} has no blocking obstacle to plan against")

    bounds = scene_config.get("walkable_bounds_xy")
    if not isinstance(bounds, Mapping):
        raise ValueError(f"scene {scene_id} manifest lacks walkable_bounds_xy")
    min_xy = _parse_xy(bounds.get("min"))
    max_xy = _parse_xy(bounds.get("max"))
    if min_xy is None or max_xy is None:
        raise ValueError(f"scene {scene_id} walkable_bounds_xy is malformed")

    route = tuple(
        point for raw in scene_config.get("route_xy", []) if (point := _parse_xy(raw)) is not None
    )
    return SceneObstacleMap(
        scene_id=scene_id,
        obstacles=tuple(obstacles),
        walkable_min_xy=min_xy,
        walkable_max_xy=max_xy,
        support_z=support_z,
        robot_clearance_height_m=robot_height,
        route_clearance_radius_m=clearance,
        declared_route_xy=route,
    )
