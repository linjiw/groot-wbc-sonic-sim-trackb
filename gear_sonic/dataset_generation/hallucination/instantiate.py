"""Deterministically instantiate a ConstraintSpec as a measured-back USDA scene pair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

from .archetypes import AuthoredPrimitive, get_archetype
from .constraint_spec import ConstraintSpec
from .stage_geometry import BindingMeasurement, measure_binding, read_stage_geometry

PLACEMENT_TOLERANCE_MM = 0.5


class InstantiationError(ValueError):
    """Authored geometry failed a mandatory measure-back preflight."""


@dataclass(frozen=True)
class SceneInstantiation:
    cell: str
    path: Path
    commanded_coordinate_m: float
    measurement: BindingMeasurement
    binding_face_offset_mm: float
    binding_station_offset_mm: float
    sha256: str
    primitives: tuple[AuthoredPrimitive, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell,
            "path": str(self.path),
            "sha256": self.sha256,
            "binding_face_commanded_m": self.commanded_coordinate_m,
            "binding_face_realized_m": self.measurement.coordinate_m,
            "binding_face_offset_mm": self.binding_face_offset_mm,
            "binding_station_realized_xy_m": list(self.measurement.station_xy_m),
            "binding_station_offset_mm": self.binding_station_offset_mm,
            "binding_paths": list(self.measurement.binding_paths),
            "authored_primitives": [primitive.to_dict() for primitive in self.primitives],
        }


@dataclass(frozen=True)
class InstantiationResult:
    spec_id: str
    archetype_id: str
    seed: int
    out_dir: Path
    manifest_path: Path
    easy: SceneInstantiation
    hard: SceneInstantiation

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "lfh_instantiation_v1",
            "spec_id": self.spec_id,
            "archetype_id": self.archetype_id,
            # Named for what it does: it draws non-binding *collision* dimensions
            # (hanging-panel thickness, ibeam web height), not only appearance.
            "geometry_seed": self.seed,
            "visual_seed": self.seed,  # deprecated alias; read geometry_seed
            "manifest_path": str(self.manifest_path),
            "material_provenance": "runtime_default",
            "scenes": {"easy": self.easy.to_dict(), "hard": self.hard.to_dict()},
        }


def _cube(primitive: AuthoredPrimitive) -> str:
    binding = ""
    if primitive.binding_face_id:
        binding = (
            f'            custom string lfh:bindingFaceId = "{primitive.binding_face_id}"\n'
            f'            custom string lfh:bindingSense = "{primitive.binding_sense}"\n'
        )
    color = primitive.color
    center = primitive.center_local_m
    size = primitive.size_m
    return f"""        def Cube "{primitive.name}" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {{
            custom string g1Dataset:role = "{primitive.role}"
{binding}            double size = 1
            bool physics:collisionEnabled = 1
            color3f[] primvars:displayColor = [({color[0]:.4f}, {color[1]:.4f}, {color[2]:.4f})]
            double3 xformOp:scale = ({size[0]:.7f}, {size[1]:.7f}, {size[2]:.7f})
            double3 xformOp:translate = ({center[0]:.7f}, {center[1]:.7f}, {center[2]:.7f})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }}
"""


def _world_cube(name: str, center, size, color=(0.82, 0.80, 0.74)) -> str:
    return _cube(AuthoredPrimitive(name, center, size, color, "room_shell"))


def render_scene_usda(
    spec: ConstraintSpec,
    scene_id: str,
    primitives: Iterable[AuthoredPrimitive],
) -> str:
    """Render the same self-contained Plane/Cube subset as existing counterfactual rooms."""

    half_x, half_y = spec.room_size_xy_m[0] / 2, spec.room_size_xy_m[1] / 2
    height = spec.wall_height_m
    station = spec.binding_station_xy_m
    constraint = "".join(_cube(primitive) for primitive in primitives)
    walls = "".join(
        (
            _world_cube(
                "WallWest", (-half_x - 0.05, 0.0, height / 2), (0.1, 2 * half_y + 0.2, height)
            ),
            _world_cube(
                "WallEast", (half_x + 0.05, 0.0, height / 2), (0.1, 2 * half_y + 0.2, height)
            ),
            _world_cube(
                "WallSouth", (0.0, -half_y - 0.05, height / 2), (2 * half_x + 0.2, 0.1, height)
            ),
            _world_cube(
                "WallNorth", (0.0, half_y + 0.05, height / 2), (2 * half_x + 0.2, 0.1, height)
            ),
        )
    )
    return f"""#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    timeCodesPerSecond = 60
    upAxis = "Z"
)

# LFH CPU instantiation. Physics verdicts are not authored here.
# spec_id={spec.spec_id} material_provenance=runtime_default
def Xform "World" (kind = "assembly")
{{
    custom string g1Dataset:sceneId = "{scene_id}"
    custom string g1Dataset:splitGroup = "{spec.spec_id}_hallucinated_v1"
    def Xform "Structure"
    {{
        def Plane "Floor" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {{
            custom string g1Dataset:role = "support_floor"
            uniform token axis = "Z"
            bool doubleSided = true
            double width = {spec.room_size_xy_m[0]:.7f}
            double length = {spec.room_size_xy_m[1]:.7f}
            bool physics:collisionEnabled = 1
        }}
{walls}    }}
    def Xform "ConstraintFrame"
    {{
        double3 xformOp:translate = ({station[0]:.7f}, {station[1]:.7f}, 0.0000000)
        uniform token[] xformOpOrder = ["xformOp:translate"]
{constraint}    }}
}}
"""


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _instantiate_one(
    spec: ConstraintSpec,
    archetype_id: str,
    seed: int,
    out_dir: Path,
    cell: str,
    coordinate: float,
) -> SceneInstantiation:
    archetype = get_archetype(archetype_id, spec.axis_type)
    primitives = archetype.build(spec, coordinate, seed)
    scene_id = f"{spec.spec_id}__{archetype_id}__s{seed:08d}__{cell}"
    path = out_dir / f"{scene_id}.usda"
    path.write_text(render_scene_usda(spec, scene_id, primitives), encoding="utf-8")
    path.chmod(0o664)
    measurement = measure_binding(read_stage_geometry(path), spec.axis_type, spec.route_axis)
    face_offset = 1000 * abs(measurement.coordinate_m - coordinate)
    station_offset = 1000 * max(
        abs(measurement.station_xy_m[i] - spec.binding_station_xy_m[i]) for i in range(2)
    )
    if face_offset > PLACEMENT_TOLERANCE_MM:
        raise InstantiationError(
            f"{cell}: binding face offset {face_offset:.3f} mm exceeds "
            f"{PLACEMENT_TOLERANCE_MM:.1f} mm"
        )
    if station_offset > PLACEMENT_TOLERANCE_MM:
        raise InstantiationError(
            f"{cell}: binding station offset {station_offset:.3f} mm exceeds "
            f"{PLACEMENT_TOLERANCE_MM:.1f} mm"
        )
    return SceneInstantiation(
        cell,
        path,
        coordinate,
        measurement,
        face_offset,
        station_offset,
        _sha256(path),
        primitives,
    )


def instantiate(
    spec: ConstraintSpec,
    archetype_id: str,
    seed: int,
    out_dir: Path,
) -> InstantiationResult:
    """Write easy/hard scenes and refuse if emitted geometry misses either measured target."""

    spec.validate()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.chmod(0o775)
    easy = _instantiate_one(spec, archetype_id, seed, out_dir, "easy", spec.easy_coordinate_m)
    hard = _instantiate_one(spec, archetype_id, seed, out_dir, "hard", spec.hard_coordinate_m)
    stem = f"{spec.spec_id}__{archetype_id}__s{seed:08d}"
    manifest_path = out_dir / f"{stem}.manifest.json"
    result = InstantiationResult(
        spec.spec_id, archetype_id, seed, out_dir, manifest_path, easy, hard
    )
    manifest = {
        **result.to_dict(),
        "constraint_spec_fingerprint": spec.fingerprint(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path.chmod(0o664)
    return result
