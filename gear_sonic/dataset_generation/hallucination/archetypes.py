"""Deterministic v1 LFH primitive archetypes.

All geometry is static, axis-aligned Cube collision geometry. Material attributes are deliberately
omitted so variants preserve the source shelf's runtime-default material provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

from .constraint_spec import ConstraintSpec


@dataclass(frozen=True)
class AuthoredPrimitive:
    name: str
    center_local_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    color: tuple[float, float, float]
    role: str
    binding_face_id: str | None = None
    binding_sense: str | None = None
    optional: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "center_local_m": list(self.center_local_m),
            "size_m": list(self.size_m),
            "role": self.role,
            "binding_face_id": self.binding_face_id,
            "binding_sense": self.binding_sense,
            "optional": self.optional,
        }


@dataclass(frozen=True)
class Archetype:
    archetype_id: str
    axis_type: str
    build: Callable[[ConstraintSpec, float, int], tuple[AuthoredPrimitive, ...]]


def _jitter_color(
    base: tuple[float, float, float], rng: random.Random
) -> tuple[float, float, float]:
    return tuple(max(0.05, min(0.95, value + rng.uniform(-0.06, 0.06))) for value in base)


def _axes(spec: ConstraintSpec, along: float, across: float, vertical: float):
    return (along, across, vertical) if spec.route_axis == "x" else (across, along, vertical)


def _overhead_binding(
    spec: ConstraintSpec,
    coordinate: float,
    *,
    name: str,
    thickness: float,
    color: tuple[float, float, float],
) -> AuthoredPrimitive:
    return AuthoredPrimitive(
        name=name,
        center_local_m=(0.0, 0.0, coordinate + thickness / 2),
        size_m=_axes(spec, spec.face_along_route_m, spec.face_across_route_m, thickness),
        color=color,
        role="binding_constraint",
        binding_face_id=f"{spec.spec_id}:underside",
        binding_sense="underside",
    )


def _shelf_plank(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    color = _jitter_color((0.46, 0.34, 0.22), rng)
    return (
        _overhead_binding(
            spec,
            coordinate,
            name="BindingShelfPlank",
            thickness=spec.binding_thickness_m,
            color=color,
        ),
    )


def _ibeam(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    # The historical hard-cell capsule proxy extends roughly 68 mm beyond the contact face.
    # A thin flange would therefore let the non-binding web enter the 50 mm keepout corridor,
    # even though physics stops the body at the flange. Keep the web behind a sufficiently deep
    # binding solid so the CPU certificate does not rely on that dynamic response.
    thickness = max(0.14, spec.binding_thickness_m * 1.4)
    color = _jitter_color((0.31, 0.34, 0.36), rng)
    web_height = rng.uniform(0.18, 0.26)
    return (
        _overhead_binding(
            spec,
            coordinate,
            name="BindingIBeamFlange",
            thickness=thickness,
            color=color,
        ),
        AuthoredPrimitive(
            "IBeamWeb",
            (0.0, 0.0, coordinate + thickness + web_height / 2),
            _axes(spec, spec.face_along_route_m, 0.10, web_height),
            color,
            "constraint_context",
        ),
        AuthoredPrimitive(
            "IBeamTopFlange",
            (0.0, 0.0, coordinate + thickness + web_height + 0.035),
            _axes(spec, spec.face_along_route_m, spec.face_across_route_m, 0.07),
            color,
            "constraint_context",
        ),
    )


def _hvac_duct(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    thickness = max(0.16, spec.binding_thickness_m)
    return (
        _overhead_binding(
            spec,
            coordinate,
            name="BindingHvacDuct",
            thickness=thickness,
            color=_jitter_color((0.63, 0.67, 0.69), rng),
        ),
    )


def _door_lintel(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    thickness = max(0.12, spec.binding_thickness_m)
    color = _jitter_color((0.48, 0.36, 0.24), rng)
    cross_offset = spec.face_across_route_m / 2 + 0.06
    jamb_size = _axes(spec, 0.12, 0.12, coordinate + thickness)
    if spec.route_axis == "x":
        jamb_centers = (
            (0.0, -cross_offset, (coordinate + thickness) / 2),
            (0.0, cross_offset, (coordinate + thickness) / 2),
        )
    else:
        jamb_centers = (
            (-cross_offset, 0.0, (coordinate + thickness) / 2),
            (cross_offset, 0.0, (coordinate + thickness) / 2),
        )
    return (
        _overhead_binding(
            spec,
            coordinate,
            name="BindingDoorLintel",
            thickness=thickness,
            color=color,
        ),
        AuthoredPrimitive(
            "DoorJambLeft", jamb_centers[0], jamb_size, color, "constraint_context", optional=True
        ),
        AuthoredPrimitive(
            "DoorJambRight", jamb_centers[1], jamb_size, color, "constraint_context", optional=True
        ),
    )


def _hanging_panel(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    panel_height = rng.uniform(0.28, 0.42)
    return (
        _overhead_binding(
            spec,
            coordinate,
            name="BindingHangingPanel",
            thickness=panel_height,
            color=_jitter_color((0.30, 0.43, 0.52), rng),
        ),
    )


def _lateral_binding(
    spec: ConstraintSpec,
    gap: float,
    *,
    name_prefix: str,
    thickness: float,
    color: tuple[float, float, float],
) -> tuple[AuthoredPrimitive, AuthoredPrimitive]:
    height = float(spec.face_vertical_m)
    outward_depth = max(thickness, spec.face_across_route_m)
    cross = gap / 2 + outward_depth / 2
    size = _axes(spec, spec.face_along_route_m, outward_depth, height)
    if spec.route_axis == "x":
        left_center, right_center = (0.0, -cross, height / 2), (0.0, cross, height / 2)
    else:
        left_center, right_center = (-cross, 0.0, height / 2), (cross, 0.0, height / 2)
    return (
        AuthoredPrimitive(
            f"{name_prefix}Left",
            left_center,
            size,
            color,
            "binding_constraint",
            f"{spec.spec_id}:left_inner",
            "left_inner",
        ),
        AuthoredPrimitive(
            f"{name_prefix}Right",
            right_center,
            size,
            color,
            "binding_constraint",
            f"{spec.spec_id}:right_inner",
            "right_inner",
        ),
    )


def _pinch_panels(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    return _lateral_binding(
        spec,
        coordinate,
        name_prefix="BindingPinchPanel",
        thickness=max(0.10, spec.binding_thickness_m),
        color=_jitter_color((0.38, 0.42, 0.45), rng),
    )


def _doorway(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    thickness = max(0.12, spec.binding_thickness_m)
    color = _jitter_color((0.50, 0.37, 0.25), rng)
    binding = _lateral_binding(
        spec, coordinate, name_prefix="BindingDoorway", thickness=thickness, color=color
    )
    outward_depth = max(thickness, spec.face_across_route_m)
    lintel_height = min(spec.wall_height_m - 0.05, float(spec.face_vertical_m) + 0.20)
    return binding + (
        AuthoredPrimitive(
            "DoorwayLintelContext",
            (0.0, 0.0, lintel_height),
            _axes(spec, spec.face_along_route_m, coordinate + 2 * outward_depth, 0.10),
            color,
            "constraint_context",
            optional=True,
        ),
    )


def _rack_aisle(spec: ConstraintSpec, coordinate: float, seed: int):
    rng = random.Random(seed)
    thickness = max(0.16, spec.binding_thickness_m)
    color = _jitter_color((0.35, 0.36, 0.31), rng)
    binding = _lateral_binding(
        spec, coordinate, name_prefix="BindingRack", thickness=thickness, color=color
    )
    outer = coordinate / 2 + max(thickness, spec.face_across_route_m) + 0.22
    size = _axes(spec, spec.face_along_route_m, 0.18, float(spec.face_vertical_m))
    if spec.route_axis == "x":
        centers = (
            (0.0, -outer, float(spec.face_vertical_m) / 2),
            (0.0, outer, float(spec.face_vertical_m) / 2),
        )
    else:
        centers = (
            (-outer, 0.0, float(spec.face_vertical_m) / 2),
            (outer, 0.0, float(spec.face_vertical_m) / 2),
        )
    return binding + (
        AuthoredPrimitive("RackOuterLeft", centers[0], size, color, "constraint_context"),
        AuthoredPrimitive("RackOuterRight", centers[1], size, color, "constraint_context"),
    )


ARCHETYPES: dict[str, Archetype] = {
    "shelf_plank": Archetype("shelf_plank", "overhead", _shelf_plank),
    "ibeam": Archetype("ibeam", "overhead", _ibeam),
    "hvac_duct": Archetype("hvac_duct", "overhead", _hvac_duct),
    "door_lintel": Archetype("door_lintel", "overhead", _door_lintel),
    "hanging_panel": Archetype("hanging_panel", "overhead", _hanging_panel),
    "pinch_panels": Archetype("pinch_panels", "lateral_gap", _pinch_panels),
    "doorway": Archetype("doorway", "lateral_gap", _doorway),
    "rack_aisle": Archetype("rack_aisle", "lateral_gap", _rack_aisle),
}


def get_archetype(archetype_id: str, axis_type: str) -> Archetype:
    try:
        archetype = ARCHETYPES[archetype_id]
    except KeyError as error:
        raise ValueError(f"unknown archetype {archetype_id!r}") from error
    if archetype.axis_type != axis_type:
        raise ValueError(f"archetype {archetype_id!r} is {archetype.axis_type}, not {axis_type}")
    return archetype
