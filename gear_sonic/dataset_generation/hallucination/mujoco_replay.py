"""Build a MuJoCo cross-render scene from LFH's certified USDA cube subset.

This bridge is intentionally geometric.  It reproduces LFH obstacle centres and extents in
MuJoCo so an already recorded SONIC/Isaac trajectory can be inspected in a second renderer.  It
does not step MuJoCo physics and therefore cannot create or replace a physics verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from .stage_geometry import StageGeometry


@dataclass(frozen=True)
class MujocoBoxMapping:
    """Exact USDA-cube to MuJoCo-box mapping used by a replay."""

    stage_path: str
    role: str
    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    mujoco_geom_name: str
    mujoco_half_size_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "stage_path": self.stage_path,
            "role": self.role,
            "center_m": list(self.center_m),
            "size_m": list(self.size_m),
            "mujoco_geom_name": self.mujoco_geom_name,
            "mujoco_half_size_m": list(self.mujoco_half_size_m),
        }


_ROLE_RGBA = {
    "binding_constraint": "0.10 0.35 0.95 0.82",
    "constraint_context": "0.90 0.50 0.12 0.92",
    "room_shell": "0.72 0.76 0.82 0.20",
    "unknown": "0.55 0.55 0.55 0.65",
}


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.10g}" for value in values)


def build_mujoco_scene_xml(
    robot_xml: Path,
    stage: StageGeometry,
    *,
    rendered_roles: tuple[str, ...] = ("binding_constraint", "constraint_context"),
) -> tuple[str, tuple[MujocoBoxMapping, ...]]:
    """Embed selected axis-aligned LFH cubes into a self-contained MuJoCo XML string."""

    robot_xml = robot_xml.resolve()
    root = ET.parse(robot_xml).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    raw_meshdir = compiler.get("meshdir", ".")
    meshdir = Path(raw_meshdir)
    if not meshdir.is_absolute():
        meshdir = robot_xml.parent / meshdir
    compiler.set("meshdir", str(meshdir.resolve()))

    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1920")
    global_visual.set("offheight", "1080")

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "lfh_floor_grid",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.20 0.23",
            "rgb2": "0.28 0.31 0.35",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "lfh_floor_material",
            "texture": "lfh_floor_grid",
            "texrepeat": "8 6",
            "reflectance": "0.08",
        },
    )

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"robot model has no worldbody: {robot_xml}")
    half_room = (
        tuple(value / 2.0 for value in stage.room_size_xy_m)
        if stage.room_size_xy_m is not None
        else (6.0, 5.0)
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "lfh_support_floor",
            "type": "plane",
            "pos": "0 0 -0.002",
            "size": _numbers((half_room[0], half_room[1], 0.05)),
            "material": "lfh_floor_material",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "lfh_key_light",
            "pos": "1.8 -2.5 5.0",
            "dir": "0.0 0.25 -1.0",
            "directional": "true",
            "diffuse": "0.85 0.85 0.85",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "lfh_fill_light",
            "pos": "2.0 3.0 3.0",
            "dir": "0.0 -0.4 -1.0",
            "directional": "true",
            "diffuse": "0.45 0.48 0.55",
        },
    )

    accepted_roles = set(rendered_roles)
    mappings: list[MujocoBoxMapping] = []
    for index, cube in enumerate(cube for cube in stage.cubes if cube.role in accepted_roles):
        half_size = tuple(value / 2.0 for value in cube.size_m)
        name = f"lfh_cube_{index:03d}_{cube.role}"
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": _numbers(cube.center_m),
                "size": _numbers(half_size),
                "rgba": _ROLE_RGBA.get(cube.role, _ROLE_RGBA["unknown"]),
                "contype": "0",
                "conaffinity": "0",
            },
        )
        mappings.append(
            MujocoBoxMapping(
                stage_path=cube.path,
                role=cube.role,
                center_m=cube.center_m,
                size_m=cube.size_m,
                mujoco_geom_name=name,
                mujoco_half_size_m=half_size,
            )
        )
    if not mappings:
        raise ValueError(f"stage has no cubes for roles {sorted(accepted_roles)}")

    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode"), tuple(mappings)


def mapping_error_m(mapping: MujocoBoxMapping) -> tuple[float, float]:
    """Return centre and full-size reconstruction errors for a mapping."""

    reconstructed_size = tuple(2.0 * value for value in mapping.mujoco_half_size_m)
    center_error = max(abs(a - b) for a, b in zip(mapping.center_m, mapping.center_m))
    size_error = max(abs(a - b) for a, b in zip(mapping.size_m, reconstructed_size))
    return center_error, size_error
