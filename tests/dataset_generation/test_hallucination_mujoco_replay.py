"""CPU tests for the LFH USDA-to-MuJoCo geometric bridge."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from gear_sonic.dataset_generation.hallucination.mujoco_replay import (
    build_mujoco_scene_xml,
    mapping_error_m,
)
from gear_sonic.dataset_generation.hallucination.stage_geometry import (
    StageCube,
    StageGeometry,
)


def test_build_mujoco_scene_preserves_selected_cube_geometry(tmp_path: Path) -> None:
    meshdir = tmp_path / "meshes"
    meshdir.mkdir()
    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text(
        '<mujoco><compiler meshdir="meshes"/><asset/><worldbody><body name="robot"/></worldbody></mujoco>',
        encoding="utf-8",
    )
    stage = StageGeometry(
        path=tmp_path / "scene.usda",
        meters_per_unit=1.0,
        source_up_axis="Z",
        room_size_xy_m=(8.0, 6.0),
        cubes=(
            StageCube(
                path="/World/Binding",
                role="binding_constraint",
                center_m=(1.25, -0.2, 1.4),
                size_m=(0.1, 3.0, 0.4),
                binding_face_id="underside",
                binding_sense="underside",
            ),
            StageCube(
                path="/World/Wall",
                role="room_shell",
                center_m=(4.0, 0.0, 1.5),
                size_m=(0.1, 6.0, 3.0),
                binding_face_id=None,
                binding_sense=None,
            ),
        ),
    )

    xml, mappings = build_mujoco_scene_xml(robot_xml, stage)
    root = ET.fromstring(xml)

    assert root.find("compiler").get("meshdir") == str(meshdir.resolve())
    assert root.find("visual/global").get("offwidth") == "1920"
    assert root.find("visual/global").get("offheight") == "1080"
    assert len(mappings) == 1
    geom = root.find("worldbody").find(f"geom[@name='{mappings[0].mujoco_geom_name}']")
    assert geom is not None
    assert tuple(map(float, geom.get("pos").split())) == pytest.approx((1.25, -0.2, 1.4))
    assert tuple(map(float, geom.get("size").split())) == pytest.approx((0.05, 1.5, 0.2))
    assert mapping_error_m(mappings[0]) == pytest.approx((0.0, 0.0))


def test_build_mujoco_scene_fails_without_a_selected_constraint(tmp_path: Path) -> None:
    robot_xml = tmp_path / "robot.xml"
    robot_xml.write_text("<mujoco><worldbody/></mujoco>", encoding="utf-8")
    stage = StageGeometry(tmp_path / "scene.usda", 1.0, "Z", (), None)

    with pytest.raises(ValueError, match="no cubes"):
        build_mujoco_scene_xml(robot_xml, stage)
