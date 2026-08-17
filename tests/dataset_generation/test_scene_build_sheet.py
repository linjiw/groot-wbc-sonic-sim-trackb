"""Tests for turning a released scene into a physical build sheet."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.scene_build_sheet import (
    BuildSheetError,
    build_sheet_from_scene,
    parse_scene_usda,
    rect_distance_to_route,
    render_build_sheet_markdown,
    render_build_sheet_svg,
    scene_id_from_usda,
)

USDA = """#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "World" (
    kind = "assembly"
)
{
    custom string g1Dataset:sceneId = "test_room"

    def Xform "Structure"
    {
        def Plane "Floor" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            custom string g1Dataset:role = "support_floor"
            uniform token axis = "Z"
            double width = 6.000
            double length = 4.000
        }

        def Cube "WallWest" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            custom string g1Dataset:role = "wall"
            double size = 1
            double3 xformOp:scale = (0.1000, 4.2000, 2.8000)
            double3 xformOp:translate = (-3.0500, 0.0000, 1.4000)
        }

        def Cube "Crate_00" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            custom string g1Dataset:role = "clutter"
            double size = 1
            double3 xformOp:scale = (0.4000, 0.4000, 0.5000)
            double3 xformOp:translate = (0.0000, 1.5000, 0.2500)
        }

        def Cube "Shelf_01" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            custom string g1Dataset:role = "furniture"
            double size = 1
            double3 xformOp:scale = (0.6000, 0.3000, 0.2000)
            double3 xformOp:translate = (0.0000, 0.0000, 1.6000)
        }
    }
}
"""

#: A straight route along y = 0, which the shelf overhangs and the crate does not.
SCENE_ENTRY = {
    "scene_id": "test_room",
    "file": "test_room.usda",
    "route_xy": [[-2.0, 0.0], [2.0, 0.0]],
    "route_clearance_radius_m": 0.45,
    "robot_clearance_height_m": 1.9,
}


def build() -> object:
    return build_sheet_from_scene(USDA, scene_entry=SCENE_ENTRY, package_id="test_pkg")


def test_parses_planes_and_cubes_with_roles():
    prims = {prim.name: prim for prim in parse_scene_usda(USDA)}
    assert set(prims) == {"Floor", "WallWest", "Crate_00", "Shelf_01"}
    assert prims["Floor"].size[:2] == (6.0, 4.0)
    assert prims["Crate_00"].center == (0.0, 1.5, 0.25)
    assert scene_id_from_usda(USDA) == "test_room"


def test_z_base_distinguishes_floor_standing_from_raised():
    prims = {prim.name: prim for prim in parse_scene_usda(USDA)}
    # Crate: centre 0.25 with height 0.5 sits exactly on the floor.
    assert prims["Crate_00"].z_base == pytest.approx(0.0)
    assert not prims["Crate_00"].is_raised
    # Shelf: centre 1.6 with height 0.2 has its underside at 1.5 m.
    assert prims["Shelf_01"].z_base == pytest.approx(1.5)
    assert prims["Shelf_01"].is_raised


def test_structural_prims_are_not_build_items():
    sheet = build()
    assert {item.name for item in sheet.items} == {"Crate_00", "Shelf_01"}


def test_semantic_roles_are_included_not_just_clutter():
    """The hand-authored scenes label solids "furniture"/"storage_rack"/"pallet".

    An allow-list of "clutter" silently emitted an empty sheet for those scenes, which
    is indistinguishable from a room that genuinely has nothing in it.
    """
    sheet = build()
    assert {item.role for item in sheet.items} == {"clutter", "furniture"}


def test_empty_sheet_raises_rather_than_reporting_an_empty_room():
    with pytest.raises(BuildSheetError, match="no buildable solids"):
        build_sheet_from_scene(
            USDA,
            scene_entry=SCENE_ENTRY,
            package_id="test_pkg",
            exclude_roles=("support_floor", "wall", "clutter", "furniture"),
        )


def test_tape_frame_coordinates_are_positive_and_offset_by_half_the_room():
    sheet = build()
    assert sheet.tape_origin_scene_m == pytest.approx((-3.0, -2.0))
    crate = next(item for item in sheet.items if item.name == "Crate_00")
    # Scene centre (0, 1.5), footprint 0.4 x 0.4 -> corner (-0.2, 1.3) -> tape (2.8, 3.3).
    assert crate.tape_corner_m == pytest.approx((2.8, 3.3))
    assert crate.tape_center_m == pytest.approx((3.0, 3.5))
    for item in sheet.items:
        assert item.tape_corner_m[0] >= 0.0 and item.tape_corner_m[1] >= 0.0


def test_placement_tolerance_is_distance_minus_required_clearance():
    sheet = build()
    crate = next(item for item in sheet.items if item.name == "Crate_00")
    # Footprint spans y in [1.3, 1.7]; the route is y = 0, so the clear gap is 1.3 m.
    assert crate.distance_to_route_m == pytest.approx(1.3)
    assert crate.placement_tolerance_m == pytest.approx(1.3 - 0.45)
    assert not crate.is_over_corridor


def test_piece_over_the_corridor_gets_a_negative_tolerance():
    sheet = build()
    shelf = next(item for item in sheet.items if item.name == "Shelf_01")
    assert shelf.distance_to_route_m == pytest.approx(0.0)
    assert shelf.is_over_corridor
    assert shelf.stand_height_m == pytest.approx(1.5)
    assert sheet.lowest_overhead_m == pytest.approx(1.5)


def test_tightest_tolerance_ignores_pieces_over_the_corridor():
    sheet = build()
    # Only the crate is floor-standing, so it sets the tightest tolerance; the shelf's
    # negative value would otherwise make every room look unbuildable.
    assert sheet.tightest_tolerance_m == pytest.approx(0.85)


def test_rect_distance_is_zero_when_the_route_crosses_the_footprint():
    route = np.array([[-1.0, 0.0], [1.0, 0.0]])
    assert rect_distance_to_route((-0.5, -0.5, 0.5, 0.5), route) == pytest.approx(0.0)


def test_rect_distance_uses_corner_to_segment_not_just_vertices():
    """A route whose vertices are far away can still pass close to a corner."""
    route = np.array([[-5.0, 0.5], [5.0, 0.5]])
    # Nearest vertex is >5 m away, but the segment runs 0.5 m above the rectangle top.
    assert rect_distance_to_route((-0.5, -0.5, 0.5, 0.0), route) == pytest.approx(0.5)


def test_markdown_reports_both_groups_and_the_underside_height():
    text = render_build_sheet_markdown(build())
    assert "Floor-standing pieces" in text
    assert "Raised pieces" in text
    assert "1.500" in text  # the underside a walker passes beneath
    assert "overhang the walking corridor" in text


def test_svg_is_well_formed_and_marks_the_tape_origin():
    svg = render_build_sheet_svg(build())
    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert svg.count("<svg") == 1
    assert "tape origin" in svg


def test_route_is_translated_into_the_tape_frame():
    sheet = build()
    assert sheet.route_tape_m.shape == (2, 2)
    # Scene route runs from (-2, 0) to (2, 0); origin is (-3, -2).
    assert sheet.route_tape_m[0] == pytest.approx([1.0, 2.0])
    assert sheet.route_tape_m[-1] == pytest.approx([5.0, 2.0])
