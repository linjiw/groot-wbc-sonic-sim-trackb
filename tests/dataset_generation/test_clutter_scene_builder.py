"""Gates for generating cluttered scenes around a known-good motion path.

The property that matters: the room may be as full as we like, but the corridor
the recorded episode actually walks must stay clear by construction. If that
guarantee slips, the generator manufactures episodes that collide.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gear_sonic.dataset_generation.clutter_scene_builder import (
    FURNITURE_CATALOG,
    build_clutter_scene,
    render_scene_usda,
)
from gear_sonic.dataset_generation.scene_asset_preflight import load_scene_obstacle_map

CLEARANCE = 0.75


def _straight(length: float = 4.0, points: int = 40) -> np.ndarray:
    xs = np.linspace(0.0, length, points)
    return np.stack([xs, np.zeros_like(xs)], axis=1)


def _curved(points: int = 60) -> np.ndarray:
    t = np.linspace(-math.pi / 2, math.pi / 2, points)
    return np.stack([np.sin(t) * 2.0, -np.cos(t) * 2.0], axis=1)


# --------------------------------------------------------------------------
# The core guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", [_straight(), _curved()])
def test_no_piece_ever_intrudes_on_the_motion_corridor(path):
    spec = build_clutter_scene(path, scene_id="unit", seed=3, clearance_m=CLEARANCE)

    assert spec.pieces
    for piece in spec.pieces:
        min_x, min_y, max_x, max_y = piece.rect
        for point in spec.path_xy:
            dx = max(min_x - point[0], 0.0, point[0] - max_x)
            dy = max(min_y - point[1], 0.0, point[1] - max_y)
            assert math.hypot(dx, dy) >= CLEARANCE - 1e-9


def test_pieces_do_not_overlap_each_other():
    spec = build_clutter_scene(_straight(), scene_id="unit", seed=5, clearance_m=CLEARANCE)
    rects = [piece.rect for piece in spec.pieces]
    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            overlap = not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])
            assert not overlap


def test_pieces_stay_inside_the_room():
    spec = build_clutter_scene(_straight(), scene_id="unit", seed=7, clearance_m=CLEARANCE)
    (min_x, min_y), (max_x, max_y) = spec.walkable_bounds()
    for piece in spec.pieces:
        rect = piece.rect
        assert min_x <= rect[0] and rect[2] <= max_x
        assert min_y <= rect[1] and rect[3] <= max_y


def test_the_room_actually_ends_up_cluttered():
    """A generator that satisfies clearance by placing nothing is useless."""
    spec = build_clutter_scene(
        _straight(), scene_id="unit", seed=1, clearance_m=CLEARANCE, target_pieces=24
    )
    assert len(spec.pieces) >= 12
    assert spec.metrics["clutter_occupancy"] > 0.10
    # Furniture must hug the corridor, not hide in the corners.
    assert spec.metrics["min_distance_to_path_m"] < CLEARANCE + 0.5


def test_a_larger_clearance_pushes_furniture_further_away():
    tight = build_clutter_scene(_straight(), scene_id="a", seed=2, clearance_m=0.75)
    loose = build_clutter_scene(_straight(), scene_id="b", seed=2, clearance_m=1.5,
                                max_distance_from_path_m=4.0)
    assert loose.metrics["min_distance_to_path_m"] >= 1.5 - 1e-9
    assert tight.metrics["min_distance_to_path_m"] >= 0.75 - 1e-9


def test_generation_is_deterministic_for_a_seed():
    first = build_clutter_scene(_straight(), scene_id="unit", seed=11, clearance_m=CLEARANCE)
    second = build_clutter_scene(_straight(), scene_id="unit", seed=11, clearance_m=CLEARANCE)
    assert [p.rect for p in first.pieces] == [p.rect for p in second.pieces]


def test_different_seeds_give_different_layouts():
    first = build_clutter_scene(_straight(), scene_id="unit", seed=1, clearance_m=CLEARANCE)
    second = build_clutter_scene(_straight(), scene_id="unit", seed=2, clearance_m=CLEARANCE)
    assert [p.rect for p in first.pieces] != [p.rect for p in second.pieces]


@pytest.mark.parametrize(
    "path,clearance",
    [(np.zeros((1, 2)), 0.75), (np.zeros((4, 3)), 0.75), (_straight(), 0.0)],
)
def test_invalid_inputs_are_rejected(path, clearance):
    with pytest.raises(ValueError):
        build_clutter_scene(path, scene_id="unit", clearance_m=clearance)


# --------------------------------------------------------------------------
# Round-trip through the checked-in preflight
# --------------------------------------------------------------------------


def test_generated_scene_parses_under_the_same_gate_as_hand_authored_ones(tmp_path):
    import json

    spec = build_clutter_scene(_curved(), scene_id="clutter_rt", seed=4, clearance_m=CLEARANCE)
    usda = render_scene_usda(spec)
    (tmp_path / "clutter_rt.usda").write_text(usda, encoding="utf-8")
    (min_x, min_y), (max_x, max_y) = spec.walkable_bounds()
    route = spec.path_xy[:: max(1, len(spec.path_xy) // 8)]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenes": [
                    {
                        "scene_id": "clutter_rt",
                        "file": "clutter_rt.usda",
                        "support_floor_prim": "/World/Structure/Floor",
                        "support_z_m": 0.0,
                        "robot_clearance_height_m": 1.9,
                        "route_clearance_radius_m": 0.45,
                        "walkable_bounds_xy": {"min": [min_x, min_y], "max": [max_x, max_y]},
                        "route_xy": [[float(x), float(y)] for x, y in route],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    obstacle_map = load_scene_obstacle_map("clutter_rt", tmp_path)

    # Every parsed obstacle must be one of the generated pieces or a wall.
    assert len(obstacle_map.obstacles) >= len(spec.pieces)
    clearance, _ = obstacle_map.clearance_to_obstacles(
        [(float(x), float(y)) for x, y in spec.path_xy]
    )
    assert clearance >= CLEARANCE - 1e-6


def test_rendered_usda_declares_collision_on_every_solid():
    spec = build_clutter_scene(_straight(), scene_id="unit", seed=9, clearance_m=CLEARANCE)
    usda = render_scene_usda(spec)
    cube_count = usda.count('def Cube "')
    assert cube_count == len(spec.pieces) + 4  # four walls
    assert usda.count("physics:collisionEnabled = 1") == cube_count + 1  # plus the floor
    assert usda.count('prepend apiSchemas = ["PhysicsCollisionAPI"]') == cube_count + 1
    assert 'upAxis = "Z"' in usda and "metersPerUnit = 1" in usda


def test_catalog_pieces_are_tall_enough_to_matter():
    """A solid shorter than the robot's step is not an obstacle worth recording."""
    for kind in FURNITURE_CATALOG:
        assert kind.size_min[2] >= 0.3, f"{kind.name} is too low to obstruct a walking G1"
