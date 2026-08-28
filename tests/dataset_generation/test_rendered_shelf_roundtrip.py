"""The shelf physics loads must be the shelf the search reasoned about.

These are different artefacts -- an ObstacleSpec in memory and a USD prim on disk -- and
nothing checked they agreed. They silently did not: the boundary search placed its obstacle
at the station where two motions differ most, while the renderer recomputed the position as
the path midpoint half a metre away. Both motions hit the shelf and a 0.178 m window produced
no family, which read as a scientific failure and was a plumbing one.

So the round trip is pinned here: render a scene, parse the geometry back out of the file,
and require it to match what was asked for.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.clutter_scene_builder import (
    ClutterSceneSpec,
    FurniturePiece,
    render_scene_usda,
)
from gear_sonic.dataset_generation.counterfactual_family import ObstacleSpec

MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts/research/build_counterfactual_family.py"
)
spec = importlib.util.spec_from_file_location("build_counterfactual_family", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

SHELF_SIZE = (0.5, 3.0, 0.10)


def write_scene_file(tmp_path: Path, center_xy, z_base: float) -> Path:
    piece = FurniturePiece(
        name="LowShelf_00",
        kind="WallShelf",
        center_xy=(float(center_xy[0]), float(center_xy[1])),
        size=SHELF_SIZE,
        color=(0.46, 0.34, 0.22),
        z_base=float(z_base),
        band="overhead",
    )
    path_xy = np.stack([np.linspace(0.0, 4.0, 20), np.zeros(20)], axis=1)
    scene = ClutterSceneSpec(
        scene_id="cf_test_hard",
        split_group="cf_test_family_v1",
        room_size_xy=(10.0, 5.0),
        wall_height=2.8,
        pieces=[piece],
        path_xy=path_xy,
        clearance_m=0.0,
        seed=0,
        metrics={
            "scene_start_xy": [0.0, 0.0],
            "placed_pieces": 1,
            "clutter_occupancy": 0.0,
            "min_distance_to_path_m": 0.0,
            "path_length_m": 4.0,
        },
    )
    destination = tmp_path / "cf_test_hard.usda"
    destination.write_text(render_scene_usda(scene), encoding="utf-8")
    return destination


def test_the_rendered_box_matches_the_position_asked_for(tmp_path):
    usda = write_scene_file(tmp_path, (2.68, 0.0), 1.212)
    min_x, min_y, min_z, max_x, max_y, max_z = module.rendered_shelf_box(usda)
    assert (min_x + max_x) / 2 == pytest.approx(2.68, abs=1e-6)
    assert (min_y + max_y) / 2 == pytest.approx(0.0, abs=1e-6)
    assert max_x - min_x == pytest.approx(SHELF_SIZE[0], abs=1e-6)
    assert max_y - min_y == pytest.approx(SHELF_SIZE[1], abs=1e-6)
    assert max_z - min_z == pytest.approx(SHELF_SIZE[2], abs=1e-6)


def test_the_rendered_underside_is_the_height_asked_for(tmp_path):
    """The underside is what the family is defined by, so it is what must round-trip."""
    usda = write_scene_file(tmp_path, (2.68, 0.0), 1.212)
    assert module.rendered_shelf_box(usda)[2] == pytest.approx(1.212, abs=1e-6)


def test_moving_the_shelf_moves_the_rendered_box(tmp_path):
    """The bug: the position was passed in but ignored, so this would not have changed."""
    near = module.rendered_shelf_box(write_scene_file(tmp_path / "a", (2.23, 0.0), 1.212))
    far = module.rendered_shelf_box(write_scene_file(tmp_path / "b", (2.68, 0.0), 1.212))
    assert far[0] - near[0] == pytest.approx(0.45, abs=1e-6)


def test_the_rendered_box_agrees_with_the_obstacle_spec_the_search_used(tmp_path):
    """The exact invariant the builder now checks before spending rollouts."""
    station, underside = (2.68, 1.212)
    obstacle = ObstacleSpec(
        name="LowShelf",
        size=SHELF_SIZE,
        base_center=(station, 0.0, 2.30),
        axis=(0.0, 0.0, -1.0),
        regime="overhead",
    )
    # Parameter that puts the spec's underside at the same height as the rendered scene.
    parameter = 2.30 - SHELF_SIZE[2] / 2 - underside
    searched = obstacle.box_at(parameter)
    rendered = module.rendered_shelf_box(write_scene_file(tmp_path, (station, 0.0), underside))
    for expected, actual in zip(searched, rendered):
        assert actual == pytest.approx(expected, abs=1e-6)


@pytest.fixture(autouse=True)
def _make_subdirs(tmp_path):
    (tmp_path / "a").mkdir(exist_ok=True)
    (tmp_path / "b").mkdir(exist_ok=True)
