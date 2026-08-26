"""Determinism and binding-handle gates for the v1 archetype library."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from gear_sonic.dataset_generation.hallucination.archetypes import ARCHETYPES
from gear_sonic.dataset_generation.hallucination.constraint_spec import ConstraintSpec
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ConstraintSpec.load(REPO_ROOT / "specs/hallucination/cs_duck_003.json")


def _lateral_spec() -> ConstraintSpec:
    return replace(
        GOLDEN,
        spec_id="cs_lateral_fixture",
        axis_type="lateral_gap",
        expected_link_group="shoulder_left",
        reach_orig_m=0.38,
        reach_edit_m=0.28,
        window_mm=100.0,
        easy_coordinate_m=0.8,
        hard_coordinate_m=0.6,
        face_along_route_m=1.2,
        face_across_route_m=0.8,
        face_vertical_m=2.0,
        binding_thickness_m=0.12,
        binding_keypoint=None,
        critical_frame_orig=None,
        critical_frame_edit=None,
        per_keypoint_margins_m=None,
    )


def test_library_has_five_overhead_and_three_lateral_archetypes() -> None:
    counts = {
        axis: sum(archetype.axis_type == axis for archetype in ARCHETYPES.values())
        for axis in ("overhead", "lateral_gap")
    }
    assert counts == {"overhead": 5, "lateral_gap": 3}


@pytest.mark.parametrize(
    "archetype_id",
    [name for name, archetype in ARCHETYPES.items() if archetype.axis_type == "overhead"],
)
def test_every_overhead_archetype_measures_back_to_the_golden_faces(
    tmp_path: Path, archetype_id: str
) -> None:
    result = instantiate(GOLDEN, archetype_id, 17, tmp_path / archetype_id)
    assert result.easy.measurement.coordinate_m == pytest.approx(1.390625, abs=0.0005)
    assert result.hard.measurement.coordinate_m == pytest.approx(1.2125, abs=0.0005)
    assert result.easy.binding_station_offset_mm <= 0.5
    assert result.hard.binding_station_offset_mm <= 0.5


@pytest.mark.parametrize(
    "archetype_id",
    [name for name, archetype in ARCHETYPES.items() if archetype.axis_type == "lateral_gap"],
)
def test_every_lateral_archetype_measures_back_to_both_gap_widths(
    tmp_path: Path, archetype_id: str
) -> None:
    spec = _lateral_spec()
    result = instantiate(spec, archetype_id, 23, tmp_path / archetype_id)
    assert result.easy.measurement.coordinate_m == pytest.approx(0.8, abs=0.0005)
    assert result.hard.measurement.coordinate_m == pytest.approx(0.6, abs=0.0005)
    assert result.easy.binding_station_offset_mm <= 0.5
    cross_axis = 1 if spec.route_axis == "x" else 0
    for scene in (result.easy, result.hard):
        binding = [primitive for primitive in scene.primitives if primitive.binding_face_id]
        assert len(binding) == 2
        assert all(
            primitive.size_m[cross_axis] >= spec.face_across_route_m for primitive in binding
        )


def test_same_spec_archetype_and_seed_are_byte_deterministic(tmp_path: Path) -> None:
    first = instantiate(GOLDEN, "shelf_plank", 41, tmp_path / "first")
    second = instantiate(GOLDEN, "shelf_plank", 41, tmp_path / "second")
    third = instantiate(GOLDEN, "shelf_plank", 42, tmp_path / "third")
    assert first.easy.path.read_bytes() == second.easy.path.read_bytes()
    assert first.hard.path.read_bytes() == second.hard.path.read_bytes()
    assert first.easy.path.read_bytes() != third.easy.path.read_bytes()
    assert first.easy.measurement.coordinate_m == third.easy.measurement.coordinate_m
    assert first.easy.path.name == "cs_duck_003__shelf_plank__s00000041__easy.usda"
    assert first.manifest_path.name == "cs_duck_003__shelf_plank__s00000041.manifest.json"
