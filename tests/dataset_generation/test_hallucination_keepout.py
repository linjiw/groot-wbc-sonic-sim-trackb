"""Property-style refusal gates for LFH single-cause geometry."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.archetypes import (
    AuthoredPrimitive,
    get_archetype,
)
from gear_sonic.dataset_generation.hallucination.constraint_spec import ConstraintSpec
from gear_sonic.dataset_generation.hallucination.instantiate import (
    instantiate,
    render_scene_usda,
)
from gear_sonic.dataset_generation.hallucination.validate_keepout import validate_scene
from gear_sonic.dataset_generation.swept_volume import G1_COLLISION_CAPSULES

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ConstraintSpec.load(REPO_ROOT / "specs/hallucination/cs_duck_003.json")


def _fixture_spec() -> ConstraintSpec:
    return replace(
        GOLDEN,
        spec_id="cs_keepout_fixture",
        binding_station_xy_m=(0.0, 0.0),
        easy_coordinate_m=1.6,
        hard_coordinate_m=1.4,
        reach_orig_m=1.45,
        reach_edit_m=1.25,
        window_mm=200.0,
        crossing_frames_orig=(2, 2),
        crossing_frames_edit=(2, 2),
        face_along_route_m=1.2,
        face_across_route_m=2.0,
        room_size_xy_m=(8.0, 5.0),
    )


def _payload() -> dict:
    names = sorted(G1_COLLISION_CAPSULES)
    frames = 5
    xs = np.linspace(-0.2, 0.2, frames)
    body_pos = np.zeros((frames, len(names), 3))
    body_pos[:, :, 0] = xs[:, None]
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frames, len(names), 4))
    body_quat[:, :, 0] = 1.0
    return {
        "root_pos_w": np.stack((xs, np.zeros(frames), np.ones(frames)), axis=1),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_names": names,
    }


def test_clean_binding_only_scene_passes_the_cpu_preflight(tmp_path: Path) -> None:
    spec = _fixture_spec()
    result = instantiate(spec, "shelf_plank", 0, tmp_path / "clean")
    report = validate_scene(spec, result.hard.path, "hard", _payload(), _payload())
    assert report.ok, report.to_dict()


def test_lateral_gap_uses_two_measured_binding_faces(tmp_path: Path) -> None:
    spec = replace(
        _fixture_spec(),
        axis_type="lateral_gap",
        expected_link_group="shoulder_left",
        easy_coordinate_m=0.8,
        hard_coordinate_m=0.6,
        reach_orig_m=0.38,
        reach_edit_m=0.28,
        window_mm=100.0,
        face_across_route_m=0.8,
        face_vertical_m=2.0,
        binding_thickness_m=0.12,
        binding_keypoint=None,
        critical_frame_orig=None,
        critical_frame_edit=None,
        per_keypoint_margins_m=None,
    )
    result = instantiate(spec, "pinch_panels", 0, tmp_path / "lateral")
    report = validate_scene(spec, result.hard.path, "hard", _payload(), _payload())
    assert report.ok, report.to_dict()
    assert result.hard.measurement.coordinate_m == pytest.approx(0.6)
    assert len(result.hard.measurement.binding_paths) == 2


@pytest.mark.parametrize("offset_xyz", [(0.0, 0.0, 1.0), (0.15, 0.0, 1.1), (-0.15, 0.1, 0.9)])
def test_any_injected_corridor_prim_is_refused(
    tmp_path: Path, offset_xyz: tuple[float, float, float]
) -> None:
    spec = _fixture_spec()
    archetype = get_archetype("shelf_plank", "overhead")
    primitives = archetype.build(spec, spec.hard_coordinate_m, 0) + (
        AuthoredPrimitive(
            "InjectedViolation",
            offset_xyz,
            (0.12, 0.12, 0.12),
            (0.8, 0.1, 0.1),
            "constraint_context",
        ),
    )
    scene = tmp_path / f"injected_{offset_xyz[0]}.usda"
    scene.write_text(render_scene_usda(spec, "injected", primitives))
    report = validate_scene(spec, scene, "hard", _payload(), _payload())
    assert not report.ok
    assert "keepout_violation" in report.refusal_reasons
    assert any(v.prim_path and v.prim_path.endswith("InjectedViolation") for v in report.violations)


def test_one_millimetre_station_error_is_refused(tmp_path: Path) -> None:
    spec = _fixture_spec()
    result = instantiate(spec, "shelf_plank", 0, tmp_path / "station")
    text = result.hard.path.read_text()
    result.hard.path.write_text(
        text.replace(
            "double3 xformOp:translate = (0.0000000, 0.0000000, 0.0000000)",
            "double3 xformOp:translate = (0.0010000, 0.0000000, 0.0000000)",
            1,
        )
    )
    report = validate_scene(spec, result.hard.path, "hard", _payload(), _payload())
    assert not report.ok
    assert "binding_station_mismatch" in report.refusal_reasons


def test_undersized_binding_face_is_refused(tmp_path: Path) -> None:
    spec = _fixture_spec()
    result = instantiate(spec, "shelf_plank", 0, tmp_path / "extent")
    text = result.hard.path.read_text()
    result.hard.path.write_text(
        text.replace(
            "xformOp:scale = (1.2000000, 2.0000000, 0.1000000)",
            "xformOp:scale = (0.3000000, 0.3000000, 0.1000000)",
        )
    )
    report = validate_scene(spec, result.hard.path, "hard", _payload(), _payload())
    assert not report.ok
    assert "face_extent" in report.refusal_reasons
