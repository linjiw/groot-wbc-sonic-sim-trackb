"""ConstraintSpec extraction and USD measure-back regression gates."""

from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from gear_sonic.dataset_generation.clutter_scene_builder import (
    ClutterSceneSpec,
    FurniturePiece,
    render_scene_usda as render_source_scene,
)
from gear_sonic.dataset_generation.hallucination.constraint_spec import (
    ConstraintSpec,
    ConstraintSpecError,
    MotionEvidence,
)
from gear_sonic.dataset_generation.hallucination.extract_spec import extract_spec
from gear_sonic.dataset_generation.hallucination.stage_geometry import (
    StageGeometryError,
    measure_binding,
    read_stage_geometry,
)
from gear_sonic.dataset_generation.swept_volume import G1_COLLISION_CAPSULES


def _payload(frames: int = 5) -> dict:
    names = sorted(G1_COLLISION_CAPSULES)
    xs = np.linspace(-0.2, 0.2, frames)
    body_pos = np.zeros((frames, len(names), 3), dtype=np.float64)
    body_pos[:, :, 0] = xs[:, None]
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frames, len(names), 4), dtype=np.float64)
    body_quat[:, :, 0] = 1.0
    return {
        "motion_time_s": np.arange(frames, dtype=np.float64) / 50,
        "total_frames": frames,
        "root_pos_w": np.stack((xs, np.zeros(frames), np.ones(frames)), axis=1),
        "root_quat_w": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (frames, 1)),
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "body_names": names,
    }


def _write_source_scene(path: Path, scene_id: str, underside: float) -> None:
    piece = FurniturePiece(
        "LowShelf_00",
        "WallShelf",
        (0.0, 0.0),
        (0.5, 3.0, 0.1),
        (0.46, 0.34, 0.22),
        underside,
        "overhead",
    )
    spec = ClutterSceneSpec(
        scene_id,
        f"{scene_id}_family_v1",
        (8.0, 5.0),
        2.8,
        [piece],
        np.asarray([[-1.0, 0.0], [1.0, 0.0]]),
        0.0,
        0,
        {"clutter_occupancy": 0.0},
    )
    path.write_text(render_source_scene(spec))


def test_extract_spec_uses_two_plane_artifacts_and_per_motion_frames(tmp_path: Path) -> None:
    family = tmp_path / "families" / "duck_fixture"
    scenes = tmp_path / "scenes"
    (family / "logs").mkdir(parents=True)
    scenes.mkdir()
    manifest = {
        "family_id": "cf_fixture",
        "counterfactual_established": True,
        "regime": "overhead",
        "nominal_motion": "nominal.csv",
        "adapted_motion": "adapted.csv",
        "nominal_clears_to_m": 1.4,
        "adapted_clears_to_m": 1.2,
        "easy_shelf_underside_m": 1.5,
        "hard_shelf_underside_m": 1.3,
        "window_m": 0.2,
    }
    (family / "family.json").write_text(json.dumps(manifest))
    (family / "attribution.json").write_text(
        json.dumps({"cells": {"nominal_hard": {"first_contact_body": "torso_link"}}})
    )
    for label, shift in (("nominal", 0.0), ("adapted", 0.1)):
        trajectory = family / f"probe_{label}" / "trajectories" / "000000.trajectory.pkl"
        trajectory.parent.mkdir(parents=True)
        payload = _payload()
        payload["root_pos_w"][:, 0] += shift
        payload["body_pos_w"][:, :, 0] += shift
        if label == "adapted":
            payload["body_pos_w"][:, :, 2] -= 0.2
        with trajectory.open("wb") as handle:
            pickle.dump(payload, handle)
        (family / "logs" / f"probe_{label}.log").write_text("scene=plane\n")
    for role, scene_id in (("nominal_easy", "source_easy"), ("nominal_hard", "source_hard")):
        (family / "logs" / f"{role}.log").write_text(f"scene={scene_id}\n")
    _write_source_scene(scenes / "source_easy.usda", "source_easy", 1.5)
    _write_source_scene(scenes / "source_hard.usda", "source_hard", 1.3)

    spec = extract_spec(family, scenes)

    assert spec.source_family_id == "cf_fixture"
    assert spec.expected_link_group == "head_torso"
    assert spec.window_mm == pytest.approx(200.0)
    assert spec.crossing_frames_orig != spec.crossing_frames_edit
    assert spec.face_along_route_m == pytest.approx(0.5)
    assert spec.binding_keypoint == "head_torso"


def test_extract_spec_refuses_a_missing_adapted_plane_rollout(tmp_path: Path) -> None:
    family = tmp_path / "mf_005_c08"
    family.mkdir()
    (tmp_path / "mf_005_c08.json").write_text(
        json.dumps(
            {
                "family_id": "mf_005_c08",
                "regime": "overhead",
                "nominal_motion": "nominal",
                "adapted_motion": "adapted",
            }
        )
    )
    with pytest.raises(
        ConstraintSpecError,
        match=r"missing or unsupported evidence: nominal .*; adapted",
    ):
        extract_spec(family, tmp_path)


def test_stage_measurement_composes_units_up_axis_and_nested_xforms(tmp_path: Path) -> None:
    stage_path = tmp_path / "nested_y_up.usda"
    stage_path.write_text("""#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Y"
)
def Xform "World"
{
    def Xform "Parent"
    {
        double3 xformOp:translate = (100, 200, 300)
        double3 xformOp:scale = (2, 3, 4)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        def Cube "Binding" (prepend apiSchemas = ["PhysicsCollisionAPI"])
        {
            custom string g1Dataset:role = "binding_constraint"
            custom string lfh:bindingFaceId = "fixture:underside"
            custom string lfh:bindingSense = "underside"
            double size = 1
            bool physics:collisionEnabled = 1
            double3 xformOp:translate = (10, 20, 30)
            double3 xformOp:scale = (40, 50, 60)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
}
""")

    measured = measure_binding(read_stage_geometry(stage_path), "overhead", "x")

    assert measured.coordinate_m == pytest.approx(1.85)
    assert measured.station_xy_m == pytest.approx((1.2, -4.2))
    assert measured.along_route_m == pytest.approx(0.8)
    assert measured.across_route_m == pytest.approx(2.4)


def test_stage_reader_refuses_rotated_geometry(tmp_path: Path) -> None:
    stage = tmp_path / "rotated.usda"
    stage.write_text("""#usda 1.0
(metersPerUnit = 1 upAxis = "Z")
def Cube "Bad"
{
    double size = 1
    double3 xformOp:rotateXYZ = (0, 0, 45)
    double3 xformOp:scale = (1, 1, 1)
}
""")
    with pytest.raises(StageGeometryError, match="non-axis-aligned"):
        read_stage_geometry(stage)


def test_stage_reader_refuses_unordered_transform_ops(tmp_path: Path) -> None:
    stage = tmp_path / "unordered.usda"
    stage.write_text("""#usda 1.0
(metersPerUnit = 1 upAxis = "Z")
def Cube "Bad"
{
    double size = 1
    double3 xformOp:translate = (1, 2, 3)
    double3 xformOp:scale = (1, 1, 1)
}
""")
    with pytest.raises(StageGeometryError, match="absent from xformOpOrder"):
        read_stage_geometry(stage)


def test_committed_golden_spec_round_trips() -> None:
    path = Path(__file__).resolve().parents[2] / "specs/hallucination/cs_duck_003.json"
    spec = ConstraintSpec.load(path)
    assert ConstraintSpec.from_dict(spec.to_dict()) == spec
    assert spec.easy_coordinate_m == pytest.approx(1.390625)
    assert spec.hard_coordinate_m == pytest.approx(1.2125)
    assert spec.window_mm == pytest.approx(178.125)


def test_screen_empty_evidence_requires_a_hash_pinned_adjudication() -> None:
    with pytest.raises(ConstraintSpecError, match="hash-pinned adjudication"):
        MotionEvidence(
            "motion",
            "/tmp/trajectory.pkl",
            "sha256:" + "1" * 64,
            "screen_empty",
        ).validate()

    evidence = MotionEvidence(
        "motion",
        "/tmp/trajectory.pkl",
        "sha256:" + "1" * 64,
        "screen_empty",
        adjudication_path="/tmp/run-record.json",
        adjudication_sha256="sha256:" + "2" * 64,
        adjudication_cell_id="motion__nominal",
    )
    evidence.validate()
    assert MotionEvidence.from_mapping(evidence.to_dict()) == evidence
