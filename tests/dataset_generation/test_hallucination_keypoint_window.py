"""LFH semantic critical-set and finite-footprint window regressions."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.extract_spec import extract_spec
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate
from gear_sonic.dataset_generation.hallucination.keypoints import (
    SEMANTIC_GROUPS,
    SemanticCapsuleTracks,
    validate_semantic_partition,
)
from gear_sonic.dataset_generation.hallucination.reach import (
    FaceReach,
    capsule_height_over_rectangle,
    lateral_face_reach,
    lateral_gap_reach,
)
from gear_sonic.dataset_generation.hallucination.validate_keepout import validate_pair
from gear_sonic.dataset_generation.hallucination.window import WindowError, solve_window
from gear_sonic.dataset_generation.swept_volume import G1_COLLISION_CAPSULES

REPO_ROOT = Path(__file__).resolve().parents[2]
DUCK = Path("/data/robotixx/groot-wbc-kimodo-m0/counterfactual/duck_003")
SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"


def _face(reach: float, frame: int) -> FaceReach:
    values = {group: 0.1 for group in SEMANTIC_GROUPS}
    frames = {group: 0 for group in SEMANTIC_GROUPS}
    owners = {group: group for group in SEMANTIC_GROUPS}
    values["head_torso"] = reach
    frames["head_torso"] = frame
    owners["head_torso"] = "torso_link"
    return FaceReach("overhead", (0.0, 0.0), values, frames, owners)


def test_semantic_groups_partition_every_authoritative_capsule() -> None:
    partition = validate_semantic_partition()
    owners = [owner for group in SEMANTIC_GROUPS for owner in partition[group]]
    assert sorted(owners) == sorted(G1_COLLISION_CAPSULES)
    assert sum(len(G1_COLLISION_CAPSULES[owner]) for owner in owners) == 29


def test_capsule_face_support_handles_a_rounded_rectangle_edge_exactly() -> None:
    height = capsule_height_over_rectangle(
        start=[0.2, 0.0, 0.0],
        end=[0.2, 0.0, 0.0],
        radius=0.2,
        rectangle_xy=(-0.1, -0.1, 0.1, 0.1),
    )
    assert height == pytest.approx(math.sqrt(0.2**2 - 0.1**2))


def test_window_solver_reports_binding_frames_and_all_group_margins() -> None:
    solution = solve_window(
        _face(1.4, 12),
        _face(1.2, 9),
        hard_coordinate_m=1.3,
        easy_coordinate_m=1.45,
    )
    assert solution.width_m == pytest.approx(0.2)
    assert solution.binding_keypoint == "head_torso"
    assert solution.critical_frame_orig == 12
    assert solution.critical_frame_edit == 9
    assert set(solution.per_keypoint_margins_m) == set(SEMANTIC_GROUPS)
    assert solution.per_keypoint_margins_m["head_torso"] == pytest.approx(
        {"orig_m": -0.1, "edit_m": 0.1}
    )


def test_window_solver_refuses_a_face_outside_the_executed_interval() -> None:
    with pytest.raises(WindowError, match="hard_face_outside_window"):
        solve_window(_face(1.4, 12), _face(1.2, 9), hard_coordinate_m=1.1)


def test_lateral_reach_attributes_the_one_sided_band_extremum() -> None:
    capsules = len(SEMANTIC_GROUPS)
    starts = np.zeros((1, capsules, 3))
    ends = np.zeros_like(starts)
    starts[:, :, 2] = ends[:, :, 2] = 1.0
    starts[0, SEMANTIC_GROUPS.index("shoulder_left"), 1] = 0.4
    ends[0, SEMANTIC_GROUPS.index("shoulder_left"), 1] = 0.4
    tracks = SemanticCapsuleTracks(
        starts,
        ends,
        np.full(capsules, 0.05),
        tuple(SEMANTIC_GROUPS),
        tuple(SEMANTIC_GROUPS),
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    reach = lateral_face_reach(tracks, (0.0, 0.0), "x", 0.5, (0.8, 1.2), "left")
    assert reach.binding_keypoint == "shoulder_left"
    assert reach.reach_m == pytest.approx(0.45)


def test_lateral_gap_reach_uses_full_world_fixed_face_separation() -> None:
    capsules = len(SEMANTIC_GROUPS)
    starts = np.zeros((1, capsules, 3))
    ends = np.zeros_like(starts)
    starts[:, :, 2] = ends[:, :, 2] = 1.0
    starts[0, SEMANTIC_GROUPS.index("shoulder_left"), 1] = 0.4
    ends[0, SEMANTIC_GROUPS.index("shoulder_left"), 1] = 0.4
    tracks = SemanticCapsuleTracks(
        starts,
        ends,
        np.full(capsules, 0.05),
        tuple(SEMANTIC_GROUPS),
        tuple(SEMANTIC_GROUPS),
        np.zeros((1, 3)),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    reach = lateral_gap_reach(tracks, (0.0, 0.0), "x", 0.5, (0.8, 1.2))
    assert reach.binding_keypoint == "shoulder_left"
    assert reach.reach_m == pytest.approx(0.9)


def test_lateral_gap_reach_includes_route_drift_from_the_fixed_station() -> None:
    capsules = len(SEMANTIC_GROUPS)
    starts = np.zeros((1, capsules, 3))
    ends = np.zeros_like(starts)
    starts[:, :, 2] = ends[:, :, 2] = 1.0
    starts[:, :, 1] = ends[:, :, 1] = 0.1
    tracks = SemanticCapsuleTracks(
        starts,
        ends,
        np.full(capsules, 0.05),
        tuple(SEMANTIC_GROUPS),
        tuple(SEMANTIC_GROUPS),
        np.asarray([[0.0, 0.1, 0.0]]),
        np.asarray([[1.0, 0.0, 0.0, 0.0]]),
    )
    reach = lateral_gap_reach(tracks, (0.0, 0.0), "x", 0.5, (0.8, 1.2))
    assert reach.reach_m == pytest.approx(0.3)


@pytest.mark.skipif(not DUCK.exists(), reason="local executed duck_003 evidence is absent")
def test_duck_golden_recomputes_binding_and_rejects_the_extended_plank(tmp_path: Path) -> None:
    spec = extract_spec(DUCK, SCENES)
    assert spec.face_along_route_m == pytest.approx(0.5)
    assert spec.binding_keypoint == "head_torso"
    assert spec.critical_frame_orig == 129
    assert spec.critical_frame_edit == 98

    source = instantiate(spec, "shelf_plank", 17, tmp_path / "source")
    source_report = validate_pair(spec, source.easy.path, source.hard.path)
    assert source_report["ok"], source_report
    assert source_report["binding_geometry_signs"] == {
        "easy": {"orig": "+", "edit": "+"},
        "hard": {"orig": "-", "edit": "+"},
    }
    assert source_report["binding_geometry_pattern"]["hard"]["edit"][
        "clearance_mm"
    ] == pytest.approx(66.6633, abs=0.01)

    extended_spec = replace(spec, face_along_route_m=1.0810606920641086)
    extended = instantiate(extended_spec, "shelf_plank", 17, tmp_path / "extended")
    extended_report = validate_pair(extended_spec, extended.easy.path, extended.hard.path)
    assert not extended_report["ok"]
    assert extended_report["refusal_reasons"] == ["binding_pattern_mismatch"]
    assert extended_report["binding_geometry_signs"]["hard"]["edit"] == "-"
    assert extended_report["binding_geometry_pattern"]["hard"]["edit"][
        "clearance_mm"
    ] == pytest.approx(-38.9141, abs=0.01)
