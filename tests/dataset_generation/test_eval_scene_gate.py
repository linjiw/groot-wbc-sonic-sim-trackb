"""Tests for keeping policy-specific geometry out of evaluation scenes."""

from __future__ import annotations

import pytest

from gear_sonic.dataset_generation.eval_scene_gate import (
    AUTHORED_PROVENANCE,
    EvalSceneGateError,
    assert_eval_scenes_are_eligible,
    classify_scene,
    partition_scenes,
)

AUTHORED = {
    "scene_id": "household_room",
    "provenance": AUTHORED_PROVENANCE,
    "source_motion": None,
}
GENERATED = {
    "scene_id": "density_tight",
    "provenance": "repo_generated_primitive_geometry",
    "source_motion": "01_single_text_prompt",
}


def test_authored_scene_is_eligible():
    verdict = classify_scene(AUTHORED)
    assert verdict.geometry_origin == "authored"
    assert verdict.eligible
    assert not verdict.is_policy_specific


def test_scene_generated_around_a_motion_is_not_eligible():
    verdict = classify_scene(GENERATED)
    assert verdict.geometry_origin == "executed_swept_volume"
    assert verdict.is_policy_specific
    assert not verdict.eligible
    assert "01_single_text_prompt" in verdict.reasons[0]


def test_a_scene_with_no_independent_route_is_rejected_whatever_its_manifest_says():
    """density_dense and density_tight admit no alternative 2 m route at 0.60 m clearance.

    The empirical check catches mislabelled scenes that the declared-provenance check
    would wave through.
    """
    verdict = classify_scene(AUTHORED, independent_routes=0)
    assert verdict.geometry_origin == "authored"
    assert not verdict.eligible
    assert "no independent route" in verdict.reasons[0]


def test_routability_alone_does_not_rescue_a_generated_scene():
    """Admitting other routes does not undo the fact that clearance encodes one motion."""
    verdict = classify_scene(GENERATED, independent_routes=5)
    assert not verdict.eligible


def test_untested_routability_is_not_treated_as_failure():
    assert classify_scene(AUTHORED, independent_routes=None).eligible


def test_assert_passes_for_an_all_authored_set():
    verdicts = assert_eval_scenes_are_eligible([AUTHORED, dict(AUTHORED, scene_id="factory")])
    assert len(verdicts) == 2


def test_assert_names_the_offending_scenes():
    with pytest.raises(EvalSceneGateError, match="density_tight"):
        assert_eval_scenes_are_eligible([AUTHORED, GENERATED])


def test_assert_counts_how_many_failed():
    with pytest.raises(EvalSceneGateError, match="2 of 3"):
        assert_eval_scenes_are_eligible(
            [AUTHORED, GENERATED, dict(GENERATED, scene_id="density_dense")]
        )


def test_assert_refuses_an_empty_set():
    """Zero scenes would otherwise certify vacuously."""
    with pytest.raises(EvalSceneGateError, match="empty evaluation scene set"):
        assert_eval_scenes_are_eligible([])


def test_assert_uses_the_routes_mapping():
    with pytest.raises(EvalSceneGateError, match="no independent route"):
        assert_eval_scenes_are_eligible([AUTHORED], routes_by_scene={"household_room": 0})


def test_partition_separates_without_raising():
    eligible, training_only = partition_scenes([AUTHORED, GENERATED])
    assert [v.scene_id for v in eligible] == ["household_room"]
    assert [v.scene_id for v in training_only] == ["density_tight"]


def test_partition_reports_the_route_count_it_used():
    _, training_only = partition_scenes(
        [dict(AUTHORED, scene_id="cramped")], routes_by_scene={"cramped": 0}
    )
    assert training_only[0].independent_routes == 0


def test_missing_provenance_is_treated_as_generated_when_a_source_motion_is_named():
    """Absence of a provenance string must not be a way past the gate."""
    verdict = classify_scene({"scene_id": "x", "source_motion": "some_motion"})
    assert not verdict.eligible


REFERENCE_CLEARED = {
    "scene_id": "eval_000",
    "provenance": "repo_generated_reference_clearance_geometry",
    "source_motion": None,
}


def test_reference_cleared_scenes_are_eligible_and_labelled_distinctly():
    """Generated from the commanded path, so no executed trajectory enters the geometry.

    Labelling these "authored" would misdescribe how they were made to anyone reading a
    split report, so they get their own origin.
    """
    verdict = classify_scene(REFERENCE_CLEARED, independent_routes=3)
    assert verdict.geometry_origin == "reference_clearance"
    assert verdict.eligible
    assert not verdict.is_policy_specific


def test_a_reference_cleared_scene_still_needs_routes():
    verdict = classify_scene(REFERENCE_CLEARED, independent_routes=0)
    assert not verdict.eligible
