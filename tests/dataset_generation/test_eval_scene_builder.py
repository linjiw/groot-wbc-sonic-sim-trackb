"""Tests for building rooms that may grade a policy."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.eval_scene_builder import (
    EVAL_CLEARANCE_M,
    MEASURED_BODY_HALF_WIDTH_M,
    POLICY_DRIFT_ALLOWANCE_M,
    EvalSceneError,
    build_eval_scene,
)
from gear_sonic.dataset_generation.eval_scene_gate import classify_scene

STRAIGHT = np.stack([np.linspace(-2.0, 2.0, 40), np.zeros(40)], axis=1)


def test_clearance_is_the_measured_body_plus_a_drift_allowance():
    assert EVAL_CLEARANCE_M == pytest.approx(
        MEASURED_BODY_HALF_WIDTH_M + POLICY_DRIFT_ALLOWANCE_M
    )
    # The training generator's 0.45 m body radius was exceeded on 98% of measured episodes.
    assert MEASURED_BODY_HALF_WIDTH_M > 0.45


def test_drift_allowance_exceeds_the_largest_measured_deviation():
    """An evaluation must admit policies worse than the one that produced the measurement."""
    largest_observed_drift_m = 0.357
    assert POLICY_DRIFT_ALLOWANCE_M > largest_observed_drift_m


def test_builds_a_certified_scene_from_a_reference_path():
    spec, report = build_eval_scene(STRAIGHT, scene_id="eval_a", seed=0)
    assert report.certified
    assert report.independent_routes >= 2
    assert report.scene_id == "eval_a"
    assert spec.scene_id == "eval_a"


def test_the_built_scene_leaves_at_least_the_budgeted_clearance():
    _, report = build_eval_scene(STRAIGHT, scene_id="eval_b", seed=1)
    assert report.min_route_clearance_m >= report.clearance_m - 1e-9


def test_an_eval_scene_is_roomier_than_a_training_scene():
    """The lower occupancy is the price of eligibility, not a defect -- worth asserting so
    a future tightening of the density cannot pass unnoticed."""
    _, report = build_eval_scene(STRAIGHT, scene_id="eval_c", seed=2, target_pieces=16)
    assert report.occupancy < 0.30


def test_requiring_more_routes_than_the_room_admits_is_an_error():
    with pytest.raises(EvalSceneError, match="independent route"):
        build_eval_scene(STRAIGHT, scene_id="eval_d", seed=3, min_independent_routes=99)


def test_a_degenerate_reference_path_is_rejected():
    with pytest.raises(EvalSceneError, match="at least two points"):
        build_eval_scene(np.array([[0.0, 0.0]]), scene_id="eval_e")


def test_a_scene_that_admits_no_route_raises_rather_than_returning_uncertified():
    """Returning an uncertified scene invites a caller to ship it by accident."""
    with pytest.raises(EvalSceneError):
        # A clearance wider than any room this path produces leaves nothing walkable.
        build_eval_scene(STRAIGHT, scene_id="eval_f", seed=4, drift_allowance_m=50.0)


def test_the_result_still_carries_a_source_motion_so_the_gate_sees_it():
    """The gate keys on declared provenance; an eval scene must not fake its way past it.

    Certification comes from routability, which the gate checks separately -- not from
    hiding how the scene was made.
    """
    _, report = build_eval_scene(STRAIGHT, scene_id="eval_g", seed=5)
    verdict = classify_scene(
        {"scene_id": "eval_g", "source_motion": "some_reference"},
        independent_routes=report.independent_routes,
    )
    assert verdict.geometry_origin == "executed_swept_volume"
    assert not verdict.eligible


def test_reports_are_deterministic_for_a_seed():
    first = build_eval_scene(STRAIGHT, scene_id="eval_h", seed=7)[1]
    second = build_eval_scene(STRAIGHT, scene_id="eval_h", seed=7)[1]
    assert first == second
