"""LFH coverage-target proposer tests."""

from __future__ import annotations

import pytest

from gear_sonic.dataset_generation.hallucination.keypoints import SEMANTIC_GROUPS
from gear_sonic.dataset_generation.hallucination.propose import (
    CoverageTarget,
    ProposalRefusal,
    propose,
)
from gear_sonic.dataset_generation.hallucination.reach import FaceReach


def _reach() -> FaceReach:
    values = {group: 0.2 for group in SEMANTIC_GROUPS}
    values["shoulder_left"] = 0.38
    return FaceReach(
        "lateral_gap",
        (0.0, 0.0),
        values,
        {group: 4 for group in SEMANTIC_GROUPS},
        {group: group for group in SEMANTIC_GROUPS},
    )


def _target() -> CoverageTarget:
    return CoverageTarget(
        "arms", "local_arm_tuck", "lateral_gap", "shoulder_left", 0.3, 0.4, "clear_10_25"
    )


def test_proposer_selects_the_minimum_certified_edit_and_screens_all_groups() -> None:
    orig = _reach()

    def prediction(alpha: float) -> dict[str, float]:
        values = dict(orig.per_keypoint_reach_m)
        values["shoulder_left"] = 0.38 - 0.1 * alpha
        return values

    result = propose("motion-1", _target(), orig, prediction)
    assert result.tier == 1
    assert 0 < result.alpha < 1
    assert 0.3 <= result.coordinate_m <= 0.4
    assert min(result.per_keypoint_predicted_edit_margin_m.values()) >= 0


def test_proposer_refuses_when_an_uncoupled_keypoint_sets_the_floor() -> None:
    orig = _reach()

    def prediction(alpha: float) -> dict[str, float]:
        values = dict(orig.per_keypoint_reach_m)
        values["shoulder_left"] = 0.38 - 0.1 * alpha
        values["pelvis"] = 0.39
        return values

    with pytest.raises(ProposalRefusal, match="certified_window_empty"):
        propose("motion-1", _target(), orig, prediction)


def test_target_refuses_operator_anatomy_mismatch() -> None:
    with pytest.raises(ProposalRefusal, match="operator_keypoint_mismatch"):
        CoverageTarget(
            "arms", "local_arm_tuck", "lateral_gap", "knee_left", 0.3, 0.4, "clear"
        ).validate()


def test_proposer_can_search_a_registered_scale_above_one() -> None:
    orig = _reach()

    def prediction(alpha: float) -> dict[str, float]:
        values = dict(orig.per_keypoint_reach_m)
        values["shoulder_left"] = 0.38 - 0.01 * alpha
        return values

    with pytest.raises(ProposalRefusal, match="certified_window_empty"):
        propose("motion-1", _target(), orig, prediction)
    result = propose("motion-1", _target(), orig, prediction, max_alpha=2.5)
    assert 1.0 < result.alpha <= 2.5


@pytest.mark.parametrize(
    ("bucket", "expected"),
    [("under_0.3", (0.0, 0.3)), ("at_least_0.9", (0.9, float("inf")))],
)
def test_target_parses_open_coordinate_buckets(bucket: str, expected: tuple[float, float]) -> None:
    target = CoverageTarget.from_mapping(
        {
            "edit_behaviour_class": "arms",
            "operator": "local_arm_tuck",
            "constraint_axis": "lateral_gap",
            "binding_keypoint": "wrist_right",
            "constraint_coordinate_bucket": bucket,
            "margin_bucket": "clear_10_25",
        }
    )
    assert (target.coordinate_lower_m, target.coordinate_upper_m) == expected
