"""Tests for trajectory-conditioned LFH critical support."""

from __future__ import annotations

import pytest

from gear_sonic.dataset_generation.hallucination.critical_support import (
    CriticalSupport,
    source_balanced_weights,
    widest_per_source,
)


def support(record_id: str, source: str, nominal: float, adapted: float, depth: float = 0.1):
    return CriticalSupport(
        record_id=record_id,
        source_pair_id=source,
        operator="local_crouch",
        axis_type="overhead",
        binding_keypoint="head_torso",
        route_progress=0.5,
        face_along_route_m=depth,
        face_across_route_m=3.0,
        nominal_reach_m=nominal,
        adapted_reach_m=adapted,
        clear_margin_m=0.018,
        strike_margin_m=0.018,
        context_status="accepted_screen_empty_zero_external",
    )


def test_symmetric_engineering_interval_and_normalized_position():
    record = support("a", "source_a", nominal=1.30, adapted=1.23)
    assert record.lower_m == pytest.approx(1.248)
    assert record.upper_m == pytest.approx(1.282)
    assert record.engineering_width_m == pytest.approx(0.034)
    midpoint = record.coordinate_at(0.5)
    assert midpoint == pytest.approx(1.265)
    assert record.normalized_position(midpoint) == pytest.approx(0.5)


def test_empty_engineering_interval_refuses_quantile_mapping():
    record = support("a", "source_a", nominal=1.25, adapted=1.22)
    with pytest.raises(ValueError, match="support is empty"):
        record.coordinate_at(0.5)


def test_widest_selection_is_one_per_source_and_prefers_shorter_tie():
    records = [
        support("a_long", "source_a", 1.30, 1.23, depth=0.4),
        support("a_short", "source_a", 1.30, 1.23, depth=0.1),
        support("b", "source_b", 1.31, 1.22),
    ]
    assert [row.record_id for row in widest_per_source(records)] == ["a_short", "b"]


def test_source_balancing_prevents_dense_sources_from_dominating():
    records = [
        support("a1", "source_a", 1.30, 1.23),
        support("a2", "source_a", 1.31, 1.23),
        support("b1", "source_b", 1.30, 1.23),
    ]
    weights = source_balanced_weights(records)
    assert weights == {"a1": 0.25, "a2": 0.25, "b1": 0.5}
