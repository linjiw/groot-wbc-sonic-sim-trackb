"""LFH keypoint-response schema and monotone delivery-model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from gear_sonic.dataset_generation.hallucination.delivery import (
    RESPONSE_COLUMNS,
    ResponseRow,
    append_responses,
    fit_delivery_models,
    read_responses,
)


def _row(motion: str, commanded: float, executed: float) -> ResponseRow:
    return ResponseRow(
        motion, "local_arm_tuck", commanded / 30, "wrist_left", "lateral", commanded, executed
    )


def test_response_sidecar_uses_the_declared_schema(tmp_path: Path) -> None:
    path = tmp_path / "keypoint_response.csv"
    rows = [_row("a", 0, 0), _row("b", 10, 9)]
    assert append_responses(path, rows) == 2
    assert tuple(path.read_text().splitlines()[0].split(",")) == RESPONSE_COLUMNS
    assert read_responses(path) == rows


def test_delivery_fit_is_monotone_and_carries_a_residual_interval() -> None:
    rows = [
        _row("a", 0, 0),
        _row("b", 0, 0),
        _row("a", 10, 12),
        _row("b", 10, 10),
        _row("a", 20, 9),
        _row("b", 20, 11),
        _row("a", 30, 29),
        _row("b", 30, 31),
    ]
    models, refusals = fit_delivery_models(rows)
    assert not refusals
    model = models[("local_arm_tuck", "wrist_left", "lateral")]
    assert list(model.fitted_mm) == sorted(model.fitted_mm)
    estimate, lower, upper = model.predict(15)
    assert lower <= estimate <= upper
    assert model.residual_q90_mm > 0


def test_delivery_fit_refuses_one_motion_even_with_multiple_levels() -> None:
    models, refusals = fit_delivery_models([_row("a", value, value) for value in (0, 10, 20)])
    assert not models
    assert "has 3 levels / 1 motions" in next(iter(refusals.values()))


def test_delivery_fit_pools_nearby_commands_as_matched_motion_replicates() -> None:
    def level(motion: str, alpha: float, commanded: float, executed: float) -> ResponseRow:
        return ResponseRow(
            motion,
            "local_arm_tuck",
            alpha,
            "wrist_left",
            "lateral",
            commanded,
            executed,
        )

    rows = [
        level("a", 0.25, 20.0, 1.0),
        level("b", 0.25, 20.4, 2.0),
        level("a", 0.50, 40.0, 4.0),
        level("b", 0.50, 40.5, 5.0),
        level("a", 1.00, 60.8, 5.0),
        level("b", 1.00, 61.4, 17.0),
    ]
    models, refusals = fit_delivery_models(rows)
    assert not refusals
    model = models[("local_arm_tuck", "wrist_left", "lateral")]
    assert model.samples_per_level == (2, 2, 2)
    assert model.fitted_mm[-1] == pytest.approx(11.0)
    assert model.residual_q90_mm == pytest.approx(6.0)


def test_delivery_fit_refuses_a_level_without_two_motion_replicates() -> None:
    rows = [
        _row("a", 10, 8),
        _row("b", 10, 9),
        _row("a", 20, 15),
        _row("b", 20, 16),
        _row("a", 30, 20),
    ]
    models, refusals = fit_delivery_models(rows)
    assert not models
    assert "levels with >= 2 motions each" in next(iter(refusals.values()))


def test_delivery_fit_retains_supported_range_when_later_level_has_one_motion() -> None:
    rows = [
        _row(motion, commanded, executed)
        for commanded, executed in ((10, 8), (20, 15), (30, 20))
        for motion in ("a", "b")
    ]
    rows.append(ResponseRow("a", "local_arm_tuck", 2.0, "wrist_left", "lateral", 60, 22))
    models, refusals = fit_delivery_models(rows)
    assert not refusals
    model = models[("local_arm_tuck", "wrist_left", "lateral")]
    assert model.alpha_levels == pytest.approx((1 / 3, 2 / 3, 1.0))
    assert model.unsupported_alpha_levels == (2.0,)


def test_response_refuses_an_unrecognized_semantic_group() -> None:
    with pytest.raises(ValueError, match="unknown semantic keypoint"):
        ResponseRow("a", "local_arm_tuck", 1.0, "head", "lateral", 10, 9).validate()


def test_response_allows_stronger_than_anchor_command_scale() -> None:
    ResponseRow("a", "local_arm_tuck", 2.0, "wrist_left", "lateral", 120, 20).validate()
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        ResponseRow("a", "local_arm_tuck", -0.1, "wrist_left", "lateral", 10, 2).validate()
