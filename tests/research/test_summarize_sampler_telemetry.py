from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.summarize_sampler_telemetry import (
    parse_sampler_telemetry,
    summarize_telemetry_logs,
)


def _iteration_block(iteration: int, telemetry: dict[str, str]) -> str:
    """Mimic the PPO trainer's console block: ANSI-bold header + padded Env lines."""
    header = f" \033[1m Learning iteration {iteration}  \033[0m "
    lines = [header.center(80, " "), ""]
    lines.append(f"{'Mean rewards:':>35} 0.95000")
    for key, value in telemetry.items():
        lines.append(f"{f'Env/adp_samp/{key}:':>35} {value}")
    lines.append("-" * 80)
    return "\n".join(lines) + "\n"


def test_parse_extracts_aligned_series_with_known_values() -> None:
    text = (
        _iteration_block(0, {"num_episodes_mean": "1.0000", "failure_rate_mean": "1.0000"})
        + _iteration_block(1, {"num_episodes_mean": "1.5000", "failure_rate_mean": "0.9000"})
        + _iteration_block(2, {"num_episodes_mean": "2.2500", "failure_rate_mean": "0.7500"})
    )

    summary = parse_sampler_telemetry(text)

    assert summary["adaptive_telemetry_present"] is True
    assert summary["iteration_count"] == 3
    assert summary["iteration_first"] == 0
    assert summary["iteration_last"] == 2
    episodes = summary["keys"]["num_episodes_mean"]
    assert episodes["series"] == [[0, 1.0], [1, 1.5], [2, 2.25]]
    assert episodes["first"] == 1.0
    assert episodes["last"] == 2.25
    assert episodes["min"] == 1.0
    assert episodes["max"] == 2.25
    assert episodes["slope_final_iters"] == pytest.approx(0.625)
    failure = summary["keys"]["failure_rate_mean"]
    assert failure["last"] == 0.75
    assert failure["slope_final_iters"] == pytest.approx(-0.125)


def test_parse_handles_percent_4f_truncation() -> None:
    # %.4f truncates: a true value of 0.00004 prints as 0.0000.
    text = _iteration_block(0, {"prob_min": "0.0000"})

    summary = parse_sampler_telemetry(text)

    assert summary["keys"]["prob_min"]["series"] == [[0, 0.0]]


def test_parse_keeps_alignment_when_keys_appear_late_or_drop_out() -> None:
    # prob_* keys are emitted only after the first prob recompute (hasattr guard);
    # episodes_max_over_mean is guarded by eps_mean > 0 and can drop out.
    text = (
        _iteration_block(0, {"num_episodes_mean": "1.0000"})
        + _iteration_block(1, {"num_episodes_mean": "1.5000", "prob_max_over_uniform": "2.9000"})
        + _iteration_block(2, {"num_episodes_mean": "2.0000", "prob_max_over_uniform": "3.0500"})
    )

    summary = parse_sampler_telemetry(text)

    prob = summary["keys"]["prob_max_over_uniform"]
    assert prob["series"] == [[1, 2.9], [2, 3.05]]
    assert prob["num_points"] == 2
    assert prob["first"] == 2.9
    assert summary["keys"]["num_episodes_mean"]["num_points"] == 3


def test_parse_records_nan_without_shifting_series() -> None:
    text = _iteration_block(0, {"failure_rate_mean": "nan"}) + _iteration_block(
        1, {"failure_rate_mean": "0.5000"}
    )

    summary = parse_sampler_telemetry(text)

    failure = summary["keys"]["failure_rate_mean"]
    assert failure["series"] == [[0, None], [1, 0.5]]
    assert failure["non_finite_count"] == 1
    assert failure["first"] == 0.5  # first FINITE value
    assert failure["min"] == 0.5


def test_uniform_log_yields_absent_telemetry_not_warning() -> None:
    text = _iteration_block(0, {}) + _iteration_block(1, {})

    summary = parse_sampler_telemetry(text)

    assert summary["adaptive_telemetry_present"] is False
    assert summary["keys"] == {}
    assert summary["iteration_count"] == 2


def test_summarize_multiple_logs_never_concatenates_series(tmp_path: Path) -> None:
    adaptive_log = tmp_path / "adaptive.log"
    uniform_log = tmp_path / "uniform.log"
    adaptive_log.write_text(
        _iteration_block(0, {"num_episodes_mean": "1.0000"})
        + _iteration_block(1, {"num_episodes_mean": "2.0000"}),
        encoding="utf-8",
    )
    uniform_log.write_text(_iteration_block(0, {}), encoding="utf-8")

    summary = summarize_telemetry_logs([adaptive_log, uniform_log])

    assert summary["log_count"] == 2
    adaptive_record, uniform_record = summary["logs"]
    assert adaptive_record["log_path"] == str(adaptive_log)
    assert adaptive_record["adaptive_telemetry_present"] is True
    assert adaptive_record["keys"]["num_episodes_mean"]["num_points"] == 2
    assert uniform_record["adaptive_telemetry_present"] is False
    json.dumps(summary)  # JSON-serializable round trip


def test_telemetry_before_first_iteration_header_does_not_crash() -> None:
    # A rotated/truncated log can start mid-block, with telemetry lines before
    # any "Learning iteration N" header.
    text = "Env/adp_samp/prob_max: 0.0500\n" + _iteration_block(3, {"prob_max": "0.0600"})

    summary = parse_sampler_telemetry(text)

    prob = summary["keys"]["prob_max"]
    assert prob["series"] == [[None, 0.05], [3, 0.06]]
    assert prob["first"] == 0.05
    assert prob["last"] == 0.06
    # Slope needs >= 2 iteration-tagged points; the pre-header point is excluded.
    assert prob["slope_final_iters"] is None


def test_slope_is_none_for_single_point_series() -> None:
    text = _iteration_block(5, {"prob_max": "0.0430"})

    summary = parse_sampler_telemetry(text)

    assert summary["keys"]["prob_max"]["slope_final_iters"] is None
