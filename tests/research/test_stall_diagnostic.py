"""Tests freezing the D12 stall-diagnostic classification rule (plan §3.4.2).

The rule separates capacity-limited stalls (teacher exonerated) from
curriculum-limited ones (sampler implicated) BEFORE any M5 stall is interpreted.
"""

from __future__ import annotations

import numpy as np

from scripts.research.stall_diagnostic import classify_bin, diagnose, fit_error_growth

_THRESHOLD = 0.15


def _steady_drift_series(n=50, slope=0.004):
    # Crosses 0.15 within ~37 steps of a 50-step bin: execution ceiling.
    return list(np.arange(n) * slope + 0.01)


def _late_spike_series(n=50):
    # Flat at 0.03 for 80% of the bin, spikes over threshold in the last steps.
    y = np.full(n, 0.03)
    y[int(n * 0.85) :] = 0.2
    return list(y)


def test_fit_error_growth_recovers_slope() -> None:
    fit = fit_error_growth(_steady_drift_series(slope=0.004))
    assert abs(fit["slope_per_step"] - 0.004) < 1e-9


def test_capacity_limited_classification() -> None:
    record = {
        "bin_id": 1,
        "motion_key": "hard_x2.0",
        "bin_length": 50,
        "failed": True,
        "error_series": _steady_drift_series(),
    }
    assert classify_bin(record, _THRESHOLD)["classification"] == "capacity_limited"


def test_curriculum_limited_classification() -> None:
    record = {
        "bin_id": 2,
        "motion_key": "walk_x1.0",
        "bin_length": 50,
        "failed": True,
        "error_series": _late_spike_series(),
    }
    assert classify_bin(record, _THRESHOLD)["classification"] == "curriculum_limited"


def test_survived_bin_is_not_stalled() -> None:
    record = {
        "bin_id": 3,
        "bin_length": 50,
        "failed": False,
        "error_series": list(np.full(50, 0.02)),
    }
    assert classify_bin(record, _THRESHOLD)["classification"] == "not_stalled"


def test_short_series_is_insufficient_data_never_classified() -> None:
    record = {"bin_id": 4, "failed": True, "error_series": [0.1, 0.2, 0.3]}
    assert classify_bin(record, _THRESHOLD)["classification"] == "insufficient_data"


def test_diagnose_majority_verdicts() -> None:
    capacity_record = {
        "bin_length": 50,
        "failed": True,
        "error_series": _steady_drift_series(),
    }
    curriculum_record = {
        "bin_length": 50,
        "failed": True,
        "error_series": _late_spike_series(),
    }
    payload = {
        "termination_threshold": _THRESHOLD,
        "records": [dict(capacity_record, bin_id=i) for i in range(5)]
        + [dict(curriculum_record, bin_id=10)],
    }
    result = diagnose(payload)
    assert result["verdict"] == "capacity_limited"  # 5:1 majority — teacher exonerated

    payload["records"] = [dict(curriculum_record, bin_id=i) for i in range(5)] + [
        dict(capacity_record, bin_id=10)
    ]
    assert diagnose(payload)["verdict"] == "curriculum_limited"


def test_diagnose_mixed_when_no_majority() -> None:
    payload = {
        "termination_threshold": _THRESHOLD,
        "records": [
            {"bin_id": 0, "bin_length": 50, "failed": True, "error_series": _steady_drift_series()},
            {"bin_id": 1, "bin_length": 50, "failed": True, "error_series": _late_spike_series()},
        ],
    }
    assert diagnose(payload)["verdict"] == "mixed_or_ambiguous"


def test_diagnose_no_failures_is_no_stall() -> None:
    payload = {
        "termination_threshold": _THRESHOLD,
        "records": [
            {
                "bin_id": 0,
                "bin_length": 50,
                "failed": False,
                "error_series": list(np.full(50, 0.02)),
            }
        ],
    }
    result = diagnose(payload)
    assert result["verdict"] == "no_stall"
    assert result["is_measurement"] is True  # consumes real recordings, unlike the sims
