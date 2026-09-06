"""Guard paired experimental denominators and result adjudication."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research/motion2scene_timing_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("timing_diagnostic", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows():
    return [
        {
            "generation_seed": seed,
            "condition": condition,
            "label": label,
            "status": "completed",
            "tracker_accepted": condition == "shared_clock",
            "diagnostics": {"endpoint_error_m": 0.1 if condition == "shared_clock" else 0.3},
        }
        for seed in MODULE.SEEDS
        for condition in MODULE.CONDITIONS
        for label in MODULE.LEVELS
    ]


def test_paired_improvement_and_full_denominators():
    result = MODULE.summarize(rows())
    assert all(result["predictions"].values())
    assert result["counts"]["shared_clock/d055"]["registered"] == 3
    assert all(
        r["endpoint_error_delta_m"] == pytest.approx(-0.2)
        for r in result["paired_deltas_slower_minus_original"]
    )


def test_skips_are_not_improvements_or_removed_from_denominator():
    data = rows()
    for row in data:
        if row["condition"] == "shared_clock" and row["label"] == "d055":
            row.update(status="skipped_dependency", tracker_accepted=None)
            row.pop("diagnostics")
    result = MODULE.summarize(data)
    assert result["counts"]["shared_clock/d055"]["registered"] == 3
    assert result["counts"]["shared_clock/d055"]["completed"] == 0
    assert not result["predictions"]["p1_crouch_endpoint_improves_in_at_least_two_carriers"]
    assert not result["predictions"]["p2_more_tracker_accepted_crouches"]


@pytest.mark.parametrize("mutation", ("partial", "duplicate", "running", "extra"))
def test_refuse_incomplete_or_wrong_factorial(mutation):
    data = rows()
    if mutation == "partial":
        data.pop()
    elif mutation == "duplicate":
        data[-1] = data[0]
    elif mutation == "extra":
        data.append(data[0])
    else:
        data[-1]["status"] = "running"
    with pytest.raises(ValueError):
        MODULE.summarize(data)


def test_hash_mismatch_and_no_overwrite(tmp_path):
    path = tmp_path / "artifact.json"
    MODULE.write_new(path, {"value": 1})
    expected = MODULE.sha(path)
    with pytest.raises(FileExistsError):
        MODULE.write_new(path, {"value": 2})
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        MODULE.checked(path, expected)
