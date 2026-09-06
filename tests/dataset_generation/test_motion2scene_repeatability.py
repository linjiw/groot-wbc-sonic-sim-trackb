"""Keep physics seeds as repeats and refuse incomplete admission evidence."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    return importlib.import_module("motion2scene_repeatability")


def complete_rows(diagnostic):
    return [
        {
            "runtime_seed": seed,
            "label": level,
            "status": "completed",
            "tracker_accepted": True,
            "route_retained": True,
            "paired_behavior_retained": level != "neutral",
            "state_sha256": str(seed),
        }
        for seed in diagnostic.SEEDS
        for level in diagnostic.LEVELS
    ]


def test_repeats_are_not_independent_carriers(diagnostic):
    result = diagnostic.summarize(complete_rows(diagnostic))
    assert result["strict_development_repeatability"]
    assert result["independent_carriers"] == 1
    assert result["physics_repeats"] == 3
    assert result["q4_admitted_ladders"] == 0
    assert not result["training_eligible"]


def test_one_failure_prevents_strict_repeatability(diagnostic):
    rows = complete_rows(diagnostic)
    rows[-1].update(
        status="skipped_dependency",
        tracker_accepted=None,
        route_retained=None,
        paired_behavior_retained=None,
    )
    rows[-1].pop("state_sha256")
    result = diagnostic.summarize(rows)
    assert not result["strict_development_repeatability"]
    assert result["counts"]["d055"]["registered"] == 3
    assert result["counts"]["d055"]["completed"] == 2


@pytest.mark.parametrize("bad", ("partial", "duplicate", "running"))
def test_no_partial_adjudication(diagnostic, bad):
    rows = complete_rows(diagnostic)
    if bad == "partial":
        rows.pop()
    elif bad == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["status"] = "running"
    with pytest.raises(ValueError):
        diagnostic.summarize(rows)
