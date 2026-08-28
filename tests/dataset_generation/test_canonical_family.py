"""Pin `cf_005_056`, the first physics-verified counterfactual family.

This family is the project's reference result, and its numbers appear in a paper draft, in
two published pages and in several commit messages. Everything that produced it lives on a
scratch filesystem, under a directory name a later run over the same motion pair will reuse
-- which has already happened once: a second run overwrote the first's scene USDA silently,
after which the earlier family's report was recomputed against the later family's geometry
and reported clearances for scenes that had never been rolled out.

So the artifacts are checked in and asserted here. If a number below changes, either the
pipeline changed meaning or the fixture was overwritten, and both need a person to look.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/counterfactual"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"cf_005_056_{name}.json").read_text())


@pytest.fixture(scope="module")
def family() -> dict:
    return load("family")


@pytest.fixture(scope="module")
def attribution() -> dict:
    return load("attribution")


@pytest.fixture(scope="module")
def robustness() -> dict:
    return load("robustness")


def test_the_window_is_the_one_the_station_search_found(family):
    """53 mm at the route midpoint, 178 mm at the station where the motions differ."""
    assert family["window_m"] == pytest.approx(0.178, abs=5e-4)
    assert family["nominal_clears_to_m"] == pytest.approx(1.302, abs=1e-3)
    assert family["adapted_clears_to_m"] == pytest.approx(1.123, abs=1e-3)


def test_exactly_one_cell_fails_and_it_is_the_nominal_motion_in_the_hard_scene(family):
    results = family["results"]
    assert results["nominal_hard"]["outcome"] == "rejected"
    for cell in ("nominal_easy", "adapted_easy", "adapted_hard"):
        assert results[cell]["outcome"] == "accepted", cell
        assert results[cell]["lateral_contact_n"] == 0.0, f"{cell} must be clean, not merely small"
    assert family["counterfactual_established"] is True


def test_the_geometric_predictor_lands_within_two_frames_of_physics(attribution):
    """The single number tying the swept-volume model to the simulator."""
    cell = attribution["cells"]["nominal_hard"]
    assert cell["first_contact_frame"] == 113
    assert cell["predicted_first_interference_frame"] == 112
    assert abs(cell["first_contact_frame"] - cell["predicted_first_interference_frame"]) <= 2


def test_the_contacting_body_is_the_one_the_overhead_regime_targets(attribution):
    assert attribution["cells"]["nominal_hard"]["first_contact_body"] == "torso_link"
    assert attribution["attribution_pure"] is True


def test_drift_follows_the_contact_rather_than_preceding_it(attribution):
    """If drift came first the episode failed for tracking reasons and the shelf is
    incidental. It does not: lowering the shelf moves drift onset from 175 to 120, and the
    contact is at 113."""
    cells = attribution["cells"]
    assert cells["nominal_hard"]["drift_onset_frame"] > cells["nominal_hard"]["first_contact_frame"]
    assert cells["nominal_easy"]["drift_onset_frame"] > cells["nominal_hard"]["drift_onset_frame"]


def test_the_four_force_quantities_are_recorded_separately(attribution):
    """55.4 N and 137.2 N are both true and are not the same measurement."""
    cell = attribution["cells"]["nominal_hard"]
    assert cell["first_contact_force_n"] == pytest.approx(55.4, abs=0.2)
    assert cell["peak_contact_force_n"] == pytest.approx(137.2, abs=0.2)
    assert cell["peak_contact_force_n"] > cell["first_contact_force_n"]
    assert cell["contact_impulse_ns"] > 0
    assert cell["contact_duration_s"] > 0


def test_every_start_pose_jitter_reproduces_the_same_2x2(robustness):
    assert robustness["perturbation_robust"] is True
    assert len(robustness["holds"]) == 3
    assert all(robustness["holds"].values())
    for cells in robustness["outcomes"].values():
        assert cells["nominal_hard"] == "rejected"
        assert cells["nominal_easy"] == cells["adapted_easy"] == cells["adapted_hard"] == "accepted"


def test_the_claim_is_worded_no_more_broadly_than_what_was_varied(robustness):
    """Only the start pose was varied, and only the outcome survived it."""
    assert robustness["claim_level"] == "start-pose outcome-robust"


def test_the_perturbations_are_small_enough_to_leave_the_task_unchanged(robustness):
    """Large enough to break bit-identical replay, small enough that it is the same task."""
    import math

    for perturbation in robustness["perturbations"]:
        offset = math.hypot(perturbation["dx_m"], perturbation["dy_m"])
        assert 0.005 < offset < 0.05, offset
        assert abs(math.degrees(perturbation["dyaw_rad"])) <= 1.5


def test_both_scenes_differ_only_in_the_shelf_height():
    """The whole claim rests on one scalar changing between the two rooms."""
    easy = (FIXTURES / "cf_005_056_easy.usda").read_text().splitlines()
    hard = (FIXTURES / "cf_005_056_hard.usda").read_text().splitlines()
    assert len(easy) == len(hard)
    differing = [(a, b) for a, b in zip(easy, hard) if a != b]
    translate = [d for d in differing if "translate" in d[0]]
    assert len(translate) == 1, f"more than the shelf moved: {differing}"
