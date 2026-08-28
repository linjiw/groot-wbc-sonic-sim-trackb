"""Preflight must reject, on CPU, every cell the 2026-08-19 batch spent GPU discovering.

That batch ran 21 of 56 planned rollouts and produced no verified family. Each check below is
pinned to the measurement that exposed it, so a regression cannot pass by relaxing a threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.batch_preflight import (
    CROUCH_LAG_CURVE,
    ENDPOINT_BUDGET_M,
    TUCK_LAG_CURVE,
    check_endpoint_budget,
    check_route_fits_room,
    check_window_survives_delivery,
    interpolate,
    legacy_room_for,
    room_for,
)

ROUTES = {
    "n_013": np.array([[-1.98, -2.79], [2.23, -0.01]]),
    "n_064": np.array([[-3.81, -1.71], [-2.00, -0.01]]),
    "n_065": np.array([[-2.83, -1.71], [-2.00, -0.01]]),
    "n_122": np.array([[-5.97, -2.10], [-1.99, -0.01]]),
}


@pytest.mark.parametrize("name,overrun_mm", [("n_064", 906), ("n_065", 414), ("n_122", 1981)])
def test_legacy_rooms_reject_the_three_measured_overruns(name: str, overrun_mm: int) -> None:
    """Route bounds here are rounded to the centimetre, so the overrun is matched to ±10 mm."""
    ok, why = check_route_fits_room(ROUTES[name], legacy=True)
    assert not ok
    reported = float(why.split("by")[1].split("mm")[0])
    assert reported == pytest.approx(overrun_mm, abs=10)


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_fixed_rooms_accept_every_measured_route(name: str) -> None:
    ok, _ = check_route_fits_room(ROUTES[name], legacy=False)
    assert ok


def test_n013_was_never_the_problem() -> None:
    """The nominal that produced usable families fits under both rules."""
    assert check_route_fits_room(ROUTES["n_013"], legacy=True)[0]
    assert check_route_fits_room(ROUTES["n_013"], legacy=False)[0]


def test_fixed_room_is_never_smaller_than_the_legacy_one() -> None:
    for route in ROUTES.values():
        assert (room_for(route) >= legacy_room_for(route) - 1e-9).all()


def test_endpoint_budget_rejects_a_crouch_at_the_operator_cap() -> None:
    """0.98 rad is inside the crouch's own cap and far outside the endpoint budget.

    This is the check that makes the overhead band a decision rather than a retry: clamping at the
    cap produces a clip whose rejection is certain, so the cap alone is not a sufficient guard.
    """
    ok, why = check_endpoint_budget({"operator": "local_crouch", "commanded_excursion_rad": 0.98})
    assert not ok
    assert "0.509" in why


def test_endpoint_budget_binds_before_the_crouch_cap() -> None:
    """Somewhere below the 0.98 rad cap the budget is already spent; find it and pin the ordering."""
    binding = min(
        (
            x
            for x in np.linspace(0.0, 0.98, 99)
            if interpolate(CROUCH_LAG_CURVE, x) > ENDPOINT_BUDGET_M
        ),
        default=None,
    )
    assert binding is not None
    assert binding < 0.98
    assert 0.4 < binding < 0.8


def test_endpoint_budget_never_rejects_a_tuck() -> None:
    """The tuck's lag falls as it grows, so no amplitude in the measured range is disqualified."""
    for amplitude, _lag in TUCK_LAG_CURVE:
        ok, _ = check_endpoint_budget(
            {"operator": "local_arm_tuck", "commanded_excursion_rad": amplitude}
        )
        assert ok


def test_window_check_rejects_the_chest_family_that_struck_its_wall() -> None:
    """23.8 mm predicted at a 0.70 ratio is 17 mm delivered, and it struck the obstacle."""
    ok, why = check_window_survives_delivery(
        {
            "operator": "local_arm_tuck",
            "band": "chest",
            "window_m": 0.0238,
            "delivery_ratio_used": 0.70,
        }
    )
    assert not ok
    assert "17 mm" in why


def test_window_check_accepts_the_corrected_chest_window() -> None:
    """The delivery-corrected planner produces 75.2 mm on the same nominal, which survives."""
    ok, _ = check_window_survives_delivery(
        {
            "operator": "local_arm_tuck",
            "band": "chest",
            "window_m": 0.0752,
            "delivery_ratio_used": 0.70,
        }
    )
    assert ok


def test_unmeasured_band_is_assumed_to_deliver_as_badly_as_the_worst_measured_one() -> None:
    ok, _ = check_window_survives_delivery(
        {"operator": "local_arm_tuck", "band": "knee", "window_m": 0.05}
    )
    assert not ok
