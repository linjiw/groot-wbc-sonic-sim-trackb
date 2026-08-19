"""A generated room must contain the route it was built for.

`build_graded_scene.py` sized the room from the route's *span* and the shared scene spec places
walls symmetrically about the origin, so a route that does not straddle the origin ends up partly
outside its own room. `n_064` began 418 mm beyond the wall and spent its episode being pushed back
in: 183.5 N of lateral contact and a foot against a vertical surface, in a scene whose only intended
obstacle was a ceiling. Its whole family was void, and so were the two nominals queued behind it.

The fixtures below are the measured routes, so the regression is the real geometry rather than a
constructed one.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.clutter_scene_builder import ClutterSceneSpec

# Measured route bounds, (x_min, x_max, y_min, y_max), from each nominal's own screen rollout.
ROUTES = {
    "n_001": (-1.98, -0.11, -1.60, -0.01),
    "n_013": (-1.98, 2.23, -2.79, -0.01),
    "n_014": (-2.00, 4.22, -2.20, -0.01),
    "n_064": (-3.81, -2.00, -1.71, -0.01),
    "n_065": (-2.83, -2.00, -1.71, -0.01),
    "n_122": (-5.97, -1.99, -2.10, -0.01),
    "n_126": (-1.98, 1.52, -2.00, -0.01),
}

#: Metres of walkable floor the builder leaves beyond the route on every side.
CLEARANCE_M = 2.0


def route_xy(name: str) -> np.ndarray:
    x0, x1, y0, y1 = ROUTES[name]
    return np.array([[x0, y0], [x1, y1]], dtype=np.float64)


def span_sized_room(route: np.ndarray) -> tuple[float, float]:
    """The old rule: size from the span, and let the shared spec centre it on the origin."""
    span = route.max(0) - route.min(0)
    return float(span[0] + 2 * CLEARANCE_M), float(max(span[1] + 2 * CLEARANCE_M, 5.0))


def containing_room(route: np.ndarray) -> tuple[float, float]:
    """The fixed rule: size so the origin-centred room still contains the route."""
    reach = np.maximum(np.abs(route.min(0)), np.abs(route.max(0)))
    half = reach + CLEARANCE_M
    return float(2 * half[0]), float(max(2 * half[1], 5.0))


def overrun_m(route: np.ndarray, room: tuple[float, float]) -> float:
    """How far outside the room the route reaches; ≤0 means it fits."""
    half = np.array(room) / 2.0
    return float(np.maximum(np.abs(route.min(0)) - half, np.abs(route.max(0)) - half).max())


@pytest.mark.parametrize("name", sorted(ROUTES))
def test_containing_room_holds_every_measured_route(name: str) -> None:
    route = route_xy(name)
    assert overrun_m(route, containing_room(route)) <= 0.0


def test_span_sizing_puts_n064_outside_its_own_room() -> None:
    """The defect itself, so a regression cannot pass by reverting the rule."""
    route = route_xy("n_064")
    assert overrun_m(route, span_sized_room(route)) > 0.0


def test_n064_overrun_matches_the_rollout_that_found_it() -> None:
    """0.42 m predicted from geometry; 418 mm measured in the rollout that exposed this."""
    route = np.array([[-2.83, -1.71], [-2.00, -0.01]], dtype=np.float64)
    assert overrun_m(route, span_sized_room(route)) == pytest.approx(0.415, abs=0.01)


@pytest.mark.parametrize("name", ["n_064", "n_065", "n_122"])
def test_the_three_rejected_nominals_fit_once_the_room_contains_them(name: str) -> None:
    """They were unfit for the builder's defect, not their own."""
    route = route_xy(name)
    assert overrun_m(route, span_sized_room(route)) > 0.0
    assert overrun_m(route, containing_room(route)) <= 0.0


def test_walkable_bounds_are_still_symmetric_about_the_origin() -> None:
    """The fix works by sizing, not by moving walls, so this invariant must not have changed."""
    spec = ClutterSceneSpec(
        scene_id="t",
        split_group="t",
        room_size_xy=(8.0, 6.0),
        wall_height=2.5,
        pieces=[],
        path_xy=np.zeros((2, 2)),
        clearance_m=0.0,
        seed=0,
    )
    low, high = spec.walkable_bounds()
    assert low == (-4.0, -3.0)
    assert high == (4.0, 3.0)
