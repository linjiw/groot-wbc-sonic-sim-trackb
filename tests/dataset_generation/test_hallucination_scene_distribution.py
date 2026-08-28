"""Tests for q(S | executed pair) and its eps-delta scorecard."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.scene_distribution import (
    ARCHETYPE_DIMENSION_PRIORS,
    CriticalAtom,
    SceneDistribution,
    SupportError,
    realism_score,
    score_atom,
)


def _atom(nominal: float = 1.3021430, adapted: float = 1.2319751, **kwargs) -> CriticalAtom:
    defaults = dict(
        source_id="ladder_138",
        station_xy_m=(1.9, 0.0),
        route_axis="x",
        route_progress=0.55,
        face_along_route_m=0.10,
        face_across_route_m=3.0,
        reach_nominal_m=nominal,
        reach_adapted_m=adapted,
        delta_clear_m=0.018044,
        delta_strike_m=0.018044,
    )
    defaults.update(kwargs)
    return CriticalAtom(**defaults)


def test_window_is_the_closed_form_interval():
    atom = _atom()
    assert atom.lower_m == pytest.approx(1.2319751 + 0.018044)
    assert atom.upper_m == pytest.approx(1.3021430 - 0.018044)
    assert 1000 * atom.raw_width_m == pytest.approx(70.168, abs=1e-2)
    assert 1000 * atom.width_m == pytest.approx(34.080, abs=1e-2)


def test_regret_equals_xi_times_window_and_the_adapted_clearance():
    """The identity the eps-delta scorecard rests on."""
    atom = _atom()
    for xi in (0.0, 0.25, 0.5, 0.75, 1.0):
        coordinate = atom.coordinate_at(xi)
        # Regret is the depth by which the adaptation overshot what this face required.
        required_reduction = atom.reach_nominal_m - (coordinate - atom.delta_clear_m)
        actual_reduction = atom.reach_nominal_m - atom.reach_adapted_m
        assert atom.regret_m(xi) == pytest.approx(actual_reduction - required_reduction)
        assert atom.regret_m(xi) == pytest.approx(xi * atom.width_m)
        # ... and it is exactly the adapted motion's clearance beyond the engineering margin.
        assert atom.regret_m(xi) == pytest.approx(
            coordinate - atom.delta_clear_m - atom.reach_adapted_m
        )


def test_necessity_is_the_nominal_penetration_and_shares_one_budget_with_regret():
    atom = _atom()
    for xi in (0.0, 0.3, 1.0):
        coordinate = atom.coordinate_at(xi)
        assert atom.necessity_m(xi) == pytest.approx(atom.reach_nominal_m - coordinate)
        assert atom.regret_m(xi) + atom.necessity_m(xi) == pytest.approx(atom.budget_m())
    # The budget does not move with xi; only a wider executed window enlarges it.
    assert atom.budget_m() == pytest.approx(atom.delta_strike_m + atom.width_m)
    wider = _atom(adapted=1.20)
    assert wider.budget_m() > atom.budget_m()


def test_an_atom_whose_margins_eat_the_window_refuses_rather_than_inverts():
    atom = _atom(nominal=1.30, adapted=1.29)
    assert atom.width_m < 0
    with pytest.raises(SupportError):
        atom.coordinate_at(0.5)
    with pytest.raises(SupportError):
        SceneDistribution([atom], {"shelf_plank": 1})


def test_xi_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError):
        _atom().coordinate_at(1.5)


def test_realism_scores_only_non_binding_dimensions():
    assert realism_score("shelf_plank", thickness_m=0.04, across_m=0.60) == pytest.approx(1.0)
    assert realism_score("shelf_plank", thickness_m=0.40, across_m=0.60) == pytest.approx(0.5)
    assert realism_score("not_an_archetype", thickness_m=0.04, across_m=0.6) == 0.0


def test_samples_land_in_support_and_respect_the_regret_budget():
    atom = _atom()
    counts = {"shelf_plank": 4, "ibeam": 4, "door_lintel": 3, "hanging_panel": 1}
    budget_mm = 8.0
    distribution = SceneDistribution([atom], counts, regret_budget_mm=budget_mm)
    samples = distribution.sample(400, seed=11)
    assert samples
    for sample in samples:
        assert atom.lower_m - 1e-12 <= sample.coordinate_m <= atom.upper_m + 1e-12
        assert 1000 * atom.regret_m(sample.xi) <= budget_mm + 1e-9
        prior = ARCHETYPE_DIMENSION_PRIORS[sample.archetype]
        assert prior["thickness_m"][0] <= sample.thickness_m <= prior["thickness_m"][1]


def test_source_mass_is_equal_regardless_of_how_many_atoms_a_source_contributes():
    dense = [_atom(source_id="dense", face_along_route_m=depth) for depth in (0.1, 0.2, 0.3, 0.4)]
    sparse = [_atom(source_id="sparse")]
    distribution = SceneDistribution(dense + sparse, {"shelf_plank": 1})
    weights = distribution.source_weights()
    assert weights["dense"] == pytest.approx(weights["sparse"])
    drawn = [sample.source_id for sample in distribution.sample(2000, seed=3)]
    share = np.mean([name == "dense" for name in drawn])
    assert 0.4 < share < 0.6


def test_archetype_component_carries_an_exploration_floor():
    """A never-verified archetype keeps non-zero mass, so a closed support is not read as certain."""
    distribution = SceneDistribution([_atom()], {"shelf_plank": 10}, exploration=0.10)
    index = distribution.archetype_names.index("hvac_duct")
    assert distribution.archetype_probabilities[index] == pytest.approx(
        0.10 / len(distribution.archetype_names)
    )
    assert distribution.archetype_probabilities.sum() == pytest.approx(1.0)


def test_scorecard_admits_a_tight_realistic_face_and_rejects_a_regretful_one():
    atom = _atom()
    tight = score_atom(atom, archetype="shelf_plank", xi=0.05, thickness_m=0.04, across_m=0.6)
    loose = score_atom(atom, archetype="shelf_plank", xi=0.95, thickness_m=0.04, across_m=0.6)
    rule = dict(epsilon_mm=10.0, delta_mm=20.0, realism_floor=1.0)
    assert tight.admits(**rule)
    assert not loose.admits(**rule)
    unrealistic = score_atom(atom, archetype="shelf_plank", xi=0.05, thickness_m=0.90, across_m=0.6)
    assert not unrealistic.admits(**rule)


def _inventory():
    from gear_sonic.dataset_generation.hallucination.scene_distribution import ObstacleItem

    return [
        # A plain plank: wide face, nothing protrudes back towards the body.
        ObstacleItem("plank", "overhead", 0.40, 3.00, 0.04),
        # A door lintel: jambs reach the floor, but stand 0.8 m either side of the centreline.
        ObstacleItem(
            "lintel", "overhead", 0.20, 1.60, 0.20, protrusion_m=1.2, protrusion_half_gap_m=0.80
        ),
        # A narrow gantry whose legs stand inside the corridor the body sweeps.
        ObstacleItem(
            "narrow_gantry",
            "overhead",
            0.30,
            1.00,
            0.15,
            protrusion_m=1.0,
            protrusion_half_gap_m=0.25,
        ),
        # A face shorter than the required along-route exposure.
        ObstacleItem("stub", "overhead", 0.05, 3.00, 0.05),
        # Right shape, absurd thickness.
        ObstacleItem("slab", "overhead", 0.40, 3.00, 1.80),
        # Wrong constraint axis entirely.
        ObstacleItem("side_wall", "lateral_gap", 0.40, 3.00, 0.10),
    ]


def test_inventory_filter_keeps_only_items_that_can_realise_the_atom():
    from gear_sonic.dataset_generation.hallucination.scene_distribution import admissible_items

    atom = _atom()
    verdicts = {v.item_id: v for v in admissible_items(_inventory(), atom, body_half_width_m=0.35)}
    assert verdicts["plank"].admissible
    assert verdicts["lintel"].admissible, "jambs stand outside the swept corridor"
    assert not verdicts["narrow_gantry"].admissible
    assert "protrusion_enters_swept_corridor" in verdicts["narrow_gantry"].reasons
    assert "face_too_short_along_route" in verdicts["stub"].reasons
    assert "implausible_thickness" in verdicts["slab"].reasons
    assert "wrong_axis" in verdicts["side_wall"].reasons


def test_a_wider_body_rejects_an_item_a_narrow_body_accepts():
    """Admissibility is a property of the item *and* the body that must pass it."""
    from gear_sonic.dataset_generation.hallucination.scene_distribution import (
        ObstacleItem,
        item_admissibility,
    )

    lintel = ObstacleItem(
        "lintel", "overhead", 0.20, 1.60, 0.20, protrusion_m=1.2, protrusion_half_gap_m=0.40
    )
    atom = _atom()
    assert item_admissibility(lintel, atom, body_half_width_m=0.30).admissible
    wide = item_admissibility(lintel, atom, body_half_width_m=0.40)
    assert not wide.admissible
    assert "protrusion_enters_swept_corridor" in wide.reasons
