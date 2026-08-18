from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
import pytest

from gear_sonic.research.lace.interventions import solve_fixed_budget_intervention
from gear_sonic.research.lace.schema import canonical_sha256


def _kl(probabilities: np.ndarray, base: np.ndarray) -> float:
    active = probabilities > 0.0
    return float(np.sum(probabilities[active] * np.log(probabilities[active] / base[active])))


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0.10, 0.20, 0.30, 0.40], dtype=np.float64),
        np.array([True, True, False, False]),
    )


def test_solves_target_kl_with_fixed_budget_and_canonical_digest() -> None:
    base, panel = _fixture()

    first = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=0.05,
        max_probability_ratio=3.0,
    )
    second = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=0.05,
        max_probability_ratio=3.0,
    )

    assert first == second
    assert first["panel_distribution"] == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 0.0, 0.0])
    assert first["p_base"] == first["base_probabilities"]
    assert first["r_s"] == first["panel_distribution"]
    assert first["p_plus"] == first["intervention_probabilities"]
    assert math.fsum(first["intervention_probabilities"]) == 1.0
    assert math.fsum(first["delta_p"]) == pytest.approx(0.0, abs=1e-15)
    assert first["realized_kl"] == pytest.approx(first["target_kl"], abs=1e-12)
    assert first["added_panel_exposure"] == pytest.approx(first["realized_tv"])
    assert first["fixed_budget_diagnostics"]["l1_probability_shift"] == pytest.approx(
        2.0 * first["realized_tv"]
    )
    assert first["ratio_diagnostics"]["maximum_probability_ratio"] <= 3.0 + 1e-12
    assert first["intervention_sha256"] == canonical_sha256(
        first,
        digest_field="intervention_sha256",
    )


def test_panel_distribution_contributes_support_not_hidden_within_panel_weights() -> None:
    base, mask = _fixture()
    distribution = np.array([0.01, 0.99, 0.0, 0.0], dtype=np.float64)

    from_mask = solve_fixed_budget_intervention(
        base,
        mask,
        target_kl=0.02,
        max_probability_ratio=4.0,
    )
    from_distribution = solve_fixed_budget_intervention(
        base,
        distribution,
        target_kl=0.02,
        max_probability_ratio=4.0,
    )

    assert from_distribution["panel_input_kind"] == "probability_distribution_support"
    assert from_distribution["panel_distribution"] == pytest.approx(from_mask["panel_distribution"])
    assert from_distribution["intervention_probabilities"] == pytest.approx(
        from_mask["intervention_probabilities"]
    )


def test_optional_tv_target_must_match_the_kl_solution() -> None:
    base, panel = _fixture()
    panel_mass = float(base[panel].sum())
    known_rho = 0.25
    panel_distribution = np.where(panel, base / panel_mass, 0.0)
    expected = (1.0 - known_rho) * base + known_rho * panel_distribution
    target_kl = _kl(expected, base)
    target_tv = known_rho * (1.0 - panel_mass)

    result = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=target_kl,
        target_tv=target_tv,
        tv_tolerance=1e-12,
        max_probability_ratio=4.0,
    )

    assert result["rho"] == pytest.approx(known_rho, abs=1e-12)
    assert result["realized_tv"] == pytest.approx(target_tv, abs=1e-12)

    with pytest.raises(ValueError, match="target_tv is incompatible"):
        solve_fixed_budget_intervention(
            base,
            panel,
            target_kl=target_kl,
            target_tv=target_tv + 0.05,
            tv_tolerance=1e-12,
            max_probability_ratio=4.0,
        )


def test_ratio_cap_reports_infeasible_kl_and_accepts_boundary() -> None:
    base = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    panel = np.array([True, False, False, False])
    cap = 1.2
    rho_limit = (cap - 1.0) / (1.0 / base[0] - 1.0)
    panel_distribution = np.array([1.0, 0.0, 0.0, 0.0])
    boundary = (1.0 - rho_limit) * base + rho_limit * panel_distribution
    boundary_kl = _kl(boundary, base)

    result = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=boundary_kl,
        max_probability_ratio=cap,
    )
    assert result["rho"] == pytest.approx(rho_limit)
    assert result["ratio_diagnostics"]["maximum_probability_ratio"] == pytest.approx(cap)
    assert result["ratio_diagnostics"]["ratio_cap_active"] is True

    with pytest.raises(ValueError, match="target_kl is infeasible"):
        solve_fixed_budget_intervention(
            base,
            panel,
            target_kl=boundary_kl + 0.01,
            max_probability_ratio=cap,
        )


def test_input_probability_tolerance_never_weakens_the_ratio_cap() -> None:
    base = np.array([0.95, 0.05], dtype=np.float64)

    result = solve_fixed_budget_intervention(
        base,
        [True, False],
        target_kl=0.0,
        max_probability_ratio=1.0,
        probability_tolerance=0.1,
    )

    assert result["ratio_diagnostics"]["rho_limit_from_ratio_cap"] == 0.0
    with pytest.raises(ValueError, match="target_kl is infeasible"):
        solve_fixed_budget_intervention(
            base,
            [True, False],
            target_kl=1e-6,
            max_probability_ratio=1.0,
            probability_tolerance=0.1,
        )


def test_zero_base_support_cannot_be_injected() -> None:
    base = np.array([0.5, 0.5, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="zero-base bins"):
        solve_fixed_budget_intervention(
            base,
            [True, False, True],
            target_kl=0.01,
            max_probability_ratio=2.0,
        )

    result = solve_fixed_budget_intervention(
        base,
        [True, False, False],
        target_kl=0.01,
        max_probability_ratio=2.0,
    )
    assert result["intervention_probabilities"][2] == 0.0
    assert result["ratio_diagnostics"]["ratio_by_bin"][2] is None
    assert result["ratio_diagnostics"]["zero_base_output_mass"] == 0.0


def test_full_support_panel_only_allows_the_no_op_target() -> None:
    base, _ = _fixture()
    panel = np.ones_like(base, dtype=bool)

    result = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=0.0,
        target_tv=0.0,
        max_probability_ratio=1.0,
    )
    assert result["rho"] == 0.0
    assert result["intervention_probabilities"] == pytest.approx(base)

    with pytest.raises(ValueError, match="target_kl is infeasible"):
        solve_fixed_budget_intervention(
            base,
            panel,
            target_kl=0.01,
            max_probability_ratio=10.0,
        )


def test_support_and_effective_sample_diagnostics_are_consistent() -> None:
    base = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float64)
    result = solve_fixed_budget_intervention(
        base,
        [1, 0, 1, 0],
        target_kl=0.01,
        max_probability_ratio=2.0,
    )

    assert result["support_stats"] == {
        "base_positive_bin_count": 3,
        "zero_base_bin_count": 1,
        "panel_bin_count": 2,
        "outside_panel_positive_bin_count": 1,
        "intervention_positive_bin_count": 3,
    }
    stats = result["effective_sample_stats"]
    assert stats["inverse_simpson_base"] == pytest.approx(1.0 / np.sum(base**2))
    assert 0.0 < stats["importance_weight_ess_fraction_from_base"] <= 1.0


@pytest.mark.parametrize(
    ("base", "message"),
    [
        ([0.5, -0.5, 1.0], "nonnegative"),
        ([0.5, float("nan"), 0.5], "finite"),
        ([0.2, 0.2], "sum to one"),
        ([True, False], "numeric, not boolean"),
        ([0.5 + 0.1j, 0.5 - 0.1j], "numeric, not boolean"),
        ([], "non-empty"),
    ],
)
def test_invalid_base_probabilities_fail_closed(base: list[object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solve_fixed_budget_intervention(
            base,
            [True] + [False] * max(0, len(base) - 1),
            target_kl=0.0,
            max_probability_ratio=2.0,
        )


@pytest.mark.parametrize(
    ("panel", "message"),
    [
        ([False, False, False, False], "non-empty"),
        ([1.0, -1.0, 0.0, 1.0], "nonnegative"),
        ([0.2, 0.2, 0.0, 0.0], "sum to one"),
        ([True, False], "shape"),
        (["yes", "no", "no", "no"], "boolean mask"),
        ([0.5 + 0.1j, 0.5 - 0.1j, 0.0, 0.0], "boolean mask"),
    ],
)
def test_invalid_panel_inputs_fail_closed(panel: list[object], message: str) -> None:
    base, _ = _fixture()
    with pytest.raises(ValueError, match=message):
        solve_fixed_budget_intervention(
            base,
            panel,
            target_kl=0.0,
            max_probability_ratio=2.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target_kl": -0.1, "max_probability_ratio": 2.0}, "target_kl"),
        ({"target_kl": 0.1, "max_probability_ratio": 0.9}, "max_probability_ratio"),
        (
            {"target_kl": 0.1, "max_probability_ratio": 2.0, "target_tv": 1.1},
            "target_tv",
        ),
        (
            {"target_kl": 0.1, "max_probability_ratio": 2.0, "kl_tolerance": 0.0},
            "kl_tolerance",
        ),
        (
            {"target_kl": 0.1, "max_probability_ratio": 2.0, "bisection_iterations": 0},
            "bisection_iterations",
        ),
    ],
)
def test_invalid_solver_parameters_fail_closed(kwargs: dict[str, object], message: str) -> None:
    base, panel = _fixture()
    with pytest.raises(ValueError, match=message):
        solve_fixed_budget_intervention(base, panel, **kwargs)


def test_inputs_are_not_mutated() -> None:
    base, panel = _fixture()
    original_base = base.copy()
    original_panel = panel.copy()

    solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=0.01,
        max_probability_ratio=2.0,
    )

    assert np.array_equal(base, original_base)
    assert np.array_equal(panel, original_panel)


def test_digest_binds_panel_input_provenance_even_when_support_matches() -> None:
    base, panel = _fixture()
    first = solve_fixed_budget_intervention(
        base,
        panel,
        target_kl=0.01,
        max_probability_ratio=2.0,
    )
    changed = deepcopy(first)
    changed["panel_input_kind"] = "mutated"

    assert changed["intervention_sha256"] != canonical_sha256(
        changed,
        digest_field="intervention_sha256",
    )
