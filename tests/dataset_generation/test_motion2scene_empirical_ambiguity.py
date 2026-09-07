import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_empirical_ambiguity import (
    empirical_limits,
)


def test_outcome_aliasing_does_not_force_action_error():
    labels = [[1, 1]] * 9 + [[1, 0]] * 2 + [[0, 0]] * 2
    r = empirical_limits(np.zeros((13, 2)), labels, np.ones((13, 2), bool))
    assert r["masked_bce_infimum"] > 0
    assert r["observation_action_ceiling"] == r["paired_action_ceiling"] == 11 / 13
    assert r["action_ambiguity_gap"] == 0


def test_opposing_successful_actions_create_a_real_action_gap():
    r = empirical_limits([[0], [0]], [[1, 0], [0, 1]], [[1, 1], [1, 1]])
    assert r["observation_action_ceiling"] == 0.5
    assert r["paired_action_ceiling"] == 1
    assert r["masked_bce_infimum"] == pytest.approx(np.log(2))


def test_uniform_alias_contribution_matches_analytic_loss_floor():
    x = [[0]] * 6 + [[i] for i in range(1, 6)]
    y = [[1, 1]] * 4 + [[0, 0]] * 2 + [[1, 0]] * 5
    r = empirical_limits(x, y, np.ones((11, 2), bool))
    p = 4 / 6
    assert r["masked_bce_infimum"] == pytest.approx(
        6 / 11 * (-p * np.log(p) - (1 - p) * np.log(1 - p))
    )


def test_unknown_outcomes_are_not_turned_into_failure():
    r = empirical_limits([[0], [0]], [[1, np.nan], [0, 1]], [[1, 0], [1, 1]])
    assert r["complete_pairs"] == 1 and r["known_outcomes"] == 3
    assert r["masked_bce_infimum"] == pytest.approx(2 / 3 * np.log(2))
    assert r["paired_action_ceiling"] == 1
