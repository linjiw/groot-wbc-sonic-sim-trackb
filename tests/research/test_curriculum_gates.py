from __future__ import annotations

import pytest

from scripts.research.curriculum_sampler import competence_score, hard_gates_pass
from scripts.research.update_curriculum_state import initial_state, update_state


def test_hard_gates_pass_min_and_max_thresholds() -> None:
    gates = {
        "success_rate_min": 0.80,
        "fall_rate_max": 0.05,
        "drop_rate_max": 0.10,
    }

    assert hard_gates_pass(
        {"success_rate": 0.82, "fall_rate": 0.01, "drop_rate": 0.0},
        gates,
    )
    assert not hard_gates_pass(
        {"success_rate": 0.82, "fall_rate": 0.06, "drop_rate": 0.0},
        gates,
    )
    assert not hard_gates_pass(
        {"success_rate": 0.79, "fall_rate": 0.01, "drop_rate": 0.0},
        gates,
    )


def test_competence_score_normalizes_active_terms() -> None:
    s0_score = competence_score({"schema_valid": True}, {"schema_valid_min": 1.0})
    s4_score = competence_score(
        {"success_rate": 0.80, "fall_rate": 0.0, "schema_valid": True},
        {"success_rate_min": 0.80, "fall_rate_max": 0.05, "torso_oscillation_max": 0.10},
    )

    assert s0_score == pytest.approx(1.0)
    assert s4_score == pytest.approx(1.0)


def test_update_state_unlocks_successor_after_competence_gate() -> None:
    graph = {
        "name": "test_curriculum",
        "competence_threshold": 0.85,
        "minimum_eval_episodes_per_stage": 2,
        "stages": [
            {
                "id": "S0",
                "name": "schema_validity",
                "unlock": "always",
                "gates": {"schema_valid_min": 1.0},
            },
            {
                "id": "S1",
                "name": "static_reach",
                "prerequisites": ["S0"],
                "gates": {
                    "success_rate_min": 0.90,
                    "fall_rate_max": 0.02,
                    "action_jerk_max": 0.05,
                },
            },
        ],
    }
    state = initial_state(graph)

    state = update_state(
        graph,
        state,
        {"S0": {"schema_valid": True, "eval_episodes": 2}},
        ema_alpha=0.3,
    )

    assert state["stages"]["S0"]["gate_passed"]
    assert state["stages"]["S1"]["unlocked"]


def test_update_state_does_not_pass_with_too_few_episodes() -> None:
    graph = {
        "name": "test_curriculum",
        "competence_threshold": 0.85,
        "minimum_eval_episodes_per_stage": 5,
        "stages": [
            {
                "id": "S0",
                "name": "schema_validity",
                "unlock": "always",
                "gates": {"schema_valid_min": 1.0},
            },
            {
                "id": "S1",
                "name": "static_reach",
                "prerequisites": ["S0"],
                "gates": {"success_rate_min": 0.90},
            },
        ],
    }
    state = initial_state(graph)

    state = update_state(
        graph,
        state,
        {"S0": {"schema_valid": True, "eval_episodes": 4}},
        ema_alpha=0.3,
    )

    assert not state["stages"]["S0"]["gate_passed"]
    assert not state["stages"]["S1"]["unlocked"]
