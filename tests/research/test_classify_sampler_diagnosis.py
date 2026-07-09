from __future__ import annotations

import pytest

from scripts.research.classify_sampler_diagnosis import classify_diagnosis, classify_seed


def _telemetry(seed, *, pmax, concentrated, eff_bins, eff_bins_first=70.0):
    return {
        "log_path": f"outputs/seed{seed}/adaptive_sampling_micro/train.log",
        "adaptive_telemetry_present": True,
        "keys": {
            "prob_max_over_uniform": {"first": 3.0, "last": pmax},
            "num_concentrated_bins": {"first": 0.0, "last": concentrated},
            "effective_num_bins": {"first": eff_bins_first, "last": eff_bins},
        },
    }


def _checkpoint(seed, *, prior_dominated_fraction, num_bins=70, domination_threshold=2.0):
    return {
        "checkpoint_path": f"outputs/adaptive_seed{seed}/last.pt",
        "adaptive_state_present": True,
        "num_bins": num_bins,
        "params": {"prior_domination_threshold": domination_threshold},
        "aggregates": {"prior_dominated_fraction": prior_dominated_fraction},
    }


def test_compound_under_active_when_flat_and_prior_dominated() -> None:
    # SIM-M3 working hypothesis: flat distribution + prior-dominated bins.
    result = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5),
        _checkpoint(0, prior_dominated_fraction=0.95),
    )
    assert result["verdict"] == "compound_under_active_signal_starved"
    assert result["evidence"]["flat"] is True
    assert result["evidence"]["starved"] is True


def test_plain_under_active_when_flat_but_not_starved() -> None:
    # Flat distribution but real failures accrued -> sampler too damped, M5a.
    result = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5),
        _checkpoint(0, prior_dominated_fraction=0.10),
    )
    assert result["verdict"] == "plain_under_active"
    assert result["evidence"]["starved"] is False


def test_flat_boundary_uses_frozen_absolute_floor() -> None:
    # The floor is the preregistered literal 0.9*70 = 63, NOT a fraction of the
    # observed bin count. A 69-bin checkpoint must not move the boundary.
    not_flat = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=62.0),
        _checkpoint(0, prior_dominated_fraction=0.95, num_bins=69),
    )
    assert not_flat["evidence"]["flat"] is False
    just_flat = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=63.5),
        _checkpoint(0, prior_dominated_fraction=0.95, num_bins=69),
    )
    assert just_flat["evidence"]["flat"] is True
    # eff_bins=62.5 with 69 bins: frozen floor (63) => not flat, even though
    # 0.9*69=62.1 would have said flat. Proves num_bins does not move the floor.
    frozen = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=62.5),
        _checkpoint(0, prior_dominated_fraction=0.95, num_bins=69),
    )
    assert frozen["evidence"]["flat"] is False


def test_concentrated_undetermined_without_difficulty_ranking() -> None:
    # Not flat, but no per-motion difficulty available at M4b time.
    result = classify_seed(
        _telemetry(0, pmax=14.0, concentrated=3.0, eff_bins=20.0),
        _checkpoint(0, prior_dominated_fraction=0.1),
        difficulty_rank_agrees=None,
    )
    assert result["verdict"] == "active_targeting_undetermined"


def test_concentrated_wrong_target_when_ranking_disagrees() -> None:
    result = classify_seed(
        _telemetry(0, pmax=14.0, concentrated=3.0, eff_bins=20.0),
        _checkpoint(0, prior_dominated_fraction=0.1),
        difficulty_rank_agrees=False,
    )
    assert result["verdict"] == "wrong_target"


def test_concentrated_not_useful_when_targeted_and_gate_failed() -> None:
    result = classify_seed(
        _telemetry(0, pmax=14.0, concentrated=3.0, eff_bins=20.0),
        _checkpoint(0, prior_dominated_fraction=0.1),
        difficulty_rank_agrees=True,
        effect_gate_failed=True,
    )
    assert result["verdict"] == "active_but_not_useful"


def test_flat_without_checkpoint_uses_starvation_neutral_verdict() -> None:
    # No checkpoint -> cannot confirm starvation -> route conservatively to D1->M5b
    # but with a starvation-NEUTRAL label (must not assert the starved diagnosis).
    result = classify_seed(_telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5), None)
    assert result["verdict"] == "under_active_starvation_unconfirmed"
    assert "starved" not in result["verdict"] or "unconfirmed" in result["verdict"]
    assert result["evidence"]["starved"] is None


def test_incomplete_when_checkpoint_dump_used_wrong_domination_threshold() -> None:
    # A dump generated at threshold != 2.0 makes prior_dominated_fraction
    # incomparable to the preregistered <= 2 coupling.
    result = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5),
        _checkpoint(0, prior_dominated_fraction=0.99, domination_threshold=10.0),
    )
    assert result["verdict"] == "incomplete"
    assert result["evidence"]["prior_domination_threshold_ok"] is False


def test_flatness_floor_is_frozen_even_without_checkpoint() -> None:
    # Without a checkpoint, num_bins is unknown but the frozen floor (63) still
    # decides flatness — no telemetry-proxy substitution that could move it.
    result = classify_seed(
        _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5, eff_bins_first=70.0), None
    )
    assert result["evidence"]["num_bins"] is None
    assert result["evidence"]["effective_num_bins_floor"] == 63.0
    assert result["evidence"]["flat"] is True


def test_incomplete_when_required_telemetry_key_missing() -> None:
    tele = _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5)
    del tele["keys"]["prob_max_over_uniform"]
    result = classify_seed(tele, _checkpoint(0, prior_dominated_fraction=0.95))
    assert result["verdict"] == "incomplete"


def test_majority_verdict_over_three_seeds() -> None:
    telemetry = {
        "logs": [_telemetry(s, pmax=3.0, concentrated=0.0, eff_bins=69.5) for s in (0, 1, 2)]
    }
    dump = {"checkpoints": [_checkpoint(s, prior_dominated_fraction=0.95) for s in (0, 1, 2)]}
    result = classify_diagnosis(telemetry, dump)
    assert result["overall_verdict"] == "compound_under_active_signal_starved"
    assert result["decision_routing"] == "SIM-D1 -> SIM-M5b"
    assert result["scored_seed_count"] == 3


def test_one_scored_of_three_seeds_is_not_a_majority() -> None:
    # 2 of 3 adaptive seeds have no telemetry (incomplete); 1 is flat+starved.
    # 1/3 is NOT a majority over the adaptive seeds -> insufficient_data, not a
    # diagnosis declared off a single seed.
    telemetry = {"logs": [_telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5)]}
    dump = {
        "checkpoints": [
            _checkpoint(0, prior_dominated_fraction=0.95),
            _checkpoint(1, prior_dominated_fraction=0.95),
            _checkpoint(2, prior_dominated_fraction=0.95),
        ]
    }
    result = classify_diagnosis(telemetry, dump)
    # seeds 1,2 have no telemetry -> incomplete; only seed 0 scores.
    assert result["seed_count"] == 3
    assert result["scored_seed_count"] == 1
    assert result["overall_verdict"] == "insufficient_data"


def test_no_majority_when_seeds_split_evenly() -> None:
    # 2 seeds only, disagreeing -> no strict majority.
    telemetry = {
        "logs": [
            _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5),
            _telemetry(1, pmax=3.0, concentrated=0.0, eff_bins=69.5),
        ]
    }
    dump = {
        "checkpoints": [
            _checkpoint(0, prior_dominated_fraction=0.95),  # compound
            _checkpoint(1, prior_dominated_fraction=0.10),  # plain
        ]
    }
    result = classify_diagnosis(telemetry, dump)
    assert result["overall_verdict"] == "no_majority"


def test_ignores_uniform_and_non_adaptive_records() -> None:
    telemetry = {
        "logs": [
            _telemetry(0, pmax=3.0, concentrated=0.0, eff_bins=69.5),
            {
                "log_path": "outputs/seed0/uniform_sampling_micro/train.log",
                "adaptive_telemetry_present": False,
            },
        ]
    }
    dump = {"checkpoints": [_checkpoint(0, prior_dominated_fraction=0.95)]}
    result = classify_diagnosis(telemetry, dump)
    assert result["seed_count"] == 1
    assert result["overall_verdict"] == "compound_under_active_signal_starved"


def test_raises_when_no_adaptive_seeds() -> None:
    with pytest.raises(ValueError, match="no adaptive seeds"):
        classify_diagnosis({"logs": []}, {"checkpoints": []})
