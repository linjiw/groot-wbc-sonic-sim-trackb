"""Tests for the support-preserving pilot's proposal controls and validation draw."""

from pathlib import Path
import sys

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_acquisition_pool import (
    acquisition_queues,
    draw_pool,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_support_pool import (  # noqa: E402
    MIXTURE_CHANNELS,
    PILOT_ARMS,
    fresh_candidates,
    gate_membership,
    select_pilot_encounters,
    stratum_schedule,
    support_queues,
)
from motion2scene_support_validation import (  # noqa: E402
    ancestry_groups,
    draw_validation_specifications,
    underside_bands,
    validation_cells,
)

OPTIONS = [
    "neutral",
    "short_e015_r265",
    "sustained_e015_r255",
    "prior_splice_e015_r265",
    "prior_splice_e050_r265",
    "short_e070_r265",
    "sustained_e070_r255",
]
STRATA = ("short", "sustained", "early_constraint", "short_then_short", "short_then_sustained")


def generation():
    return {
        "single_families_in_order": [
            {"name": "short", "station_range": [0.26, 0.77], "length_range_m": [0.1, 0.25]},
            {"name": "sustained", "station_range": [0.32, 0.68], "length_range_m": [0.5, 0.9]},
            {
                "name": "early_constraint",
                "station_range": [0.15, 0.35],
                "length_range_m": [0.1, 0.25],
            },
        ],
        "course_station_ranges": [[0.24, 0.35], [0.65, 0.77]],
        "common_ranges": {
            "underside_m": [1.14, 1.4],
            "width_m": [1.15, 1.55],
            "thickness_m": [0.08, 0.16],
            "lateral_offset_m": [-0.08, 0.08],
            "yaw_offset_rad": [-0.06, 0.06],
        },
    }


def geometry(candidates, *, margin=0.01, offsets=81):
    """Synthesise a screen in which a known subset lies outside each gate."""
    count, actions = len(candidates), len(OPTIONS)
    minimum = np.full((count, actions), margin + 0.05)
    evaluated = np.full((count, actions), offsets, dtype=np.int64)
    negative = np.zeros((count, actions))
    for i in range(count):
        if i % 3 == 0:  # outside the positive robustness bar on every option
            minimum[i, :] = margin - 0.005
            evaluated[i, :] = 1
        if i % 2 == 0:  # carries an interfering alternative
            negative[i, 1] = -margin - 0.02
    return minimum, evaluated, negative


def test_fresh_candidates_reproduces_the_screened_prefix_and_extends_it():
    draw = fresh_candidates(generation(), 93201, 8, OPTIONS, prior_rounds=4)
    shallow = draw_pool(generation(), 93201, 4, OPTIONS)
    assert draw["screened_prefix"] == shallow
    assert len(draw["fresh"]) == len(draw["drawn"]) - len(shallow)
    assert {row["stratum_round"] for row in draw["fresh"]} == {4, 5, 6, 7}
    assert not {row["candidate_id"] for row in draw["fresh"]} & {
        row["candidate_id"] for row in shallow
    }


def test_fresh_candidates_rejects_a_reservoir_that_does_not_extend():
    with pytest.raises(ValueError):
        fresh_candidates(generation(), 93201, 4, OPTIONS, prior_rounds=4)


def test_strict_queue_is_the_unchanged_contrast_rule():
    candidates = draw_pool(generation(), 93201, 6, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    original = acquisition_queues(candidates, OPTIONS, minimum, evaluated, negative)
    assert [row["candidate_id"] for row in queues["support_strict"]] == [
        row["candidate_id"] for row in original["analytic_contrast"]
    ]
    assert [row["geometric_slack_m"] for row in queues["support_strict"]] == [
        row["geometric_slack_m"] for row in original["analytic_contrast"]
    ]


def test_broad_queue_keeps_every_candidate_in_draw_order():
    candidates = draw_pool(generation(), 93202, 6, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    assert len(queues["support_broad"]) == len(candidates)
    assert [row["candidate_index"] for row in queues["support_broad"]] == sorted(
        row["candidate_index"] for row in candidates
    )


def test_broad_queue_has_support_outside_both_original_screens():
    candidates = draw_pool(generation(), 93203, 6, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    broad = queues["support_broad"]
    assert any(not row["inside_positive_screen"] for row in broad)
    assert any(not row["inside_negative_screen"] for row in broad)
    # Everything the strict rule can reach is inside both screens, so the broad
    # channel is a strict superset and the pilot really does restore support.
    strict = {row["candidate_id"] for row in queues["support_strict"]}
    assert strict <= {row["candidate_id"] for row in broad}
    assert all(
        row["inside_positive_screen"] and row["inside_negative_screen"]
        for row in broad
        if row["candidate_id"] in strict
    )


def test_gate_membership_reports_each_screen_independently():
    inside = gate_membership([0.05] * 3, [81] * 3, [-0.02, 0.0, 0.0])
    assert inside["inside_positive_screen"] and inside["inside_negative_screen"]
    nominal_only = gate_membership([0.05] * 3, [1] * 3, [0.0] * 3)
    assert not nominal_only["inside_positive_screen"]
    assert not nominal_only["inside_negative_screen"]


def test_stratum_schedule_balances_from_the_shared_prefix():
    prefix = ["short", "sustained", "early_constraint", "short_then_sustained"]
    assert stratum_schedule(prefix, 4, STRATA) == [
        "short_then_short",
        "short",
        "sustained",
        "early_constraint",
    ]


def test_stratum_schedule_is_identical_for_every_arm_by_construction():
    prefix = ["short", "sustained", "short_then_short", "short_then_sustained"]
    first = stratum_schedule(prefix, 4, STRATA)
    assert first == stratum_schedule(list(prefix), 4, STRATA)
    assert first[0] == "early_constraint"


def test_selection_matches_strata_across_arms_and_realises_half_the_mixture():
    candidates = draw_pool(generation(), 93201, 12, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    schedule = ["short", "sustained", "early_constraint", "short_then_short"]
    chosen = select_pilot_encounters(queues, schedule)
    assert set(chosen["selected"]) == set(PILOT_ARMS)
    for arm in PILOT_ARMS:
        assert [row["stratum"] for row in chosen["selected"][arm]] == schedule
    assert chosen["realised_mixture_strict_fraction"] == 0.5
    assert [row["proposal_channel"] for row in chosen["selected"]["support_mixture"]] == list(
        MIXTURE_CHANNELS
    )
    assert not chosen["shortfalls"]
    assert chosen["physical_outcomes_consulted"] is False


def test_mixture_draws_each_slot_from_the_same_candidate_as_its_source_arm():
    candidates = draw_pool(generation(), 93202, 12, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    schedule = ["short", "sustained", "early_constraint", "short_then_short"]
    chosen = select_pilot_encounters(queues, schedule)
    by_arm = {arm: chosen["selected"][arm] for arm in PILOT_ARMS}
    for slot, channel in enumerate(MIXTURE_CHANNELS):
        assert (
            by_arm["support_mixture"][slot]["candidate_id"] == by_arm[channel][slot]["candidate_id"]
        )


def test_selection_retains_a_shortfall_instead_of_refilling_from_the_other_channel():
    candidates = draw_pool(generation(), 93203, 4, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    # Ask for more encounters in one stratum than the strict channel can supply.
    schedule = ["short"] * 12
    chosen = select_pilot_encounters(queues, schedule, mixture_channels=("support_strict",) * 12)
    assert chosen["shortfalls"]
    assert {row["arm"] for row in chosen["shortfalls"]} <= set(PILOT_ARMS)
    strict_ids = {row["candidate_id"] for row in queues["support_strict"]}
    assert {row["candidate_id"] for row in chosen["selected"]["support_strict"]} <= strict_ids


def test_validation_bands_split_the_declared_underside_range_at_its_midpoint():
    bands = underside_bands(generation()["common_ranges"])
    assert bands["low"] == [1.14, 1.27]
    assert bands["high"] == [1.27, 1.4]


def test_validation_sample_covers_every_family_and_band_without_any_query():
    drawn = draw_validation_specifications(generation())
    assert len(drawn) == len(validation_cells(generation()))
    assert len(drawn) == 10
    assert {(row["stratum"], row["underside_band"]) for row in drawn} == {
        (name, band) for name in STRATA for band in ("low", "high")
    }
    for row in drawn:
        low, high = row["underside_band_range_m"]
        for beam in row["beam_specifications"]:
            assert low <= beam["underside_m"] <= high
        assert row["clearance_queries"] == 0
        assert row["physical_outcomes_used"] is False
        assert row["selected_by_policy_failure"] is False
        assert row["rejection_sampling"] is False


def test_validation_sample_is_deterministic_and_seed_sensitive():
    assert draw_validation_specifications(generation()) == draw_validation_specifications(
        generation()
    )
    other = draw_validation_specifications(generation(), seed=940911)
    assert [row["beam_specifications"] for row in other] != [
        row["beam_specifications"] for row in draw_validation_specifications(generation())
    ]


def test_validation_layouts_are_each_their_own_base_layout():
    drawn = draw_validation_specifications(generation())
    groups = ancestry_groups(drawn)
    assert len(groups) == len(drawn)
    assert all(len(members) == 1 for members in groups.values())


def test_validation_draw_does_not_reuse_a_training_reservoir_specification():
    drawn = draw_validation_specifications(generation())
    reservoir = draw_pool(generation(), 93201, 16, OPTIONS)
    training = {
        tuple(sorted(beam.items())) for row in reservoir for beam in row["beam_specifications"]
    }
    for row in drawn:
        for beam in row["beam_specifications"]:
            assert tuple(sorted(beam.items())) not in training


def test_strict_rows_record_their_screen_membership_explicitly():
    candidates = draw_pool(generation(), 93201, 12, OPTIONS)
    minimum, evaluated, negative = geometry(candidates)
    queues = support_queues(candidates, OPTIONS, minimum, evaluated, negative)
    assert queues["support_strict"], "the synthetic screen must leave strict support"
    for row in queues["support_strict"]:
        # A strict row is inside both screens by construction; the field must say
        # so rather than being absent, which a reader would take for "outside".
        assert row["inside_positive_screen"] is True
        assert row["inside_negative_screen"] is True
        assert row["robust_positive_option_count"] >= 1
        assert row["interfering_negative_option_count"] >= 1
