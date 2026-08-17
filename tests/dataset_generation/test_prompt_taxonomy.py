"""Tests for the structured prompt taxonomy."""

from __future__ import annotations

import pytest

from gear_sonic.dataset_generation.prompt_taxonomy import (
    BODY_MODES,
    NON_WALK_MODES,
    SPEED_STYLES,
    TURN_STYLES,
    build_taxonomy,
)


def test_default_taxonomy_lands_in_the_planned_size_band():
    taxonomy = build_taxonomy()
    assert 100 <= len(taxonomy.specs) <= 200


def test_prompts_are_unique():
    taxonomy = build_taxonomy()
    assert len(set(taxonomy.prompts)) == len(taxonomy.prompts)


def test_every_axis_value_is_represented():
    """A taxonomy that silently drops an axis value is worse than a shorter list."""
    taxonomy = build_taxonomy()
    assert set(taxonomy.counts["body_mode"]) == set(BODY_MODES)
    assert set(taxonomy.counts["speed"]) == set(SPEED_STYLES)
    assert set(taxonomy.counts["turn"]) == set(TURN_STYLES)


def test_majority_of_prompts_leave_the_plain_forward_walk():
    """The corpus's whole problem is that every episode is a forward walk."""
    taxonomy = build_taxonomy()
    non_walk = sum(spec.is_non_walk for spec in taxonomy.specs)
    assert non_walk > len(taxonomy.specs) / 2


def test_specs_carry_the_axis_values_that_produced_them():
    taxonomy = build_taxonomy(body_modes=["walk"], speeds=["brisk"], turns=["straight"])
    (spec,) = taxonomy.specs
    assert (spec.body_mode, spec.speed, spec.turn) == ("walk", "brisk", "straight")
    assert spec.expected_speed_mps == SPEED_STYLES["brisk"][1]
    assert spec.axis_key == "walk|brisk|straight"


def test_speed_phrase_appears_in_the_prompt():
    taxonomy = build_taxonomy(body_modes=["walk"], speeds=["slow"], turns=["straight"])
    assert "slowly" in taxonomy.specs[0].prompt


def test_direction_locked_modes_are_not_crossed_with_turns():
    """"A person steps sideways, then turns sharply left" describes two different things."""
    taxonomy = build_taxonomy(body_modes=["side_step"], speeds=["steady"], turns=list(TURN_STYLES))
    assert {spec.turn for spec in taxonomy.specs} == {"straight"}


def test_terminal_action_modes_skip_sharp_turns_but_keep_gentle_curves():
    taxonomy = build_taxonomy(body_modes=["walk_to_stop"], speeds=["steady"], turns=list(TURN_STYLES))
    turns = {spec.turn for spec in taxonomy.specs}
    assert not any(turn.startswith("sharp") for turn in turns)
    assert "gentle_left" in turns


def test_prompts_are_third_person_descriptions_not_robot_commands():
    """The model was trained on descriptions of captured human motion, not on commands."""
    for spec in build_taxonomy().specs:
        assert spec.prompt.startswith("A person ")


def test_prompts_have_no_stray_punctuation_or_doubled_spaces():
    for spec in build_taxonomy().specs:
        assert "  " not in spec.prompt
        assert " ," not in spec.prompt
        assert not spec.prompt.endswith((",", ".", " "))


def test_non_walk_modes_are_flagged_consistently():
    for spec in build_taxonomy().specs:
        assert spec.is_non_walk == (spec.body_mode in NON_WALK_MODES)


def test_unknown_axis_values_are_named_in_the_error():
    with pytest.raises(ValueError, match="unknown body mode"):
        build_taxonomy(body_modes=["moonwalk"])
    with pytest.raises(ValueError, match="unknown speed style"):
        build_taxonomy(speeds=["sprinting"])
    with pytest.raises(ValueError, match="unknown turn style"):
        build_taxonomy(turns=["u_turn"])


def test_by_axis_groups_every_spec():
    taxonomy = build_taxonomy()
    grouped = taxonomy.by_axis("speed")
    assert sum(len(v) for v in grouped.values()) == len(taxonomy.specs)
