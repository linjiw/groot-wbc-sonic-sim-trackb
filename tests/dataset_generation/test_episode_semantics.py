"""Tests for deriving goal objects and task language from episode geometry."""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.episode_semantics import (
    SceneObject,
    SemanticsError,
    derive_semantics,
    motion_style_phrase,
)

STRAIGHT = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])


def test_goal_is_the_object_nearest_the_endpoint():
    objects = [
        SceneObject("Sofa_00", (-0.4, -0.4, 0.4, 0.4)),  # near the start
        SceneObject("Bookshelf_01", (3.2, -0.4, 3.8, 0.4)),  # near the end
    ]
    semantics = derive_semantics(STRAIGHT, objects)
    assert semantics.goal_object == "Bookshelf_01"
    assert "stop at the bookshelf" in semantics.task


def test_nearest_to_path_is_not_used_as_the_goal():
    """A route brushes many objects; only one is where it stopped."""
    objects = [
        SceneObject("Chair_00", (1.4, -0.15, 1.6, 0.15)),  # closest to the path overall
        SceneObject("Cabinet_01", (3.4, -0.4, 4.0, 0.4)),  # closest to the endpoint
    ]
    assert derive_semantics(STRAIGHT, objects).goal_object == "Cabinet_01"


def test_no_goal_is_claimed_when_the_endpoint_is_in_open_floor():
    """Inventing a referent would put a false claim into the training language."""
    far = [SceneObject("Sofa_00", (-6.0, -6.0, -5.0, -5.0))]
    semantics = derive_semantics(STRAIGHT, far)
    assert not semantics.has_goal
    assert semantics.goal_object is None
    assert "stop at" not in semantics.task
    assert "sofa" not in semantics.task


def test_goal_radius_decides_whether_an_object_counts():
    objects = [SceneObject("Fridge_00", (4.2, -0.3, 4.8, 0.3))]  # 1.2 m past the endpoint
    assert derive_semantics(STRAIGHT, objects, goal_radius_m=1.5).has_goal
    assert not derive_semantics(STRAIGHT, objects, goal_radius_m=1.0).has_goal


def test_passed_objects_exclude_the_goal():
    objects = [
        SceneObject("Chair_00", (1.4, 0.4, 1.6, 0.7)),
        SceneObject("Cabinet_01", (3.2, -0.3, 3.7, 0.3)),
    ]
    semantics = derive_semantics(STRAIGHT, objects)
    assert semantics.goal_object == "Cabinet_01"
    assert "Cabinet_01" not in semantics.passed_objects


def test_two_flanking_objects_produce_a_between_instruction():
    objects = [
        SceneObject("Bookshelf_00", (1.4, 0.4, 1.8, 0.8)),
        SceneObject("Sofa_01", (1.4, -0.8, 1.8, -0.4)),
        SceneObject("Cabinet_02", (3.2, -0.3, 3.7, 0.3)),
    ]
    task = derive_semantics(STRAIGHT, objects).task
    assert "between the" in task and "then stop at the cabinet" in task


def test_camel_case_catalog_names_become_spoken_words():
    objects = [SceneObject("CoffeeTable_03", (3.2, -0.3, 3.7, 0.3))]
    semantics = derive_semantics(STRAIGHT, objects)
    assert semantics.goal_object_kind == "CoffeeTable"
    assert "coffee table" in semantics.task
    assert "CoffeeTable" not in semantics.task


def test_unlisted_catalog_names_still_read_as_words():
    objects = [SceneObject("StorageRack_00", (3.2, -0.3, 3.7, 0.3))]
    assert "storage rack" in derive_semantics(STRAIGHT, objects).task


def test_distance_is_zero_inside_a_footprint():
    obstacle = SceneObject("Sofa_00", (-1.0, -1.0, 1.0, 1.0))
    assert obstacle.distance_to((0.0, 0.0)) == 0.0
    assert obstacle.distance_to((2.0, 0.0)) == pytest.approx(1.0)


def test_crouching_is_described_from_root_height():
    low = np.full(len(STRAIGHT), 0.48)
    task = derive_semantics(STRAIGHT, [], root_z=low).task
    assert "crouch" in task


def test_turning_is_described_from_the_path():
    turn = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    assert "turn" in derive_semantics(turn, []).task


def test_explicit_motion_style_overrides_the_derived_description():
    """The instruction should say what was asked for, not only what the path looks like."""
    task = derive_semantics(STRAIGHT, [], motion_style="backward").task
    assert task.startswith("walk backwards")


def test_unknown_motion_style_is_rejected_with_the_known_list():
    with pytest.raises(SemanticsError, match="unknown motion style"):
        motion_style_phrase("moonwalk")
    with pytest.raises(SemanticsError):
        derive_semantics(STRAIGHT, [], motion_style="moonwalk")


def test_every_taxonomy_body_mode_has_a_phrase():
    from gear_sonic.dataset_generation.prompt_taxonomy import BODY_MODES

    for mode in BODY_MODES:
        assert motion_style_phrase(mode)


def test_a_one_point_path_is_rejected():
    with pytest.raises(SemanticsError, match="at least two path points"):
        derive_semantics(np.array([[0.0, 0.0]]), [])


def test_empty_scene_yields_a_task_with_no_object_reference():
    semantics = derive_semantics(STRAIGHT, [])
    assert semantics.task == "walk forward across the room"
    assert semantics.passed_objects == ()
