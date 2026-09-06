"""Check finite extent, all-source constraints, and the declared jitter denominator."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def teacher(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    return importlib.import_module("motion2scene_beam_teacher")


def sphere(z, label):
    return {
        "starts": np.array([[[0.0, 0.0, z]]]),
        "ends": np.array([[[0.0, 0.0, z]]]),
        "radii": np.array([0.1]),
        "label": label,
    }


def test_common_interval_includes_worst_execution(teacher):
    states = {"walk": sphere(1.3, "neutral"), "duck": sphere(1.1, "d055")}
    result = teacher.interval(states, [0, 0], 0, 0.1, 1.2)
    assert result["lower_m"] == pytest.approx(1.22, abs=2e-6)
    assert result["upper_m"] == pytest.approx(1.38, abs=2e-6)
    assert result["nonempty"]
    states["bad_repeat"] = sphere(1.31, "d055")
    result = teacher.interval(states, [0, 0], 0, 0.1, 1.2)
    assert not result["nonempty"]


def test_finite_beam_is_not_infinite_roof(teacher):
    local = teacher.local_capsules(sphere(1.6, "neutral"), [0, 0], 0, 0.1, 1.2)
    assert teacher.clearance(local, 1.2, 0.1, 1.2, roof=True) < 0
    assert teacher.clearance(local, 1.2, 0.1, 1.2) > 0


def test_all_81_jitter_points_and_both_motion_roles(teacher):
    states = {"walk": sphere(1.4, "neutral"), "duck": sphere(1.1, "d055")}
    candidate = {"center_xy_m": [0, 0], "yaw_rad": 0, "length_m": 0.1, "width_m": 1.2}
    result = teacher.jitter_audit(states, candidate, 1.3)
    assert result["tested_placements"] == 81
    assert result["all_passed"]
    states["escaping_walk"] = sphere(0.9, "neutral")
    assert not teacher.jitter_audit(states, candidate, 1.3)["all_passed"]


def test_world_frame_translation_is_shared(teacher):
    state = sphere(1.3, "neutral")
    shift = np.array([4.0, -7.0, 0.0])
    translated = {**state, "starts": state["starts"] + shift, "ends": state["ends"] + shift}
    first = teacher.local_capsules(state, [0, 0], 0.37, 0.1, 1.2)
    second = teacher.local_capsules(translated, shift[:2], 0.37, 0.1, 1.2)
    assert all(np.allclose(a, b) for a, b in zip(first, second))
