"""Geometric sensing and legal-switch contracts without an Isaac runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.dataset_generation.hallucination.motion2scene_overhang_observer import (
    observe_overhang,
    transition_decision,
)


class BoxQuery:
    def __init__(self, boxes):
        self.boxes = boxes

    def raycast_all(self, origin, direction, distance, callback):
        origin, direction = np.asarray(origin), np.asarray(direction)
        for path, low, high in self.boxes:
            enter, leave = 0.0, distance
            for i in range(3):
                if abs(direction[i]) < 1e-9:
                    if not low[i] <= origin[i] <= high[i]:
                        leave = -1
                        break
                else:
                    a, b = sorted(
                        ((low[i] - origin[i]) / direction[i], (high[i] - origin[i]) / direction[i])
                    )
                    enter, leave = max(enter, a), min(leave, b)
            if enter <= leave:
                hit = SimpleNamespace(
                    collision=path, distance=enter, position=origin + enter * direction
                )
                if not callback(hit):
                    return


def box(zlow, zhigh, path="/World/object", x=2.5):
    return path, np.array([x, -0.6, zlow]), np.array([x + 0.1, 0.6, zhigh])


def test_beam_has_lower_free_space_but_full_wall_does_not():
    root = np.array([0.0, 0.0, 0.74])
    occupied, _, rays = observe_overhang(root, [1, 0, 0, 0], BoxQuery([box(1.26, 1.36)]))
    assert occupied
    assert any(r["upper_candidate"] for r in rays)
    occupied, _, rays = observe_overhang(root, [1, 0, 0, 0], BoxQuery([box(0, 3)]))
    assert not occupied
    assert any(r["upper_candidate"] for r in rays)
    assert any(h["hit"] for r in rays for h in r["lower_rays"])


def test_empty_high_and_blocked_underpass_are_rejected_without_identity_filter():
    root = np.array([0.0, 0.0, 0.74])
    for boxes in ([], [box(2, 2.1)], [box(1.26, 1.36), box(0.2, 1.1, x=1.5)]):
        assert not observe_overhang(root, [1, 0, 0, 0], BoxQuery(boxes))[0]
    assert observe_overhang(root, [1, 0, 0, 0], BoxQuery([box(1.26, 1.36, "/World/renamed")]))[0]


def test_guard_checks_phase_and_reference_jump_independently():
    assert transition_decision("reactive", 0.2, True, 0, 0, 0) == (1, True, [])
    assert transition_decision("reactive", 3.3, False, 1, 0, 0) == (0, True, [])
    assert transition_decision("reactive", 2.6, True, 0, 0, 0) == (
        1,
        False,
        ["outside_legal_phase"],
    )
    assert transition_decision("reactive", 0.2, True, 0, 0.06, 0) == (
        1,
        False,
        ["joint_reference_jump"],
    )
    assert transition_decision("reactive", 0.2, True, 0, 0, 0.02) == (
        1,
        False,
        ["root_reference_jump"],
    )
    assert transition_decision("late_oracle", 2.6, False, 0, 0.63, 0)[1] is False


def test_blind_does_not_switch_and_active_skill_latches():
    assert transition_decision("blind", 0.2, True, 0, 0, 0) == (0, True, [])
    assert transition_decision("reactive", 2, False, 1, 1, 1) == (1, True, [])
    with pytest.raises(ValueError):
        transition_decision("reactive", float("nan"), True, 0, 0, 0)
