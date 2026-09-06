"""Check online latching and force-cadence validation without Isaac imports."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_reactive_interface import physics_windows

from gear_sonic.dataset_generation.hallucination.motion2scene_reactive_rule import select_skill


def test_sensor_latches_and_returns_only_at_registered_phase():
    active = 0
    outputs = []
    for time, hit in ((0.1, True), (0.2, True), (1.0, False), (3.28, False), (3.3, True)):
        active = select_skill("reactive", time, hit, active)
        outputs.append(active)
    assert outputs == [0, 1, 1, 1, 0]
    assert select_skill("reactive", 1.0, False, 0) == 0
    assert select_skill("blind", 1.0, True, 0) == 0
    assert select_skill("oracle", 0.2, False, 0) == 1


def test_rejects_invalid_interface_inputs():
    with pytest.raises(ValueError):
        select_skill("trained", 1, True, 0)
    with pytest.raises(ValueError):
        select_skill("reactive", float("nan"), True, 0)


def fixture():
    force = np.zeros((8, 30, 3))
    force[1, 4, 0] = 12  # A transient missed by a 50-Hz sample.
    return {
        "physics_steps": np.arange(1, 9),
        "control_steps": np.array([4, 8]),
        "force_w": force,
        "physics_dt": np.array(0.005),
    }, force[[3, 7]].copy()


def test_detects_substep_contact_missed_by_control_sampling():
    physics, sampled = fixture()
    _, aggregate, error = physics_windows(physics, sampled)
    assert sampled.max() == 0
    assert aggregate[0, 4, 0] == 12
    assert error == 0


def test_rejects_missing_substeps_and_stale_force_reads():
    physics, sampled = fixture()
    physics["physics_steps"][2] = 2
    with pytest.raises(ValueError, match="nonconsecutive"):
        physics_windows(physics, sampled)
    physics, sampled = fixture()
    sampled[0, 0, 0] = 1
    with pytest.raises(ValueError, match="mismatch"):
        physics_windows(physics, sampled)
