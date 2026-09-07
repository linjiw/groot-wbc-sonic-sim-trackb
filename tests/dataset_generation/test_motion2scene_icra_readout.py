"""Comparator isolation and exact preservation of the frozen learned readout."""

import hashlib
import json

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_icra_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    readout as original,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    OutcomePredictor,
)


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_script_uses_only_actual_packet_and_does_not_override_learned(tmp_path):
    model = OutcomePredictor()
    for p in model.parameters():
        torch.nn.init.zeros_(p)
    path = tmp_path / "policy.pt"
    torch.save({"model": model.state_dict(), "mean": np.zeros(214), "std": np.ones(214)}, path)
    before = torch.random.get_rng_state()
    policy = load_readout(path, digest(path))
    assert torch.equal(before, torch.random.get_rng_state())
    expected = original(policy["value"], np.zeros(214))
    result = readout(policy, np.zeros(214), packet={"occupied": True})
    assert result["selected_action"] == expected["selected_action"] == 0
    assert result["probabilities"] == expected["probabilities"]
    scripted = {"kind": "scripted_rays"}
    assert readout(scripted, np.zeros(214), packet={"occupied": True})["requested_action"] == 1
    assert readout(scripted, np.ones(214), packet={"occupied": False})["requested_action"] == 0
    with pytest.raises(ValueError):
        readout(scripted, np.zeros(214))


def test_privileged_refusal_remains_walking_and_hash_is_bound(tmp_path):
    path = tmp_path / "privileged.json"
    path.write_text(json.dumps({"kind": "privileged_geometry", "selected_action": -1}))
    policy = load_readout(path, digest(path))
    result = readout(policy, np.zeros(214))
    assert result["refusal"] and result["requested_action"] == 0
    assert result["probabilities"] is None
    with pytest.raises(ValueError):
        load_readout(path, "sha256:wrong")
