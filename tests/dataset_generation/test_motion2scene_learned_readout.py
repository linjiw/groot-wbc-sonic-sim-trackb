import hashlib

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    OutcomePredictor,
)


def test_checkpoint_readout_preserves_rng_and_refusal_is_neutral_fallback(tmp_path):
    model = OutcomePredictor()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias[:] = torch.tensor([-5.0, -5.0])
    path = tmp_path / "policy.pt"
    torch.save(
        {"model": model.state_dict(), "mean": torch.zeros(214), "std": torch.ones(214)}, path
    )
    digest = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    before = torch.get_rng_state().clone()
    policy = load_readout(path, digest)
    assert torch.equal(before, torch.get_rng_state())
    decision = readout(policy, np.zeros(214))
    assert decision["selected_action"] == -1 and decision["requested_action"] == 0
    assert decision["refusal"] and decision["fallback"] == "neutral_commitment_not_stop"
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="hash mismatch"):
        load_readout(path, "sha256:invalid")


def test_both_feasible_prefers_walk_and_only_crouch_requests_it():
    model = OutcomePredictor()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.network[-1].bias[:] = torch.tensor([5.0, 5.0])
    policy = (model, torch.zeros(214), torch.ones(214))
    assert readout(policy, np.zeros(214))["requested_action"] == 0
    with torch.no_grad():
        model.network[-1].bias[0] = -5
    assert readout(policy, np.zeros(214))["requested_action"] == 1
