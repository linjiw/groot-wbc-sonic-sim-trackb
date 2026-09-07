import hashlib
import importlib
from pathlib import Path

import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_learned_readout import (
    load_readout,
    readout,
)
from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import select_action


def test_linear_control_survives_runtime_checkpoint_and_all_action_regions(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/research"))
    embed = importlib.import_module("motion2scene_linear_execution_v2").embed_linear
    rng = np.random.default_rng(19)
    weights = rng.normal(0, 0.1, (214, 2)).astype(np.float32)
    bias = np.array([0.2, -0.3], np.float32)
    scale = np.ones(214, np.float32)
    scale[185:] = 5
    before = torch.get_rng_state().clone()
    model = embed(weights, bias)
    assert torch.equal(before, torch.get_rng_state())
    path = tmp_path / "linear.pt"
    torch.save({"model": model.state_dict(), "mean": np.zeros(214), "std": scale}, path)
    policy = load_readout(path, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest())
    actions = set()
    for signs in ((-3, -3), (-3, 3), (3, -3), (3, 3)):
        # Construct physical inputs for four different feasibility regions.
        z = np.linalg.lstsq(weights.T, np.array(signs) - bias, rcond=None)[0].astype(np.float32)
        features = z * scale
        expected = torch.tensor(z @ weights + bias).sigmoid().numpy()
        actual = readout(policy, features)
        np.testing.assert_allclose(actual["probabilities"], expected, atol=1e-6, rtol=0)
        assert actual["selected_action"] == int(select_action(expected))
        actions.add(actual["selected_action"])
    assert actions == {-1, 0, 1}
    assert torch.equal(before, torch.get_rng_state())
