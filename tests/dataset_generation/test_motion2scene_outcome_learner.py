import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_outcome_learner import (
    FEATURE_COUNT,
    fit,
    masked_loss,
    select_action,
)


def test_unknown_nan_target_cannot_affect_gradient():
    logits = torch.zeros((2, 2), requires_grad=True)
    labels = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    masked_loss(logits, labels, torch.tensor([[1, 0], [1, 1]])).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1] == 0
    with pytest.raises(ValueError):
        masked_loss(logits, labels, torch.zeros((2, 2)))


def test_all_four_outcomes_and_refusal_are_distinct():
    assert select_action([[0.9, 0.9], [0.1, 0.9], [0.9, 0.1], [0.1, 0.1]]).tolist() == [0, 1, 0, -1]


def test_two_outcome_heads_can_learn_without_forcing_a_success_label():
    torch.set_num_threads(1)
    x = np.zeros((16, FEATURE_COUNT), dtype=np.float32)
    y = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32), (4, 1))
    x[:, :2] = y * 2 - 1
    model, mean, std, loss = fit(x, y, np.ones_like(y), 17, steps=100)
    with torch.no_grad():
        prediction = model((torch.tensor(x) - mean) / std).sigmoid().numpy()
    np.testing.assert_array_equal(prediction >= 0.5, y.astype(bool))
    assert loss < 0.5 * np.log(2)
