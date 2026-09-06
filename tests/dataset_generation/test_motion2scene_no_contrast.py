import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_no_contrast import (
    target_only_accept,
    target_only_pattern,
    target_penalty,
)


def test_loss_and_rejection_preserve_noncritical_target_clear_scenes():
    target = torch.tensor([[0.02, 0.03], [0.009, 0.02]], requires_grad=True)
    penalty = target_penalty(target)
    assert target_only_accept(target).tolist() == [True, False]
    penalty.sum().backward()
    assert target.grad[1, 0] < 0 and target.grad[0].abs().sum() == 0


def test_search_uses_target_only_and_keeps_query_budget():
    initial = torch.tensor([[0.5, 1.2]], dtype=torch.float64)
    calls = []

    def target_query(scenes):
        calls.append(scenes.clone())
        return (0.04 - (scenes[:, 0] - 0.54).abs())[:, None]

    output, trace = target_only_pattern(initial, target_query)
    assert len(calls) == 17
    assert abs(output[0, 0] - 0.54) < abs(initial[0, 0] - 0.54)
    assert trace["neutral_channel_is_sentinel"]
