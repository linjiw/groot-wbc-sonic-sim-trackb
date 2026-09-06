import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import (
    EventMixture,
    event_features,
    fit_normalization,
    physical_at_anchor,
    sample_scenes,
    stratified_scenes,
)


def test_horizontal_translation_invariance_and_vertical_event_preservation():
    motion = np.zeros((20, 2, 7))
    route = np.c_[np.linspace(0, 3, 20), np.zeros(20)]
    motion[..., :2] = route[:, None]
    motion[5:10, :, 2] = 0.2
    shifted = motion.copy()
    shifted[..., :2] += [12, -5]
    np.testing.assert_allclose(
        event_features(motion, route), event_features(shifted, route + [12, -5])
    )
    assert event_features(motion, route)[:, 2].max() == pytest.approx(0.2)
    with pytest.raises(ValueError, match="stationary"):
        event_features(motion, np.zeros_like(route))


def test_normalization_does_not_read_excluded_features():
    features = {"train": np.arange(128).reshape(64, 2), "test": np.full((64, 2), np.nan)}
    mean, std = fit_normalization(features, ["train"])
    np.testing.assert_allclose(mean, [63, 64])
    assert np.isfinite(std).all()
    with pytest.raises(ValueError, match="unique"):
        fit_normalization(features, ["train", "train"])


def test_anchor_bounds_and_mixture_gradients():
    latent = torch.tensor([[-100.0, -100], [100, 100]], dtype=torch.float64)
    result = physical_at_anchor(latent, torch.tensor([0, 7]))
    torch.testing.assert_close(result, torch.tensor([[0.1, 1.1], [0.9, 1.45]], dtype=torch.float64))
    torch.manual_seed(14)
    model = EventMixture(features=3).double()
    feature = torch.randn(64, 3, dtype=torch.float64)
    scenes, weights = stratified_scenes(model(feature), torch.Generator().manual_seed(22))
    assert scenes.shape == (8, 2) and weights.sum().item() == pytest.approx(1)
    loss = (weights * ((scenes[:, 0] - 0.3).square() + (scenes[:, 1] - 1.23).square())).sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.base.grad[:, 0].abs() > 1e-9).all()
    assert model.input[0].weight.grad.abs().max() > 1e-9
    samples = sample_scenes(model(feature), 500, torch.Generator().manual_seed(41))
    assert ((samples[:, 0] >= 0.1) & (samples[:, 0] <= 0.9)).all()


def test_local_encoder_retains_event_position_and_pooled_control_discards_interior_shift():
    torch.manual_seed(20)
    local = EventMixture(features=3).double()
    pooled = EventMixture(features=3, pooled=True).double()
    pooled.load_state_dict(local.state_dict())
    early = torch.zeros(64, 3, dtype=torch.float64)
    late = early.clone()
    early[13:19, 0] = 2
    late[45:51, 0] = 2
    assert (local(early)[1] - local(late)[1]).abs().max() > 1e-5
    torch.testing.assert_close(pooled(early)[1], pooled(late)[1], rtol=0, atol=1e-12)
