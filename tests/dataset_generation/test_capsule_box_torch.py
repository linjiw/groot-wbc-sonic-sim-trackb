"""Independent geometry agreement and derivatives needed for scene optimization."""

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation import capsule_box_exact as oracle, capsule_box_torch as geometry


def tensor(value, grad=False):
    return torch.tensor(value, dtype=torch.float64, requires_grad=grad)


def test_random_batched_boxes_against_numpy():
    rng = np.random.default_rng(597)
    starts, ends = rng.normal(size=(2, 31, 3)), rng.normal(size=(2, 31, 3))
    lo = rng.normal(size=(5, 3))
    hi = lo + rng.uniform(0.01, 1.0, size=(5, 3))
    expected = np.stack(
        [oracle.segment_box_distance(starts, ends, low, high) for low, high in zip(lo, hi)]
    )
    actual = geometry.segment_box_distance(
        tensor(starts)[None],
        tensor(ends)[None],
        tensor(lo)[:, None, None],
        tensor(hi)[:, None, None],
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12)


def test_missed_axis_sample_contact_and_finite_backward():
    args = [
        tensor(x, True) for x in ([0, 0, 0], [1, 1, 0], [0.30, 0.32, -0.01], [0.40, 0.34, 0.01])
    ]
    result = geometry.capsule_box_clearance(args[0], args[1], tensor(0.005), *args[2:])
    assert result.item() == pytest.approx(-0.005)
    result.backward()
    assert all(torch.isfinite(x.grad).all() for x in args)


@pytest.mark.parametrize(
    "start,end",
    [([2, 2, 2], [2, 2, 2]), ([-2, 2, 0.5], [2, 2, 0.5]), ([0.2, 0.3, 0.4], [0.8, 0.7, 0.6])],
)
def test_degenerate_and_parallel_backward(start, end):
    args = [tensor(x, True) for x in (start, end, [0, 0, 0], [1, 1, 1])]
    value = geometry.segment_box_distance(*args)
    assert value.item() == pytest.approx(
        oracle.segment_box_distance(start, end, [0, 0, 0], [1, 1, 1])
    )
    value.backward()
    assert all(torch.isfinite(x.grad).all() for x in args)


def test_endpoint_box_gradients_against_finite_difference():
    args = [tensor(x, True) for x in ([0.1, 4.1, 0.23], [4.2, 0.2, 0.37], [0, 0, -1], [1, 1, 1])]
    assert torch.autograd.gradcheck(geometry.segment_box_distance, tuple(args), atol=1e-5)


def test_yaw_box_pose_gradients_and_rigid_invariance():
    start, end, radius = tensor([0.1, 4.1, 0.23]), tensor([4.2, 0.2, 0.37]), tensor(0.03)
    center, half, yaw = (
        tensor([0.1, -0.2, 0.05], True),
        tensor([1, 0.7, 1], True),
        tensor(0.23, True),
    )

    def query(c, h, y):
        return geometry.capsule_yaw_box_clearance(start, end, radius, c, h, y)

    assert torch.autograd.gradcheck(query, (center, half, yaw), atol=1e-5)
    expected = query(center, half, yaw)
    angle, shift = 0.61, tensor([8, -9, 2])
    rotation = tensor(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    actual = geometry.capsule_yaw_box_clearance(
        start @ rotation.T + shift,
        end @ rotation.T + shift,
        radius,
        center @ rotation.T + shift,
        half,
        yaw + angle,
    )
    assert actual.item() == pytest.approx(expected.item(), abs=1e-12)


def test_invalid_bounds_radii_and_nonfinite_rejected():
    with pytest.raises(ValueError):
        geometry.segment_box_distance(
            tensor([0, 0, 0]), tensor([1, 1, 1]), tensor([1, 0, 0]), tensor([0, 1, 1])
        )
    with pytest.raises(ValueError):
        geometry.capsule_box_clearance(
            tensor([0, 0, 0]), tensor([1, 1, 1]), tensor(-0.1), tensor([0, 0, 0]), tensor([1, 1, 1])
        )
    with pytest.raises(ValueError):
        geometry.segment_box_distance(
            tensor([float("nan"), 0, 0]), tensor([1, 1, 1]), tensor([0, 0, 0]), tensor([1, 1, 1])
        )
