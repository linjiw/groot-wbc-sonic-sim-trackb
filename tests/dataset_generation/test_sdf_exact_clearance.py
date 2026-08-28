"""The reported clearance must be a hard minimum, never the training-time soft minimum.

``SdfChoiceDecoder.clearance`` returns a soft minimum so that gradients reach the nearest few
points instead of a single one. A soft minimum is a weighted average, so it is never below the true
minimum -- which means a scene can score positive on it while a point of the robot is physically
inside an obstacle. Every reported number must use ``exact_clearance``.
"""

import numpy as np
import torch

from gear_sonic.dataset_generation.hallucination.sdf_decoder import SdfChoiceDecoder


def _box(centre_z: float, half_vertical: float = 0.15) -> dict[str, torch.Tensor]:
    return {
        "centre_x": torch.tensor([0.0]),
        "centre_y": torch.tensor([0.0]),
        "centre_z": torch.tensor([centre_z]),
        "half_along_m": torch.tensor([0.10]),
        "half_lateral_m": torch.tensor([0.45]),
        "half_vertical_m": torch.tensor([half_vertical]),
        "yaw": torch.tensor([0.0]),
    }


def test_soft_minimum_is_never_below_the_hard_minimum():
    decoder = SdfChoiceDecoder()
    rng = np.random.default_rng(11)
    for _ in range(25):
        points = torch.tensor(rng.uniform(-1.0, 2.0, size=(200, 3)), dtype=torch.float32)
        radii = torch.full((200,), 0.05)
        box = _box(float(rng.uniform(0.8, 1.8)))
        soft = decoder.clearance(points, radii, box)
        hard = decoder.exact_clearance(points, radii, box)
        assert torch.all(hard <= soft + 1e-6)


def test_a_penetrating_cloud_can_still_score_positive_on_the_soft_minimum():
    """The defect this metric split fixes, exhibited rather than asserted in prose.

    The realistic shape of the failure is not a deep intrusion -- that dominates the soft minimum
    too. It is a *grazing* contact: one point a couple of millimetres inside the box, inside a
    cloud of hundreds that are comfortably clear. The clear crowd carries enough total softmax
    weight to pull the average positive, so the scene is reported clear while the robot is touching
    the obstacle. A real body cloud has exactly this shape.
    """
    decoder = SdfChoiceDecoder()
    grazing = torch.tensor([[0.0, 0.0, 1.172]])  # 2 mm inside the underside, after the radius
    clear_crowd = torch.tensor([[0.0, 0.0, 1.12] for _ in range(1200)])
    points = torch.cat((grazing, clear_crowd), dim=0)
    radii = torch.full((points.shape[0],), 0.02)
    box = _box(1.34, half_vertical=0.15)  # underside at z = 1.19
    soft = float(decoder.clearance(points, radii, box).min())
    hard = float(decoder.exact_clearance(points, radii, box).min())
    assert hard < 0.0, "the grazing point is inside the box"
    assert soft > 0.0, "the soft minimum calls a penetrating cloud clear -- hence the split"
    assert soft - hard > 0.02, "and the disagreement is tens of millimetres, not rounding"


def test_hard_minimum_matches_a_direct_computation_for_an_axis_aligned_box():
    decoder = SdfChoiceDecoder()
    points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]])
    radii = torch.tensor([0.10, 0.10])
    box = _box(1.0, half_vertical=0.20)  # spans z in [0.80, 1.20]
    hard = float(decoder.exact_clearance(points, radii, box).min())
    # Nearest point is the one at z = 0.5: gap to the underside is 0.30, minus the 0.10 radius.
    assert abs(hard - 0.20) < 1e-6


def test_clearance_distinguishes_a_wall_across_the_path_from_one_in_the_sky():
    """The original decoder scored these identically; this is the regression that motivated it."""
    decoder = SdfChoiceDecoder()
    walk = torch.tensor([[float(x) * 0.05, 0.0, 1.0] for x in range(40)])
    radii = torch.full((walk.shape[0],), 0.10)
    across = decoder.exact_clearance(walk, radii, _box(1.0)).min()
    sky = decoder.exact_clearance(walk, radii, _box(10.0)).min()
    assert float(across) < 0.0 < float(sky)
    assert float(sky) > 8.0
