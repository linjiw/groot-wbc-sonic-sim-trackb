from pathlib import Path
import sys

import numpy as np

from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
from gear_sonic.dataset_generation.hallucination.motion2scene_temporal_geometry import (
    clearance_lower_bounds,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/research"))
from motion2scene_fresh_native_temporal import HALF, values  # noqa: E402


def test_filtered_query_matches_unfiltered_rotated_clearances():
    rng = np.random.default_rng(9081)
    a = rng.uniform(-1, 1, (7, 20, 3))
    b = a + rng.uniform(-0.5, 0.5, a.shape)
    radii = rng.uniform(0.01, 0.2, 20)
    center = np.array([0.1, -0.2, 0.3])
    yaw = 0.37
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    baseline = capsule_box_clearance(
        (a - center) @ rotation, (b - center) @ rotation, radii, -HALF, HALF
    )
    assert (baseline < 0).any() and (baseline > 0.02).any()
    np.testing.assert_allclose(values(a, b, radii, center, rotation), np.minimum(0.02, baseline))


def test_clearance_cap_can_leave_distant_motion_unresolved():
    # A sphere remains almost a metre from the box, while its interval displacement
    # exceeds the positive score cap. Unresolved is not an observed collision.
    a = np.array([[[1.0, 0, 0]], [[1.08, 0, 0]]])
    exact = capsule_box_clearance(a, a, np.array([0.01]), -HALF, HALF)
    displacement = np.array([[0.04]])
    assert clearance_lower_bounds(exact, displacement).min() > 0.8
    assert clearance_lower_bounds(np.minimum(0.02, exact), displacement).min() < 0
