"""Tests for multi-obstacle LfLH: the decoder must decide, and the losses must oppose collapse."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gear_sonic.dataset_generation.hallucination.lflh import (
    ChoiceDecoder,
    MultiObstacleHallucinator,
    ObstacleGeometry,
    sample_scenes,
    train,
)

STATIONS = 12
LABELS = ("nominal", "crouch", "tuck_left", "tuck_right")
COSTS = np.asarray([0.0, 1.0, 0.6, 0.6], dtype=np.float64)


def _candidate(up: float, left: float, right: float) -> np.ndarray:
    return np.stack(
        (np.full(STATIONS, up), np.full(STATIONS, left), np.full(STATIONS, right)), axis=0
    )


def _extents() -> np.ndarray:
    return np.stack(
        (
            _candidate(1.30, 0.35, 0.35),
            _candidate(1.22, 0.35, 0.35),
            _candidate(1.30, 0.22, 0.35),
            _candidate(1.30, 0.35, 0.22),
        )
    )


def _box(station, lateral, height, *, half_along=0.6, half_lateral=0.10, half_vertical=0.06):
    values = dict(
        station=station,
        lateral_m=lateral,
        height_m=height,
        half_along_m=half_along,
        half_lateral_m=half_lateral,
        half_vertical_m=half_vertical,
    )
    return {key: torch.tensor([value], dtype=torch.float32) for key, value in values.items()}


def _choose(box) -> str:
    weights = ChoiceDecoder()(
        torch.tensor(_extents(), dtype=torch.float32), box, torch.tensor(COSTS, dtype=torch.float32)
    )
    return LABELS[int(weights.argmax())]


def test_an_empty_scene_selects_the_nominal():
    """With nothing in the way the cheapest candidate must win, or nothing is being decided."""
    assert _choose(_box(6.0, 0.0, 0.20, half_vertical=0.05)) == "nominal"


def test_a_wide_overhead_bar_selects_the_crouch():
    assert _choose(_box(6.0, 0.0, 1.32, half_lateral=1.0)) == "crouch"


def test_a_left_obstacle_selects_the_left_tuck_and_a_right_one_the_right_tuck():
    """The directional result: which side the obstacle is on decides which edit is preferred."""
    assert _choose(_box(6.0, 0.38, 0.95, half_lateral=0.10, half_vertical=0.45)) == "tuck_left"
    assert _choose(_box(6.0, -0.38, 0.95, half_lateral=0.10, half_vertical=0.45)) == "tuck_right"


def test_a_side_obstacle_is_not_scored_as_overhead():
    """The bug that made every obstacle block every candidate.

    A box at lateral 0.38 with half-width 0.10 spans [0.28, 0.48] and never sits above the head,
    so it must not contribute overhead depth however tall it is.
    """
    decoder = ChoiceDecoder()
    extents = torch.tensor(_extents(), dtype=torch.float32)
    side = decoder.penetration(
        extents, _box(6.0, 0.38, 0.95, half_lateral=0.10, half_vertical=0.45)
    )
    # The crouch has the same lateral extent as the nominal, so a side obstacle blocks it equally.
    assert side[1] == pytest.approx(float(side[0]), rel=1e-3)
    # The left tuck escapes it; the right tuck does not.
    assert side[2] < 0.5 * float(side[0])
    assert side[3] == pytest.approx(float(side[0]), rel=1e-3)


def test_an_obstacle_binds_only_where_the_body_is_tall():
    """Station support must be local, or every obstacle binds everywhere on the route.

    The body is given a bulge at station 6 and is short elsewhere, so an overhead box at station 6
    must block far more than the same box at station 0.
    """
    decoder = ChoiceDecoder()
    profile = np.full(STATIONS, 0.95)
    profile[5:8] = 1.30
    tall = np.stack((profile, np.full(STATIONS, 0.35), np.full(STATIONS, 0.35)), axis=0)
    extents = torch.tensor(np.stack((tall,)), dtype=torch.float32)
    near = decoder.penetration(extents, _box(6.0, 0.0, 1.32, half_lateral=1.0, half_along=0.3))
    far = decoder.penetration(extents, _box(0.0, 0.0, 1.32, half_lateral=1.0, half_along=0.3))
    assert float(far[0]) < 0.5 * float(near[0])


def test_the_decoder_has_no_learnable_parameters_and_no_view_of_the_answer():
    """Anti-collusion, and the retracted surrogate's other failure: it knew the observed index."""
    decoder = ChoiceDecoder()
    assert not isinstance(decoder, torch.nn.Module)
    signature = ChoiceDecoder.__call__.__annotations__
    assert "observed" not in signature and "target" not in signature


def test_the_geometry_anchors_are_not_derived_from_a_feasibility_answer():
    """The retracted version anchored its coordinate on the window's lower edge."""
    geometry = ObstacleGeometry(stations=STATIONS)
    latent = torch.zeros((3, 6))
    decoded = geometry.decode(latent)
    assert decoded["height_m"].tolist() == pytest.approx([1.30, 1.30, 1.30])
    assert decoded["lateral_m"].abs().max().item() == pytest.approx(0.0)
    for key in ("half_along_m", "half_lateral_m", "half_vertical_m"):
        assert (decoded[key] >= geometry.min_half_extent_m - 1e-9).all()
        assert (decoded[key] <= geometry.max_half_extent_m + 1e-9).all()


def test_the_variance_floor_is_permissive_enough_not_to_be_the_answer():
    """The retracted headline was exactly its clamp floor; this one cannot be."""
    model = MultiObstacleHallucinator(STATIONS)
    assert model.min_log_sigma <= -20.0


@pytest.mark.parametrize("observed", (1, 2, 3), ids=("crouch", "tuck_left", "tuck_right"))
def test_training_places_obstacles_that_make_the_observed_motion_preferred(observed):
    """The capability the whole method rests on.

    Given one observed edit, the hallucinator must find obstacles under which that edit -- and not
    the cheaper nominal, and not the other edits -- is what the decoder selects. Run per direction,
    because succeeding only on the crouch would mean the scene cannot express sidedness.

    Whether one model can do this for *many* clips at once is a different and harder question
    (does the encoder condition on its input at all); `train_lflh.py` measures that with an
    input-ablation control rather than asserting it here.
    """
    extents = _extents()
    model, report = train(
        [extents], [COSTS], [observed], obstacles=3, steps=260, samples=3, seed=0, kl_weight=0.004
    )
    assert report.reconstruction < 0.7, "the observed motion should usually win"
    scenes = sample_scenes(model, extents, COSTS, observed, "m", count=60, seed=observed)
    assert scenes.reconstruction_rate > 0.6, (
        f"sampled scenes should mostly select {LABELS[observed]}, "
        f"got {scenes.reconstruction_rate:.2f}"
    )


def test_the_kl_term_carries_the_log_sigma_that_opposes_collapse():
    """The term the retracted version omitted.

    A Gaussian KL to a unit prior contains ``-log sigma``, which diverges as sigma falls, so the
    objective can no longer drive the variance to zero for free. Asserted on the term itself rather
    than on a training run, because the direction it pushes depends on where sigma starts.
    """
    from gear_sonic.dataset_generation.hallucination.lflh import _kl

    mean = torch.zeros((1, 3, 6))
    small = _kl(mean, torch.full((1, 3, 6), -8.0))
    large = _kl(mean, torch.full((1, 3, 6), -2.0))
    assert float(small) > float(large), "shrinking sigma must cost more, not less"
    assert float(small) > 5.0
