from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from gear_sonic.research.lace.normalizer import (
    NORMALIZER_METHOD,
    fit_d_atlas_normalizer,
)
from gear_sonic.research.lace.schema import canonical_sha256

MECHANISMS = ("contact", "slip")


def _episode(
    *,
    failed: bool,
    contact: float,
    slip: float,
    partition: str = "D_atlas",
    index: int = 0,
) -> dict[str, Any]:
    return {
        "rollout_id": f"rollout-{index}",
        "motion_key": f"motion-{index}",
        "partition": partition,
        "failed": failed,
        "mechanism_scores": {"contact": contact, "slip": slip},
    }


def _calibration_episodes() -> list[dict[str, Any]]:
    episodes = [
        _episode(failed=True, contact=float(index), slip=float(2 * index), index=index)
        for index in range(1, 6)
    ]
    # Successful positive scores are valid, but must not calibrate either scale.
    episodes.append(_episode(failed=False, contact=1000.0, slip=2000.0, index=6))
    return episodes


def test_fits_positive_failed_d_atlas_q90_with_frozen_provenance() -> None:
    episodes = _calibration_episodes()

    normalizer = fit_d_atlas_normalizer(
        episodes,
        mechanism_names=MECHANISMS,
        minimum_positive_observations=5,
    )

    assert normalizer["frozen"] is True
    assert normalizer["method"] == NORMALIZER_METHOD
    assert normalizer["quantile"] == 0.9
    assert normalizer["quantile_method"] == "linear"
    assert normalizer["fit_partition"] == "D_atlas"
    assert normalizer["mechanism_names"] == ["contact", "slip"]
    assert normalizer["mechanism_scales"] == pytest.approx({"contact": 4.6, "slip": 9.2})
    assert normalizer["positive_observation_counts"] == {"contact": 5, "slip": 5}
    assert normalizer["minimum_positive_observations"] == {"contact": 5, "slip": 5}
    assert normalizer["episode_count"] == 6
    assert normalizer["failure_count"] == 5
    assert normalizer["input_sha256"] == canonical_sha256(
        {
            "fit_partition": "D_atlas",
            "mechanism_names": list(MECHANISMS),
            "episodes": episodes,
        }
    )


def test_per_mechanism_minimums_are_exact_and_configurable() -> None:
    episodes = _calibration_episodes()
    episodes[0]["mechanism_scores"]["slip"] = 0.0
    episodes[1]["mechanism_scores"]["slip"] = 0.0

    normalizer = fit_d_atlas_normalizer(
        (episode for episode in episodes),
        mechanism_names=MECHANISMS,
        minimum_positive_observations={"contact": 5, "slip": 3},
    )

    assert normalizer["positive_observation_counts"] == {"contact": 5, "slip": 3}
    assert normalizer["mechanism_scales"]["slip"] == pytest.approx(9.6)


def test_does_not_mutate_inputs_and_digest_binds_successful_episodes() -> None:
    episodes = _calibration_episodes()
    original = deepcopy(episodes)

    first = fit_d_atlas_normalizer(
        episodes,
        mechanism_names=MECHANISMS,
        minimum_positive_observations=5,
    )
    assert episodes == original
    episodes[-1]["mechanism_scores"]["contact"] = 9999.0
    second = fit_d_atlas_normalizer(
        episodes,
        mechanism_names=MECHANISMS,
        minimum_positive_observations=5,
    )

    assert original[-1]["mechanism_scores"]["contact"] == 1000.0
    assert first["mechanism_scales"] == second["mechanism_scales"]
    assert first["input_sha256"] != second["input_sha256"]


@pytest.mark.parametrize("partition", ["D_geometry", "D_test", "", None])
def test_rejects_mixed_or_unlabelled_partitions(partition: object) -> None:
    episode = _episode(failed=True, contact=1.0, slip=1.0)
    episode["partition"] = partition

    with pytest.raises(ValueError, match="partition must be 'D_atlas'"):
        fit_d_atlas_normalizer(
            [episode],
            mechanism_names=MECHANISMS,
            minimum_positive_observations=1,
        )


@pytest.mark.parametrize("failed", [0, 1, "false", None])
def test_failed_flag_must_be_a_strict_boolean(failed: object) -> None:
    episode = _episode(failed=True, contact=1.0, slip=1.0)
    episode["failed"] = failed

    with pytest.raises(ValueError, match="failed must be bool"):
        fit_d_atlas_normalizer([episode], mechanism_names=MECHANISMS)


def test_mechanism_channels_must_match_exactly() -> None:
    missing = _episode(failed=True, contact=1.0, slip=1.0)
    del missing["mechanism_scores"]["slip"]
    with pytest.raises(ValueError, match=r"channel mismatch.*missing=\['slip'\]"):
        fit_d_atlas_normalizer([missing], mechanism_names=MECHANISMS)

    unexpected = _episode(failed=True, contact=1.0, slip=1.0)
    unexpected["mechanism_scores"]["drift"] = 0.5
    with pytest.raises(ValueError, match=r"channel mismatch.*unexpected=\['drift'\]"):
        fit_d_atlas_normalizer([unexpected], mechanism_names=MECHANISMS)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), "1.0", True])
def test_scores_must_be_numeric_finite_and_nonnegative(value: object) -> None:
    episode = _episode(failed=True, contact=1.0, slip=1.0)
    episode["mechanism_scores"]["contact"] = value

    with pytest.raises(ValueError, match=r"mechanism_scores\['contact'\].*must be"):
        fit_d_atlas_normalizer([episode], mechanism_names=MECHANISMS)


def test_no_failures_or_no_positive_failure_evidence_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least one failed D_atlas episode"):
        fit_d_atlas_normalizer(
            [_episode(failed=False, contact=5.0, slip=5.0)],
            mechanism_names=MECHANISMS,
        )

    with pytest.raises(ValueError, match="no positive mechanism evidence"):
        fit_d_atlas_normalizer(
            [_episode(failed=True, contact=0.0, slip=0.0)],
            mechanism_names=MECHANISMS,
        )


def test_each_mechanism_must_meet_its_positive_evidence_floor() -> None:
    episodes = [
        _episode(failed=True, contact=1.0, slip=0.0, index=1),
        _episode(failed=True, contact=2.0, slip=1.0, index=2),
    ]

    with pytest.raises(
        ValueError,
        match=r"insufficient positive mechanism evidence.*slip.*observed.*1.*required.*2",
    ):
        fit_d_atlas_normalizer(
            episodes,
            mechanism_names=MECHANISMS,
            minimum_positive_observations=2,
        )


def test_scientific_default_requires_twenty_positive_observations_per_channel() -> None:
    with pytest.raises(ValueError, match=r"required.*20"):
        fit_d_atlas_normalizer(_calibration_episodes(), mechanism_names=MECHANISMS)


@pytest.mark.parametrize(
    "minimum",
    [
        0,
        True,
        {"contact": 1},
        {"contact": 1, "slip": 1, "drift": 1},
        {"contact": 1, "slip": 0},
    ],
)
def test_invalid_positive_evidence_configuration_is_rejected(minimum: object) -> None:
    with pytest.raises(ValueError, match="minimum_positive_observations"):
        fit_d_atlas_normalizer(
            _calibration_episodes(),
            mechanism_names=MECHANISMS,
            minimum_positive_observations=minimum,
        )


def test_fitter_is_q90_only_and_requires_json_hashable_provenance() -> None:
    with pytest.raises(ValueError, match="q90-only"):
        fit_d_atlas_normalizer(
            _calibration_episodes(),
            mechanism_names=MECHANISMS,
            quantile=0.5,
        )

    episode = _episode(failed=True, contact=1.0, slip=1.0)
    episode["non_json_metadata"] = {1, 2, 3}
    with pytest.raises(ValueError, match="canonical-JSON serializable"):
        fit_d_atlas_normalizer(
            [episode],
            mechanism_names=MECHANISMS,
            minimum_positive_observations=1,
        )
