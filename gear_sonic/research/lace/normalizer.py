"""Fit frozen LACE mechanism scales without importing simulator dependencies.

The fitter is intentionally narrow: its input must already be restricted and
labelled as ``D_atlas``.  It never guesses partition membership, never treats a
successful rollout as calibration evidence, and never replaces an unsupported
channel with an identity scale.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

from gear_sonic.research.lace.schema import (
    SCIENTIFIC_NORMALIZER_METHOD,
    SCIENTIFIC_NORMALIZER_MINIMUM_POSITIVE_OBSERVATIONS,
    canonical_sha256,
)
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

FIT_PARTITION = "D_atlas"
Q90 = 0.90
QUANTILE_METHOD = "linear"
NORMALIZER_METHOD = SCIENTIFIC_NORMALIZER_METHOD
DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS = SCIENTIFIC_NORMALIZER_MINIMUM_POSITIVE_OBSERVATIONS


def _validate_mechanism_names(mechanism_names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(mechanism_names, (str, bytes)):
        raise ValueError("mechanism_names must be a sequence of channel names")
    names = tuple(mechanism_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("mechanism_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("mechanism_names must be unique")
    return names


def _validate_q90(quantile: float) -> float:
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise ValueError("quantile must be the numeric q90 value 0.9")
    numeric = float(quantile)
    if not math.isfinite(numeric) or not math.isclose(
        numeric,
        Q90,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("quantile must be 0.9; this fitter is intentionally q90-only")
    return Q90


def _positive_minimums(
    value: int | Mapping[str, int],
    mechanism_names: Sequence[str],
) -> dict[str, int]:
    if isinstance(value, Integral) and not isinstance(value, bool):
        minimum = int(value)
        if minimum < 1:
            raise ValueError("minimum_positive_observations must be at least one")
        return {name: minimum for name in mechanism_names}

    if not isinstance(value, Mapping):
        raise ValueError(
            "minimum_positive_observations must be an integer or per-mechanism mapping"
        )
    missing = [name for name in mechanism_names if name not in value]
    unexpected = [name for name in value if name not in mechanism_names]
    if missing or unexpected:
        raise ValueError(
            "minimum_positive_observations must exactly match mechanism_names; "
            f"missing={missing}, unexpected={unexpected}"
        )

    result: dict[str, int] = {}
    for name in mechanism_names:
        minimum = value[name]
        if isinstance(minimum, bool) or not isinstance(minimum, Integral) or int(minimum) < 1:
            raise ValueError(f"minimum_positive_observations[{name!r}] must be a positive integer")
        result[name] = int(minimum)
    return result


def _linear_quantile(values: Sequence[float], quantile: float) -> float:
    """Return NumPy-compatible linear quantiles using only the standard library."""

    if not values:
        raise ValueError("cannot compute a quantile without positive observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(ordered[lower_index])
    weight = position - lower_index
    return float(ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight)


def _score(value: Any, *, episode_index: int, mechanism_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"episodes[{episode_index}].mechanism_scores[{mechanism_name!r}] " "must be numeric"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"episodes[{episode_index}].mechanism_scores[{mechanism_name!r}] "
            "must be finite and nonnegative"
        )
    return numeric


def normalizer_input_sha256(
    episodes: Sequence[Mapping[str, Any]],
    mechanism_names: Sequence[str],
) -> str:
    """Bind the exact ordered calibration episodes and channel contract."""

    payload = {
        "fit_partition": FIT_PARTITION,
        "mechanism_names": list(mechanism_names),
        "episodes": [dict(episode) for episode in episodes],
    }
    try:
        return canonical_sha256(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "normalizer inputs must be canonical-JSON serializable without NaN or infinity"
        ) from exc


def fit_d_atlas_normalizer(
    episodes: Iterable[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str] = DEFAULT_MECHANISMS,
    minimum_positive_observations: int | Mapping[str, int] = DEFAULT_MINIMUM_POSITIVE_OBSERVATIONS,
    quantile: float = Q90,
) -> dict[str, Any]:
    """Fit per-mechanism q90 scales from positive failed ``D_atlas`` scores.

    Every episode must contain ``partition``, a strict boolean ``failed``, and
    ``mechanism_scores`` with exactly ``mechanism_names``.  Successful episodes
    are validated and committed to provenance, but do not contribute values or
    counts.  Zero scores are valid observations but are excluded from the
    positive-score calibration distribution.

    ``minimum_positive_observations`` may be one positive integer shared by all
    mechanisms or an exact per-mechanism mapping.  The function raises rather
    than emitting a fallback scale when any channel lacks sufficient evidence.
    """

    names = _validate_mechanism_names(mechanism_names)
    q90 = _validate_q90(quantile)
    minimums = _positive_minimums(minimum_positive_observations, names)

    try:
        materialized = list(episodes)
    except TypeError as exc:
        raise ValueError("episodes must be an iterable of mappings") from exc
    if not materialized:
        raise ValueError("episodes must be non-empty")

    validated_episodes: list[Mapping[str, Any]] = []
    positive_values = {name: [] for name in names}
    failure_count = 0

    for index, episode in enumerate(materialized):
        if not isinstance(episode, Mapping):
            raise ValueError(f"episodes[{index}] must be a mapping")
        if episode.get("partition") != FIT_PARTITION:
            raise ValueError(
                f"episodes[{index}].partition must be {FIT_PARTITION!r}; "
                "the D_atlas fitter refuses mixed or unlabelled partitions"
            )
        failed = episode.get("failed")
        if not isinstance(failed, bool):
            raise ValueError(f"episodes[{index}].failed must be bool")
        raw_scores = episode.get("mechanism_scores")
        if not isinstance(raw_scores, Mapping):
            raise ValueError(f"episodes[{index}].mechanism_scores must be a mapping")
        missing = [name for name in names if name not in raw_scores]
        unexpected = [name for name in raw_scores if name not in names]
        if missing or unexpected:
            raise ValueError(
                f"episodes[{index}].mechanism_scores channel mismatch; "
                f"missing={missing}, unexpected={unexpected}"
            )

        scores = {
            name: _score(raw_scores[name], episode_index=index, mechanism_name=name)
            for name in names
        }
        if failed:
            failure_count += 1
            for name, value in scores.items():
                if value > 0.0:
                    positive_values[name].append(value)
        validated_episodes.append(episode)

    if failure_count == 0:
        raise ValueError("normalizer requires at least one failed D_atlas episode")
    positive_counts = {name: len(positive_values[name]) for name in names}
    if sum(positive_counts.values()) == 0:
        raise ValueError("failed D_atlas episodes contain no positive mechanism evidence")
    insufficient = {
        name: {"observed": positive_counts[name], "required": minimums[name]}
        for name in names
        if positive_counts[name] < minimums[name]
    }
    if insufficient:
        raise ValueError(f"insufficient positive mechanism evidence: {insufficient}")

    scales = {name: _linear_quantile(positive_values[name], q90) for name in names}
    for name, scale in scales.items():
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"computed mechanism scale for {name!r} is not finite and positive")

    return {
        "schema_version": 1,
        "frozen": True,
        "method": NORMALIZER_METHOD,
        "quantile": q90,
        "quantile_method": QUANTILE_METHOD,
        "fit_partition": FIT_PARTITION,
        "mechanism_names": list(names),
        "mechanism_scales": scales,
        "positive_observation_counts": positive_counts,
        "minimum_positive_observations": minimums,
        "episode_count": len(validated_episodes),
        "failure_count": failure_count,
        "input_digest_method": "canonical_json_sha256_v1",
        "input_sha256": normalizer_input_sha256(validated_episodes, names),
    }
