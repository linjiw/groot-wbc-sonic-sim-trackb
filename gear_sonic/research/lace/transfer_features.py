"""Representation-matched features for LACE source-to-target transfer.

The causal treatment in RQ1 is a signed change in motion exposure, not a
source-cluster label.  These helpers therefore summarize the complete
``delta_p = p_plus - p_base`` vector, including the diffuse exposure removed
from motions outside the upweighted panel.  Every candidate representation
uses the same ordered ``K``-dimensional construction.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from gear_sonic.research.lace.interventions import (
    INTERVENTION_KIND,
    INTERVENTION_SCHEMA_VERSION,
)
from gear_sonic.research.lace.schema import canonical_sha256

FloatArray = NDArray[np.float64]

TRANSFER_FEATURE_KIND = "lace_signed_exposure_transfer_features"
TRANSFER_FEATURE_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identifier(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, name: str) -> str:
    result = _identifier(value, name)
    _require(len(result) == 64, f"{name} must be a lowercase SHA-256")
    try:
        int(result, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    _require(result == result.lower(), f"{name} must be a lowercase SHA-256")
    return result


def _tolerance(value: Any) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, bool),
        "probability_tolerance must be a positive finite scalar",
    )
    result = float(value)
    _require(
        math.isfinite(result) and result > 0.0,
        "probability_tolerance must be a positive finite scalar",
    )
    return result


def _probabilities(values: Any, name: str, size: int, tolerance: float) -> FloatArray:
    raw = np.asarray(values)
    _require(raw.ndim == 1 and raw.shape == (size,), f"{name} must have shape [{size}]")
    _require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        f"{name} must contain real numeric probabilities",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(
        np.all(np.isfinite(result)) and np.all(result >= 0.0),
        f"{name} must contain finite nonnegative probabilities",
    )
    _require(
        abs(math.fsum(float(value) for value in result) - 1.0) <= tolerance,
        f"{name} must sum to one within probability_tolerance",
    )
    return result


def _signed_shift(values: Any, size: int, tolerance: float) -> FloatArray:
    raw = np.asarray(values)
    _require(raw.ndim == 1 and raw.shape == (size,), f"delta_p must have shape [{size}]")
    _require(
        np.issubdtype(raw.dtype, np.number)
        and not np.issubdtype(raw.dtype, np.bool_)
        and not np.issubdtype(raw.dtype, np.complexfloating),
        "delta_p must contain real numeric values",
    )
    result = np.asarray(raw, dtype=np.float64)
    _require(np.all(np.isfinite(result)), "delta_p must contain finite values")
    _require(
        abs(math.fsum(float(value) for value in result)) <= tolerance,
        "delta_p must sum to zero within probability_tolerance",
    )
    return result


def _membership_matrix(
    assignments: Mapping[str, Sequence[float]],
    *,
    ordered_keys: Sequence[str],
    dimension: int | None,
    name: str,
    tolerance: float,
) -> tuple[FloatArray, int]:
    _require(isinstance(assignments, Mapping), f"{name} must be a mapping")
    _require(
        set(assignments) == set(ordered_keys),
        f"{name} keys must exactly match the declared ordered keys",
    )
    rows: list[FloatArray] = []
    expected_dimension = dimension
    for key in ordered_keys:
        raw = np.asarray(assignments[key])
        _require(raw.ndim == 1 and raw.size > 0, f"{name}[{key!r}] must be one-dimensional")
        if expected_dimension is None:
            expected_dimension = int(raw.size)
        _require(
            raw.shape == (expected_dimension,),
            f"{name}[{key!r}] must have shape [{expected_dimension}]",
        )
        rows.append(_probabilities(raw, f"{name}[{key!r}]", expected_dimension, tolerance))
    assert expected_dimension is not None
    _require(expected_dimension >= 2, "representation dimension K must be at least two")
    return np.stack(rows), expected_dimension


def _validate_intervention(
    intervention: Mapping[str, Any],
    *,
    motion_count: int,
    tolerance: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, str]:
    _require(isinstance(intervention, Mapping), "intervention must be a mapping")
    _require(intervention.get("kind") == INTERVENTION_KIND, "intervention kind mismatch")
    _require(
        intervention.get("schema_version") == INTERVENTION_SCHEMA_VERSION,
        "intervention schema version mismatch",
    )
    digest = _sha256(intervention.get("intervention_sha256"), "intervention_sha256")
    _require(
        digest == canonical_sha256(intervention, digest_field="intervention_sha256"),
        "intervention_sha256 mismatch",
    )
    _require(intervention.get("num_bins") == motion_count, "intervention num_bins mismatch")
    base = _probabilities(
        intervention.get("p_base"), "intervention.p_base", motion_count, tolerance
    )
    panel = _probabilities(intervention.get("r_s"), "intervention.r_s", motion_count, tolerance)
    plus = _probabilities(
        intervention.get("p_plus"), "intervention.p_plus", motion_count, tolerance
    )
    delta = _signed_shift(intervention.get("delta_p"), motion_count, tolerance)
    _require(
        np.allclose(plus - base, delta, rtol=0.0, atol=tolerance),
        "intervention delta_p does not equal p_plus - p_base",
    )
    return base, panel, plus, delta, digest


def build_signed_exposure_transfer_features(
    intervention: Mapping[str, Any],
    *,
    panel_id: str,
    motion_keys: Sequence[str],
    source_memberships: Mapping[str, Sequence[float]],
    target_memberships: Mapping[str, Sequence[float]],
    target_source_groups: Mapping[str, str],
    representation_id: str,
    representation_artifact_sha256: str,
    component_names: Sequence[str],
    probability_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Build matched-``K`` transfer features from a complete signed treatment.

    Membership vectors may be soft (failure mechanisms) or one-hot (matched
    semantic, kinematic, feasibility, and seeded-random partitions), but every
    vector must be a probability simplex of the same dimension.  The primary
    cell feature is

    ``delta_centroid[k] * (target[k] - base_centroid[k])``.

    This retains the mechanism-resolved signed treatment while making the sum
    an exposure-alignment statistic.  It cannot mistake an upweighted panel
    centroid for the treatment because ``delta_centroid`` also includes all
    exposure removed elsewhere.
    """

    tolerance = _tolerance(probability_tolerance)
    resolved_panel_id = _identifier(panel_id, "panel_id")
    resolved_representation_id = _identifier(representation_id, "representation_id")
    representation_digest = _sha256(
        representation_artifact_sha256,
        "representation_artifact_sha256",
    )

    _require(
        isinstance(motion_keys, Sequence) and not isinstance(motion_keys, (str, bytes)),
        "motion_keys must be a non-empty ordered sequence",
    )
    ordered_motion_keys = list(motion_keys)
    _require(
        bool(ordered_motion_keys)
        and all(isinstance(key, str) and key for key in ordered_motion_keys),
        "motion_keys must contain non-empty strings",
    )
    _require(
        len(ordered_motion_keys) == len(set(ordered_motion_keys)),
        "motion_keys must be unique",
    )
    base, panel, plus, delta, intervention_digest = _validate_intervention(
        intervention,
        motion_count=len(ordered_motion_keys),
        tolerance=tolerance,
    )

    source_matrix, dimension = _membership_matrix(
        source_memberships,
        ordered_keys=ordered_motion_keys,
        dimension=None,
        name="source_memberships",
        tolerance=tolerance,
    )
    _require(
        isinstance(target_memberships, Mapping) and bool(target_memberships),
        "target_memberships must be a non-empty mapping",
    )
    ordered_target_ids = sorted(target_memberships)
    _require(
        all(isinstance(target_id, str) and target_id for target_id in ordered_target_ids),
        "target_memberships keys must be non-empty strings",
    )
    target_matrix, target_dimension = _membership_matrix(
        target_memberships,
        ordered_keys=ordered_target_ids,
        dimension=dimension,
        name="target_memberships",
        tolerance=tolerance,
    )
    _require(target_dimension == dimension, "source and target representation dimensions differ")
    _require(
        isinstance(target_source_groups, Mapping)
        and set(target_source_groups) == set(ordered_target_ids),
        "target_source_groups keys must exactly match target_memberships",
    )
    groups = {
        target_id: _identifier(
            target_source_groups[target_id],
            f"target_source_groups[{target_id!r}]",
        )
        for target_id in ordered_target_ids
    }

    _require(
        isinstance(component_names, Sequence) and not isinstance(component_names, (str, bytes)),
        "component_names must be an ordered sequence",
    )
    ordered_components = list(component_names)
    _require(
        len(ordered_components) == dimension
        and all(isinstance(name, str) and name for name in ordered_components)
        and len(set(ordered_components)) == dimension,
        f"component_names must contain {dimension} unique non-empty names",
    )

    base_centroid = base @ source_matrix
    panel_centroid = panel @ source_matrix
    plus_centroid = plus @ source_matrix
    delta_centroid = delta @ source_matrix
    _require(
        np.allclose(plus_centroid - base_centroid, delta_centroid, rtol=0.0, atol=tolerance),
        "signed representation shift is inconsistent with the intervention",
    )
    _require(
        abs(float(delta_centroid.sum())) <= tolerance,
        "simplex memberships and zero-sum delta_p must yield a zero-sum centroid shift",
    )

    feature_names = [
        f"{resolved_representation_id}:signed_exposure_interaction:{name}"
        for name in ordered_components
    ]
    targets: list[dict[str, Any]] = []
    for target_id, target, group_id in zip(
        ordered_target_ids,
        target_matrix,
        (groups[target_id] for target_id in ordered_target_ids),
        strict=True,
    ):
        centered_target = target - base_centroid
        interactions = delta_centroid * centered_target
        base_distance = float(np.sum(np.square(base_centroid - target)))
        plus_distance = float(np.sum(np.square(plus_centroid - target)))
        panel_distance = float(np.sum(np.square(panel_centroid - target)))
        targets.append(
            {
                "target_motion_id": target_id,
                "target_source_group_id": group_id,
                "target_membership": [float(value) for value in target],
                "centered_target_membership": [float(value) for value in centered_target],
                "feature_vector": [float(value) for value in interactions],
                "signed_exposure_alignment": float(np.sum(interactions)),
                "base_to_target_squared_l2": base_distance,
                "panel_to_target_squared_l2": panel_distance,
                "intervention_to_target_squared_l2": plus_distance,
                "signed_squared_l2_change": plus_distance - base_distance,
            }
        )

    result: dict[str, Any] = {
        "kind": TRANSFER_FEATURE_KIND,
        "schema_version": TRANSFER_FEATURE_SCHEMA_VERSION,
        "panel_id": resolved_panel_id,
        "intervention_sha256": intervention_digest,
        "representation_id": resolved_representation_id,
        "representation_artifact_sha256": representation_digest,
        "motion_order": ordered_motion_keys,
        "target_motion_order": ordered_target_ids,
        "representation_dimension": dimension,
        "component_names": ordered_components,
        "feature_names": feature_names,
        "feature_rule": (
            "delta_membership_centroid[k] * " "(target_membership[k] - base_membership_centroid[k])"
        ),
        "matched_k_required_across_candidate_representations": True,
        "base_membership_centroid": [float(value) for value in base_centroid],
        "panel_membership_centroid": [float(value) for value in panel_centroid],
        "intervention_membership_centroid": [float(value) for value in plus_centroid],
        "delta_membership_centroid": [float(value) for value in delta_centroid],
        "signed_delta_sum": float(math.fsum(float(value) for value in delta)),
        "removed_exposure_is_included": True,
        "probability_tolerance": tolerance,
        "targets": targets,
    }
    result["transfer_feature_sha256"] = canonical_sha256(result)
    return result
