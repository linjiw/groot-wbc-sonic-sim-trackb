"""Report-only cross-policy transport diagnostics for frozen LACE signatures.

Released SONIC versus SONIC-Lite changes architecture, training history,
capacity, and mastery together.  This module therefore measures transport of a
failure coordinate system without presenting the comparison as a causal model-
capacity test.  Missing mechanism signatures stay missing and all uncertainty
resamples source recording groups rather than individual clips.
"""

from __future__ import annotations

from collections import defaultdict
import math
from numbers import Integral, Real
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.signatures import DEFAULT_MECHANISMS

TRANSPORT_PROTOCOL_KIND = "lace_cross_policy_transport_protocol"
TRANSPORT_PROTOCOL_SCHEMA_VERSION = 1
TRANSPORT_PROTOCOL_DIGEST_FIELD = "transport_protocol_sha256"
TRANSPORT_REPORT_KIND = "lace_cross_policy_transport_report"
TRANSPORT_REPORT_SCHEMA_VERSION = 1
TRANSPORT_REPORT_DIGEST_FIELD = "transport_report_sha256"
TRANSPORT_BINDING_FIELDS = {
    "reference_artifact_sha256",
    "trainee_artifact_sha256",
    "parent_normalizer_sha256",
    "measurement_protocol_sha256",
    "condition_grid_sha256",
}

_PROTOCOL_FIELDS = {
    "kind",
    "schema_version",
    "frozen",
    "scientific_use",
    "report_only",
    "causal_capacity_claim_allowed",
    "declared_before_primary_atlas_outcomes",
    "split_selection_sha256",
    "partition",
    "mechanism_names",
    "reference_policy_id",
    "trainee_policy_ids",
    "comparison_rule",
    "normalizer_rule",
    "missingness_rule",
    "rollouts_per_motion_policy",
    "minimum_resolved_failures",
    "neighbor_k",
    "bootstrap_unit",
    "bootstrap_replicates",
    "bootstrap_seed",
    "confidence_level",
    "minimum_common_support_motion_count",
    "minimum_common_support_fraction",
    "minimum_common_support_source_group_count",
    "low_transport_action",
    TRANSPORT_PROTOCOL_DIGEST_FIELD,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    _require(
        isinstance(value, Integral)
        and not isinstance(value, (bool, np.bool_))
        and int(value) >= minimum,
        f"{name} must be an integer >= {minimum}",
    )
    return int(value)


def _fraction(value: Any, name: str, *, include_zero: bool = False) -> float:
    _require(
        isinstance(value, Real) and not isinstance(value, (bool, np.bool_)),
        f"{name} must be real",
    )
    result = float(value)
    lower = result >= 0.0 if include_zero else result > 0.0
    _require(math.isfinite(result) and lower and result <= 1.0, f"{name} must be in (0, 1]")
    return result


def _names(values: Any, name: str) -> tuple[str, ...]:
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{name} must be a sequence",
    )
    result = tuple(values)
    _require(
        bool(result) and all(isinstance(value, str) and value for value in result),
        f"{name} must contain non-empty strings",
    )
    _require(len(result) == len(set(result)), f"{name} must be unique")
    return result


def validate_transport_protocol(
    protocol: Mapping[str, Any],
    *,
    verify_digest: bool = True,
) -> None:
    """Validate the frozen, explicitly non-causal P1b analysis contract."""

    _require(isinstance(protocol, Mapping), "transport protocol must be a mapping")
    _require(set(protocol) == _PROTOCOL_FIELDS, "transport protocol fields are invalid")
    _require(protocol.get("kind") == TRANSPORT_PROTOCOL_KIND, "protocol kind mismatch")
    _require(
        protocol.get("schema_version") == TRANSPORT_PROTOCOL_SCHEMA_VERSION,
        "protocol schema version mismatch",
    )
    _require(
        protocol.get("frozen") is True
        and protocol.get("scientific_use") is True
        and protocol.get("report_only") is True
        and protocol.get("causal_capacity_claim_allowed") is False
        and protocol.get("declared_before_primary_atlas_outcomes") is True,
        "transport protocol scientific/interpretation flags are invalid",
    )
    _sha256(protocol.get("split_selection_sha256"), "split_selection_sha256")
    _require(protocol.get("partition") == "D_atlas", "transport partition must be D_atlas")
    mechanisms = _names(protocol.get("mechanism_names"), "mechanism_names")
    _require(
        mechanisms == tuple(DEFAULT_MECHANISMS),
        "transport mechanisms must use the canonical ordered channels",
    )
    reference = protocol.get("reference_policy_id")
    _require(isinstance(reference, str) and reference, "reference_policy_id missing")
    trainees = _names(protocol.get("trainee_policy_ids"), "trainee_policy_ids")
    _require(reference not in trainees, "reference policy cannot be a trainee policy")
    _require(
        protocol.get("comparison_rule")
        == "reference_vs_each_trainee_stage_separately_no_policy_axis_averaging",
        "comparison_rule drifted",
    )
    _require(
        protocol.get("normalizer_rule")
        == "apply_one_primary_lite_d_atlas_normalizer_to_both_policies_no_refit",
        "normalizer_rule drifted",
    )
    _require(
        protocol.get("missingness_rule") == "complete_pair_support_no_imputation",
        "missingness_rule drifted",
    )
    rollouts = _positive_integer(
        protocol.get("rollouts_per_motion_policy"),
        "rollouts_per_motion_policy",
        minimum=2,
    )
    minimum_resolved = _positive_integer(
        protocol.get("minimum_resolved_failures"),
        "minimum_resolved_failures",
    )
    _require(
        minimum_resolved <= rollouts,
        "minimum_resolved_failures cannot exceed rollouts_per_motion_policy",
    )
    _positive_integer(protocol.get("neighbor_k"), "neighbor_k")
    _require(protocol.get("bootstrap_unit") == "source_group", "bootstrap_unit drifted")
    _positive_integer(protocol.get("bootstrap_replicates"), "bootstrap_replicates", minimum=100)
    _require(
        isinstance(protocol.get("bootstrap_seed"), int)
        and not isinstance(protocol.get("bootstrap_seed"), bool),
        "bootstrap_seed must be an integer",
    )
    _fraction(protocol.get("confidence_level"), "confidence_level")
    _positive_integer(
        protocol.get("minimum_common_support_motion_count"),
        "minimum_common_support_motion_count",
        minimum=3,
    )
    _fraction(
        protocol.get("minimum_common_support_fraction"),
        "minimum_common_support_fraction",
    )
    _positive_integer(
        protocol.get("minimum_common_support_source_group_count"),
        "minimum_common_support_source_group_count",
        minimum=3,
    )
    _require(
        protocol.get("low_transport_action")
        == "continue_trainee_geometry_and_narrow_external_validity_not_an_h1_kill",
        "low_transport_action drifted",
    )
    expected = _sha256(
        protocol.get(TRANSPORT_PROTOCOL_DIGEST_FIELD),
        TRANSPORT_PROTOCOL_DIGEST_FIELD,
    )
    if verify_digest:
        actual = canonical_sha256(
            protocol,
            digest_field=TRANSPORT_PROTOCOL_DIGEST_FIELD,
        )
        _require(expected == actual, "transport protocol digest mismatch")


def _validate_bindings(raw: Mapping[str, Any]) -> dict[str, str]:
    _require(isinstance(raw, Mapping), "artifact_bindings must be a mapping")
    _require(set(raw) == TRANSPORT_BINDING_FIELDS, "artifact_bindings fields are invalid")
    result = {name: _sha256(raw[name], f"artifact_bindings.{name}") for name in sorted(raw)}
    _require(
        result["reference_artifact_sha256"] != result["trainee_artifact_sha256"],
        "reference and trainee artifacts must be distinct",
    )
    return result


def _index_policy_signatures(
    records: Sequence[Mapping[str, Any]],
    *,
    policy_id: str,
    mechanisms: tuple[str, ...],
    expected_rollouts: int,
    minimum_resolved_failures: int,
    label: str,
) -> dict[str, dict[str, Any]]:
    _require(
        isinstance(records, Sequence) and not isinstance(records, (str, bytes)),
        f"{label} must be a sequence",
    )
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"{label}[{index}] must be a mapping")
        if record.get("probe_policy_id") != policy_id:
            continue
        motion_key = record.get("motion_key")
        _require(
            isinstance(motion_key, str) and motion_key,
            f"{label}[{index}].motion_key missing",
        )
        _require(motion_key not in indexed, f"{label} duplicates motion {motion_key!r}")
        num_rollouts = _positive_integer(
            record.get("num_rollouts"),
            f"{label}[{index}].num_rollouts",
        )
        _require(
            num_rollouts == expected_rollouts,
            f"{label}[{index}] rollout count drifted from protocol",
        )
        num_resolved = record.get("num_resolved_failures")
        _require(
            isinstance(num_resolved, int)
            and not isinstance(num_resolved, bool)
            and 0 <= num_resolved <= num_rollouts,
            f"{label}[{index}].num_resolved_failures is invalid",
        )
        q_raw = record.get("q")
        q: tuple[float, ...] | None
        if q_raw is None:
            q = None
            _require(
                num_resolved < minimum_resolved_failures,
                f"{label}[{index}] missing q despite sufficient resolved failures",
            )
        else:
            _require(
                num_resolved >= minimum_resolved_failures,
                f"{label}[{index}] has q without sufficient resolved failures",
            )
            _require(
                isinstance(q_raw, Sequence) and not isinstance(q_raw, (str, bytes)),
                f"{label}[{index}].q must be a sequence or null",
            )
            _require(len(q_raw) == len(mechanisms), f"{label}[{index}].q length mismatch")
            q = tuple(float(value) for value in q_raw)
            _require(
                all(math.isfinite(value) and value >= 0.0 for value in q),
                f"{label}[{index}].q must be finite and nonnegative",
            )
            _require(
                math.isclose(math.fsum(q), 1.0, rel_tol=0.0, abs_tol=1e-9),
                f"{label}[{index}].q must sum to one",
            )
        indexed[motion_key] = {
            "q": q,
            "num_rollouts": num_rollouts,
            "num_resolved_failures": int(num_resolved),
        }
    _require(bool(indexed), f"{label} contains no signatures for policy {policy_id!r}")
    return indexed


def _js(left: np.ndarray, right: np.ndarray) -> float:
    distance = float(jensenshannon(left, right, base=2.0))
    return distance * distance


def _distance_matrix(vectors: np.ndarray) -> np.ndarray:
    count = len(vectors)
    result = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = _js(vectors[left], vectors[right])
            result[left, right] = value
            result[right, left] = value
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size < 3 or np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    result = float(spearmanr(left, right).statistic)
    return result if math.isfinite(result) else None


def _interval(values: Sequence[float], *, level: float) -> list[float | None]:
    if not values:
        return [None, None, None]
    tail = (1.0 - level) / 2.0
    array = np.asarray(values, dtype=np.float64)
    return [
        float(np.quantile(array, tail, method="linear")),
        float(np.quantile(array, 0.5, method="linear")),
        float(np.quantile(array, 1.0 - tail, method="linear")),
    ]


def _nearest(
    distances: np.ndarray,
    motion_keys: tuple[str, ...],
    index: int,
    k: int,
) -> tuple[str, ...]:
    candidates = [candidate for candidate in range(len(motion_keys)) if candidate != index]
    candidates.sort(
        key=lambda candidate: (float(distances[index, candidate]), motion_keys[candidate])
    )
    return tuple(motion_keys[candidate] for candidate in candidates[:k])


def build_cross_policy_transport_report(
    reference_signatures: Sequence[Mapping[str, Any]],
    trainee_signatures: Sequence[Mapping[str, Any]],
    *,
    trainee_policy_id: str,
    source_group_by_motion: Mapping[str, str],
    artifact_bindings: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare release signatures with one Lite stage on complete pair support."""

    validate_transport_protocol(protocol)
    mechanisms = tuple(protocol["mechanism_names"])
    reference_policy_id = str(protocol["reference_policy_id"])
    _require(
        trainee_policy_id in protocol["trainee_policy_ids"],
        "trainee_policy_id is not declared by the protocol",
    )
    bindings = _validate_bindings(artifact_bindings)
    expected_rollouts = int(protocol["rollouts_per_motion_policy"])
    minimum_resolved_failures = int(protocol["minimum_resolved_failures"])
    reference = _index_policy_signatures(
        reference_signatures,
        policy_id=reference_policy_id,
        mechanisms=mechanisms,
        expected_rollouts=expected_rollouts,
        minimum_resolved_failures=minimum_resolved_failures,
        label="reference_signatures",
    )
    trainee = _index_policy_signatures(
        trainee_signatures,
        policy_id=trainee_policy_id,
        mechanisms=mechanisms,
        expected_rollouts=expected_rollouts,
        minimum_resolved_failures=minimum_resolved_failures,
        label="trainee_signatures",
    )
    _require(
        set(reference) == set(trainee),
        "reference and trainee motion universes must exactly match",
    )
    all_motion_keys = tuple(sorted(reference))
    _require(
        set(source_group_by_motion) == set(all_motion_keys),
        "source_group_by_motion must exactly match the motion universe",
    )
    _require(
        all(isinstance(group, str) and group for group in source_group_by_motion.values()),
        "source_group_by_motion values must be non-empty strings",
    )

    common_motion_keys: list[str] = []
    excluded: list[dict[str, Any]] = []
    reference_vectors: list[tuple[float, ...]] = []
    trainee_vectors: list[tuple[float, ...]] = []
    for motion_key in all_motion_keys:
        missing_reference = reference[motion_key]["q"] is None
        missing_trainee = trainee[motion_key]["q"] is None
        if missing_reference or missing_trainee:
            excluded.append(
                {
                    "motion_key": motion_key,
                    "reference_q_missing": missing_reference,
                    "trainee_q_missing": missing_trainee,
                }
            )
            continue
        common_motion_keys.append(motion_key)
        reference_vectors.append(reference[motion_key]["q"])
        trainee_vectors.append(trainee[motion_key]["q"])

    _require(len(common_motion_keys) >= 3, "at least three common-support motions are required")
    common_keys = tuple(common_motion_keys)
    reference_array = np.asarray(reference_vectors, dtype=np.float64)
    trainee_array = np.asarray(trainee_vectors, dtype=np.float64)
    groups_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, motion_key in enumerate(common_keys):
        groups_to_indices[source_group_by_motion[motion_key]].append(index)
    group_names = tuple(sorted(groups_to_indices))
    _require(len(group_names) >= 3, "at least three common-support source groups are required")

    motion_js = np.asarray(
        [_js(reference_array[index], trainee_array[index]) for index in range(len(common_keys))],
        dtype=np.float64,
    )
    reference_distances = _distance_matrix(reference_array)
    trainee_distances = _distance_matrix(trainee_array)
    upper = np.triu_indices(len(common_keys), k=1)
    distance_spearman = _spearman(
        reference_distances[upper],
        trainee_distances[upper],
    )
    channel_points = {
        mechanism: _spearman(reference_array[:, index], trainee_array[:, index])
        for index, mechanism in enumerate(mechanisms)
    }

    effective_k = min(int(protocol["neighbor_k"]), len(common_keys) - 1)
    neighbor_rows: list[dict[str, Any]] = []
    overlaps: list[float] = []
    for index, motion_key in enumerate(common_keys):
        reference_neighbors = _nearest(reference_distances, common_keys, index, effective_k)
        trainee_neighbors = _nearest(trainee_distances, common_keys, index, effective_k)
        overlap = len(set(reference_neighbors) & set(trainee_neighbors)) / effective_k
        overlaps.append(overlap)
        neighbor_rows.append(
            {
                "motion_key": motion_key,
                "reference_neighbors": list(reference_neighbors),
                "trainee_neighbors": list(trainee_neighbors),
                "overlap_fraction": overlap,
            }
        )

    replicates = int(protocol["bootstrap_replicates"])
    rng = np.random.default_rng(int(protocol["bootstrap_seed"]))
    channel_samples: dict[str, list[float]] = {mechanism: [] for mechanism in mechanisms}
    distance_samples: list[float] = []
    mean_js_samples: list[float] = []
    for _ in range(replicates):
        sampled_groups = rng.integers(0, len(group_names), size=len(group_names))
        sampled_indices: list[int] = []
        for group_index in sampled_groups:
            sampled_indices.extend(groups_to_indices[group_names[int(group_index)]])
        indices = np.asarray(sampled_indices, dtype=np.int64)
        mean_js_samples.append(float(np.mean(motion_js[indices])))
        for mechanism_index, mechanism in enumerate(mechanisms):
            value = _spearman(
                reference_array[indices, mechanism_index],
                trainee_array[indices, mechanism_index],
            )
            if value is not None:
                channel_samples[mechanism].append(value)
        sampled_reference_distances = reference_distances[np.ix_(indices, indices)]
        sampled_trainee_distances = trainee_distances[np.ix_(indices, indices)]
        sampled_upper = np.triu_indices(len(indices), k=1)
        value = _spearman(
            sampled_reference_distances[sampled_upper],
            sampled_trainee_distances[sampled_upper],
        )
        if value is not None:
            distance_samples.append(value)

    level = float(protocol["confidence_level"])
    channel_rows = {
        mechanism: {
            "point_spearman": channel_points[mechanism],
            "valid_bootstrap_replicates": len(channel_samples[mechanism]),
            "valid_bootstrap_fraction": len(channel_samples[mechanism]) / replicates,
            "bootstrap_ci_lower_median_upper": _interval(
                channel_samples[mechanism],
                level=level,
            ),
        }
        for mechanism in mechanisms
    }
    common_fraction = len(common_keys) / len(all_motion_keys)
    coverage = {
        "minimum_motion_count": int(protocol["minimum_common_support_motion_count"]),
        "minimum_motion_fraction": float(protocol["minimum_common_support_fraction"]),
        "minimum_source_group_count": int(protocol["minimum_common_support_source_group_count"]),
        "passes": (
            len(common_keys) >= int(protocol["minimum_common_support_motion_count"])
            and common_fraction >= float(protocol["minimum_common_support_fraction"])
            and len(group_names) >= int(protocol["minimum_common_support_source_group_count"])
        ),
    }
    canonical_reference = [{"motion_key": key, **reference[key]} for key in sorted(reference)]
    canonical_trainee = [{"motion_key": key, **trainee[key]} for key in sorted(trainee)]
    input_payload = {
        "protocol_sha256": protocol[TRANSPORT_PROTOCOL_DIGEST_FIELD],
        "trainee_policy_id": trainee_policy_id,
        "source_group_by_motion": dict(sorted(source_group_by_motion.items())),
        "artifact_bindings": bindings,
        "reference_signatures": canonical_reference,
        "trainee_signatures": canonical_trainee,
    }
    report: dict[str, Any] = {
        "kind": TRANSPORT_REPORT_KIND,
        "schema_version": TRANSPORT_REPORT_SCHEMA_VERSION,
        "scientific_use": True,
        "report_only": True,
        "causal_capacity_claim_allowed": False,
        "interpretation": (
            "Cross-policy coordinate transport only; architecture, training history, capacity, "
            "and mastery are jointly changed."
        ),
        "low_transport_action": protocol["low_transport_action"],
        "transport_protocol_sha256": protocol[TRANSPORT_PROTOCOL_DIGEST_FIELD],
        "artifact_bindings": bindings,
        "input_sha256": canonical_sha256(input_payload),
        "mechanism_names": list(mechanisms),
        "reference_policy_id": reference_policy_id,
        "trainee_policy_id": trainee_policy_id,
        "num_motion_keys_total": len(all_motion_keys),
        "num_motion_keys_common_support": len(common_keys),
        "common_support_fraction": common_fraction,
        "num_source_groups_common_support": len(group_names),
        "common_support_motion_keys": list(common_keys),
        "excluded_motion_keys": excluded,
        "coverage": coverage,
        "mean_motion_js_divergence_bits": float(np.mean(motion_js)),
        "median_motion_js_divergence_bits": float(np.median(motion_js)),
        "q90_motion_js_divergence_bits": float(np.quantile(motion_js, 0.9, method="linear")),
        "mean_motion_js_bootstrap_ci_lower_median_upper": _interval(
            mean_js_samples,
            level=level,
        ),
        "per_channel_spearman": channel_rows,
        "distance_matrix_spearman": distance_spearman,
        "distance_matrix_valid_bootstrap_replicates": len(distance_samples),
        "distance_matrix_valid_bootstrap_fraction": len(distance_samples) / replicates,
        "distance_matrix_bootstrap_ci_lower_median_upper": _interval(
            distance_samples,
            level=level,
        ),
        "neighbor_k_requested": int(protocol["neighbor_k"]),
        "neighbor_k_effective": effective_k,
        "mean_neighbor_overlap_fraction": float(np.mean(overlaps)),
        "median_neighbor_overlap_fraction": float(np.median(overlaps)),
        "neighbor_rows": neighbor_rows,
        "bootstrap_unit": "source_group",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": int(protocol["bootstrap_seed"]),
        "confidence_level": level,
    }
    report[TRANSPORT_REPORT_DIGEST_FIELD] = canonical_sha256(
        report,
        digest_field=TRANSPORT_REPORT_DIGEST_FIELD,
    )
    return report
