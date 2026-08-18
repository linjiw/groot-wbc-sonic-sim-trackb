"""Leakage-resistant diagnostics for a frozen LACE failure geometry.

The functions in this module compare two independently aggregated halves of an
atlas. They deliberately report measurements rather than choosing thresholds:
the stability threshold, policy order, and neighbourhood size must be frozen
before any transfer outcome is opened.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from gear_sonic.research.lace.schema import canonical_sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    result = tuple(values)
    _require(bool(result), f"{name} must be non-empty")
    _require(
        all(isinstance(value, str) and value for value in result),
        f"{name} must contain non-empty strings",
    )
    _require(len(result) == len(set(result)), f"{name} must be unique")
    return result


def _index_signatures(
    records: Sequence[Mapping[str, Any]],
    *,
    mechanism_names: tuple[str, ...],
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    _require(bool(records), f"{label} must be non-empty")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"{label}[{index}] must be a mapping")
        motion_key = record.get("motion_key")
        policy_id = record.get("probe_policy_id")
        _require(
            isinstance(motion_key, str) and motion_key,
            f"{label}[{index}].motion_key must be non-empty",
        )
        _require(
            isinstance(policy_id, str) and policy_id,
            f"{label}[{index}].probe_policy_id must be non-empty",
        )
        key = (motion_key, policy_id)
        _require(key not in indexed, f"{label} contains duplicate signature {key}")

        q_value = record.get("q")
        q: tuple[float, ...] | None
        if q_value is None:
            q = None
        else:
            _require(
                isinstance(q_value, Sequence) and not isinstance(q_value, (str, bytes)),
                f"{label}[{index}].q must be a sequence or null",
            )
            _require(
                len(q_value) == len(mechanism_names),
                f"{label}[{index}].q length must match mechanism_names",
            )
            q = tuple(float(value) for value in q_value)
            _require(
                all(math.isfinite(value) and value >= 0.0 for value in q),
                f"{label}[{index}].q must be finite and nonnegative",
            )
            _require(
                math.isclose(sum(q), 1.0, rel_tol=0.0, abs_tol=1e-9),
                f"{label}[{index}].q must sum to one",
            )

        num_rollouts = record.get("num_rollouts")
        num_resolved = record.get("num_resolved_failures")
        _require(
            isinstance(num_rollouts, int)
            and not isinstance(num_rollouts, bool)
            and num_rollouts > 0,
            f"{label}[{index}].num_rollouts must be a positive integer",
        )
        _require(
            isinstance(num_resolved, int)
            and not isinstance(num_resolved, bool)
            and 0 <= num_resolved <= num_rollouts,
            f"{label}[{index}].num_resolved_failures is invalid",
        )
        indexed[key] = {
            "q": q,
            "num_rollouts": num_rollouts,
            "num_resolved_failures": num_resolved,
        }
    return indexed


def _js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    # scipy returns the square root of the Jensen-Shannon divergence.
    distance = float(jensenshannon(left, right, base=2.0))
    return distance * distance


def _pairwise_js(vectors: np.ndarray) -> np.ndarray:
    count = vectors.shape[0]
    result = np.zeros((count, count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            value = _js_divergence(vectors[left], vectors[right])
            result[left, right] = value
            result[right, left] = value
    return result


def _nearest_neighbours(
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


def _spearman_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    if np.allclose(left, left[0]) or np.allclose(right, right[0]):
        return None
    correlation = float(spearmanr(left, right).statistic)
    return correlation if math.isfinite(correlation) else None


def compare_split_half_signatures(
    signatures_a: Sequence[Mapping[str, Any]],
    signatures_b: Sequence[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str],
    policy_order: Sequence[str],
    neighbor_k: int,
) -> dict[str, Any]:
    """Compare independently aggregated atlas halves on common support.

    A motion enters the primary comparison only when every frozen probe policy
    has a resolved ``q`` in both halves. This prevents implicit imputation from
    changing support. Ordered policy-axis concatenation is normalized by the
    number of policies before pairwise Jensen-Shannon distances are computed.

    The result is JSON-ready and binds its exact inputs. It intentionally makes
    no pass/fail decision.
    """

    mechanisms = _validate_names(mechanism_names, "mechanism_names")
    policies = _validate_names(policy_order, "policy_order")
    _require(
        isinstance(neighbor_k, int) and not isinstance(neighbor_k, bool) and neighbor_k >= 1,
        "neighbor_k must be a positive integer",
    )
    index_a = _index_signatures(
        signatures_a,
        mechanism_names=mechanisms,
        label="signatures_a",
    )
    index_b = _index_signatures(
        signatures_b,
        mechanism_names=mechanisms,
        label="signatures_b",
    )
    _require(
        set(index_a) == set(index_b),
        "split halves must contain the same motion-policy signature keys",
    )
    observed_policies = {policy for _, policy in index_a}
    _require(
        observed_policies == set(policies),
        "policy_order must exactly match the observed probe policies",
    )

    all_motion_keys = tuple(sorted({motion for motion, _ in index_a}))
    complete_motion_keys: list[str] = []
    excluded: list[dict[str, Any]] = []
    vectors_a: list[list[float]] = []
    vectors_b: list[list[float]] = []
    motion_policy_js: list[float] = []
    per_policy_js: dict[str, list[float]] = {policy: [] for policy in policies}

    for motion_key in all_motion_keys:
        missing_a = [policy for policy in policies if index_a[(motion_key, policy)]["q"] is None]
        missing_b = [policy for policy in policies if index_b[(motion_key, policy)]["q"] is None]
        if missing_a or missing_b:
            excluded.append(
                {
                    "motion_key": motion_key,
                    "unresolved_policy_ids_a": missing_a,
                    "unresolved_policy_ids_b": missing_b,
                }
            )
            continue

        concat_a: list[float] = []
        concat_b: list[float] = []
        for policy in policies:
            q_a = index_a[(motion_key, policy)]["q"]
            q_b = index_b[(motion_key, policy)]["q"]
            assert q_a is not None and q_b is not None
            array_a = np.asarray(q_a, dtype=np.float64)
            array_b = np.asarray(q_b, dtype=np.float64)
            divergence = _js_divergence(array_a, array_b)
            motion_policy_js.append(divergence)
            per_policy_js[policy].append(divergence)
            concat_a.extend(value / len(policies) for value in q_a)
            concat_b.extend(value / len(policies) for value in q_b)
        complete_motion_keys.append(motion_key)
        vectors_a.append(concat_a)
        vectors_b.append(concat_b)

    _require(
        len(complete_motion_keys) >= 3,
        "at least three complete common-support motions are required",
    )
    effective_neighbor_k = min(neighbor_k, len(complete_motion_keys) - 1)
    array_a = np.asarray(vectors_a, dtype=np.float64)
    array_b = np.asarray(vectors_b, dtype=np.float64)
    per_policy_channel_spearman: dict[str, dict[str, float | None]] = {}
    for policy_index, policy in enumerate(policies):
        channel_results: dict[str, float | None] = {}
        for mechanism_index, mechanism in enumerate(mechanisms):
            column = policy_index * len(mechanisms) + mechanism_index
            values_a = array_a[:, column]
            values_b = array_b[:, column]
            channel_results[mechanism] = _spearman_or_none(values_a, values_b)
        per_policy_channel_spearman[policy] = channel_results
    distances_a = _pairwise_js(array_a)
    distances_b = _pairwise_js(array_b)
    upper = np.triu_indices(len(complete_motion_keys), k=1)
    flat_a = distances_a[upper]
    flat_b = distances_b[upper]
    distance_degenerate = bool(np.allclose(flat_a, flat_a[0]) or np.allclose(flat_b, flat_b[0]))
    if distance_degenerate:
        distance_spearman = None
    else:
        correlation = float(spearmanr(flat_a, flat_b).statistic)
        distance_spearman = correlation if math.isfinite(correlation) else None

    overlaps: list[float] = []
    neighbour_rows: list[dict[str, Any]] = []
    keys = tuple(complete_motion_keys)
    for index, motion_key in enumerate(keys):
        neighbours_a = _nearest_neighbours(distances_a, keys, index, effective_neighbor_k)
        neighbours_b = _nearest_neighbours(distances_b, keys, index, effective_neighbor_k)
        overlap = len(set(neighbours_a) & set(neighbours_b)) / effective_neighbor_k
        overlaps.append(overlap)
        neighbour_rows.append(
            {
                "motion_key": motion_key,
                "neighbors_a": list(neighbours_a),
                "neighbors_b": list(neighbours_b),
                "overlap_fraction": overlap,
            }
        )

    input_payload = {
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "neighbor_k_requested": neighbor_k,
        "signatures_a": list(signatures_a),
        "signatures_b": list(signatures_b),
    }
    return {
        "schema_version": 1,
        "kind": "lace_split_half_signature_stability",
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "input_sha256": canonical_sha256(input_payload),
        "num_motion_keys_total": len(all_motion_keys),
        "num_motion_keys_common_support": len(complete_motion_keys),
        "common_support_fraction": len(complete_motion_keys) / len(all_motion_keys),
        "common_support_motion_keys": complete_motion_keys,
        "excluded_motion_keys": excluded,
        "mean_motion_policy_js_divergence_bits": float(np.mean(motion_policy_js)),
        "median_motion_policy_js_divergence_bits": float(np.median(motion_policy_js)),
        "q90_motion_policy_js_divergence_bits": float(
            np.quantile(motion_policy_js, 0.9, method="linear")
        ),
        "per_policy_mean_js_divergence_bits": {
            policy: float(np.mean(per_policy_js[policy])) for policy in policies
        },
        "per_policy_channel_spearman": per_policy_channel_spearman,
        "distance_matrix_spearman": distance_spearman,
        "distance_matrix_degenerate": distance_degenerate,
        "neighbor_k_requested": neighbor_k,
        "neighbor_k_effective": effective_neighbor_k,
        "mean_neighbor_overlap_fraction": float(np.mean(overlaps)),
        "median_neighbor_overlap_fraction": float(np.median(overlaps)),
        "neighbor_rows": neighbour_rows,
    }


def bootstrap_split_half_channel_spearman(
    signatures_a: Sequence[Mapping[str, Any]],
    signatures_b: Sequence[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str],
    policy_order: Sequence[str],
    source_group_by_motion: Mapping[str, str],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap split-half channel correlations by source recording group.

    Source groups, rather than individual motions, are sampled with replacement.
    A replicate may be degenerate for a rare channel; those replicates are counted
    and reported instead of being replaced or assigned a favorable correlation.
    """

    mechanisms = _validate_names(mechanism_names, "mechanism_names")
    policies = _validate_names(policy_order, "policy_order")
    _require(
        isinstance(bootstrap_replicates, int)
        and not isinstance(bootstrap_replicates, bool)
        and bootstrap_replicates >= 100,
        "bootstrap_replicates must be an integer of at least 100",
    )
    _require(
        isinstance(bootstrap_seed, int) and not isinstance(bootstrap_seed, bool),
        "bootstrap_seed must be an integer",
    )
    level = float(confidence_level)
    _require(
        math.isfinite(level) and 0.0 < level < 1.0,
        "confidence_level must lie strictly between zero and one",
    )
    index_a = _index_signatures(
        signatures_a,
        mechanism_names=mechanisms,
        label="signatures_a",
    )
    index_b = _index_signatures(
        signatures_b,
        mechanism_names=mechanisms,
        label="signatures_b",
    )
    _require(
        set(index_a) == set(index_b),
        "split halves must contain the same motion-policy signature keys",
    )
    observed_policies = {policy for _, policy in index_a}
    _require(
        observed_policies == set(policies),
        "policy_order must exactly match the observed probe policies",
    )
    all_motion_keys = tuple(sorted({motion for motion, _ in index_a}))
    _require(
        set(source_group_by_motion) == set(all_motion_keys),
        "source_group_by_motion must exactly match observed motion keys",
    )
    _require(
        all(isinstance(group, str) and group for group in source_group_by_motion.values()),
        "source_group_by_motion values must be non-empty strings",
    )

    complete_motion_keys = tuple(
        motion_key
        for motion_key in all_motion_keys
        if all(
            index_a[(motion_key, policy)]["q"] is not None
            and index_b[(motion_key, policy)]["q"] is not None
            for policy in policies
        )
    )
    _require(
        len(complete_motion_keys) >= 3,
        "at least three complete common-support motions are required",
    )
    groups_to_motion_indices: dict[str, list[int]] = defaultdict(list)
    for motion_index, motion_key in enumerate(complete_motion_keys):
        groups_to_motion_indices[source_group_by_motion[motion_key]].append(motion_index)
    group_names = tuple(sorted(groups_to_motion_indices))
    _require(
        len(group_names) >= 3,
        "at least three complete common-support source groups are required",
    )

    arrays_a = np.empty(
        (len(policies), len(mechanisms), len(complete_motion_keys)),
        dtype=np.float64,
    )
    arrays_b = np.empty_like(arrays_a)
    for motion_index, motion_key in enumerate(complete_motion_keys):
        for policy_index, policy in enumerate(policies):
            q_a = index_a[(motion_key, policy)]["q"]
            q_b = index_b[(motion_key, policy)]["q"]
            assert q_a is not None and q_b is not None
            arrays_a[policy_index, :, motion_index] = q_a
            arrays_b[policy_index, :, motion_index] = q_b

    rng = np.random.default_rng(bootstrap_seed)
    samples: list[list[list[float]]] = [[[] for _ in mechanisms] for _ in policies]
    for _ in range(bootstrap_replicates):
        sampled_group_indices = rng.integers(0, len(group_names), size=len(group_names))
        sampled_motion_indices: list[int] = []
        for sampled_group_index in sampled_group_indices:
            sampled_motion_indices.extend(
                groups_to_motion_indices[group_names[int(sampled_group_index)]]
            )
        indices = np.asarray(sampled_motion_indices, dtype=np.int64)
        for policy_index in range(len(policies)):
            for mechanism_index in range(len(mechanisms)):
                correlation = _spearman_or_none(
                    arrays_a[policy_index, mechanism_index, indices],
                    arrays_b[policy_index, mechanism_index, indices],
                )
                if correlation is not None:
                    samples[policy_index][mechanism_index].append(correlation)

    tail = (1.0 - level) / 2.0
    channel_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for policy_index, policy in enumerate(policies):
        mechanism_rows: dict[str, dict[str, Any]] = {}
        for mechanism_index, mechanism in enumerate(mechanisms):
            point = _spearman_or_none(
                arrays_a[policy_index, mechanism_index],
                arrays_b[policy_index, mechanism_index],
            )
            values = np.asarray(samples[policy_index][mechanism_index], dtype=np.float64)
            if values.size:
                interval: list[float | None] = [
                    float(np.quantile(values, tail, method="linear")),
                    float(np.quantile(values, 0.5, method="linear")),
                    float(np.quantile(values, 1.0 - tail, method="linear")),
                ]
            else:
                interval = [None, None, None]
            mechanism_rows[mechanism] = {
                "point_spearman": point,
                "valid_bootstrap_replicates": int(values.size),
                "valid_bootstrap_fraction": float(values.size / bootstrap_replicates),
                "bootstrap_ci_lower_median_upper": interval,
            }
        channel_rows[policy] = mechanism_rows

    input_payload = {
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "source_group_by_motion": dict(sorted(source_group_by_motion.items())),
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": level,
        "signatures_a": list(signatures_a),
        "signatures_b": list(signatures_b),
    }
    return {
        "schema_version": 1,
        "kind": "lace_split_half_channel_spearman_bootstrap",
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "input_sha256": canonical_sha256(input_payload),
        "bootstrap_unit": "source_group",
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "confidence_level": level,
        "num_motion_keys_total": len(all_motion_keys),
        "num_motion_keys_common_support": len(complete_motion_keys),
        "num_source_groups_common_support": len(group_names),
        "channels": channel_rows,
    }


def summarize_mechanism_incidence(
    signatures: Sequence[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str],
    policy_order: Sequence[str],
    source_group_by_motion: Mapping[str, str],
) -> dict[str, Any]:
    """Report attributed mechanism mass and source-group concentration."""

    mechanisms = _validate_names(mechanism_names, "mechanism_names")
    policies = _validate_names(policy_order, "policy_order")
    indexed = _index_signatures(
        signatures,
        mechanism_names=mechanisms,
        label="signatures",
    )
    observed_policies = {policy for _, policy in indexed}
    _require(
        observed_policies == set(policies),
        "policy_order must exactly match the observed probe policies",
    )
    motion_keys = tuple(sorted({motion for motion, _ in indexed}))
    _require(
        set(source_group_by_motion) == set(motion_keys),
        "source_group_by_motion must exactly match observed motion keys",
    )
    original_by_key = {
        (str(record.get("motion_key")), str(record.get("probe_policy_id"))): record
        for record in signatures
    }
    channel_mass = np.zeros(len(mechanisms), dtype=np.float64)
    group_channel_mass: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros(len(mechanisms), dtype=np.float64)
    )
    supported_signatures = 0
    for key in sorted(indexed):
        q = indexed[key]["q"]
        if q is None:
            continue
        record = original_by_key[key]
        raw_rate = record.get("attributed_failure_rate")
        _require(
            not isinstance(raw_rate, bool),
            f"attributed_failure_rate for {key} must be numeric",
        )
        try:
            rate = float(raw_rate)
        except (TypeError, ValueError) as error:
            raise ValueError(f"attributed_failure_rate for {key} must be numeric") from error
        _require(
            math.isfinite(rate) and 0.0 <= rate <= 1.0,
            f"attributed_failure_rate for {key} must lie in [0, 1]",
        )
        expected_rate = indexed[key]["num_resolved_failures"] / indexed[key]["num_rollouts"]
        _require(
            math.isclose(rate, expected_rate, rel_tol=0.0, abs_tol=1e-9),
            f"attributed_failure_rate for {key} does not match counts",
        )
        mass = rate * np.asarray(q, dtype=np.float64)
        channel_mass += mass
        group_channel_mass[source_group_by_motion[key[0]]] += mass
        supported_signatures += 1

    total_mass = float(channel_mass.sum())
    _require(total_mass > 0.0, "signatures contain no attributed mechanism mass")
    mechanism_rows: list[dict[str, Any]] = []
    for mechanism_index, mechanism in enumerate(mechanisms):
        mass = float(channel_mass[mechanism_index])
        source_rows = [
            {
                "source_group_id": group,
                "mass": float(values[mechanism_index]),
            }
            for group, values in sorted(group_channel_mass.items())
            if float(values[mechanism_index]) > 0.0
        ]
        source_total = sum(row["mass"] for row in source_rows)
        maximum_source_share = (
            max(row["mass"] for row in source_rows) / source_total if source_total > 0.0 else None
        )
        mechanism_rows.append(
            {
                "mechanism": mechanism,
                "attributed_mass": mass,
                "global_mass_fraction": mass / total_mass,
                "positive_source_group_count": len(source_rows),
                "maximum_single_source_group_share": maximum_source_share,
                "source_groups": source_rows,
            }
        )

    input_payload = {
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "source_group_by_motion": dict(sorted(source_group_by_motion.items())),
        "signatures": list(signatures),
    }
    return {
        "schema_version": 1,
        "kind": "lace_mechanism_incidence",
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "input_sha256": canonical_sha256(input_payload),
        "num_motion_keys": len(motion_keys),
        "num_signatures": len(indexed),
        "num_supported_signatures": supported_signatures,
        "total_attributed_mechanism_mass": total_mass,
        "mechanisms": mechanism_rows,
    }


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    ridge_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean(axis=0)
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    gram = centered_x.T @ centered_x
    penalty = ridge_alpha * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.pinv(gram + penalty) @ centered_x.T @ centered_y
    prediction = (test_x - x_mean) @ coefficients + y_mean
    baseline = np.broadcast_to(y_mean, prediction.shape).copy()
    return prediction, baseline


def cross_validated_difficulty_reconstruction(
    signatures: Sequence[Mapping[str, Any]],
    *,
    mechanism_names: Sequence[str],
    policy_order: Sequence[str],
    source_group_by_motion: Mapping[str, str],
    fold_by_source_group: Mapping[str, int],
    ridge_alpha: float,
) -> dict[str, Any]:
    """Measure how well scalar difficulty reconstructs the mechanism geometry.

    The predictor is the ordered vector of policy-specific failure frequencies;
    the target is the ordered policy-axis concatenation of ``q``. Folds are
    supplied by source recording group, rather than generated here, so the exact
    grouping can be frozen and shared with other representation diagnostics.
    Missing mechanism signatures are excluded motion-wise on common policy
    support and are reported, never imputed.
    """

    mechanisms = _validate_names(mechanism_names, "mechanism_names")
    policies = _validate_names(policy_order, "policy_order")
    _require(not isinstance(ridge_alpha, bool), "ridge_alpha must be numeric, not boolean")
    try:
        alpha = float(ridge_alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("ridge_alpha must be numeric") from error
    _require(math.isfinite(alpha) and alpha >= 0.0, "ridge_alpha must be finite and nonnegative")
    indexed = _index_signatures(
        signatures,
        mechanism_names=mechanisms,
        label="signatures",
    )
    original_by_key = {
        (str(record.get("motion_key")), str(record.get("probe_policy_id"))): record
        for record in signatures
    }
    observed_policies = {policy for _, policy in indexed}
    _require(
        observed_policies == set(policies),
        "policy_order must exactly match the observed probe policies",
    )
    all_motion_keys = tuple(sorted({motion for motion, _ in indexed}))
    _require(
        set(source_group_by_motion) == set(all_motion_keys),
        "source_group_by_motion must exactly match observed motion keys",
    )
    _require(
        all(isinstance(group, str) and group for group in source_group_by_motion.values()),
        "source_group_by_motion values must be non-empty strings",
    )
    observed_groups = set(source_group_by_motion.values())
    _require(
        set(fold_by_source_group) == observed_groups,
        "fold_by_source_group must exactly match observed source groups",
    )
    fold_values = list(fold_by_source_group.values())
    _require(
        all(
            isinstance(fold, int) and not isinstance(fold, bool) and fold >= 0
            for fold in fold_values
        ),
        "fold assignments must be nonnegative integers",
    )
    unique_folds = sorted(set(fold_values))
    _require(
        unique_folds == list(range(len(unique_folds))) and len(unique_folds) >= 3,
        "fold assignments must contain at least three contiguous folds starting at zero",
    )

    included: list[str] = []
    excluded: list[dict[str, Any]] = []
    predictors: list[list[float]] = []
    targets: list[list[float]] = []
    groups: list[str] = []
    folds: list[int] = []
    for motion_key in all_motion_keys:
        unresolved = [policy for policy in policies if indexed[(motion_key, policy)]["q"] is None]
        if unresolved:
            excluded.append(
                {
                    "motion_key": motion_key,
                    "unresolved_policy_ids": unresolved,
                }
            )
            continue
        difficulty_values: list[float] = []
        q_values: list[float] = []
        for policy in policies:
            # Difficulty is required here even though the split-half comparator
            # intentionally needs only q and count fields.
            raw_difficulty = original_by_key[(motion_key, policy)].get("difficulty")
            _require(
                not isinstance(raw_difficulty, bool),
                f"difficulty for {(motion_key, policy)} must be numeric",
            )
            try:
                difficulty = float(raw_difficulty)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"difficulty for {(motion_key, policy)} must be numeric"
                ) from error
            _require(
                math.isfinite(difficulty) and 0.0 <= difficulty <= 1.0,
                f"difficulty for {(motion_key, policy)} must lie in [0, 1]",
            )
            q = indexed[(motion_key, policy)]["q"]
            assert q is not None
            difficulty_values.append(difficulty)
            q_values.extend(q)
        group = source_group_by_motion[motion_key]
        included.append(motion_key)
        predictors.append(difficulty_values)
        targets.append(q_values)
        groups.append(group)
        folds.append(fold_by_source_group[group])

    _require(
        len(included) > len(policies),
        "too few complete motions for difficulty reconstruction",
    )
    x = np.asarray(predictors, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    fold_array = np.asarray(folds, dtype=np.int64)
    prediction = np.empty_like(y)
    baseline = np.empty_like(y)
    fold_rows: list[dict[str, Any]] = []
    for fold in unique_folds:
        test_mask = fold_array == fold
        train_mask = ~test_mask
        train_groups = {groups[index] for index in np.flatnonzero(train_mask)}
        test_groups = {groups[index] for index in np.flatnonzero(test_mask)}
        _require(bool(test_groups), f"fold {fold} has no test source group")
        _require(
            len(train_groups) >= 2 and train_groups.isdisjoint(test_groups),
            f"fold {fold} must have at least two disjoint training source groups",
        )
        fold_prediction, fold_baseline = _ridge_predict(
            x[train_mask],
            y[train_mask],
            x[test_mask],
            alpha,
        )
        prediction[test_mask] = fold_prediction
        baseline[test_mask] = fold_baseline
        fold_rows.append(
            {
                "fold": fold,
                "train_source_groups": sorted(train_groups),
                "test_source_groups": sorted(test_groups),
                "num_train_motions": int(train_mask.sum()),
                "num_test_motions": int(test_mask.sum()),
            }
        )

    model_sse = float(np.square(y - prediction).sum())
    baseline_sse = float(np.square(y - baseline).sum())
    reconstruction_r2 = None if baseline_sse <= 0.0 else 1.0 - model_sse / baseline_sse
    input_payload = {
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "source_group_by_motion": dict(sorted(source_group_by_motion.items())),
        "fold_by_source_group": dict(sorted(fold_by_source_group.items())),
        "ridge_alpha": alpha,
        "signatures": list(signatures),
    }
    return {
        "schema_version": 1,
        "kind": "lace_difficulty_reconstruction",
        "mechanism_names": list(mechanisms),
        "policy_order": list(policies),
        "input_sha256": canonical_sha256(input_payload),
        "ridge_alpha": alpha,
        "num_motion_keys_total": len(all_motion_keys),
        "num_motion_keys_common_support": len(included),
        "common_support_fraction": len(included) / len(all_motion_keys),
        "common_support_motion_keys": included,
        "excluded_motion_keys": excluded,
        "folds": fold_rows,
        "model_squared_error": model_sse,
        "intercept_baseline_squared_error": baseline_sse,
        "cross_validated_reconstruction_r2": reconstruction_r2,
        "target_geometry_degenerate": baseline_sse <= 0.0,
    }


_STABILITY_THRESHOLD_KEYS = {
    "minimum_common_support_motion_count",
    "minimum_common_support_fraction",
    "minimum_common_support_source_group_count",
    "maximum_median_js_divergence_bits",
    "minimum_distance_matrix_spearman",
    "minimum_mean_neighbor_overlap_fraction",
    "minimum_channel_point_spearman",
    "minimum_channel_bootstrap_ci_lower",
    "minimum_channel_valid_bootstrap_fraction",
    "minimum_global_mechanism_mass_fraction",
    "maximum_global_mechanism_mass_fraction",
    "minimum_positive_source_groups_per_mechanism",
    "maximum_single_source_group_share",
    "maximum_difficulty_reconstruction_r2",
}


def evaluate_stability_gate(
    stability_report: Mapping[str, Any],
    bootstrap_report: Mapping[str, Any],
    incidence_report: Mapping[str, Any],
    reconstruction_report: Mapping[str, Any],
    *,
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a frozen, outcome-blind atlas screening gate to four reports."""

    _require(
        gate_config.get("kind") == "lace_signature_stability_gate_config",
        "gate_config.kind is invalid",
    )
    _require(gate_config.get("schema_version") == 1, "gate_config.schema_version must be one")
    _require(gate_config.get("frozen") is True, "gate_config must be frozen")
    expected_digest = gate_config.get("gate_config_sha256")
    _require(
        isinstance(expected_digest, str) and len(expected_digest) == 64,
        "gate_config_sha256 must be a SHA-256",
    )
    _require(
        expected_digest == canonical_sha256(gate_config, digest_field="gate_config_sha256"),
        "gate_config_sha256 mismatch",
    )
    mechanisms = _validate_names(gate_config.get("mechanism_names", ()), "mechanism_names")
    policies = _validate_names(gate_config.get("policy_order", ()), "policy_order")
    thresholds = gate_config.get("thresholds")
    _require(isinstance(thresholds, Mapping), "gate_config.thresholds must be a mapping")
    _require(
        set(thresholds) == _STABILITY_THRESHOLD_KEYS,
        "gate_config.thresholds must exactly match the supported threshold schema",
    )
    parsed_thresholds: dict[str, float] = {}
    for name, value in thresholds.items():
        _require(not isinstance(value, bool), f"threshold {name} must be numeric")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"threshold {name} must be numeric") from error
        _require(math.isfinite(parsed), f"threshold {name} must be finite")
        parsed_thresholds[name] = parsed
    analysis = gate_config.get("analysis")
    _require(isinstance(analysis, Mapping), "gate_config.analysis must be a mapping")
    _require(
        set(analysis)
        == {
            "neighbor_k",
            "bootstrap_replicates",
            "bootstrap_seed",
            "bootstrap_confidence_level",
            "difficulty_ridge_alpha",
        },
        "gate_config.analysis must exactly match the supported analysis schema",
    )
    for name in (
        "minimum_common_support_motion_count",
        "minimum_common_support_source_group_count",
        "minimum_positive_source_groups_per_mechanism",
    ):
        _require(
            parsed_thresholds[name].is_integer() and parsed_thresholds[name] >= 1.0,
            f"threshold {name} must be a positive integer",
        )
    for name in (
        "minimum_common_support_fraction",
        "minimum_mean_neighbor_overlap_fraction",
        "minimum_channel_valid_bootstrap_fraction",
        "minimum_global_mechanism_mass_fraction",
        "maximum_global_mechanism_mass_fraction",
        "maximum_single_source_group_share",
    ):
        _require(
            0.0 <= parsed_thresholds[name] <= 1.0,
            f"threshold {name} must lie in [0, 1]",
        )

    reports = (
        (stability_report, "lace_split_half_signature_stability", "stability_report"),
        (
            bootstrap_report,
            "lace_split_half_channel_spearman_bootstrap",
            "bootstrap_report",
        ),
        (incidence_report, "lace_mechanism_incidence", "incidence_report"),
        (
            reconstruction_report,
            "lace_difficulty_reconstruction",
            "reconstruction_report",
        ),
    )
    for report, expected_kind, name in reports:
        _require(report.get("kind") == expected_kind, f"{name}.kind is invalid")
        _require(
            tuple(report.get("mechanism_names", ())) == mechanisms,
            f"{name}.mechanism_names do not match gate_config",
        )
        _require(
            tuple(report.get("policy_order", ())) == policies,
            f"{name}.policy_order does not match gate_config",
        )
    _require(
        stability_report.get("neighbor_k_requested") == analysis["neighbor_k"],
        "stability_report neighbor_k does not match gate_config",
    )
    _require(
        bootstrap_report.get("bootstrap_replicates") == analysis["bootstrap_replicates"],
        "bootstrap_report replicate count does not match gate_config",
    )
    _require(
        bootstrap_report.get("bootstrap_seed") == analysis["bootstrap_seed"],
        "bootstrap_report seed does not match gate_config",
    )
    _require(
        math.isclose(
            float(bootstrap_report.get("confidence_level", math.nan)),
            float(analysis["bootstrap_confidence_level"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "bootstrap_report confidence level does not match gate_config",
    )
    _require(
        math.isclose(
            float(reconstruction_report.get("ridge_alpha", math.nan)),
            float(analysis["difficulty_ridge_alpha"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "reconstruction_report ridge alpha does not match gate_config",
    )

    checks: list[dict[str, Any]] = []

    def add_check(name: str, observed: Any, operator: str, threshold: float, passed: bool) -> None:
        checks.append(
            {
                "name": name,
                "observed": observed,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        )

    minimum_motions = parsed_thresholds["minimum_common_support_motion_count"]
    minimum_fraction = parsed_thresholds["minimum_common_support_fraction"]
    common_motions = int(stability_report.get("num_motion_keys_common_support", -1))
    common_fraction = float(stability_report.get("common_support_fraction", -1.0))
    add_check(
        "common_support_motion_count",
        common_motions,
        ">=",
        minimum_motions,
        common_motions >= minimum_motions,
    )
    add_check(
        "common_support_fraction",
        common_fraction,
        ">=",
        minimum_fraction,
        common_fraction >= minimum_fraction,
    )
    common_groups = int(bootstrap_report.get("num_source_groups_common_support", -1))
    minimum_groups = parsed_thresholds["minimum_common_support_source_group_count"]
    add_check(
        "common_support_source_group_count",
        common_groups,
        ">=",
        minimum_groups,
        common_groups >= minimum_groups,
    )

    median_js = float(stability_report.get("median_motion_policy_js_divergence_bits", math.inf))
    maximum_js = parsed_thresholds["maximum_median_js_divergence_bits"]
    add_check("median_js_divergence_bits", median_js, "<=", maximum_js, median_js <= maximum_js)
    distance_spearman = stability_report.get("distance_matrix_spearman")
    minimum_distance = parsed_thresholds["minimum_distance_matrix_spearman"]
    add_check(
        "distance_matrix_spearman",
        distance_spearman,
        ">=",
        minimum_distance,
        distance_spearman is not None and float(distance_spearman) >= minimum_distance,
    )
    neighbor_overlap = float(stability_report.get("mean_neighbor_overlap_fraction", -1.0))
    minimum_overlap = parsed_thresholds["minimum_mean_neighbor_overlap_fraction"]
    add_check(
        "mean_neighbor_overlap_fraction",
        neighbor_overlap,
        ">=",
        minimum_overlap,
        neighbor_overlap >= minimum_overlap,
    )

    channel_map = bootstrap_report.get("channels")
    _require(isinstance(channel_map, Mapping), "bootstrap_report.channels must be a mapping")
    minimum_point = parsed_thresholds["minimum_channel_point_spearman"]
    minimum_lower = parsed_thresholds["minimum_channel_bootstrap_ci_lower"]
    minimum_valid = parsed_thresholds["minimum_channel_valid_bootstrap_fraction"]
    for policy in policies:
        policy_channels = channel_map.get(policy)
        _require(
            isinstance(policy_channels, Mapping) and set(policy_channels) == set(mechanisms),
            f"bootstrap_report.channels[{policy!r}] must exactly match mechanisms",
        )
        for mechanism in mechanisms:
            row = policy_channels[mechanism]
            _require(isinstance(row, Mapping), "bootstrap channel row must be a mapping")
            point = row.get("point_spearman")
            valid_fraction = float(row.get("valid_bootstrap_fraction", -1.0))
            interval = row.get("bootstrap_ci_lower_median_upper")
            _require(
                isinstance(interval, Sequence)
                and not isinstance(interval, (str, bytes))
                and len(interval) == 3,
                "bootstrap channel interval must have lower, median, and upper values",
            )
            lower = interval[0]
            prefix = f"channel.{policy}.{mechanism}"
            add_check(
                f"{prefix}.point_spearman",
                point,
                ">=",
                minimum_point,
                point is not None and float(point) >= minimum_point,
            )
            add_check(
                f"{prefix}.bootstrap_ci_lower",
                lower,
                ">=",
                minimum_lower,
                lower is not None and float(lower) >= minimum_lower,
            )
            add_check(
                f"{prefix}.valid_bootstrap_fraction",
                valid_fraction,
                ">=",
                minimum_valid,
                valid_fraction >= minimum_valid,
            )

    incidence_rows = incidence_report.get("mechanisms")
    _require(isinstance(incidence_rows, Sequence), "incidence_report.mechanisms must be a sequence")
    _require(
        [row.get("mechanism") for row in incidence_rows if isinstance(row, Mapping)]
        == list(mechanisms),
        "incidence_report mechanism rows must match gate order",
    )
    minimum_mass = parsed_thresholds["minimum_global_mechanism_mass_fraction"]
    maximum_mass = parsed_thresholds["maximum_global_mechanism_mass_fraction"]
    minimum_positive_groups = parsed_thresholds["minimum_positive_source_groups_per_mechanism"]
    maximum_source_share = parsed_thresholds["maximum_single_source_group_share"]
    for row in incidence_rows:
        assert isinstance(row, Mapping)
        mechanism = str(row["mechanism"])
        mass_fraction = float(row.get("global_mass_fraction", -1.0))
        positive_groups = int(row.get("positive_source_group_count", -1))
        source_share = row.get("maximum_single_source_group_share")
        prefix = f"incidence.{mechanism}"
        add_check(
            f"{prefix}.minimum_mass_fraction",
            mass_fraction,
            ">=",
            minimum_mass,
            mass_fraction >= minimum_mass,
        )
        add_check(
            f"{prefix}.maximum_mass_fraction",
            mass_fraction,
            "<=",
            maximum_mass,
            mass_fraction <= maximum_mass,
        )
        add_check(
            f"{prefix}.positive_source_group_count",
            positive_groups,
            ">=",
            minimum_positive_groups,
            positive_groups >= minimum_positive_groups,
        )
        add_check(
            f"{prefix}.maximum_single_source_group_share",
            source_share,
            "<=",
            maximum_source_share,
            source_share is not None and float(source_share) <= maximum_source_share,
        )

    reconstruction_r2 = reconstruction_report.get("cross_validated_reconstruction_r2")
    maximum_reconstruction = parsed_thresholds["maximum_difficulty_reconstruction_r2"]
    add_check(
        "difficulty_reconstruction_r2",
        reconstruction_r2,
        "<=",
        maximum_reconstruction,
        reconstruction_r2 is not None and float(reconstruction_r2) <= maximum_reconstruction,
    )

    failed_checks = [check["name"] for check in checks if not check["passed"]]
    input_payload = {
        "gate_config": dict(gate_config),
        "stability_report": dict(stability_report),
        "bootstrap_report": dict(bootstrap_report),
        "incidence_report": dict(incidence_report),
        "reconstruction_report": dict(reconstruction_report),
    }
    return {
        "schema_version": 1,
        "kind": "lace_signature_stability_gate_result",
        "gate_config_sha256": expected_digest,
        "input_sha256": canonical_sha256(input_payload),
        "passed": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "failure_action": gate_config.get("failure_action"),
    }
