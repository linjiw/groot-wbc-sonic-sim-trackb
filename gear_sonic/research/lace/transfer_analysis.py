"""Scientific CPU-only inference primitives for the LACE RQ1 transfer test.

The v2 contract keeps motion-level outcomes while treating source intervention
panels, paired training seeds, and target recording groups as clustered units.
Every fitted prediction is crossed out of sample in both its source-panel row
and target-source-group column.  No simulator or trainer dependency is used.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import math
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from gear_sonic.research.lace.schema import canonical_sha256

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

TRANSFER_ANALYSIS_KIND = "lace_crossed_transfer_ridge_analysis"
TRANSFER_ANALYSIS_SCHEMA_VERSION = 2
CLUSTER_BOOTSTRAP_KIND = "lace_hierarchical_transfer_cluster_bootstrap"
CLUSTER_BOOTSTRAP_SCHEMA_VERSION = 2
SOURCE_ROW_PERMUTATION_KIND = "lace_whole_source_row_failure_permutation"
SOURCE_ROW_PERMUTATION_SCHEMA_VERSION = 1

DEFAULT_ALPHA_GRID = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
DEFAULT_FEATURE_SCALE_TOLERANCE = 1e-12
MINIMUM_CROSSED_LEVELS = 4
MAXIMUM_EXACT_PERMUTATION_PANELS = 8

TRANSFER_FIELDS = frozenset(
    {
        "source_panel_id",
        "target_motion_id",
        "target_source_group_id",
        "paired_seed",
        "gain",
        "independent_headroom",
        "baseline_features",
        "failure_features",
    }
)


def _real(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _positive_integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer at least {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _seed(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(value)
    if result < 0 or result >= 2**64:
        raise ValueError(f"{name} must be in [0, 2**64)")
    return result


def _seed_set(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of unique seeds")
    try:
        materialized = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of unique seeds") from exc
    if not materialized:
        raise ValueError(f"{name} must be non-empty")
    result = tuple(_seed(value, f"{name}[{index}]") for index, value in enumerate(materialized))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicate seeds")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must be in strictly increasing canonical order")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    if value.lower() != value or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _feature_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a non-empty sequence of unique names")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence of unique names") from exc
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique names")
    return result


def _feature_vector(value: Any, name: str) -> list[float]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional feature vector")
    if (
        np.issubdtype(raw.dtype, np.bool_)
        or not np.issubdtype(raw.dtype, np.number)
        or np.issubdtype(raw.dtype, np.complexfloating)
    ):
        raise ValueError(f"{name} must contain real numeric features")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite features")
    return [float(item) for item in result]


def _alpha_grid(values: Sequence[float]) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("alpha_grid must be a non-empty increasing sequence")
    try:
        materialized = list(values)
    except TypeError as exc:
        raise ValueError("alpha_grid must be a non-empty increasing sequence") from exc
    if not materialized:
        raise ValueError("alpha_grid must be non-empty")
    result = tuple(
        _real(value, f"alpha_grid[{index}]", positive=True)
        for index, value in enumerate(materialized)
    )
    if any(right <= left for left, right in zip(result, result[1:], strict=False)):
        raise ValueError("alpha_grid must be strictly increasing with no duplicates")
    return result


def _materialize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_feature_names: tuple[str, ...],
    failure_feature_names: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    dict[str, str],
]:
    try:
        raw_records = list(records)
    except TypeError as exc:
        raise ValueError("records must be an iterable of mappings") from exc
    if not raw_records:
        raise ValueError("records must be non-empty")

    canonical: list[dict[str, Any]] = []
    seen_cells: set[tuple[str, str, int]] = set()
    target_motion_to_group: dict[str, str] = {}

    for index, record in enumerate(raw_records):
        if not isinstance(record, Mapping):
            raise ValueError(f"records[{index}] must be a mapping")
        fields = set(record)
        if fields != TRANSFER_FIELDS:
            missing = sorted(TRANSFER_FIELDS - fields)
            unexpected = sorted(fields - TRANSFER_FIELDS)
            raise ValueError(
                f"records[{index}] field mismatch; missing={missing}, unexpected={unexpected}"
            )

        source_panel_id = _identifier(
            record["source_panel_id"],
            f"records[{index}].source_panel_id",
        )
        target_motion_id = _identifier(
            record["target_motion_id"],
            f"records[{index}].target_motion_id",
        )
        target_source_group_id = _identifier(
            record["target_source_group_id"],
            f"records[{index}].target_source_group_id",
        )
        prior_group = target_motion_to_group.setdefault(target_motion_id, target_source_group_id)
        if prior_group != target_source_group_id:
            raise ValueError(
                f"target motion {target_motion_id!r} maps to multiple source groups: "
                f"{prior_group!r}, {target_source_group_id!r}"
            )
        paired_seed = _seed(record["paired_seed"], f"records[{index}].paired_seed")
        gain = _real(record["gain"], f"records[{index}].gain")
        independent_headroom = _real(
            record["independent_headroom"],
            f"records[{index}].independent_headroom",
            positive=True,
        )
        normalized_gain = gain / independent_headroom
        if not math.isfinite(normalized_gain):
            raise ValueError(f"records[{index}] gain/headroom is not finite")
        baseline_features = _feature_vector(
            record["baseline_features"],
            f"records[{index}].baseline_features",
        )
        failure_features = _feature_vector(
            record["failure_features"],
            f"records[{index}].failure_features",
        )
        if len(baseline_features) != len(baseline_feature_names):
            raise ValueError(
                f"records[{index}].baseline_features dimension does not match "
                "baseline_feature_names"
            )
        if len(failure_features) != len(failure_feature_names):
            raise ValueError(
                f"records[{index}].failure_features dimension does not match "
                "failure_feature_names"
            )

        cell = (source_panel_id, target_motion_id, paired_seed)
        if cell in seen_cells:
            raise ValueError(f"duplicate transfer cell: {cell}")
        seen_cells.add(cell)
        canonical.append(
            {
                "source_panel_id": source_panel_id,
                "target_motion_id": target_motion_id,
                "target_source_group_id": target_source_group_id,
                "paired_seed": paired_seed,
                "gain": gain,
                "independent_headroom": independent_headroom,
                "normalized_gain": normalized_gain,
                "baseline_features": baseline_features,
                "failure_features": failure_features,
            }
        )

    canonical.sort(
        key=lambda record: (
            record["source_panel_id"],
            record["target_source_group_id"],
            record["target_motion_id"],
            record["paired_seed"],
        )
    )
    source_panels = tuple(sorted({record["source_panel_id"] for record in canonical}))
    target_motions = tuple(sorted({record["target_motion_id"] for record in canonical}))
    target_groups = tuple(sorted({record["target_source_group_id"] for record in canonical}))
    paired_seeds = tuple(sorted({record["paired_seed"] for record in canonical}))
    if len(source_panels) < MINIMUM_CROSSED_LEVELS:
        raise ValueError(
            f"crossed nested CV requires at least {MINIMUM_CROSSED_LEVELS} source panels"
        )
    if len(target_groups) < MINIMUM_CROSSED_LEVELS:
        raise ValueError(
            f"crossed nested CV requires at least {MINIMUM_CROSSED_LEVELS} target source groups"
        )

    expected_cells = {
        (source, target_motion, seed)
        for source in source_panels
        for target_motion in target_motions
        for seed in paired_seeds
    }
    missing_cells = sorted(expected_cells - seen_cells)
    if missing_cells:
        raise ValueError(
            "records must exactly cover source panels x target motions x paired seeds; "
            f"missing={missing_cells[:10]}"
        )

    headroom_by_target_seed: dict[tuple[str, int], float] = {}
    features_by_pair: dict[tuple[str, str], tuple[list[float], list[float]]] = {}
    for record in canonical:
        headroom_key = (record["target_motion_id"], record["paired_seed"])
        prior_headroom = headroom_by_target_seed.setdefault(
            headroom_key,
            record["independent_headroom"],
        )
        if record["independent_headroom"] != prior_headroom:
            raise ValueError(
                "independent_headroom must be invariant across source panels for each "
                f"target-motion/paired-seed unit: {headroom_key}"
            )

        feature_key = (record["source_panel_id"], record["target_motion_id"])
        feature_pair = (record["baseline_features"], record["failure_features"])
        prior_features = features_by_pair.setdefault(feature_key, feature_pair)
        if feature_pair != prior_features:
            raise ValueError(
                "baseline and failure features must be frozen across paired seeds for "
                f"source-panel/target-motion unit: {feature_key}"
            )

    return (
        canonical,
        source_panels,
        target_motions,
        target_groups,
        paired_seeds,
        dict(sorted(target_motion_to_group.items())),
    )


def _cell_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_panel_id": record["source_panel_id"],
        "target_motion_id": record["target_motion_id"],
        "target_source_group_id": record["target_source_group_id"],
        "paired_seed": record["paired_seed"],
    }


def _crossed_masks(
    records: Sequence[Mapping[str, Any]],
    *,
    held_source: str,
    held_target_group: str,
) -> tuple[BoolArray, BoolArray, BoolArray]:
    source_match = np.asarray(
        [record["source_panel_id"] == held_source for record in records],
        dtype=np.bool_,
    )
    target_match = np.asarray(
        [record["target_source_group_id"] == held_target_group for record in records],
        dtype=np.bool_,
    )
    test = source_match & target_match
    train = ~source_match & ~target_match
    embargo = ~(train | test)
    if np.any(train & test) or np.any(train & embargo) or np.any(test & embargo):
        raise RuntimeError("crossed fold masks overlap")
    if not np.all(train | test | embargo):
        raise RuntimeError("crossed fold masks do not partition records")
    return train, test, embargo


def _standardize_training_features(
    features: FloatArray,
    *,
    scale_tolerance: float,
    context: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError(
            f"degenerate {context}: training feature matrix has shape {features.shape}"
        )
    mean = np.mean(features, axis=0, dtype=np.float64)
    scale = np.std(features, axis=0, dtype=np.float64)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError(f"degenerate {context}: feature moments are non-finite")
    degenerate = np.flatnonzero(scale <= scale_tolerance)
    if degenerate.size:
        raise ValueError(
            f"degenerate {context}: feature columns have training-only scale <= "
            f"{scale_tolerance}: {degenerate.tolist()}"
        )
    return (features - mean) / scale, mean, scale


def _fit_ridge(
    features: FloatArray,
    outcomes: FloatArray,
    *,
    alpha: float,
    scale_tolerance: float,
    context: str,
) -> dict[str, FloatArray | float]:
    standardized, mean, scale = _standardize_training_features(
        features,
        scale_tolerance=scale_tolerance,
        context=context,
    )
    if outcomes.ndim != 1 or outcomes.shape[0] != features.shape[0]:
        raise ValueError(f"degenerate {context}: outcome shape does not match features")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError(f"degenerate {context}: outcomes are non-finite")

    design = np.column_stack([np.ones(features.shape[0], dtype=np.float64), standardized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    try:
        coefficients = np.linalg.solve(
            design.T @ design + penalty,
            design.T @ outcomes,
        )
    except np.linalg.LinAlgError as exc:  # pragma: no cover - positive ridge is defensive
        raise ValueError(f"degenerate {context}: ridge system is singular") from exc
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"degenerate {context}: ridge coefficients are non-finite")
    return {
        "feature_mean": mean,
        "feature_scale": scale,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
    }


def _predict_ridge(model: Mapping[str, Any], features: FloatArray) -> FloatArray:
    standardized = (features - model["feature_mean"]) / model["feature_scale"]
    predictions = model["intercept"] + standardized @ model["coefficients"]
    if not np.all(np.isfinite(predictions)):
        raise ValueError("ridge predictions are non-finite")
    return np.asarray(predictions, dtype=np.float64)


def _model_manifest(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_mean": [float(value) for value in model["feature_mean"]],
        "feature_scale": [float(value) for value in model["feature_scale"]],
        "intercept": float(model["intercept"]),
        "standardized_coefficients": [float(value) for value in model["coefficients"]],
    }


def _cluster_weighted_mean(
    values: FloatArray,
    records: Sequence[Mapping[str, Any]],
) -> float:
    """Average motions within panel/group/seed, then weight all clusters equally."""

    if values.ndim != 1 or len(values) != len(records):
        raise ValueError("cluster-weighted values must align exactly with records")
    clusters: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for value, record in zip(values, records, strict=True):
        if not math.isfinite(float(value)):
            raise ValueError("cluster-weighted values must be finite")
        key = (
            record["source_panel_id"],
            record["target_source_group_id"],
            record["paired_seed"],
        )
        clusters[key].append(float(value))
    if not clusters:
        raise ValueError("cluster-weighted aggregation requires non-empty clusters")
    cluster_means = [math.fsum(items) / len(items) for _, items in sorted(clusters.items())]
    return float(math.fsum(cluster_means) / len(cluster_means))


def _select_alpha_nested_crossed(
    features: FloatArray,
    outcomes: FloatArray,
    records: Sequence[Mapping[str, Any]],
    *,
    alpha_grid: tuple[float, ...],
    scale_tolerance: float,
    context: str,
) -> tuple[float, list[dict[str, float]], dict[str, Any]]:
    sources = tuple(sorted({record["source_panel_id"] for record in records}))
    target_groups = tuple(sorted({record["target_source_group_id"] for record in records}))
    paired_seeds = tuple(sorted({record["paired_seed"] for record in records}))
    if len(sources) < 3 or len(target_groups) < 3:
        raise ValueError(
            f"degenerate {context}: nested crossed CV needs at least three training "
            "source-panel and target-group units"
        )

    errors = np.full((len(alpha_grid), len(outcomes)), np.nan, dtype=np.float64)
    train_counts: list[int] = []
    validation_counts: list[int] = []
    for held_source in sources:
        for held_target_group in target_groups:
            train, validation, _ = _crossed_masks(
                records,
                held_source=held_source,
                held_target_group=held_target_group,
            )
            if not np.any(train) or not np.any(validation):
                raise ValueError(f"degenerate {context}: empty inner crossed fold")
            target_motion_count = len(
                {
                    record["target_motion_id"]
                    for record in records
                    if record["target_source_group_id"] == held_target_group
                }
            )
            expected_validation = target_motion_count * len(paired_seeds)
            remaining_motion_count = len(
                {
                    record["target_motion_id"]
                    for record in records
                    if record["target_source_group_id"] != held_target_group
                }
            )
            expected_train = (len(sources) - 1) * remaining_motion_count * len(paired_seeds)
            if int(validation.sum()) != expected_validation or int(train.sum()) != expected_train:
                raise ValueError(f"degenerate {context}: inner crossed fold support mismatch")
            train_counts.append(int(train.sum()))
            validation_counts.append(int(validation.sum()))

            for alpha_index, alpha in enumerate(alpha_grid):
                model = _fit_ridge(
                    features[train],
                    outcomes[train],
                    alpha=alpha,
                    scale_tolerance=scale_tolerance,
                    context=f"{context}/inner/{held_source}/{held_target_group}",
                )
                predictions = _predict_ridge(model, features[validation])
                errors[alpha_index, validation] = (outcomes[validation] - predictions) ** 2

    if np.any(~np.isfinite(errors)):
        raise RuntimeError(f"{context}: inner validation did not predict every training record")
    scores = []
    for alpha_index, alpha in enumerate(alpha_grid):
        loss = _cluster_weighted_mean(errors[alpha_index], records)
        scores.append({"alpha": alpha, "cluster_weighted_mse": loss})
    selected = min(scores, key=lambda score: (score["cluster_weighted_mse"], score["alpha"]))
    return (
        selected["alpha"],
        scores,
        {
            "fold_count": len(sources) * len(target_groups),
            "training_record_count_min": min(train_counts),
            "training_record_count_max": max(train_counts),
            "validation_record_count_min": min(validation_counts),
            "validation_record_count_max": max(validation_counts),
            "variable_target_group_motion_counts_allowed": True,
        },
    )


def _fit_oos_model(
    records: Sequence[Mapping[str, Any]],
    features: FloatArray,
    outcomes: FloatArray,
    *,
    alpha_grid: tuple[float, ...],
    scale_tolerance: float,
    model_name: str,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    source_panels = tuple(sorted({record["source_panel_id"] for record in records}))
    target_groups = tuple(sorted({record["target_source_group_id"] for record in records}))
    target_motions = tuple(sorted({record["target_motion_id"] for record in records}))
    paired_seeds = tuple(sorted({record["paired_seed"] for record in records}))
    if len(source_panels) < MINIMUM_CROSSED_LEVELS or len(target_groups) < MINIMUM_CROSSED_LEVELS:
        raise ValueError(
            f"{model_name} requires at least {MINIMUM_CROSSED_LEVELS} source panels and "
            "target source groups"
        )
    if features.shape[0] != len(records) or outcomes.shape != (len(records),):
        raise ValueError(f"{model_name} features/outcomes do not align with records")

    predictions = np.full(len(records), np.nan, dtype=np.float64)
    fold_manifests: list[dict[str, Any]] = []
    total_count = len(records)
    for source_index, held_source in enumerate(source_panels):
        for target_index, held_target_group in enumerate(target_groups):
            fold_id = f"outer_s{source_index:03d}_g{target_index:03d}"
            train, test, embargo = _crossed_masks(
                records,
                held_source=held_source,
                held_target_group=held_target_group,
            )
            held_motion_count = len(
                {
                    record["target_motion_id"]
                    for record in records
                    if record["target_source_group_id"] == held_target_group
                }
            )
            expected_test = held_motion_count * len(paired_seeds)
            expected_train = (
                (len(source_panels) - 1)
                * (len(target_motions) - held_motion_count)
                * len(paired_seeds)
            )
            expected_embargo = total_count - expected_test - expected_train
            if (
                int(test.sum()) != expected_test
                or int(train.sum()) != expected_train
                or int(embargo.sum()) != expected_embargo
            ):
                raise ValueError(f"degenerate {fold_id}: outer crossed support mismatch")
            if any(
                record["source_panel_id"] == held_source
                or record["target_source_group_id"] == held_target_group
                for record, selected in zip(records, train, strict=True)
                if selected
            ):
                raise ValueError(f"leakage in {fold_id}: held unit appears in training")
            if any(
                record["source_panel_id"] != held_source
                or record["target_source_group_id"] != held_target_group
                for record, selected in zip(records, test, strict=True)
                if selected
            ):
                raise ValueError(f"leakage in {fold_id}: test is not the held intersection")

            train_records = [
                record for record, selected in zip(records, train, strict=True) if selected
            ]
            selected_alpha, inner_scores, inner_balance = _select_alpha_nested_crossed(
                features[train],
                outcomes[train],
                train_records,
                alpha_grid=alpha_grid,
                scale_tolerance=scale_tolerance,
                context=f"{fold_id}/{model_name}",
            )
            model = _fit_ridge(
                features[train],
                outcomes[train],
                alpha=selected_alpha,
                scale_tolerance=scale_tolerance,
                context=f"{fold_id}/{model_name}/outer_refit",
            )
            if np.any(np.isfinite(predictions[test])):
                raise RuntimeError(f"{model_name} predicted an outer cell more than once")
            predictions[test] = _predict_ridge(model, features[test])

            training_cells = [
                _cell_payload(record)
                for record, selected in zip(records, train, strict=True)
                if selected
            ]
            test_cells = [
                _cell_payload(record)
                for record, selected in zip(records, test, strict=True)
                if selected
            ]
            fold_manifests.append(
                {
                    "fold_id": fold_id,
                    "held_source_panel_id": held_source,
                    "held_target_source_group_id": held_target_group,
                    "held_target_motion_count": held_motion_count,
                    "training_record_count": int(train.sum()),
                    "test_record_count": int(test.sum()),
                    "embargo_record_count": int(embargo.sum()),
                    "training_cells_sha256": canonical_sha256({"cells": training_cells}),
                    "test_cells_sha256": canonical_sha256({"cells": test_cells}),
                    "selected_alpha": selected_alpha,
                    "inner_cv_scores": inner_scores,
                    "inner_cv_balance": inner_balance,
                    "outer_refit": _model_manifest(model),
                }
            )

    if np.any(~np.isfinite(predictions)):
        raise RuntimeError(f"{model_name} did not produce exactly one OOS prediction per record")
    return predictions, fold_manifests


def _merge_fold_manifests(
    baseline_folds: Sequence[Mapping[str, Any]],
    augmented_folds: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(baseline_folds) != len(augmented_folds):
        raise RuntimeError("baseline and augmented outer fold counts differ")
    merged = []
    structural_fields = (
        "fold_id",
        "held_source_panel_id",
        "held_target_source_group_id",
        "held_target_motion_count",
        "training_record_count",
        "test_record_count",
        "embargo_record_count",
        "training_cells_sha256",
        "test_cells_sha256",
    )
    for baseline, augmented in zip(baseline_folds, augmented_folds, strict=True):
        if any(baseline[field] != augmented[field] for field in structural_fields):
            raise RuntimeError("baseline and augmented outer fold structure differs")
        merged.append(
            {
                **{field: baseline[field] for field in structural_fields},
                "baseline": {
                    key: baseline[key]
                    for key in (
                        "selected_alpha",
                        "inner_cv_scores",
                        "inner_cv_balance",
                        "outer_refit",
                    )
                },
                "baseline_plus_failure": {
                    key: augmented[key]
                    for key in (
                        "selected_alpha",
                        "inner_cv_scores",
                        "inner_cv_balance",
                        "outer_refit",
                    )
                },
            }
        )
    return merged


def _loss_summaries(
    records: Sequence[Mapping[str, Any]],
    baseline_squared_errors: FloatArray,
    augmented_squared_errors: FloatArray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    improvements = baseline_squared_errors - augmented_squared_errors
    clustered: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        clustered[
            (
                record["source_panel_id"],
                record["target_source_group_id"],
                record["paired_seed"],
            )
        ].append(index)

    cluster_summaries = []
    for (source_panel, target_group, paired_seed), indices in sorted(clustered.items()):
        cluster_summaries.append(
            {
                "source_panel_id": source_panel,
                "target_source_group_id": target_group,
                "paired_seed": paired_seed,
                "target_motion_count": len(indices),
                "baseline_mse": float(np.mean(baseline_squared_errors[indices], dtype=np.float64)),
                "baseline_plus_failure_mse": float(
                    np.mean(augmented_squared_errors[indices], dtype=np.float64)
                ),
                "mean_paired_loss_improvement": float(
                    np.mean(improvements[indices], dtype=np.float64)
                ),
            }
        )

    panel_seed_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for summary in cluster_summaries:
        panel_seed_groups[(summary["source_panel_id"], summary["paired_seed"])].append(summary)
    panel_seed_summaries = []
    for (source_panel, paired_seed), summaries in sorted(panel_seed_groups.items()):
        panel_seed_summaries.append(
            {
                "source_panel_id": source_panel,
                "paired_seed": paired_seed,
                "target_source_group_count": len(summaries),
                "mean_paired_loss_improvement": float(
                    math.fsum(item["mean_paired_loss_improvement"] for item in summaries)
                    / len(summaries)
                ),
            }
        )

    baseline_loss = math.fsum(item["baseline_mse"] for item in cluster_summaries) / len(
        cluster_summaries
    )
    augmented_loss = math.fsum(
        item["baseline_plus_failure_mse"] for item in cluster_summaries
    ) / len(cluster_summaries)
    improvement = math.fsum(
        item["mean_paired_loss_improvement"] for item in cluster_summaries
    ) / len(cluster_summaries)
    metrics = {
        "inference_clusters": [
            "source_panel_id",
            "global_paired_seed",
            "target_source_group_id",
        ],
        "individual_motion_cell_inference_permitted": False,
        "aggregation": (
            "mean_motions_within_panel_x_target_group_x_seed_then_equal_weight_clusters"
        ),
        "baseline_cluster_weighted_mse": float(baseline_loss),
        "baseline_plus_failure_cluster_weighted_mse": float(augmented_loss),
        "paired_loss_improvement": float(improvement),
        "positive_means_failure_features_improve_oos_loss": True,
    }
    return cluster_summaries, panel_seed_summaries, metrics


def _run_outcome_analysis(
    records: Sequence[Mapping[str, Any]],
    *,
    outcome_field: str,
    outcome_name: str,
    alpha_grid: tuple[float, ...],
    scale_tolerance: float,
) -> dict[str, Any]:
    baseline_features = np.asarray(
        [record["baseline_features"] for record in records],
        dtype=np.float64,
    )
    failure_features = np.asarray(
        [record["failure_features"] for record in records],
        dtype=np.float64,
    )
    augmented_features = np.concatenate([baseline_features, failure_features], axis=1)
    outcomes = np.asarray([record[outcome_field] for record in records], dtype=np.float64)

    baseline_predictions, baseline_folds = _fit_oos_model(
        records,
        baseline_features,
        outcomes,
        alpha_grid=alpha_grid,
        scale_tolerance=scale_tolerance,
        model_name=f"{outcome_name}/baseline",
    )
    augmented_predictions, augmented_folds = _fit_oos_model(
        records,
        augmented_features,
        outcomes,
        alpha_grid=alpha_grid,
        scale_tolerance=scale_tolerance,
        model_name=f"{outcome_name}/baseline_plus_failure",
    )
    baseline_errors = (outcomes - baseline_predictions) ** 2
    augmented_errors = (outcomes - augmented_predictions) ** 2
    if np.any(~np.isfinite(baseline_errors)) or np.any(~np.isfinite(augmented_errors)):
        raise ValueError(f"{outcome_name} outer squared losses are non-finite")

    predictions = []
    for index, record in enumerate(records):
        predictions.append(
            {
                **_cell_payload(record),
                "gain": record["gain"],
                "independent_headroom": record["independent_headroom"],
                "outcome": float(outcomes[index]),
                "baseline_prediction": float(baseline_predictions[index]),
                "baseline_plus_failure_prediction": float(augmented_predictions[index]),
                "baseline_squared_error": float(baseline_errors[index]),
                "baseline_plus_failure_squared_error": float(augmented_errors[index]),
                "paired_loss_improvement": float(baseline_errors[index] - augmented_errors[index]),
            }
        )
    cluster_summaries, panel_seed_summaries, metrics = _loss_summaries(
        records,
        baseline_errors,
        augmented_errors,
    )
    return {
        "status": "estimated",
        "outcome_name": outcome_name,
        "outcome_field": outcome_field,
        "record_count": len(records),
        "target_motion_ids": sorted({record["target_motion_id"] for record in records}),
        "target_source_group_ids": sorted({record["target_source_group_id"] for record in records}),
        "outer_folds": _merge_fold_manifests(baseline_folds, augmented_folds),
        "predictions": predictions,
        "cluster_loss_improvements": cluster_summaries,
        "panel_seed_loss_improvements": panel_seed_summaries,
        "metrics": metrics,
    }


def analyze_transfer_predictiveness(
    records: Iterable[Mapping[str, Any]],
    *,
    baseline_feature_names: Sequence[str],
    failure_feature_names: Sequence[str],
    baseline_feature_provenance_sha256: str,
    failure_feature_provenance_sha256: str,
    gain_evaluation_artifact_sha256: str,
    headroom_evaluation_artifact_sha256: str,
    gain_rollout_seeds: Sequence[int],
    headroom_rollout_seeds: Sequence[int],
    h_min: float,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    feature_scale_tolerance: float = DEFAULT_FEATURE_SCALE_TOLERANCE,
) -> dict[str, Any]:
    """Run v2 raw-gain primary and normalized-headroom sensitivity analyses.

    The gain and headroom artifacts must be distinct and bind disjoint rollout
    seed sets.  Motions with any paired-seed headroom below ``h_min`` remain in
    the raw primary analysis but are excluded as whole target motions from the
    normalized sensitivity; no denominator clipping is performed.
    """

    baseline_names = _feature_names(baseline_feature_names, "baseline_feature_names")
    failure_names = _feature_names(failure_feature_names, "failure_feature_names")
    baseline_provenance = _sha256(
        baseline_feature_provenance_sha256,
        "baseline_feature_provenance_sha256",
    )
    failure_provenance = _sha256(
        failure_feature_provenance_sha256,
        "failure_feature_provenance_sha256",
    )
    gain_artifact = _sha256(
        gain_evaluation_artifact_sha256,
        "gain_evaluation_artifact_sha256",
    )
    headroom_artifact = _sha256(
        headroom_evaluation_artifact_sha256,
        "headroom_evaluation_artifact_sha256",
    )
    if gain_artifact == headroom_artifact:
        raise ValueError("gain and headroom evaluation artifact hashes must be distinct")
    gain_seeds = _seed_set(gain_rollout_seeds, "gain_rollout_seeds")
    headroom_seeds = _seed_set(headroom_rollout_seeds, "headroom_rollout_seeds")
    seed_overlap = sorted(set(gain_seeds) & set(headroom_seeds))
    if seed_overlap:
        raise ValueError(
            "gain and headroom rollout seed sets must be disjoint; " f"overlap={seed_overlap}"
        )
    minimum_headroom = _real(h_min, "h_min", positive=True)
    alphas = _alpha_grid(alpha_grid)
    scale_tolerance = _real(
        feature_scale_tolerance,
        "feature_scale_tolerance",
        positive=True,
    )

    (
        canonical,
        source_panels,
        target_motions,
        target_groups,
        paired_seeds,
        target_motion_to_group,
    ) = _materialize_records(
        records,
        baseline_feature_names=baseline_names,
        failure_feature_names=failure_names,
    )

    primary_raw = _run_outcome_analysis(
        canonical,
        outcome_field="gain",
        outcome_name="primary_raw_gain",
        alpha_grid=alphas,
        scale_tolerance=scale_tolerance,
    )

    minimum_headroom_by_motion = {
        motion: min(
            record["independent_headroom"]
            for record in canonical
            if record["target_motion_id"] == motion
        )
        for motion in target_motions
    }
    unsupported_motions = tuple(
        motion for motion in target_motions if minimum_headroom_by_motion[motion] < minimum_headroom
    )
    supported_motion_set = set(target_motions) - set(unsupported_motions)
    supported_records = [
        record for record in canonical if record["target_motion_id"] in supported_motion_set
    ]
    supported_groups = sorted({record["target_source_group_id"] for record in supported_records})
    unsupported_manifest = [
        {
            "target_motion_id": motion,
            "target_source_group_id": target_motion_to_group[motion],
            "minimum_independent_headroom": minimum_headroom_by_motion[motion],
            "reason": "minimum_headroom_below_h_min",
        }
        for motion in unsupported_motions
    ]
    if len(supported_groups) < MINIMUM_CROSSED_LEVELS:
        normalized_sensitivity: dict[str, Any] = {
            "status": "not_estimable",
            "outcome_name": "normalized_headroom_sensitivity",
            "reason": ("fewer_than_four_target_source_groups_remain_after_h_min_filter"),
            "h_min": minimum_headroom,
            "denominator_clipping": None,
            "supported_target_motion_ids": sorted(supported_motion_set),
            "unsupported_targets": unsupported_manifest,
            "supported_target_source_group_count": len(supported_groups),
        }
    else:
        normalized_sensitivity = _run_outcome_analysis(
            supported_records,
            outcome_field="normalized_gain",
            outcome_name="normalized_headroom_sensitivity",
            alpha_grid=alphas,
            scale_tolerance=scale_tolerance,
        )
        normalized_sensitivity.update(
            {
                "h_min": minimum_headroom,
                "denominator_clipping": None,
                "supported_target_motion_ids": sorted(supported_motion_set),
                "unsupported_targets": unsupported_manifest,
            }
        )

    input_payload = {
        "record_fields": sorted(TRANSFER_FIELDS),
        "records": canonical,
        "baseline_feature_names": list(baseline_names),
        "failure_feature_names": list(failure_names),
        "baseline_feature_provenance_sha256": baseline_provenance,
        "failure_feature_provenance_sha256": failure_provenance,
        "gain_evaluation_artifact_sha256": gain_artifact,
        "headroom_evaluation_artifact_sha256": headroom_artifact,
        "gain_rollout_seeds": list(gain_seeds),
        "headroom_rollout_seeds": list(headroom_seeds),
        "h_min": minimum_headroom,
    }
    result: dict[str, Any] = {
        "kind": TRANSFER_ANALYSIS_KIND,
        "schema_version": TRANSFER_ANALYSIS_SCHEMA_VERSION,
        "input_sha256": canonical_sha256(input_payload),
        "canonical_records": canonical,
        "feature_schemas": {
            "baseline": {
                "names": list(baseline_names),
                "dimension": len(baseline_names),
                "frozen": True,
                "provenance_sha256": baseline_provenance,
            },
            "failure": {
                "names": list(failure_names),
                "dimension": len(failure_names),
                "frozen": True,
                "provenance_sha256": failure_provenance,
            },
        },
        "evaluation_provenance": {
            "gain_evaluation_artifact_sha256": gain_artifact,
            "headroom_evaluation_artifact_sha256": headroom_artifact,
            "gain_rollout_seeds": list(gain_seeds),
            "headroom_rollout_seeds": list(headroom_seeds),
            "rollout_seed_sets_disjoint": True,
        },
        "levels": {
            "source_panel_ids": list(source_panels),
            "target_motion_ids": list(target_motions),
            "target_source_group_ids": list(target_groups),
            "paired_seeds": list(paired_seeds),
            "target_motion_to_source_group": target_motion_to_group,
        },
        "cross_validation": {
            "outer_strategy": "leave_one_source_panel_x_leave_one_target_group_crossed_v2",
            "outer_training_rule": "exclude_records_containing_either_held_unit",
            "outer_test_rule": "all_motions_in_held_panel_x_target_group_intersection",
            "inner_strategy": "same_crossed_group_rule_within_outer_training_only",
            "target_motion_predictions": True,
            "minimum_source_and_target_group_levels": MINIMUM_CROSSED_LEVELS,
            "alpha_grid": list(alphas),
            "alpha_selection": "minimum_cluster_weighted_inner_mse_then_smallest_alpha",
            "feature_standardization": "training_mean_and_population_sd_per_inner_or_outer_fit",
            "feature_scale_tolerance": scale_tolerance,
            "ridge_objective": "sum_squared_error + alpha * squared_l2_standardized_coefficients",
            "intercept_penalized": False,
        },
        "primary_raw": primary_raw,
        "normalized_headroom_sensitivity": normalized_sensitivity,
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def _validated_analysis_outcome(
    analysis: Mapping[str, Any],
    outcome: str,
) -> Mapping[str, Any]:
    if not isinstance(analysis, Mapping):
        raise ValueError("analysis must be a transfer-analysis mapping")
    if analysis.get("kind") != TRANSFER_ANALYSIS_KIND:
        raise ValueError(f"analysis kind must be {TRANSFER_ANALYSIS_KIND!r}")
    if analysis.get("schema_version") != TRANSFER_ANALYSIS_SCHEMA_VERSION:
        raise ValueError("unsupported transfer-analysis schema version")
    digest = analysis.get("analysis_sha256")
    if not isinstance(digest, str) or digest != canonical_sha256(
        analysis,
        digest_field="analysis_sha256",
    ):
        raise ValueError("analysis_sha256 mismatch")
    outcome_fields = {
        "primary_raw": "primary_raw",
        "normalized_headroom_sensitivity": "normalized_headroom_sensitivity",
    }
    if outcome not in outcome_fields:
        raise ValueError(f"outcome must be one of {sorted(outcome_fields)}")
    result = analysis.get(outcome_fields[outcome])
    if not isinstance(result, Mapping) or result.get("status") != "estimated":
        raise ValueError(f"analysis outcome {outcome!r} is not estimable")
    return result


def bootstrap_clustered_loss_improvement(
    analysis: Mapping[str, Any],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    outcome: str = "primary_raw",
) -> dict[str, Any]:
    """Bootstrap intact source panels, global paired seeds, and target groups.

    Motion-level losses are averaged within each panel/group/seed cluster by the
    analysis.  Each bootstrap draw independently samples the source-panel and
    target-group axes, while paired seed IDs are sampled globally and therefore
    remain synchronized across every sampled panel and target group.
    """

    outcome_result = _validated_analysis_outcome(analysis, outcome)
    rng_seed = _seed(seed, "seed")
    resample_count = _positive_integer(resamples, "resamples", minimum=2)
    confidence = _real(confidence_level, "confidence_level", positive=True)
    if confidence >= 1.0:
        raise ValueError("confidence_level must be less than one")

    levels = analysis.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("analysis levels are missing")
    source_panels = tuple(levels.get("source_panel_ids", ()))
    target_groups = tuple(
        sorted(
            {item["target_source_group_id"] for item in outcome_result["cluster_loss_improvements"]}
        )
    )
    paired_seeds = tuple(levels.get("paired_seeds", ()))
    if (
        len(source_panels) < 2
        or len(target_groups) < 2
        or len(paired_seeds) < 1
        or len(set(source_panels)) != len(source_panels)
        or len(set(target_groups)) != len(target_groups)
        or len(set(paired_seeds)) != len(paired_seeds)
    ):
        raise ValueError("analysis has degenerate cluster levels")

    summaries = outcome_result.get("cluster_loss_improvements")
    if not isinstance(summaries, list):
        raise ValueError("analysis cluster_loss_improvements are missing")
    effect_by_cluster: dict[tuple[str, str, int], float] = {}
    for index, summary in enumerate(summaries):
        if not isinstance(summary, Mapping):
            raise ValueError(f"cluster_loss_improvements[{index}] must be a mapping")
        key = (
            _identifier(summary.get("source_panel_id"), f"cluster[{index}].source_panel_id"),
            _identifier(
                summary.get("target_source_group_id"),
                f"cluster[{index}].target_source_group_id",
            ),
            _seed(summary.get("paired_seed"), f"cluster[{index}].paired_seed"),
        )
        if key in effect_by_cluster:
            raise ValueError(f"duplicate loss cluster: {key}")
        effect_by_cluster[key] = _real(
            summary.get("mean_paired_loss_improvement"),
            f"cluster[{index}].mean_paired_loss_improvement",
        )
    expected = {
        (source, target, paired_seed)
        for source in source_panels
        for target in target_groups
        for paired_seed in paired_seeds
    }
    if set(effect_by_cluster) != expected:
        raise ValueError("loss summaries do not exactly cover panel x target-group x paired-seed")

    effect_tensor = np.empty(
        (len(source_panels), len(target_groups), len(paired_seeds)),
        dtype=np.float64,
    )
    for source_index, source in enumerate(source_panels):
        for target_index, target in enumerate(target_groups):
            for seed_index, paired_seed in enumerate(paired_seeds):
                effect_tensor[source_index, target_index, seed_index] = effect_by_cluster[
                    (source, target, paired_seed)
                ]
    observed = float(np.mean(effect_tensor, dtype=np.float64))
    reported = outcome_result.get("metrics", {}).get("paired_loss_improvement")
    if (
        isinstance(reported, (bool, np.bool_))
        or not isinstance(reported, Real)
        or not math.isclose(
            observed,
            float(reported),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("cluster effects do not match reported paired loss improvement")

    rng = np.random.Generator(np.random.PCG64(rng_seed))
    bootstrap_means = np.empty(resample_count, dtype=np.float64)
    for draw_index in range(resample_count):
        sampled_sources = rng.integers(0, len(source_panels), size=len(source_panels))
        sampled_targets = rng.integers(0, len(target_groups), size=len(target_groups))
        sampled_seeds = rng.integers(0, len(paired_seeds), size=len(paired_seeds))
        bootstrap_means[draw_index] = np.mean(
            effect_tensor[np.ix_(sampled_sources, sampled_targets, sampled_seeds)],
            dtype=np.float64,
        )
    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail, 1.0 - tail],
        method="linear",
    )
    result: dict[str, Any] = {
        "kind": CLUSTER_BOOTSTRAP_KIND,
        "schema_version": CLUSTER_BOOTSTRAP_SCHEMA_VERSION,
        "analysis_sha256": analysis["analysis_sha256"],
        "outcome": outcome,
        "resampling_design": ("crossed_source_panel_x_global_paired_seed_x_target_source_group"),
        "motion_cell_resampling": False,
        "source_panel_resampling": True,
        "global_paired_seed_resampling": True,
        "target_source_group_resampling": True,
        "source_panel_count": len(source_panels),
        "paired_seed_count": len(paired_seeds),
        "target_source_group_count": len(target_groups),
        "observed_paired_loss_improvement": observed,
        "seed": rng_seed,
        "rng": "numpy.PCG64",
        "resamples": resample_count,
        "confidence_level": confidence,
        "interval_method": "percentile_linear_quantile",
        "confidence_interval": [float(lower), float(upper)],
        "bootstrap_standard_error": float(np.std(bootstrap_means, ddof=1, dtype=np.float64)),
        "bootstrap_distribution_sha256": canonical_sha256(
            {"bootstrap_means": [float(value) for value in bootstrap_means]}
        ),
    }
    result["bootstrap_sha256"] = canonical_sha256(result)
    return result


def bootstrap_source_panel_loss_improvement(
    analysis: Mapping[str, Any],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    outcome: str = "primary_raw",
) -> dict[str, Any]:
    """Compatibility name for the v2 three-axis clustered bootstrap."""

    return bootstrap_clustered_loss_improvement(
        analysis,
        seed=seed,
        resamples=resamples,
        confidence_level=confidence_level,
        outcome=outcome,
    )


def _permuted_failure_features(
    records: Sequence[Mapping[str, Any]],
    source_panels: tuple[str, ...],
    donor_order: tuple[str, ...],
) -> FloatArray:
    if len(source_panels) != len(donor_order) or set(source_panels) != set(donor_order):
        raise ValueError("source-row permutation must be a bijection over source panels")
    donor_for_source = dict(zip(source_panels, donor_order, strict=True))
    feature_by_source_motion: dict[tuple[str, str], list[float]] = {}
    for record in records:
        feature_by_source_motion.setdefault(
            (record["source_panel_id"], record["target_motion_id"]),
            record["failure_features"],
        )
    return np.asarray(
        [
            feature_by_source_motion[
                (donor_for_source[record["source_panel_id"]], record["target_motion_id"])
            ]
            for record in records
        ],
        dtype=np.float64,
    )


def permute_whole_source_failure_rows(
    analysis: Mapping[str, Any],
    *,
    method: str,
    seed: int = 0,
    monte_carlo_permutations: int = 999,
) -> dict[str, Any]:
    """Test failure-feature alignment by permuting complete source rows.

    Baseline predictions are held fixed because they do not use failure
    features.  For every permutation, the augmented model repeats all crossed
    outer fits, training-only standardization, nested alpha selection, and outer
    refits.  The identity mapping is evaluated explicitly.
    """

    primary = _validated_analysis_outcome(analysis, "primary_raw")
    if method not in {"exact", "monte_carlo"}:
        raise ValueError("method must be 'exact' or 'monte_carlo'")
    levels = analysis.get("levels")
    records = analysis.get("canonical_records")
    cross_validation = analysis.get("cross_validation")
    if (
        not isinstance(levels, Mapping)
        or not isinstance(records, list)
        or not isinstance(cross_validation, Mapping)
    ):
        raise ValueError("analysis is missing permutation inputs")
    source_panels = tuple(levels.get("source_panel_ids", ()))
    if len(source_panels) < 2 or len(set(source_panels)) != len(source_panels):
        raise ValueError("analysis source panels are degenerate")
    if method == "exact" and len(source_panels) > MAXIMUM_EXACT_PERMUTATION_PANELS:
        raise ValueError(
            f"exact permutation is limited to {MAXIMUM_EXACT_PERMUTATION_PANELS} source panels"
        )
    alphas = _alpha_grid(cross_validation.get("alpha_grid", ()))
    scale_tolerance = _real(
        cross_validation.get("feature_scale_tolerance"),
        "analysis feature_scale_tolerance",
        positive=True,
    )
    outcomes = np.asarray([record["gain"] for record in records], dtype=np.float64)
    baseline_features = np.asarray(
        [record["baseline_features"] for record in records],
        dtype=np.float64,
    )
    baseline_errors = np.asarray(
        [prediction["baseline_squared_error"] for prediction in primary["predictions"]],
        dtype=np.float64,
    )
    observed = _real(
        primary["metrics"]["paired_loss_improvement"],
        "observed paired_loss_improvement",
    )
    identity = tuple(source_panels)
    rng_seed = _seed(seed, "seed")
    if method == "exact":
        donor_orders: Iterable[tuple[str, ...]] = itertools.permutations(source_panels)
        requested_random_count = None
    else:
        random_count = _positive_integer(
            monte_carlo_permutations,
            "monte_carlo_permutations",
        )
        rng = np.random.Generator(np.random.PCG64(rng_seed))

        def random_nonidentity_orders() -> Iterable[tuple[str, ...]]:
            for _ in range(random_count):
                while True:
                    donor_order = tuple(
                        str(value) for value in rng.permutation(source_panels).tolist()
                    )
                    if donor_order != identity:
                        yield donor_order
                        break

        donor_orders = itertools.chain(
            [identity],
            random_nonidentity_orders(),
        )
        requested_random_count = random_count

    statistics: list[float] = []
    fit_audit: list[dict[str, Any]] = []
    for permutation_index, donor_order in enumerate(donor_orders):
        permuted_failure = _permuted_failure_features(records, source_panels, donor_order)
        augmented_features = np.concatenate([baseline_features, permuted_failure], axis=1)
        augmented_predictions, augmented_folds = _fit_oos_model(
            records,
            augmented_features,
            outcomes,
            alpha_grid=alphas,
            scale_tolerance=scale_tolerance,
            model_name=f"source_row_permutation_{permutation_index:06d}",
        )
        augmented_errors = (outcomes - augmented_predictions) ** 2
        statistic = _cluster_weighted_mean(
            baseline_errors - augmented_errors,
            records,
        )
        statistics.append(float(statistic))
        fit_audit.append(
            {
                "donor_order": list(donor_order),
                "statistic": float(statistic),
                "selected_alpha_sha256": canonical_sha256(
                    {"selected_alphas": [fold["selected_alpha"] for fold in augmented_folds]}
                ),
            }
        )

    if not statistics:
        raise RuntimeError("permutation generated no statistics")
    identity_statistic = statistics[0]
    if not math.isclose(identity_statistic, observed, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError(
            "identity permutation does not reproduce observed paired loss improvement"
        )
    numerical_tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    if method == "exact":
        exceedances = int(sum(value >= observed - numerical_tolerance for value in statistics))
        p_value = exceedances / len(statistics)
        p_rule = "exact_fraction_including_identity"
        random_draw_count = None
    else:
        random_statistics = statistics[1:]
        exceedances = int(
            sum(value >= observed - numerical_tolerance for value in random_statistics)
        )
        p_value = (1 + exceedances) / (1 + len(random_statistics))
        p_rule = "plus_one_one_sided_monte_carlo"
        random_draw_count = requested_random_count

    result: dict[str, Any] = {
        "kind": SOURCE_ROW_PERMUTATION_KIND,
        "schema_version": SOURCE_ROW_PERMUTATION_SCHEMA_VERSION,
        "analysis_sha256": analysis["analysis_sha256"],
        "outcome": "primary_raw",
        "method": method,
        "permutation_unit": "whole_source_panel_failure_feature_row",
        "target_motion_blocks_preserved": True,
        "paired_seed_blocks_preserved": True,
        "identity_included": True,
        "identity_position": 0,
        "source_panel_count": len(source_panels),
        "evaluated_permutation_count": len(statistics),
        "monte_carlo_random_draw_count": random_draw_count,
        "monte_carlo_sampling_frame": (
            None if method == "exact" else "uniform_with_replacement_over_nonidentity_permutations"
        ),
        "seed": None if method == "exact" else rng_seed,
        "rng": None if method == "exact" else "numpy.PCG64",
        "observed_paired_loss_improvement": observed,
        "identity_statistic": identity_statistic,
        "alternative": "failure_features_reduce_cluster_weighted_oos_mse",
        "exceedance_count": exceedances,
        "p_value_one_sided": float(p_value),
        "p_value_rule": p_rule,
        "minimum_achievable_p": (
            1.0 / len(statistics) if method == "exact" else 1.0 / (1 + len(statistics) - 1)
        ),
        "null_statistics": [float(value) for value in statistics],
        "permutation_fit_audit_sha256": canonical_sha256({"fits": fit_audit}),
    }
    result["permutation_sha256"] = canonical_sha256(result)
    return result
