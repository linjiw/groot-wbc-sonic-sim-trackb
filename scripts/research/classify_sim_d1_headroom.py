#!/usr/bin/env python3
"""Classify the preregistered SIM-D1 dataset-headroom gate.

The input is the ``metrics_eval.json`` written by SONIC's all-motion evaluation
callback.  The per-motion table must contain ``motion_keys``, ``mpjpe_g``,
``terminated``, and ``progress`` under ``eval/all_metrics_dict``.

The frozen defaults come from ``fable-next.md`` Phase 2.  A dataset passes iff:

1. MPJPE-G ``p90 - p10 >= 20`` OR ``p90 / p10 >= 1.5``;
2. at least 20% of motions have ``success == 0`` OR ``progress < 0.9``; and
3. at least 30% have ``success == 1`` AND complete progress.

Threshold flags exist for transparent sensitivity checks.  Changing any frozen
default marks the artifact ``exploratory``; it must not be reported as the
preregistered gate.  MPJPE-G values are used in their recorded units (no hidden
conversion).  The plan requires a frozen easiest-first ranking but does not
specify its formula, so this tool records its operational choice explicitly:
ascending MPJPE-G with lexical motion-key tie-breaking.

The output is schema-versioned and exposes a top-level ``ranking`` list, making
it directly consumable by ``summarize_sonic_logs.py --difficulty-ranking-json``.
A valid scientific FAIL returns exit status 0; malformed/incomplete input raises
an error and returns nonzero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

_SCHEMA_VERSION = 1
_TABLE_KEY = "eval/all_metrics_dict"

_DEFAULT_MPJPE_SPREAD_MIN = 20.0
_DEFAULT_MPJPE_RATIO_MIN = 1.5
_DEFAULT_HARD_FRACTION_MIN = 0.20
_DEFAULT_MASTERED_FRACTION_MIN = 0.30
_DEFAULT_INCOMPLETE_PROGRESS_BELOW = 0.90
_DEFAULT_COMPLETE_PROGRESS_MIN = 1.0


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _finite_number(value: Any, *, field: str, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field}[{index}] must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field}[{index}] must be finite")
    return result


def _numeric_vector(
    table: dict[str, Any], field: str, length: int, *, minimum: float | None = None
) -> list[float]:
    raw = table.get(field)
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"{_TABLE_KEY}.{field} must be a list of length {length}")
    values = [_finite_number(value, field=field, index=i) for i, value in enumerate(raw)]
    if minimum is not None and any(value < minimum for value in values):
        raise ValueError(f"{_TABLE_KEY}.{field} values must be >= {minimum}")
    return values


def _termination_vector(table: dict[str, Any], length: int) -> list[bool]:
    raw = table.get("terminated")
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError(f"{_TABLE_KEY}.terminated must be a list of length {length}")
    result: list[bool] = []
    for i, value in enumerate(raw):
        if isinstance(value, bool):
            result.append(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)) and value in (0, 1):
            result.append(bool(value))
        else:
            raise ValueError(f"terminated[{i}] must be boolean or numeric 0/1")
    return result


def _validate_thresholds(
    *,
    mpjpe_spread_min: float,
    mpjpe_ratio_min: float,
    hard_fraction_min: float,
    mastered_fraction_min: float,
    incomplete_progress_below: float,
    complete_progress_min: float,
) -> None:
    values = {
        "mpjpe_spread_min": mpjpe_spread_min,
        "mpjpe_ratio_min": mpjpe_ratio_min,
        "hard_fraction_min": hard_fraction_min,
        "mastered_fraction_min": mastered_fraction_min,
        "incomplete_progress_below": incomplete_progress_below,
        "complete_progress_min": complete_progress_min,
    }
    for name, value in values.items():
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if mpjpe_spread_min < 0:
        raise ValueError("mpjpe_spread_min must be >= 0")
    if mpjpe_ratio_min <= 0:
        raise ValueError("mpjpe_ratio_min must be > 0")
    for name, value in (
        ("hard_fraction_min", hard_fraction_min),
        ("mastered_fraction_min", mastered_fraction_min),
        ("incomplete_progress_below", incomplete_progress_below),
        ("complete_progress_min", complete_progress_min),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")


def _linear_percentile(values: Sequence[float], quantile: float) -> float:
    """Return the linearly interpolated percentile (NumPy's default method)."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_expected_coverage(
    motion_keys: list[str],
    *,
    expected_motion_count: int | None,
    expected_motion_keys: Sequence[str] | None,
) -> dict[str, Any]:
    if expected_motion_count is not None:
        if expected_motion_count <= 0:
            raise ValueError("expected_motion_count must be positive")
        if len(motion_keys) != expected_motion_count:
            raise ValueError(
                f"all-motion coverage mismatch: evaluated {len(motion_keys)}, "
                f"expected {expected_motion_count}"
            )

    exact_key_set_verified = expected_motion_keys is not None
    if expected_motion_keys is not None:
        expected = list(expected_motion_keys)
        if not expected or not all(isinstance(key, str) and key for key in expected):
            raise ValueError("expected_motion_keys must contain non-empty strings")
        if len(expected) != len(set(expected)):
            raise ValueError("expected_motion_keys contains duplicates")
        missing = sorted(set(expected) - set(motion_keys))
        unexpected = sorted(set(motion_keys) - set(expected))
        if missing or unexpected:
            raise ValueError(
                "all-motion key coverage mismatch: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

    independently_verified = expected_motion_count is not None or exact_key_set_verified
    record: dict[str, Any] = {
        "evaluated_motion_count": len(motion_keys),
        "expected_motion_count": expected_motion_count,
        "exact_key_set_verified": exact_key_set_verified,
        "all_motion_coverage_independently_verified": independently_verified,
    }
    if not independently_verified:
        record["warning"] = (
            "The table is structurally complete, but all-dataset coverage cannot be "
            "proved without --expected-motion-count or --expected-motion-keys-json."
        )
    return record


def classify_sim_d1(
    metrics_eval: dict[str, Any],
    *,
    dataset: str = "unspecified",
    dataset_kind: str = "unspecified",
    expected_motion_count: int | None = None,
    expected_motion_keys: Sequence[str] | None = None,
    mpjpe_spread_min: float = _DEFAULT_MPJPE_SPREAD_MIN,
    mpjpe_ratio_min: float = _DEFAULT_MPJPE_RATIO_MIN,
    hard_fraction_min: float = _DEFAULT_HARD_FRACTION_MIN,
    mastered_fraction_min: float = _DEFAULT_MASTERED_FRACTION_MIN,
    incomplete_progress_below: float = _DEFAULT_INCOMPLETE_PROGRESS_BELOW,
    complete_progress_min: float = _DEFAULT_COMPLETE_PROGRESS_MIN,
) -> dict[str, Any]:
    """Validate one all-motion eval and apply the SIM-D1 gate."""
    _validate_thresholds(
        mpjpe_spread_min=mpjpe_spread_min,
        mpjpe_ratio_min=mpjpe_ratio_min,
        hard_fraction_min=hard_fraction_min,
        mastered_fraction_min=mastered_fraction_min,
        incomplete_progress_below=incomplete_progress_below,
        complete_progress_min=complete_progress_min,
    )
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("dataset must be a non-empty string")
    if dataset_kind not in {"real", "synthetic", "sample_data", "unspecified"}:
        raise ValueError("dataset_kind must be real, synthetic, sample_data, or unspecified")

    table = metrics_eval.get(_TABLE_KEY)
    if not isinstance(table, dict):
        raise ValueError(f"metrics_eval has no {_TABLE_KEY!r} per-motion table")
    raw_keys = table.get("motion_keys")
    if not isinstance(raw_keys, list) or not raw_keys:
        raise ValueError(f"{_TABLE_KEY}.motion_keys must be a non-empty list")
    if not all(isinstance(key, str) and key for key in raw_keys):
        raise ValueError(f"{_TABLE_KEY}.motion_keys must contain non-empty strings")
    motion_keys = list(raw_keys)
    if len(motion_keys) != len(set(motion_keys)):
        raise ValueError(f"{_TABLE_KEY}.motion_keys contains duplicates")

    count = len(motion_keys)
    mpjpe_g = _numeric_vector(table, "mpjpe_g", count, minimum=0.0)
    progress = _numeric_vector(table, "progress", count, minimum=0.0)
    if any(value > 1.0 for value in progress):
        raise ValueError(f"{_TABLE_KEY}.progress values must be <= 1.0")
    terminated = _termination_vector(table, count)
    success = [not value for value in terminated]

    coverage = _validate_expected_coverage(
        motion_keys,
        expected_motion_count=expected_motion_count,
        expected_motion_keys=expected_motion_keys,
    )

    p10 = _linear_percentile(mpjpe_g, 0.10)
    p90 = _linear_percentile(mpjpe_g, 0.90)
    spread = p90 - p10
    ratio_evaluable = p10 > 0.0
    ratio = p90 / p10 if ratio_evaluable else None
    spread_condition = spread >= mpjpe_spread_min
    ratio_condition = ratio is not None and ratio >= mpjpe_ratio_min

    hard_mask = [
        (not motion_success) or motion_progress < incomplete_progress_below
        for motion_success, motion_progress in zip(success, progress, strict=True)
    ]
    mastered_mask = [
        motion_success and motion_progress >= complete_progress_min
        for motion_success, motion_progress in zip(success, progress, strict=True)
    ]
    hard_count = sum(hard_mask)
    mastered_count = sum(mastered_mask)
    hard_fraction = hard_count / count
    mastered_fraction = mastered_count / count

    gates = {
        "difficulty_spread": {
            "pass": spread_condition or ratio_condition,
            "logic": "p90_minus_p10 >= minimum OR p90_over_p10 >= minimum",
            "absolute_condition_pass": spread_condition,
            "ratio_condition_pass": ratio_condition,
            "ratio_evaluable": ratio_evaluable,
        },
        "frontier_exists": {
            "pass": hard_fraction >= hard_fraction_min,
            "logic": "fraction(success == 0 OR progress < threshold) >= minimum",
            "motion_count": hard_count,
            "fraction": hard_fraction,
        },
        "mastered_anchor_exists": {
            "pass": mastered_fraction >= mastered_fraction_min,
            "logic": "fraction(success == 1 AND progress >= threshold) >= minimum",
            "motion_count": mastered_count,
            "fraction": mastered_fraction,
        },
    }
    passed = all(gate["pass"] for gate in gates.values())
    failed_gates = [name for name, gate in gates.items() if not gate["pass"]]

    thresholds = {
        "mpjpe_g_p90_minus_p10_min": mpjpe_spread_min,
        "mpjpe_g_p90_over_p10_min": mpjpe_ratio_min,
        "frontier_motion_fraction_min": hard_fraction_min,
        "mastered_motion_fraction_min": mastered_fraction_min,
        "incomplete_progress_below": incomplete_progress_below,
        "complete_progress_min": complete_progress_min,
    }
    defaults = {
        "mpjpe_g_p90_minus_p10_min": _DEFAULT_MPJPE_SPREAD_MIN,
        "mpjpe_g_p90_over_p10_min": _DEFAULT_MPJPE_RATIO_MIN,
        "frontier_motion_fraction_min": _DEFAULT_HARD_FRACTION_MIN,
        "mastered_motion_fraction_min": _DEFAULT_MASTERED_FRACTION_MIN,
        "incomplete_progress_below": _DEFAULT_INCOMPLETE_PROGRESS_BELOW,
        "complete_progress_min": _DEFAULT_COMPLETE_PROGRESS_MIN,
    }
    preregistered_defaults_used = thresholds == defaults

    eligibility_blockers = list(failed_gates)
    if not preregistered_defaults_used:
        eligibility_blockers.append("thresholds_are_not_preregistered_defaults")
    if not coverage["all_motion_coverage_independently_verified"]:
        eligibility_blockers.append("all_motion_coverage_not_independently_verified")

    order = sorted(range(count), key=lambda i: (mpjpe_g[i], motion_keys[i]))
    ranking = [motion_keys[i] for i in order]
    ranking_entries = [
        {
            "rank": rank,
            "motion_key": motion_keys[i],
            "mpjpe_g": mpjpe_g[i],
            "success": int(success[i]),
            "progress": progress[i],
        }
        for rank, i in enumerate(order)
    ]

    return {
        "schema_version": _SCHEMA_VERSION,
        "kind": "sim_d1_headroom_classification",
        "dataset": dataset,
        "dataset_kind": dataset_kind,
        "classification_mode": (
            "preregistered" if preregistered_defaults_used else "exploratory_threshold_override"
        ),
        "preregistered_defaults_used": preregistered_defaults_used,
        "pass": passed,
        "verdict": "PASS" if passed else "FAIL",
        "eligible_for_effect_experiment": (
            passed
            and preregistered_defaults_used
            and coverage["all_motion_coverage_independently_verified"]
        ),
        "eligibility_blockers": eligibility_blockers,
        "failed_gates": failed_gates,
        "coverage": coverage,
        "thresholds": thresholds,
        "summary": {
            "motion_count": count,
            "mpjpe_g_units": "as_recorded",
            "percentile_method": "linear",
            "mpjpe_g_p10": p10,
            "mpjpe_g_p90": p90,
            "mpjpe_g_p90_minus_p10": spread,
            "mpjpe_g_p90_over_p10": ratio,
            "frontier_motion_count": hard_count,
            "frontier_motion_fraction": hard_fraction,
            "mastered_motion_count": mastered_count,
            "mastered_motion_fraction": mastered_fraction,
        },
        "gates": gates,
        "ranking": ranking,
        "ranking_definition": {
            "order": "easiest_first",
            "primary_key": "mpjpe_g_ascending",
            "tie_break": "motion_key_lexical_ascending",
            "status": "operational_assumption_not_frozen_by_the_preregistered_gate",
        },
        "ranking_entries": ranking_entries,
        "preregistration_reference": "fable-next.md Phase 2 — SIM-D1",
    }


def _load_expected_motion_keys(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if isinstance(value, dict):
        value = value.get("motion_keys", value.get("ranking"))
    if not isinstance(value, list):
        raise ValueError(
            f"{path} must contain a JSON list or an object with 'motion_keys'/'ranking'"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-eval-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--dataset", required=True, help="Stable dataset label recorded in the artifact."
    )
    parser.add_argument(
        "--dataset-kind",
        choices=("real", "synthetic", "sample_data", "unspecified"),
        default="unspecified",
        help="Provenance label; D-B variants must use 'synthetic'.",
    )
    parser.add_argument(
        "--expected-motion-count",
        type=int,
        help="Optional independent all-motion coverage check.",
    )
    parser.add_argument(
        "--expected-motion-keys-json",
        type=Path,
        help="Optional exact all-motion key-set check (list or object with motion_keys/ranking).",
    )
    parser.add_argument("--mpjpe-spread-min", type=float, default=_DEFAULT_MPJPE_SPREAD_MIN)
    parser.add_argument("--mpjpe-ratio-min", type=float, default=_DEFAULT_MPJPE_RATIO_MIN)
    parser.add_argument("--hard-fraction-min", type=float, default=_DEFAULT_HARD_FRACTION_MIN)
    parser.add_argument(
        "--mastered-fraction-min", type=float, default=_DEFAULT_MASTERED_FRACTION_MIN
    )
    parser.add_argument(
        "--incomplete-progress-below",
        type=float,
        default=_DEFAULT_INCOMPLETE_PROGRESS_BELOW,
    )
    parser.add_argument(
        "--complete-progress-min", type=float, default=_DEFAULT_COMPLETE_PROGRESS_MIN
    )
    args = parser.parse_args()

    expected_keys = (
        _load_expected_motion_keys(args.expected_motion_keys_json)
        if args.expected_motion_keys_json
        else None
    )
    result = classify_sim_d1(
        _load_json_object(args.metrics_eval_json),
        dataset=args.dataset,
        dataset_kind=args.dataset_kind,
        expected_motion_count=args.expected_motion_count,
        expected_motion_keys=expected_keys,
        mpjpe_spread_min=args.mpjpe_spread_min,
        mpjpe_ratio_min=args.mpjpe_ratio_min,
        hard_fraction_min=args.hard_fraction_min,
        mastered_fraction_min=args.mastered_fraction_min,
        incomplete_progress_below=args.incomplete_progress_below,
        complete_progress_min=args.complete_progress_min,
    )
    result["source"] = {
        "metrics_eval_path": str(args.metrics_eval_json),
        "metrics_eval_sha256": _sha256(args.metrics_eval_json),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")

    print(
        f"SIM-D1 {result['verdict']}: {result['dataset']} "
        f"({result['summary']['motion_count']} motions)"
    )
    if result["failed_gates"]:
        print("failed gates: " + ", ".join(result["failed_gates"]))
    print(f"wrote classification and ranking to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
