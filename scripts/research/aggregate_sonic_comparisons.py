#!/usr/bin/env python3
"""Aggregate seed-level SONIC paired-comparison artifacts.

Variant pairing is parameterized: variant A is the treatment arm and variant B
is the control arm, so deltas are ``a_minus_b`` and improvement is a negative
delta (lower MPJPE is better). Defaults preserve the SIM-M2/M3 pairing
(A = adaptive_sampling_micro, B = uniform_sampling_micro).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.paired_stats import paired_delta_statistics  # noqa: E402
from scripts.research.summarize_sonic_logs import (  # noqa: E402
    ADP_SAMP_CLASSIFICATION_KEYS,
)

_DEFAULT_VARIANT_A = "adaptive_sampling_micro"  # treatment
_DEFAULT_VARIANT_B = "uniform_sampling_micro"  # control
_METRIC_KEYS = ("train.mean_rewards", "eval.ok", "eval.all.mpjpe_g")
# Final-value sampler telemetry for the SIM-M4 classification rule; present only
# on adaptive arms, so these stay informational (never part of any validity gate).
_TELEMETRY_KEYS = tuple(f"train.adp_samp_{key}" for key in ADP_SAMP_CLASSIFICATION_KEYS)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    if value is None:
        return ""
    return str(value)


def _sorted_unique(values: list[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None})


def _delta_key(metric: str) -> str:
    return f"delta.{metric}.a_minus_b"


def _build_effect_summary(
    records: list[dict[str, Any]],
    *,
    effect_metric: str,
    a_minus_b_threshold: float,
    min_improved_seeds: int,
) -> dict[str, Any]:
    delta_key = _delta_key(effect_metric)
    deltas = [row.get(delta_key) for row in records]
    # json.load parses bare NaN, so a non-finite delta must count as missing
    # (failing the completeness check below), not as a numeric observation.
    numeric_deltas = [
        float(value)
        for value in deltas
        if isinstance(value, (int, float))
        and value == value
        and value not in (float("inf"), float("-inf"))
    ]
    improved_seed_count = sum(1 for value in numeric_deltas if value < 0.0)
    mean_delta = sum(numeric_deltas) / len(numeric_deltas) if numeric_deltas else None
    passes = (
        mean_delta is not None
        and len(numeric_deltas) == len(records)
        and mean_delta <= a_minus_b_threshold
        and improved_seed_count >= min_improved_seeds
    )
    summary = {
        "metric": effect_metric,
        "a_minus_b_threshold": a_minus_b_threshold,
        "min_improved_seeds": min_improved_seeds,
        "mean_delta_a_minus_b": mean_delta,
        "improved_seed_count": improved_seed_count,
        "seed_count": len(records),
        "passes_preregistered_effect_gate": passes,
    }
    if numeric_deltas:
        summary["statistics"] = paired_delta_statistics(numeric_deltas)
    return summary


def _row_for_comparison(
    comparison: dict[str, Any], *, variant_a: str, variant_b: str
) -> dict[str, Any]:
    seeds = comparison.get("seeds") or []
    seed = seeds[0] if len(seeds) == 1 else None
    by_variant: dict[str, dict[str, Any]] = {}
    for row in comparison.get("rows", []):
        if isinstance(row, dict) and row.get("variant") is not None:
            by_variant[str(row["variant"])] = row

    aggregate_row: dict[str, Any] = {
        "seed": seed,
        "ok_for_causal_comparison": comparison.get("ok_for_causal_comparison") is True,
    }
    for variant_label, variant_name in [("a", variant_a), ("b", variant_b)]:
        # Name matching is tracked separately from metric presence: a typo'd
        # variant name and a failed eval are different problems.
        aggregate_row[f"{variant_label}.variant_matched"] = variant_name in by_variant
        metrics = by_variant.get(variant_name, {}).get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        for metric in _METRIC_KEYS + _TELEMETRY_KEYS:
            aggregate_row[f"{variant_label}.{metric}"] = metrics.get(metric)

    b_mpjpe = aggregate_row.get("b.eval.all.mpjpe_g")
    a_mpjpe = aggregate_row.get("a.eval.all.mpjpe_g")
    if isinstance(b_mpjpe, (int, float)) and isinstance(a_mpjpe, (int, float)):
        aggregate_row[_delta_key("eval.all.mpjpe_g")] = a_mpjpe - b_mpjpe
    else:
        aggregate_row[_delta_key("eval.all.mpjpe_g")] = None
    return aggregate_row


def build_aggregate_comparison(
    comparison_paths: list[Path],
    *,
    variant_a: str = _DEFAULT_VARIANT_A,
    variant_b: str = _DEFAULT_VARIANT_B,
    effect_metric: str | None = None,
    a_minus_b_threshold: float = -0.5,
    min_improved_seeds: int = 2,
) -> dict[str, Any]:
    """Build an aggregate over seed-level paired comparison JSON files.

    The aggregate is deliberately conservative: every seed-level comparison must
    already be causally valid before the aggregate can be considered valid.
    """
    if not comparison_paths:
        raise ValueError("at least one comparison path is required")
    if variant_a == variant_b:
        raise ValueError("variant_a and variant_b must differ")
    # Deltas (and their improvement-is-negative direction) are only computed for
    # this metric; any other request would silently score an all-None gate.
    if effect_metric is not None and effect_metric != "eval.all.mpjpe_g":
        raise ValueError(
            f"unsupported effect metric {effect_metric!r}: only eval.all.mpjpe_g "
            "has a computed a_minus_b delta with lower-is-better semantics"
        )
    resolved = [str(Path(path).resolve()) for path in comparison_paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate comparison paths: {comparison_paths}")

    records: list[dict[str, Any]] = []
    seed_level_ok: dict[str, bool] = {}
    all_seeds: list[Any] = []
    all_variants: list[Any] = []
    warning_counts = {
        "control_mismatches": 0,
        "metric_warnings": 0,
        "checkpoint_warnings": 0,
        "validation_errors": 0,
    }

    for path in comparison_paths:
        comparison = _load_json(path)
        seeds = comparison.get("seeds") or []
        all_seeds.extend(seeds)
        all_variants.extend(comparison.get("variants") or [])
        for key in warning_counts:
            value = comparison.get(key, [])
            warning_counts[key] += len(value) if isinstance(value, list) else 0
        row = _row_for_comparison(comparison, variant_a=variant_a, variant_b=variant_b)
        row["comparison_path"] = str(path)
        records.append(row)
        seed_key = str(row.get("seed"))
        seed_level_ok[seed_key] = row["ok_for_causal_comparison"]

    def _seed_sort_key(row: dict[str, Any]) -> int:
        seed = row.get("seed")
        return seed if isinstance(seed, int) else -1

    records.sort(key=_seed_sort_key)
    seeds = _sorted_unique(all_seeds)
    variants = _sorted_unique(all_variants)
    # A row whose variant_a/variant_b name matched nothing means the requested
    # names don't exist in that comparison (typo or wrong pairing) — a validity
    # failure, distinct from a matched variant whose eval metrics are missing
    # (already covered by seed-level metric_warnings).
    missing_variant_rows = [
        str(row.get("seed"))
        for row in records
        if not (row.get("a.variant_matched") and row.get("b.variant_matched"))
    ]
    row_seeds = [row.get("seed") for row in records]
    duplicate_seed_rows = sorted(
        {str(seed) for seed in row_seeds if seed is not None and row_seeds.count(seed) > 1}
    )
    aggregate = {
        "schema_version": 2,
        "kind": "sonic_aggregate_comparison",
        "variant_a": variant_a,
        "variant_b": variant_b,
        "comparison_count": len(records),
        "seeds": seeds,
        "variants": variants,
        "seed_level_ok": seed_level_ok,
        "warning_counts": warning_counts,
        "missing_variant_rows": missing_variant_rows,
        "duplicate_seed_rows": duplicate_seed_rows,
        "rows": records,
        "ok_for_causal_comparison": all(seed_level_ok.values())
        and not any(warning_counts.values())
        and not missing_variant_rows
        and not duplicate_seed_rows
        and len(records) >= 1,
        "interpretation_guardrail": (
            "No sampling-mechanism performance claim unless all seed-level gates pass "
            "and the metric pattern supports it."
        ),
    }
    if effect_metric is not None:
        aggregate["effect_summary"] = _build_effect_summary(
            records,
            effect_metric=effect_metric,
            a_minus_b_threshold=a_minus_b_threshold,
            min_improved_seeds=min_improved_seeds,
        )
    return aggregate


def write_aggregate_json(path: Path, aggregate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)
        f.write("\n")


def write_aggregate_table_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "seed",
        "ok_for_causal_comparison",
        "b.train.mean_rewards",
        "a.train.mean_rewards",
        "b.eval.ok",
        "a.eval.ok",
        "b.eval.all.mpjpe_g",
        "a.eval.all.mpjpe_g",
        "delta.eval.all.mpjpe_g.a_minus_b",
    ]
    telemetry_columns = ["seed"] + [f"a.{key}" for key in _TELEMETRY_KEYS]
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Aggregate Comparison\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in [
            "comparison_count",
            "seeds",
            "variants",
            "variant_a",
            "variant_b",
            "missing_variant_rows",
            "duplicate_seed_rows",
            "ok_for_causal_comparison",
        ]:
            f.write(f"| `{key}` | {_format_value(aggregate.get(key))} |\n")
        f.write("\n")
        f.write(
            f"> {aggregate.get('interpretation_guardrail', 'No sampling-mechanism performance claim.')}\n\n"
        )
        f.write(
            "> Deltas are `variant_a - variant_b` (A = treatment, B = control); improvement is negative.\n\n"
        )
        effect_summary = aggregate.get("effect_summary")
        if isinstance(effect_summary, dict):
            f.write("## Pre-registered effect gate\n\n")
            f.write("| Field | Value |\n|---|---|\n")
            for key in [
                "metric",
                "a_minus_b_threshold",
                "min_improved_seeds",
                "mean_delta_a_minus_b",
                "improved_seed_count",
                "seed_count",
                "passes_preregistered_effect_gate",
            ]:
                f.write(f"| `{key}` | {_format_value(effect_summary.get(key))} |\n")
            statistics = effect_summary.get("statistics")
            if isinstance(statistics, dict):
                f.write("\n### Paired statistics (reported, non-gating)\n\n")
                f.write("| Field | Value |\n|---|---|\n")
                for key in [
                    "n",
                    "permutation_p_one_sided",
                    "min_achievable_p",
                    "bootstrap_ci_95",
                    "power_note",
                ]:
                    f.write(f"| `{key}` | {_format_value(statistics.get(key))} |\n")
            f.write("\n")
        f.write("## Warning counts\n\n")
        f.write("| Warning type | Count |\n|---|---:|\n")
        for key, value in aggregate.get("warning_counts", {}).items():
            f.write(f"| `{key}` | {_format_value(value)} |\n")
        rows = aggregate.get("rows", [])
        f.write("\n## Seed rows\n\n")
        f.write("| " + " | ".join(f"`{column}`" for column in columns) + " |\n")
        f.write("|" + "---:|" * len(columns) + "\n")
        for row in rows:
            f.write(
                "| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |\n"
            )
        if any(row.get(column) is not None for row in rows for column in telemetry_columns[1:]):
            f.write("\n## Sampler telemetry (variant A, final values)\n\n")
            f.write("| " + " | ".join(f"`{column}`" for column in telemetry_columns) + " |\n")
            f.write("|" + "---:|" * len(telemetry_columns) + "\n")
            for row in rows:
                f.write(
                    "| "
                    + " | ".join(_format_value(row.get(column)) for column in telemetry_columns)
                    + " |\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        type=Path,
        action="append",
        required=True,
        help="Seed-level comparison JSON, repeatable.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--fail-on-invalid", action="store_true")
    parser.add_argument(
        "--variant-a",
        default=_DEFAULT_VARIANT_A,
        help="Treatment variant name (deltas are a_minus_b).",
    )
    parser.add_argument("--variant-b", default=_DEFAULT_VARIANT_B, help="Control variant name.")
    parser.add_argument(
        "--effect-metric",
        default=None,
        help="Optional metric key for a pre-registered a-minus-b effect gate, e.g. eval.all.mpjpe_g.",
    )
    parser.add_argument(
        "--a-minus-b-threshold",
        type=float,
        default=-0.5,
        help="Mean a-minus-b delta must be <= this value to pass the effect gate. "
        "Lower MPJPE is better, so beneficial thresholds are negative.",
    )
    parser.add_argument("--min-improved-seeds", type=int, default=2)
    args = parser.parse_args()

    aggregate = build_aggregate_comparison(
        args.comparison,
        variant_a=args.variant_a,
        variant_b=args.variant_b,
        effect_metric=args.effect_metric,
        a_minus_b_threshold=args.a_minus_b_threshold,
        min_improved_seeds=args.min_improved_seeds,
    )
    write_aggregate_json(args.output_json, aggregate)
    write_aggregate_table_markdown(args.output_md, aggregate)
    print(f"wrote SONIC aggregate comparison to {args.output_json} and {args.output_md}")
    if aggregate["missing_variant_rows"]:
        print(
            "WARNING: no metrics matched variant_a/variant_b for seed(s) "
            f"{', '.join(aggregate['missing_variant_rows'])} — check --variant-a/--variant-b spelling"
        )
    if args.fail_on_invalid and not aggregate["ok_for_causal_comparison"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
