#!/usr/bin/env python3
"""Aggregate seed-level SONIC paired-comparison artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


_UNIFORM_VARIANT = "uniform_sampling_micro"
_ADAPTIVE_VARIANT = "adaptive_sampling_micro"
_METRIC_KEYS = ("train.mean_rewards", "eval.ok", "eval.all.mpjpe_g")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _sorted_unique(values: list[Any]) -> list[Any]:
    return sorted({value for value in values if value is not None})


def _row_for_comparison(comparison: dict[str, Any]) -> dict[str, Any]:
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
    for variant_label, variant_name in [("uniform", _UNIFORM_VARIANT), ("adaptive", _ADAPTIVE_VARIANT)]:
        metrics = by_variant.get(variant_name, {}).get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        for metric in _METRIC_KEYS:
            aggregate_row[f"{variant_label}.{metric}"] = metrics.get(metric)

    uniform_mpjpe = aggregate_row.get("uniform.eval.all.mpjpe_g")
    adaptive_mpjpe = aggregate_row.get("adaptive.eval.all.mpjpe_g")
    if isinstance(uniform_mpjpe, (int, float)) and isinstance(adaptive_mpjpe, (int, float)):
        aggregate_row["delta.eval.all.mpjpe_g.adaptive_minus_uniform"] = adaptive_mpjpe - uniform_mpjpe
    else:
        aggregate_row["delta.eval.all.mpjpe_g.adaptive_minus_uniform"] = None
    return aggregate_row


def build_aggregate_comparison(comparison_paths: list[Path]) -> dict[str, Any]:
    """Build an aggregate over seed-level paired comparison JSON files.

    The aggregate is deliberately conservative: every seed-level comparison must
    already be causally valid before the aggregate can be considered valid.
    """
    if not comparison_paths:
        raise ValueError("at least one comparison path is required")

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
        row = _row_for_comparison(comparison)
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
    return {
        "schema_version": 1,
        "kind": "sonic_aggregate_comparison",
        "comparison_count": len(records),
        "seeds": seeds,
        "variants": variants,
        "seed_level_ok": seed_level_ok,
        "warning_counts": warning_counts,
        "rows": records,
        "ok_for_causal_comparison": all(seed_level_ok.values()) and not any(warning_counts.values()) and len(records) >= 1,
        "interpretation_guardrail": "No adaptive-sampling performance claim unless all seed-level gates pass and the metric pattern supports it.",
    }


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
        "uniform.train.mean_rewards",
        "adaptive.train.mean_rewards",
        "uniform.eval.ok",
        "adaptive.eval.ok",
        "uniform.eval.all.mpjpe_g",
        "adaptive.eval.all.mpjpe_g",
        "delta.eval.all.mpjpe_g.adaptive_minus_uniform",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Aggregate Comparison\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in ["comparison_count", "seeds", "variants", "ok_for_causal_comparison"]:
            f.write(f"| `{key}` | {_format_value(aggregate.get(key))} |\n")
        f.write("\n")
        f.write(f"> {aggregate.get('interpretation_guardrail', 'No adaptive-sampling performance claim.')}\n\n")
        f.write("## Warning counts\n\n")
        f.write("| Warning type | Count |\n|---|---:|\n")
        for key, value in aggregate.get("warning_counts", {}).items():
            f.write(f"| `{key}` | {_format_value(value)} |\n")
        f.write("\n## Seed rows\n\n")
        f.write("| " + " | ".join(f"`{column}`" for column in columns) + " |\n")
        f.write("|" + "---:|" * len(columns) + "\n")
        for row in aggregate.get("rows", []):
            f.write("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, action="append", required=True, help="Seed-level comparison JSON. Repeat for each seed.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    aggregate = build_aggregate_comparison(args.comparison)
    write_aggregate_json(args.output_json, aggregate)
    write_aggregate_table_markdown(args.output_md, aggregate)
    print(f"wrote SONIC aggregate comparison to {args.output_json} and {args.output_md}")
    if args.fail_on_invalid and not aggregate["ok_for_causal_comparison"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
