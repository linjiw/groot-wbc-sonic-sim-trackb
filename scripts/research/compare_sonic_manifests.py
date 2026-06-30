#!/usr/bin/env python3
"""Compare SONIC experiment manifests for controlled paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from scripts.research.sonic_experiment_manifest import validate_manifest

_CONTROL_GROUPS = ("controlled_variables", "datasets")
_CONTROL_FIELDS = ("checkpoint",)
_METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    ("metrics", "train", "ok"),
    ("metrics", "train", "learning_iteration"),
    ("metrics", "train", "mean_rewards"),
    ("metrics", "train", "total_timesteps"),
    ("metrics", "train", "traceback_count"),
    ("metrics", "eval", "ok"),
    ("metrics", "eval", "all", "mpjpe_g"),
    ("metrics", "eval", "all", "mpjpe_l"),
    ("metrics", "eval", "all", "mpjpe_pa"),
    ("metrics", "eval", "terminated_final"),
    ("metrics", "eval", "success_rate_final"),
    ("metrics", "eval", "traceback_count"),
)
_PRIMARY_METRIC_PATHS: tuple[tuple[str, ...], ...] = (
    ("metrics", "train", "ok"),
    ("metrics", "train", "mean_rewards"),
    ("metrics", "eval", "ok"),
    ("metrics", "eval", "all", "mpjpe_g"),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _get_nested(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _metric_key(path: tuple[str, ...]) -> str:
    return ".".join(path[1:]) if path and path[0] == "metrics" else ".".join(path)


def _manifest_label(manifest: dict[str, Any]) -> str:
    experiment_id = manifest.get("experiment_id")
    variant = manifest.get("variant")
    seed = manifest.get("seed")
    if experiment_id:
        return str(experiment_id)
    return f"{variant or 'unknown_variant'}:seed{seed}"


def _flatten_control_values(manifest: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for group in _CONTROL_GROUPS:
        group_value = manifest.get(group, {})
        if isinstance(group_value, dict):
            for key, value in group_value.items():
                values[f"{group}.{key}"] = value
        else:
            values[group] = group_value
    for field in _CONTROL_FIELDS:
        values[field] = manifest.get(field)
    return values


def _find_control_mismatches(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not manifests:
        return []
    labels = [_manifest_label(m) for m in manifests]
    all_keys: set[str] = set()
    for manifest in manifests:
        all_keys.update(_flatten_control_values(manifest).keys())

    mismatches: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        values = [_flatten_control_values(manifest).get(key) for manifest in manifests]
        unique_values = {json.dumps(value, sort_keys=True, default=str) for value in values}
        if len(unique_values) > 1:
            mismatches.append(
                {
                    "field": key,
                    "values": {label: value for label, value in zip(labels, values, strict=True)},
                }
            )
    return mismatches


def _find_metric_warnings(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find missing/failed primary metrics that make a paper comparison incomplete."""
    warnings: list[dict[str, Any]] = []
    for manifest in manifests:
        label = _manifest_label(manifest)
        for path in _PRIMARY_METRIC_PATHS:
            value = _get_nested(manifest, path)
            if value is None:
                warnings.append({"experiment": label, "field": _metric_key(path), "problem": "missing"})
            elif path[-1] == "ok" and value is not True:
                warnings.append({"experiment": label, "field": _metric_key(path), "problem": "not_true", "value": value})
    return warnings


def build_comparison(manifest_paths: list[Path]) -> dict[str, Any]:
    """Build a comparison summary from one or more SONIC experiment manifests."""
    if not manifest_paths:
        raise ValueError("at least one manifest path is required")

    records: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    for path in manifest_paths:
        manifest = _load_json(path)
        errors = validate_manifest(manifest)
        if errors:
            validation_errors.append({"path": str(path), "errors": errors})
        records.append({"path": str(path), "manifest": manifest})

    manifests = [record["manifest"] for record in records]
    rows: list[dict[str, Any]] = []
    for record in records:
        manifest = record["manifest"]
        metrics = {_metric_key(path): _get_nested(manifest, path) for path in _METRIC_PATHS}
        rows.append(
            {
                "path": record["path"],
                "experiment_id": manifest.get("experiment_id"),
                "variant": manifest.get("variant"),
                "seed": manifest.get("seed"),
                "status": manifest.get("status"),
                "git_commit": manifest.get("git_commit"),
                "interpretation": manifest.get("interpretation"),
                "metrics": metrics,
            }
        )

    variants = sorted({str(m.get("variant")) for m in manifests if m.get("variant") is not None})
    seeds = sorted({m.get("seed") for m in manifests if m.get("seed") is not None})
    mismatches = _find_control_mismatches(manifests)
    metric_warnings = _find_metric_warnings(manifests)
    return {
        "schema_version": 1,
        "kind": "sonic_manifest_comparison",
        "manifest_count": len(records),
        "variants": variants,
        "seeds": seeds,
        "control_mismatches": mismatches,
        "metric_warnings": metric_warnings,
        "validation_errors": validation_errors,
        "rows": rows,
        "ok_for_causal_comparison": not validation_errors and not mismatches and not metric_warnings and len(records) >= 2,
    }


def write_comparison_json(path: Path, comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)
        f.write("\n")


def write_comparison_markdown(path: Path, comparison: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Manifest Comparison\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in ["manifest_count", "variants", "seeds", "ok_for_causal_comparison"]:
            f.write(f"| `{key}` | {_format_value(comparison.get(key))} |\n")

        mismatches = comparison.get("control_mismatches", [])
        f.write("\n## Control Mismatches\n\n")
        if mismatches:
            f.write("| Field | Values |\n|---|---|\n")
            for mismatch in mismatches:
                f.write(f"| `{mismatch['field']}` | `{json.dumps(mismatch['values'], sort_keys=True)}` |\n")
        else:
            f.write("None. Controlled variables, dataset paths, and checkpoint match across supplied manifests.\n")

        validation_errors = comparison.get("validation_errors", [])
        f.write("\n## Validation Errors\n\n")
        if validation_errors:
            f.write("| Manifest | Errors |\n|---|---|\n")
            for item in validation_errors:
                f.write(f"| `{item['path']}` | `{json.dumps(item['errors'])}` |\n")
        else:
            f.write("None.\n")

        metric_warnings = comparison.get("metric_warnings", [])
        f.write("\n## Metric Warnings\n\n")
        if metric_warnings:
            f.write("| Experiment | Field | Problem | Value |\n|---|---|---|---|\n")
            for item in metric_warnings:
                f.write(
                    f"| `{item.get('experiment')}` | `{item.get('field')}` | `{item.get('problem')}` | {_format_value(item.get('value'))} |\n"
                )
        else:
            f.write("None. Primary train/eval metrics are present and healthy.\n")

        metric_keys = [_metric_key(path) for path in _METRIC_PATHS]
        f.write("\n## Experiment Rows\n\n")
        f.write("| Experiment | Variant | Seed | Status | " + " | ".join(f"`{k}`" for k in metric_keys) + " |\n")
        f.write("|---|---|---:|---|" + "---:|" * len(metric_keys) + "\n")
        for row in comparison.get("rows", []):
            metric_values = [row.get("metrics", {}).get(key) for key in metric_keys]
            f.write(
                "| "
                + " | ".join(
                    [
                        _format_value(row.get("experiment_id")),
                        _format_value(row.get("variant")),
                        _format_value(row.get("seed")),
                        _format_value(row.get("status")),
                        *[_format_value(value) for value in metric_values],
                    ]
                )
                + " |\n"
            )

        f.write("\n## Interpretations\n\n")
        for row in comparison.get("rows", []):
            f.write(f"- `{_format_value(row.get('experiment_id'))}`: {_format_value(row.get('interpretation'))}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True, help="Manifest JSON path. Repeat for comparisons.")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--fail-on-mismatch", action="store_true", help="Return nonzero if validation or control mismatches exist.")
    args = parser.parse_args()

    comparison = build_comparison(args.manifest)
    write_comparison_json(args.output_json, comparison)
    if args.output_md:
        write_comparison_markdown(args.output_md, comparison)
    print(f"wrote SONIC manifest comparison to {args.output_json}")

    has_problems = bool(comparison["validation_errors"] or comparison["control_mismatches"])
    if args.fail_on_mismatch and has_problems:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
