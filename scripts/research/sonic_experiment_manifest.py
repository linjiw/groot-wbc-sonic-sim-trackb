#!/usr/bin/env python3
"""Build and validate SONIC experiment manifests for paired paper comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

DEFAULT_CONTROLLED_VARIABLES: dict[str, str] = {
    "simulator": "IsaacLab / Isaac Sim headless",
    "env": "env_isaaclab",
    "robot": "Unitree G1 29-DoF dex model",
    "sonic_config": "manager/universal_token/all_modes/sonic_release",
    "action_interface": "64D SONIC motion token + 7D left hand + 7D right hand",
    "hardware": "excluded until simulation gates pass",
}

_REQUIRED_PATHS = [
    ("hypothesis",),
    ("variant",),
    ("seed",),
    ("datasets", "robot_motion"),
    ("datasets", "smpl_motion"),
    ("artifacts", "summary_json"),
    ("commands", "eval"),
    ("controlled_variables", "action_interface"),
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(
    *,
    experiment_id: str,
    hypothesis: str,
    variant: str,
    seed: int,
    dataset_robot: str,
    dataset_smpl: str,
    checkpoint: str,
    summary_json: Path,
    train_command: str | None,
    eval_command: str,
    interpretation: str,
    git_commit: str | None = None,
    controlled_variables: dict[str, str] | None = None,
    checkpoint_source: str = "configured_checkpoint",
    checkpoint_provenance: dict[str, Any] | None = None,
    status: str = "needs_review",
) -> dict[str, Any]:
    """Create a JSON-serializable manifest from an existing summary artifact."""
    summary = _load_json(summary_json)
    controls = dict(DEFAULT_CONTROLLED_VARIABLES)
    if controlled_variables:
        controls.update(controlled_variables)

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "hypothesis": hypothesis,
        "variant": variant,
        "seed": seed,
        "status": status,
        "interpretation": interpretation,
        "git_commit": git_commit or _git_commit(),
        "controlled_variables": controls,
        "datasets": {
            "robot_motion": dataset_robot,
            "smpl_motion": dataset_smpl,
        },
        "checkpoint": checkpoint,
        "checkpoint_source": checkpoint_source,
        "checkpoint_provenance": checkpoint_provenance or {},
        "commands": {
            "train": train_command,
            "eval": eval_command,
        },
        "artifacts": {
            "summary_json": str(summary_json),
            "train_log": summary.get("train", {}).get("log_path"),
            "eval_log": summary.get("eval", {}).get("log_path"),
        },
        "metrics": {
            "train": summary.get("train", {}),
            "eval": summary.get("eval", {}),
        },
    }


def _get_nested(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return validation errors for a manifest. Empty list means valid."""
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if not manifest.get("experiment_id"):
        errors.append("missing experiment_id")
    for path in _REQUIRED_PATHS:
        value = _get_nested(manifest, path)
        if value is None or value == "":
            errors.append("missing " + ".".join(path))
    metrics = manifest.get("metrics")
    if not isinstance(metrics, dict):
        errors.append("missing metrics")
    else:
        if not isinstance(metrics.get("eval"), dict):
            errors.append("missing metrics.eval")
        if not isinstance(metrics.get("train"), dict):
            errors.append("missing metrics.train")
    return errors


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def write_manifest_markdown(path: Path, manifest: dict[str, Any]) -> None:
    """Write a Markdown companion report for a SONIC experiment manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Experiment Manifest\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in ["experiment_id", "variant", "seed", "status", "git_commit", "interpretation"]:
            f.write(f"| `{key}` | {_format_value(manifest.get(key))} |\n")
        f.write("\n## Hypothesis\n\n")
        f.write(str(manifest.get("hypothesis", "")) + "\n\n")

        f.write("## Controlled Variables\n\n| Variable | Value |\n|---|---|\n")
        for key, value in manifest.get("controlled_variables", {}).items():
            f.write(f"| `{key}` | {_format_value(value)} |\n")

        f.write("\n## Datasets and Artifacts\n\n| Field | Value |\n|---|---|\n")
        for group in ["datasets", "artifacts"]:
            for key, value in manifest.get(group, {}).items():
                f.write(f"| `{group}.{key}` | {_format_value(value)} |\n")
        f.write(f"| `checkpoint` | {_format_value(manifest.get('checkpoint'))} |\n")

        f.write("\n## Commands\n\n")
        for key, value in manifest.get("commands", {}).items():
            if value:
                f.write(f"### {key}\n\n```bash\n{value}\n```\n\n")

        eval_metrics = manifest.get("metrics", {}).get("eval", {})
        all_metrics = eval_metrics.get("all", {}) if isinstance(eval_metrics, dict) else {}
        if all_metrics:
            f.write("## Evaluation `all` Metrics\n\n| Metric | Value |\n|---|---:|\n")
            for key, value in all_metrics.items():
                f.write(f"| `{key}` | {_format_value(value)} |\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset-robot", required=True)
    parser.add_argument("--dataset-smpl", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--train-command")
    parser.add_argument("--eval-command", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--git-commit")
    parser.add_argument("--status", default="needs_review")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(
        experiment_id=args.experiment_id,
        hypothesis=args.hypothesis,
        variant=args.variant,
        seed=args.seed,
        dataset_robot=args.dataset_robot,
        dataset_smpl=args.dataset_smpl,
        checkpoint=args.checkpoint,
        summary_json=args.summary_json,
        train_command=args.train_command,
        eval_command=args.eval_command,
        interpretation=args.interpretation,
        git_commit=args.git_commit,
        status=args.status,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not args.validate_only:
        _write_json(args.output_json, manifest)
        if args.output_md:
            write_manifest_markdown(args.output_md, manifest)
        print(f"wrote SONIC experiment manifest to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
