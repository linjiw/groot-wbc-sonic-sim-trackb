#!/usr/bin/env python3
"""Run or plan paired SONIC experiments from an explicit paper-validation spec.

The launcher is intentionally dry-run friendly: use --dry-run to validate and materialize
commands/manifests/comparisons from existing summaries without starting IsaacLab jobs.
Use --execute only when the spec commands are ready for a real compute run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from scripts.research.compare_sonic_manifests import (
    build_comparison,
    write_comparison_json,
    write_comparison_markdown,
)
from scripts.research.sonic_experiment_manifest import (
    build_manifest,
    validate_manifest,
    write_manifest_markdown,
)
from scripts.research.summarize_sonic_logs import summarize_logs, write_markdown as write_summary_markdown


_REQUIRED_TOP_LEVEL = ("experiment_group", "hypothesis", "seed", "dataset_robot", "dataset_smpl", "checkpoint", "variants")
_REQUIRED_VARIANT = ("name", "eval_command", "interpretation")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    if value is None:
        return ""
    return str(value)


def validate_spec(spec: dict[str, Any]) -> list[str]:
    """Validate a paired-experiment spec without touching IsaacLab."""
    errors: list[str] = []
    for key in _REQUIRED_TOP_LEVEL:
        if key not in spec or spec[key] in (None, ""):
            errors.append(f"missing {key}")
    variants = spec.get("variants")
    if not isinstance(variants, list) or not variants:
        errors.append("variants must be a non-empty list")
        return errors

    names: list[str] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            errors.append(f"variants[{index}] must be an object")
            continue
        for key in _REQUIRED_VARIANT:
            if key not in variant or variant[key] in (None, ""):
                errors.append(f"variants[{index}] missing {key}")
        name = variant.get("name")
        if name in names:
            errors.append(f"duplicate variant name: {name}")
        if name is not None:
            names.append(str(name))
        has_summary = bool(variant.get("summary_json"))
        has_logs = bool(variant.get("train_log") or variant.get("eval_log"))
        if not has_summary and not has_logs and not (variant.get("train_command") or variant.get("eval_command")):
            errors.append(f"variants[{index}] must provide summary_json, logs, or commands")
    return errors


def _run_command(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return process.returncode


def _variant_id(group: str, variant_name: str, seed: int) -> str:
    safe_group = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in group)
    safe_variant = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in variant_name)
    return f"{safe_group}_{safe_variant}_seed{seed}"


def _relative_or_absolute(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


def _summary_is_stale(summary_json: Path, train_log: Path, eval_log: Path) -> bool:
    if not summary_json.exists():
        return True
    summary_mtime = summary_json.stat().st_mtime
    return any(path.exists() and path.stat().st_mtime > summary_mtime for path in (train_log, eval_log))


def materialize_paired_experiment(
    spec: dict[str, Any],
    *,
    output_dir: Path,
    dry_run: bool,
    repo_root: Path,
) -> dict[str, Any]:
    """Create plan/summary/manifest/comparison artifacts for a paired experiment spec."""
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid spec: " + "; ".join(errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    group = str(spec["experiment_group"])
    seed = int(spec["seed"])
    manifest_paths: list[Path] = []
    variant_results: list[dict[str, Any]] = []

    for variant in spec["variants"]:
        name = str(variant["name"])
        variant_dir = output_dir / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_log = Path(variant.get("train_log") or variant_dir / "train.log")
        eval_log = Path(variant.get("eval_log") or variant_dir / "eval.log")
        summary_json = Path(variant.get("summary_json") or variant_dir / "summary.json")
        summary_md = variant_dir / "summary.md"
        manifest_json = variant_dir / "manifest.json"
        manifest_md = variant_dir / "manifest.md"

        command_results: list[dict[str, Any]] = []
        if not dry_run:
            if variant.get("train_command"):
                rc = _run_command(str(variant["train_command"]), train_log, cwd=repo_root)
                command_results.append({"kind": "train", "returncode": rc, "log": str(train_log)})
            if variant.get("eval_command"):
                rc = _run_command(str(variant["eval_command"]), eval_log, cwd=repo_root)
                command_results.append({"kind": "eval", "returncode": rc, "log": str(eval_log)})

        if not _summary_is_stale(summary_json, train_log, eval_log):
            # Existing summary is newer than logs; keep it immutable and use it as source of truth.
            summary_source = summary_json
        elif train_log.exists() or eval_log.exists():
            summary = summarize_logs(train_log if train_log.exists() else None, eval_log if eval_log.exists() else None)
            _write_json(summary_json, summary)
            write_summary_markdown(summary_md, summary)
            summary_source = summary_json
        else:
            summary_source = None

        manifest_errors: list[str] = []
        if summary_source is not None:
            checkpoint = str(variant.get("checkpoint") or spec["checkpoint"])
            manifest = build_manifest(
                experiment_id=str(variant.get("experiment_id") or _variant_id(group, name, seed)),
                hypothesis=str(spec["hypothesis"]),
                variant=name,
                seed=seed,
                dataset_robot=str(spec["dataset_robot"]),
                dataset_smpl=str(spec["dataset_smpl"]),
                checkpoint=checkpoint,
                summary_json=summary_source,
                train_command=variant.get("train_command"),
                eval_command=str(variant["eval_command"]),
                interpretation=str(variant["interpretation"]),
                git_commit=spec.get("git_commit"),
                checkpoint_source=str(variant.get("checkpoint_source") or spec.get("checkpoint_source") or "configured_checkpoint"),
                checkpoint_provenance=variant.get("checkpoint_provenance"),
                status=str(variant.get("status", "needs_review")),
            )
            manifest_errors = validate_manifest(manifest)
            if not manifest_errors:
                _write_json(manifest_json, manifest)
                write_manifest_markdown(manifest_md, manifest)
                manifest_paths.append(manifest_json)

        variant_results.append(
            {
                "name": name,
                "dry_run": dry_run,
                "commands": {
                    "train": variant.get("train_command"),
                    "eval": variant.get("eval_command"),
                },
                "command_results": command_results,
                "summary_json": str(summary_source) if summary_source else None,
                "manifest_json": str(manifest_json) if not manifest_errors and summary_source else None,
                "manifest_errors": manifest_errors,
            }
        )

    comparison_json = output_dir / "comparison.json"
    comparison_md = output_dir / "comparison.md"
    comparison: dict[str, Any] | None = None
    if manifest_paths:
        comparison = build_comparison(manifest_paths)
        write_comparison_json(comparison_json, comparison)
        write_comparison_markdown(comparison_md, comparison)

    plan = {
        "schema_version": 1,
        "kind": "sonic_paired_experiment_run",
        "experiment_group": group,
        "dry_run": dry_run,
        "seed": seed,
        "dataset_robot": spec["dataset_robot"],
        "dataset_smpl": spec["dataset_smpl"],
        "checkpoint": spec["checkpoint"],
        "output_dir": str(output_dir),
        "variants": variant_results,
        "comparison_json": str(comparison_json) if comparison is not None else None,
        "ok_for_causal_comparison": comparison.get("ok_for_causal_comparison") if comparison else False,
    }
    _write_json(output_dir / "run_plan.json", plan)
    write_run_markdown(output_dir / "run_plan.md", plan)
    return plan


def write_run_markdown(path: Path, plan: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Paired Experiment Run Plan\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in ["experiment_group", "dry_run", "seed", "dataset_robot", "dataset_smpl", "checkpoint", "ok_for_causal_comparison"]:
            f.write(f"| `{key}` | {_format_value(plan.get(key))} |\n")
        f.write("\n## Variants\n\n")
        for variant in plan.get("variants", []):
            f.write(f"### {variant['name']}\n\n")
            f.write("| Field | Value |\n|---|---|\n")
            for key in ["summary_json", "manifest_json", "manifest_errors"]:
                f.write(f"| `{key}` | {_format_value(variant.get(key))} |\n")
            commands = variant.get("commands", {})
            for kind in ["train", "eval"]:
                if commands.get(kind):
                    f.write(f"\n#### {kind} command\n\n```bash\n{commands[kind]}\n```\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    spec = _load_json(args.spec)
    try:
        plan = materialize_paired_experiment(
            spec,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            repo_root=args.repo_root,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote SONIC paired experiment plan to {args.output_dir / 'run_plan.json'}")
    if plan.get("comparison_json"):
        print(f"wrote comparison to {plan['comparison_json']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
