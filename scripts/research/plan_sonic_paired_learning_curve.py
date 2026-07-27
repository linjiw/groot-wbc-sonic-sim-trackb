#!/usr/bin/env python3
"""Plan and aggregate matched official SONIC learning-curve evaluations.

This module is deliberately CPU-only.  It turns an already validated official
failure-rate-vs-ZPD paired spec into symmetric ImEvalCallback commands at the
release initialization and iterations 50/100/150/200.  Once those commands
have produced ``metrics_eval.json`` files, ``--aggregate`` writes a paired
learning-curve and trapezoidal AUC table over environment transitions.  The
explicit ``--execute`` mode verifies all ten rows with fail-closed checkpoint,
output-freshness, and manifest-coverage checks.  The identical iteration-zero
checkpoint is simulated once and its content-identical metrics are reused for
the second arm, so execution requires nine GPU evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_text = str(REPO_ROOT)
if repo_root_text in sys.path:
    sys.path.remove(repo_root_text)
sys.path.insert(0, repo_root_text)

from scripts.research.run_sonic_paired_experiment import (  # noqa: E402
    _PAIR_EVAL_OUTPUT_KEYS,
    _command_environment,
    _command_overrides,
    _paired_command_signature,
    _parse_command,
    _verify_metrics_motion_coverage,
    _verify_official_zpd_pair_preflight,
    validate_spec,
)

PROTOCOL = "official_im_eval_paired_learning_curve_v2"
OFFICIAL_PAIR_CONTRACT = "official_failure_rate_vs_zpd_learnability_v1"
CHECKPOINT_ITERATIONS = (0, 50, 100, 150, 200)
METRIC_SOURCES = {
    "success_rate": "eval/success/success_rate",
    "progress_rate": "eval/success/progress_rate",
    "mpjpe_l": "eval/all/mpjpe_l",
    "mpjpe_g": "eval/all/mpjpe_g",
}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seeded(text: str, seed: int) -> str:
    return text.replace("{seed}", str(seed))


def _single_override(command: str, key: str, *, label: str) -> str:
    values = _command_overrides(command).get(key, [])
    if len(values) != 1:
        raise ValueError(f"{label} must set {key} exactly once; got {values}")
    return values[0]


def _replace_eval_routing(command: str, *, checkpoint: str, output_dir: str) -> str:
    """Replace only checkpoint and eval_output_dir in a shell-free command."""
    environment, argv = _parse_command(command)
    replacements = {"checkpoint": checkpoint, "eval_output_dir": output_dir}
    counts = {key: 0 for key in replacements}
    rewritten: list[str] = []
    for token in argv:
        normalized = token.lstrip("+~")
        if "=" not in normalized:
            rewritten.append(token)
            continue
        key, _ = normalized.split("=", 1)
        if key not in replacements:
            rewritten.append(token)
            continue
        raw_key = token.split("=", 1)[0]
        rewritten.append(f"{raw_key}={replacements[key]}")
        counts[key] += 1
    for key, count in counts.items():
        if count != 1:
            raise ValueError(f"eval command must set {key} exactly once; got {count}")
    tokens = [f"{key}={value}" for key, value in environment.items()] + rewritten
    return shlex.join(tokens)


def _learning_curve_config(spec: dict[str, Any]) -> tuple[int, tuple[int, ...], int]:
    config = spec.get("learning_curve")
    if not isinstance(config, dict):
        raise ValueError("spec must define a learning_curve object")
    if config.get("protocol") != PROTOCOL:
        raise ValueError(f"learning_curve.protocol must be {PROTOCOL!r}")
    steps = config.get("checkpoint_iterations")
    if steps != list(CHECKPOINT_ITERATIONS):
        raise ValueError(
            "learning_curve.checkpoint_iterations must be exactly "
            f"{list(CHECKPOINT_ITERATIONS)}"
        )
    rollout_steps = config.get("rollout_steps_per_iteration")
    if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int):
        raise ValueError("learning_curve.rollout_steps_per_iteration must be a positive integer")
    if rollout_steps <= 0:
        raise ValueError("learning_curve.rollout_steps_per_iteration must be a positive integer")
    timeout_seconds = config.get("eval_timeout_seconds")
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise ValueError("learning_curve.eval_timeout_seconds must be a positive integer")
    if timeout_seconds <= 0:
        raise ValueError("learning_curve.eval_timeout_seconds must be a positive integer")
    return rollout_steps, tuple(steps), timeout_seconds


def build_learning_curve_plan(
    spec: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Build symmetric checkpoint eval commands without touching checkpoints."""
    errors = validate_spec(spec)
    if errors:
        raise ValueError("invalid paired spec: " + "; ".join(errors))
    if spec.get("sampler_pair_contract") != OFFICIAL_PAIR_CONTRACT:
        raise ValueError(
            f"learning-curve protocol requires sampler_pair_contract={OFFICIAL_PAIR_CONTRACT!r}"
        )

    rollout_steps, checkpoint_iterations, eval_timeout_seconds = _learning_curve_config(spec)
    final_iteration = int(spec["expected_training_iterations"])
    if final_iteration != checkpoint_iterations[-1]:
        raise ValueError(
            "expected_training_iterations must equal the final learning-curve checkpoint "
            f"({checkpoint_iterations[-1]})"
        )
    num_envs = int(spec["training_num_envs"])
    seed = int(spec["seed"])
    initialization_checkpoint = str(spec["training_initialization_checkpoint"])

    variants = {str(variant["name"]): variant for variant in spec["variants"]}
    baseline_name = str(spec["variant_b"])
    treatment_name = str(spec["variant_a"])
    ordered_names = (baseline_name, treatment_name)
    if set(variants) != set(ordered_names):
        raise ValueError("variant_a and variant_b must identify exactly the paired variants")

    rows: list[dict[str, Any]] = []
    signatures: set[tuple[Any, ...] | None] = set()
    experiment_dirs: dict[str, str] = {}
    for variant_name in ordered_names:
        variant = variants[variant_name]
        train_command = _seeded(str(variant["train_command"]), seed)
        eval_command = _seeded(str(variant["eval_command"]), seed)
        experiment_dir = _single_override(
            train_command,
            "experiment_dir",
            label=f"{variant_name}.train_command",
        )
        experiment_dirs[variant_name] = experiment_dir

        for key, expected in (
            ("algo.config.num_steps_per_env", str(rollout_steps)),
            ("callbacks.model_save.save_frequency", "50"),
            ("callbacks.model_save.save_last_frequency", "50"),
        ):
            observed = _single_override(
                train_command,
                key,
                label=f"{variant_name}.train_command",
            )
            if observed != expected:
                raise ValueError(
                    f"{variant_name}.train_command must set {key}={expected}; got {observed}"
                )

        for iteration in checkpoint_iterations:
            if iteration == 0:
                checkpoint = initialization_checkpoint
                checkpoint_source = "shared_release_initialization"
            else:
                checkpoint = str(Path(experiment_dir) / f"model_step_{iteration:06d}.pt")
                checkpoint_source = "trained_iteration_checkpoint"
            route = output_dir / variant_name / f"iteration_{iteration:06d}"
            eval_output_dir = route / "eval_metrics"
            command = _replace_eval_routing(
                eval_command,
                checkpoint=checkpoint,
                output_dir=str(eval_output_dir),
            )
            signature = _paired_command_signature(
                command,
                ignored_override_keys=_PAIR_EVAL_OUTPUT_KEYS,
            )
            signatures.add(signature)
            rows.append(
                {
                    "variant": variant_name,
                    "role": "baseline" if variant_name == baseline_name else "treatment",
                    "checkpoint_iteration": iteration,
                    "checkpoint": checkpoint,
                    "checkpoint_source": checkpoint_source,
                    "sample_budget_env_transitions": iteration * num_envs * rollout_steps,
                    "eval_output_dir": str(eval_output_dir),
                    "metrics_eval_json": str(eval_output_dir / "metrics_eval.json"),
                    "eval_log": str(route / "eval.log"),
                    "eval_command": command,
                    "shared_evaluation_id": (
                        "release_initialization" if iteration == 0 else None
                    ),
                }
            )

    if None in signatures or len(signatures) != 1:
        raise ValueError(
            "learning-curve eval commands differ outside checkpoint and eval_output_dir routing"
        )

    for iteration in checkpoint_iterations:
        checkpoint_rows = [row for row in rows if row["checkpoint_iteration"] == iteration]
        budgets = {row["sample_budget_env_transitions"] for row in checkpoint_rows}
        if len(checkpoint_rows) != 2 or len(budgets) != 1:
            raise ValueError(f"checkpoint iteration {iteration} is not exactly paired")
        if iteration == 0 and len({row["checkpoint"] for row in checkpoint_rows}) != 1:
            raise ValueError("iteration-zero arms must use the identical release checkpoint")

    return {
        "schema_version": 1,
        "kind": "sonic_paired_learning_curve_plan",
        "protocol": PROTOCOL,
        "paired_spec_canonical_sha256": _canonical_sha256(spec),
        "experiment_group": spec["experiment_group"],
        "seed": seed,
        "baseline_variant": baseline_name,
        "treatment_variant": treatment_name,
        "checkpoint_iterations": list(checkpoint_iterations),
        "training_num_envs": num_envs,
        "rollout_steps_per_iteration": rollout_steps,
        "eval_timeout_seconds": eval_timeout_seconds,
        "sample_budget_unit": "policy_environment_transitions",
        "sample_budget_formula": "checkpoint_iteration * training_num_envs * rollout_steps_per_iteration",
        "metric_sources": dict(METRIC_SOURCES),
        "auc_rule": "trapezoidal_over_sample_budget; normalized_auc=raw_auc/final_budget",
        "experiment_dirs": experiment_dirs,
        "commands_equal_outside_checkpoint_and_eval_output_dir": True,
        "gpu_eval_invocations_expected": 9,
        "shared_initialization_reuse": True,
        "rows": rows,
    }


def _resolve(path_text: str, repo_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_root / path


def _load_official_metrics(path: Path) -> dict[str, float]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read official ImEvalCallback metrics {path}: {exc}") from exc
    metrics: dict[str, float] = {}
    for metric, source in METRIC_SOURCES.items():
        value = payload.get(source) if isinstance(payload, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path} must contain numeric {source}")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{path} contains non-finite {source}")
        if metric.endswith("_rate") and not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{path} {source} must be in [0, 1]")
        if metric.startswith("mpjpe_") and numeric < 0.0:
            raise ValueError(f"{path} {source} must be non-negative")
        metrics[metric] = numeric
    return metrics


def _trapezoidal_auc(points: list[tuple[int, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("AUC requires at least two points")
    points = sorted(points)
    if len({x for x, _ in points}) != len(points):
        raise ValueError("AUC sample budgets must be unique")
    raw = sum(
        (right_x - left_x) * (left_y + right_y) / 2.0
        for (left_x, left_y), (right_x, right_y) in zip(points, points[1:])
    )
    span = points[-1][0] - points[0][0]
    if span <= 0:
        raise ValueError("AUC sample-budget span must be positive")
    return raw, raw / span


def aggregate_learning_curve(plan: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Aggregate complete official callback JSONs into paired rows and AUCs."""
    if plan.get("protocol") != PROTOCOL:
        raise ValueError(f"plan protocol must be {PROTOCOL!r}")
    baseline_name = str(plan["baseline_variant"])
    treatment_name = str(plan["treatment_variant"])
    expected_iterations = list(CHECKPOINT_ITERATIONS)
    if plan.get("checkpoint_iterations") != expected_iterations:
        raise ValueError(f"plan checkpoints must be exactly {expected_iterations}")

    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for row in plan.get("rows", []):
        key = (str(row["variant"]), int(row["checkpoint_iteration"]))
        if key in by_key:
            raise ValueError(f"duplicate learning-curve row {key}")
        by_key[key] = row

    paired_rows: list[dict[str, Any]] = []
    arm_points = {
        baseline_name: {metric: [] for metric in METRIC_SOURCES},
        treatment_name: {metric: [] for metric in METRIC_SOURCES},
    }
    for iteration in expected_iterations:
        arm_metrics: dict[str, dict[str, float]] = {}
        budgets: set[int] = set()
        for variant_name in (baseline_name, treatment_name):
            key = (variant_name, iteration)
            if key not in by_key:
                raise ValueError(f"missing learning-curve row {key}")
            source_row = by_key[key]
            budget = int(source_row["sample_budget_env_transitions"])
            budgets.add(budget)
            metrics = _load_official_metrics(
                _resolve(str(source_row["metrics_eval_json"]), repo_root)
            )
            arm_metrics[variant_name] = metrics
            for metric, value in metrics.items():
                arm_points[variant_name][metric].append((budget, value))
        if len(budgets) != 1:
            raise ValueError(f"iteration {iteration} has mismatched sample budgets")
        baseline = arm_metrics[baseline_name]
        treatment = arm_metrics[treatment_name]
        deltas = {metric: treatment[metric] - baseline[metric] for metric in METRIC_SOURCES}
        paired_rows.append(
            {
                "checkpoint_iteration": iteration,
                "sample_budget_env_transitions": budgets.pop(),
                "baseline": baseline,
                "treatment": treatment,
                "delta_treatment_minus_baseline": deltas,
                "zpd_advantage": {
                    "success_rate": deltas["success_rate"],
                    "progress_rate": deltas["progress_rate"],
                    "mpjpe_l": -deltas["mpjpe_l"],
                    "mpjpe_g": -deltas["mpjpe_g"],
                },
            }
        )

    auc: dict[str, Any] = {}
    for metric in METRIC_SOURCES:
        baseline_raw, baseline_normalized = _trapezoidal_auc(
            arm_points[baseline_name][metric]
        )
        treatment_raw, treatment_normalized = _trapezoidal_auc(
            arm_points[treatment_name][metric]
        )
        direct_delta = treatment_normalized - baseline_normalized
        auc[metric] = {
            "beneficial_direction": "higher" if metric.endswith("_rate") else "lower",
            "baseline_raw_auc": baseline_raw,
            "treatment_raw_auc": treatment_raw,
            "baseline_normalized_auc": baseline_normalized,
            "treatment_normalized_auc": treatment_normalized,
            "normalized_auc_delta_treatment_minus_baseline": direct_delta,
            "normalized_auc_zpd_advantage": (
                direct_delta if metric.endswith("_rate") else -direct_delta
            ),
        }

    return {
        "schema_version": 1,
        "kind": "sonic_paired_learning_curve",
        "protocol": PROTOCOL,
        "experiment_group": plan["experiment_group"],
        "seed": plan["seed"],
        "baseline_variant": baseline_name,
        "treatment_variant": treatment_name,
        "sample_budget_unit": plan["sample_budget_unit"],
        "metric_sources": dict(METRIC_SOURCES),
        "rows": paired_rows,
        "auc": auc,
        "complete": True,
    }


def _execution_preflight(
    plan: dict[str, Any],
    *,
    spec: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fail before the first eval if any immutable input or output route is unsafe."""
    if plan.get("paired_spec_canonical_sha256") != _canonical_sha256(spec):
        raise ValueError("learning-curve plan does not match the supplied paired spec")
    if plan.get("protocol") != PROTOCOL:
        raise ValueError(f"plan protocol must be {PROTOCOL!r}")
    _, _, expected_timeout_seconds = _learning_curve_config(spec)
    if plan.get("eval_timeout_seconds") != expected_timeout_seconds:
        raise ValueError("learning-curve plan timeout does not match the supplied paired spec")

    expected_pairs = {
        (variant, iteration)
        for variant in (str(plan["baseline_variant"]), str(plan["treatment_variant"]))
        for iteration in CHECKPOINT_ITERATIONS
    }
    rows = plan.get("rows")
    if not isinstance(rows, list) or len(rows) != len(expected_pairs):
        raise ValueError("execute requires exactly 10 planned evaluation rows")
    actual_pairs = {
        (str(row.get("variant")), row.get("checkpoint_iteration"))
        for row in rows
        if isinstance(row, dict)
    }
    if actual_pairs != expected_pairs:
        raise ValueError("execute rows do not exactly cover both arms at all five checkpoints")
    shared_rows = [
        row for row in rows if row.get("shared_evaluation_id") == "release_initialization"
    ]
    if (
        len(shared_rows) != 2
        or {row.get("checkpoint_iteration") for row in shared_rows} != {0}
        or len({row.get("checkpoint") for row in shared_rows}) != 1
        or any(
            row.get("shared_evaluation_id") not in (None, "release_initialization")
            for row in rows
        )
    ):
        raise ValueError(
            "iteration-zero arms must be the only two rows sharing release_initialization"
        )

    # This verifies the manifest hash, every robot/SMPL file hash, exact flat
    # inventory, command dataset bindings, release checkpoint, and paired axis.
    input_provenance = _verify_official_zpd_pair_preflight(spec, repo_root=repo_root)
    manifest = input_provenance.get("dataset_manifest")
    motion_keys = manifest.get("motion_keys") if isinstance(manifest, dict) else None
    if not isinstance(motion_keys, list) or not motion_keys:
        raise ValueError("verified dataset manifest did not provide motion_keys")

    prepared: list[dict[str, Any]] = []
    seen_logs: set[Path] = set()
    seen_metrics: set[Path] = set()
    seen_routes: set[Path] = set()
    signatures: set[tuple[Any, ...] | None] = set()
    for index, row in enumerate(rows):
        command = str(row.get("eval_command") or "")
        environment, argv = _parse_command(command)
        signatures.add(
            _paired_command_signature(
                command,
                ignored_override_keys=_PAIR_EVAL_OUTPUT_KEYS,
            )
        )
        checkpoint_text = str(row.get("checkpoint") or "")
        eval_output_text = str(row.get("eval_output_dir") or "")
        if _command_overrides(command).get("checkpoint") != [checkpoint_text]:
            raise ValueError(f"rows[{index}] command checkpoint does not match its plan row")
        if _command_overrides(command).get("eval_output_dir") != [eval_output_text]:
            raise ValueError(f"rows[{index}] command eval_output_dir does not match its plan row")

        checkpoint = _resolve(checkpoint_text, repo_root).resolve()
        eval_log = _resolve(str(row.get("eval_log") or ""), repo_root).resolve()
        metrics = _resolve(str(row.get("metrics_eval_json") or ""), repo_root).resolve()
        eval_output = _resolve(eval_output_text, repo_root).resolve()
        route = eval_log.parent
        if not checkpoint.is_file():
            raise ValueError(f"planned evaluation checkpoint does not exist: {checkpoint}")
        if eval_log.exists():
            raise ValueError(f"refusing to overwrite existing eval log: {eval_log}")
        if metrics.exists():
            raise ValueError(f"refusing to overwrite existing eval metrics: {metrics}")
        if eval_output.exists():
            raise ValueError(f"refusing to overwrite existing eval output route: {eval_output}")
        if route.exists():
            raise ValueError(f"refusing to overwrite existing eval output route: {route}")
        if eval_output.parent != route:
            raise ValueError(f"rows[{index}] eval_output_dir must be directly inside its eval route")
        if metrics != eval_output / "metrics_eval.json":
            raise ValueError(f"rows[{index}] metrics_eval_json must be inside eval_output_dir")
        if eval_log in seen_logs or metrics in seen_metrics or route in seen_routes:
            raise ValueError("planned evaluation output routes must be pairwise unique")
        seen_logs.add(eval_log)
        seen_metrics.add(metrics)
        seen_routes.add(route)
        prepared.append(
            {
                "row": row,
                "environment": environment,
                "argv": argv,
                "checkpoint": checkpoint,
                "eval_log": eval_log,
                "metrics": metrics,
                "eval_output": eval_output,
                "route": route,
            }
        )

    if None in signatures or len(signatures) != 1:
        raise ValueError(
            "execute eval commands differ outside checkpoint and eval_output_dir routing"
        )

    for name in (
        "learning_curve.json",
        "learning_curve.csv",
        "learning_curve.md",
        "learning_curve_execution.json",
    ):
        target = _resolve(str(output_dir / name), repo_root).resolve()
        if target.exists():
            raise ValueError(f"refusing to overwrite existing learning-curve output: {target}")
    return prepared, input_provenance


def execute_learning_curve(
    plan: dict[str, Any],
    *,
    spec: dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Run all planned eval rows sequentially and aggregate only after all pass."""
    prepared, input_provenance = _execution_preflight(
        plan,
        spec=spec,
        output_dir=output_dir,
        repo_root=repo_root,
    )
    expected_motion_keys = list(input_provenance["dataset_manifest"]["motion_keys"])
    execution_rows: list[dict[str, Any]] = []
    shared_metrics: dict[str, Path] = {}

    for index, item in enumerate(prepared):
        route = item["route"]
        eval_log = item["eval_log"]
        metrics = item["metrics"]
        shared_id = item["row"].get("shared_evaluation_id")
        route.mkdir(parents=True, exist_ok=False)
        reused_shared_initialization = bool(shared_id and shared_id in shared_metrics)
        if reused_shared_initialization:
            source_metrics = shared_metrics[str(shared_id)]
            metrics.parent.mkdir(parents=True, exist_ok=False)
            shutil.copy2(source_metrics, metrics)
            eval_log.write_text(
                "Reused content-identical official ImEval metrics from the shared "
                f"release initialization: {source_metrics}\n",
                encoding="utf-8",
            )
            if _file_sha256(metrics) != _file_sha256(source_metrics):
                raise ValueError("shared initialization metrics copy changed content")
            returncode = 0
        else:
            try:
                with eval_log.open("x", encoding="utf-8") as log_file:
                    completed = subprocess.run(
                        list(item["argv"]),
                        cwd=str(repo_root),
                        env=_command_environment(item["environment"]),
                        text=True,
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        check=False,
                        shell=False,
                        timeout=int(plan["eval_timeout_seconds"]),
                    )
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    f"evaluation row {index} exceeded timeout "
                    f"{plan['eval_timeout_seconds']}s; log={eval_log}"
                ) from exc
            returncode = completed.returncode
            if returncode != 0:
                raise ValueError(
                    f"evaluation row {index} failed with return code {returncode}; "
                    f"log={eval_log}"
                )
        if not metrics.is_file():
            raise ValueError(
                f"evaluation row {index} did not create fresh official metrics: {metrics}"
            )
        if shared_id and shared_id not in shared_metrics:
            shared_metrics[str(shared_id)] = metrics
        coverage = _verify_metrics_motion_coverage(metrics, expected_motion_keys)
        # Validate the curve metrics immediately so a malformed row stops
        # the sequence instead of failing only after all GPU evaluations finish.
        official_metrics = _load_official_metrics(metrics)
        checkpoint_stat = item["checkpoint"].stat()
        execution_rows.append(
            {
                "variant": item["row"]["variant"],
                "checkpoint_iteration": item["row"]["checkpoint_iteration"],
                "checkpoint": str(item["checkpoint"]),
                "checkpoint_size_bytes": checkpoint_stat.st_size,
                "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                "eval_log": str(eval_log),
                "returncode": returncode,
                "shell": False,
                "reused_shared_initialization": reused_shared_initialization,
                "metrics_sha256": _file_sha256(metrics),
                "metrics": official_metrics,
                "metrics_coverage": coverage,
                "fresh_metrics_created_in_new_route": True,
            }
        )

    if len(execution_rows) != 10:
        raise ValueError("all 10 learning-curve evaluations must pass before aggregation")
    result = aggregate_learning_curve(plan, repo_root=repo_root)
    result["execution"] = {
        "mode": "sequential_shell_free_with_shared_initialization",
        "all_10_rows_passed": True,
        "gpu_eval_invocations": sum(
            not row["reused_shared_initialization"] for row in execution_rows
        ),
        "shared_initialization_reuses": sum(
            row["reused_shared_initialization"] for row in execution_rows
        ),
        "refused_existing_outputs": True,
        "dataset_manifest": input_provenance["dataset_manifest"],
        "rows": execution_rows,
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_plan_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        "# SONIC Paired Learning-Curve Plan",
        "",
        f"Protocol: `{plan['protocol']}`. Sample budget is measured in policy environment transitions.",
        "",
        "| Arm | Iteration | Sample budget | Checkpoint | Metrics JSON |",
        "|---|---:|---:|---|---|",
    ]
    for row in plan["rows"]:
        lines.append(
            f"| `{row['variant']}` | {row['checkpoint_iteration']} | "
            f"{row['sample_budget_env_transitions']} | `{row['checkpoint']}` | "
            f"`{row['metrics_eval_json']}` |"
        )
    lines.extend(["", "## Evaluation Commands", ""])
    for row in plan["rows"]:
        lines.extend(
            [
                f"### {row['variant']} at iteration {row['checkpoint_iteration']}",
                "",
                "```bash",
                row["eval_command"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_curve_outputs(output_dir: Path, result: dict[str, Any]) -> None:
    _write_json(output_dir / "learning_curve.json", result)
    fieldnames = ["checkpoint_iteration", "sample_budget_env_transitions"]
    for prefix in ("baseline", "treatment", "delta_treatment_minus_baseline", "zpd_advantage"):
        fieldnames.extend(f"{prefix}.{metric}" for metric in METRIC_SOURCES)
    with (output_dir / "learning_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["rows"]:
            flat = {
                "checkpoint_iteration": row["checkpoint_iteration"],
                "sample_budget_env_transitions": row["sample_budget_env_transitions"],
            }
            for prefix in (
                "baseline",
                "treatment",
                "delta_treatment_minus_baseline",
                "zpd_advantage",
            ):
                for metric, value in row[prefix].items():
                    flat[f"{prefix}.{metric}"] = value
            writer.writerow(flat)

    baseline = result["baseline_variant"]
    treatment = result["treatment_variant"]
    lines = [
        "# SONIC Paired Learning Curve",
        "",
        "| Iteration | Env transitions | Baseline success | ZPD success | Δ success | "
        "Baseline progress | ZPD progress | Δ progress | "
        "Baseline MPJPE-L | ZPD MPJPE-L | Δ MPJPE-L | Baseline MPJPE-G | ZPD MPJPE-G | Δ MPJPE-G |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        b = row["baseline"]
        t = row["treatment"]
        d = row["delta_treatment_minus_baseline"]
        lines.append(
            f"| {row['checkpoint_iteration']} | {row['sample_budget_env_transitions']} | "
            f"{b['success_rate']:.6g} | {t['success_rate']:.6g} | {d['success_rate']:.6g} | "
            f"{b['progress_rate']:.6g} | {t['progress_rate']:.6g} | {d['progress_rate']:.6g} | "
            f"{b['mpjpe_l']:.6g} | {t['mpjpe_l']:.6g} | {d['mpjpe_l']:.6g} | "
            f"{b['mpjpe_g']:.6g} | {t['mpjpe_g']:.6g} | {d['mpjpe_g']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Baseline is `{baseline}`; treatment is `{treatment}`. Deltas are treatment minus baseline.",
            "",
            "## Normalized AUC",
            "",
            "| Metric | Better | Baseline | Treatment | Treatment − baseline | ZPD advantage |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for metric, record in result["auc"].items():
        lines.append(
            f"| `{metric}` | {record['beneficial_direction']} | "
            f"{record['baseline_normalized_auc']:.6g} | "
            f"{record['treatment_normalized_auc']:.6g} | "
            f"{record['normalized_auc_delta_treatment_minus_baseline']:.6g} | "
            f"{record['normalized_auc_zpd_advantage']:.6g} |"
        )
    (output_dir / "learning_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--aggregate",
        action="store_true",
        help="Require all planned metrics_eval.json files and write learning-curve/AUC tables.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Fail-closed verification of all 10 rows using 9 GPU evals, then aggregate.",
    )
    args = parser.parse_args()

    try:
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        plan = build_learning_curve_plan(spec, output_dir=args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "learning_curve_plan.json", plan)
        write_plan_markdown(args.output_dir / "learning_curve_plan.md", plan)
        if args.execute:
            result = execute_learning_curve(
                plan,
                spec=spec,
                output_dir=args.output_dir,
                repo_root=args.repo_root.resolve(),
            )
            write_curve_outputs(args.output_dir, result)
            _write_json(
                args.output_dir / "learning_curve_execution.json",
                result["execution"],
            )
        elif args.aggregate:
            result = aggregate_learning_curve(plan, repo_root=args.repo_root)
            write_curve_outputs(args.output_dir, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"wrote learning-curve plan to {args.output_dir / 'learning_curve_plan.json'}")
    if args.aggregate or args.execute:
        print(f"wrote paired learning curve to {args.output_dir / 'learning_curve.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
