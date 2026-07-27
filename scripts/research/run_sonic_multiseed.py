#!/usr/bin/env python3
"""Run a multi-seed paired SONIC experiment from one seed-templated spec.

Replaces the untracked per-milestone ``run_sim_m*.py`` orchestrators: one
tracked spec template with literal ``{seed}`` placeholders is rendered per seed,
each seed goes through ``run_sonic_paired_experiment.materialize_paired_experiment``
(same dry-run/execute semantics), and the resulting seed-level comparisons are
aggregated with the preregistered effect gate and paired statistics.

The template must contain ``{seed}`` in at least one string value (commands and
exp_var names must differ per seed) — a template without it would silently run
identical seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from scripts.research.aggregate_sonic_comparisons import (  # noqa: E402
    build_aggregate_comparison,
    validate_effect_metric,
    write_aggregate_json,
    write_aggregate_table_markdown,
)
from scripts.research.run_sonic_paired_experiment import (  # noqa: E402
    _activation_launch_contract_errors,
    _command_overrides,
    _execute_preflight_errors,
    _load_json,
    _parse_command,
    _relative_or_absolute,
    _sha256_file,
    _verify_activation_preflight,
    _verify_flat_paired_inventory as _verify_flat_paired_inventory,
    _write_json,
    materialize_paired_experiment,
    validate_spec,
)
from scripts.research.summarize_sampler_telemetry import (  # noqa: E402
    parse_sampler_telemetry,
)

_SEED_PLACEHOLDER = "{seed}"
_DEFAULT_VARIANT_A = "adaptive_sampling_micro"
_DEFAULT_VARIANT_B = "uniform_sampling_micro"
_EXPECTED_ACTIVATION_SEEDS = frozenset((0, 1, 2))
_TRAIN_IDENTITY_OVERRIDES = frozenset({"seed", "exp_var", "experiment_dir"})
_EVAL_IDENTITY_OVERRIDES = frozenset({"checkpoint", "eval_output_dir"})
_ZPD_ACTIVATION_ARMS = frozenset({"m5_l", "m5_a"})
_BASE_SEED_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint_sha256",
        "checkpoint_global_step",
        "checkpoint_dump_sha256",
        "difficulty_ranking_sha256",
        "telemetry_summary_sha256",
        "dataset_manifest_sha256",
        "dataset_manifest_sha256_kind",
        "paired_dataset_sha256",
    }
)
_ZPD_SEED_PROVENANCE_KEYS = frozenset({"sampler_schema", "sampler_config"})


def _render(value: Any, seed: int) -> Any:
    if isinstance(value, str):
        return value.replace(_SEED_PLACEHOLDER, str(seed))
    if isinstance(value, dict):
        return {key: _render(item, seed) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, seed) for item in value]
    return value


def _template_has_placeholder(template: dict[str, Any]) -> bool:
    # Rendering changes the template iff a {seed} placeholder exists somewhere.
    return _render(template, 0) != template


def render_spec_for_seed(template: dict[str, Any], seed: int) -> dict[str, Any]:
    """Render one seed's spec: substitute {seed} in strings, set the seed field."""
    spec = _render(template, seed)
    spec["seed"] = seed
    return spec


def validate_m5_multiseed_contract(spec: dict[str, Any]) -> list[str]:
    """Compatibility wrapper around the paired activation launch contract."""
    errors = _activation_launch_contract_errors(spec)
    for index, variant in enumerate(spec.get("variants", [])):
        if not isinstance(variant, dict):
            continue
        for field in ("train_command", "eval_command"):
            try:
                _parse_command(str(variant.get(field) or ""))
            except ValueError as exc:
                errors.append(f"variants[{index}].{field} is unsafe or unparseable: {exc}")
    return errors


def _verify_training_runtime_inputs(
    spec: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Run the complete non-GPU M5 launch preflight and return provenance."""
    errors = validate_spec(spec)
    errors.extend(_execute_preflight_errors(spec, repo_root=repo_root))
    if errors:
        raise ValueError("M5 preflight failed: " + "; ".join(errors))
    return _verify_activation_preflight(spec, repo_root=repo_root)


def _command_scientific_signature(command: str, *, command_kind: str) -> tuple[Any, ...]:
    environment, argv = _parse_command(command)
    identity_keys = _TRAIN_IDENTITY_OVERRIDES if command_kind == "train" else _EVAL_IDENTITY_OVERRIDES
    normalized_argv: list[str] = []
    for token in argv:
        normalized = token.lstrip("+~")
        prefix = token[: len(token) - len(normalized)]
        if "=" in normalized:
            key, _ = normalized.split("=", 1)
            if key in identity_keys:
                token = f"{prefix}{key}=<seed_or_output_identity>"
        normalized_argv.append(token)
    return tuple(sorted(environment.items())), tuple(normalized_argv)


def _verify_multiseed_scientific_invariants(rendered_specs: list[dict[str, Any]], *, repo_root: Path) -> None:
    """Allow seed/output identities to vary while freezing scientific settings."""
    baseline: tuple[Any, ...] | None = None
    output_paths: list[Path] = []
    metrics_paths: list[Path] = []
    initialization_path = _relative_or_absolute(
        str(rendered_specs[0]["training_initialization_checkpoint"]), repo_root
    ).resolve()
    for spec in rendered_specs:
        variant_signatures: list[tuple[Any, ...]] = []
        for variant in spec["variants"]:
            output_path = _relative_or_absolute(str(variant["checkpoint"]), repo_root).resolve()
            output_paths.append(output_path)
            metrics_paths.append(_relative_or_absolute(str(variant["metrics_eval_json"]), repo_root).resolve())
            variant_signatures.append(
                (
                    variant.get("name"),
                    variant.get("checkpoint_source"),
                    variant.get("interpretation"),
                    _command_scientific_signature(str(variant["train_command"]), command_kind="train"),
                    _command_scientific_signature(str(variant["eval_command"]), command_kind="eval"),
                )
            )
        signature = tuple(variant_signatures)
        if baseline is None:
            baseline = signature
        elif signature != baseline:
            raise ValueError(
                "M5 rendered seeds vary in scientific/sampler/schedule command "
                "settings outside the allowed seed/output identity fields"
            )
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("M5 trained output checkpoints must be unique across all arms and seeds")
    if initialization_path in output_paths:
        raise ValueError("M5 trained output checkpoint must not collide with the initialization checkpoint")
    if len(metrics_paths) != len(set(metrics_paths)):
        raise ValueError("M5 eval metric outputs must be unique across all arms and seeds")


def preflight_multiseed_experiment(
    template: dict[str, Any],
    *,
    seeds: list[int],
    output_dir: Path,
    repo_root: Path,
    allow_identical_existing: bool = False,
) -> dict[str, Any]:
    """Validate all M5 launch inputs without running training or evaluation.

    Explicit preflight calls fail closed when ``preflight.json`` exists. Execute
    may opt into reuse, but only after freshly recomputing the complete report
    and proving semantic equality without rewriting the existing artifact.
    """
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    if set(seeds) != _EXPECTED_ACTIVATION_SEEDS:
        raise ValueError("M5 preflight requires the preregistered seeds 0 1 2")
    if template.get("requires_activation_gate") is not True:
        raise ValueError("--preflight is reserved for activation-gated M5 templates")
    if not _template_has_placeholder(template):
        raise ValueError("spec template must contain a literal {seed} placeholder")

    rendered_specs = [render_spec_for_seed(template, seed) for seed in seeds]
    shared_fields = (
        "activation_arm",
        "variant_a",
        "variant_b",
        "training_python_executable",
        "training_initialization_checkpoint",
        "dataset_robot",
        "dataset_smpl",
        "dataset_manifest_json",
        "dataset_manifest_sha256",
        "difficulty_ranking_json",
        "difficulty_ranking_sha256",
        "sim_d1_classification_json",
        "sim_d1_classification_sha256",
        "sim_d1_source_checkpoint_sha256",
        "expected_training_iterations",
        "training_num_envs",
        "retention_metric",
        "retention_a_minus_b_threshold",
    )
    for field in shared_fields:
        values = {str(spec.get(field)) for spec in rendered_specs}
        if len(values) != 1:
            raise ValueError(f"M5 seeds must share one frozen {field}")

    for spec in rendered_specs:
        errors = validate_spec(spec)
        errors.extend(_execute_preflight_errors(spec, repo_root=repo_root))
        if errors:
            raise ValueError(f"M5 seed {spec['seed']} preflight failed: " + "; ".join(errors))

    _verify_multiseed_scientific_invariants(rendered_specs, repo_root=repo_root)
    report_path = (output_dir / "preflight.json").resolve()
    input_paths = {
        _relative_or_absolute(str(rendered_specs[0][field]), repo_root).resolve()
        for field in (
            "training_python_executable",
            "training_initialization_checkpoint",
            "dataset_manifest_json",
            "difficulty_ranking_json",
            "sim_d1_classification_json",
        )
    }
    if report_path in input_paths:
        raise ValueError(f"preflight report path collides with a verified input: {report_path}")
    existing_report: dict[str, Any] | None = None
    if report_path.exists() and not allow_identical_existing:
        raise ValueError(f"refusing to overwrite existing preflight report: {report_path}")
    if report_path.exists():
        existing_report = _load_json(report_path)

    input_provenance_by_seed: dict[str, Any] = {}
    for spec in rendered_specs:
        input_provenance_by_seed[str(spec["seed"])] = _verify_training_runtime_inputs(spec, repo_root=repo_root)
    report = {
        "schema_version": 1,
        "kind": "sonic_multiseed_preflight",
        "mode": "preflight_only",
        "seeds": seeds,
        "passes_preflight": True,
        "execution_started": False,
        "result_ready": False,
        "input_provenance_by_seed": input_provenance_by_seed,
    }
    if existing_report is not None:
        if existing_report != report:
            raise ValueError(
                "existing preflight report is stale or does not match the freshly "
                f"recomputed report: {report_path}"
            )
        return existing_report
    _write_json(report_path, report)
    return report


def _activation_gate_summary(
    template: dict[str, Any],
    *,
    activation_json: Path | None,
    repo_root: Path,
    seeds: list[int],
) -> dict[str, Any]:
    """Load the classifier output required by M5 and fail closed on any mismatch."""
    required = template.get("requires_activation_gate") is True
    expected_arm = template.get("activation_arm")
    configured_path = activation_json or template.get("activation_json")
    summary: dict[str, Any] = {
        "required": required,
        "expected_arm": expected_arm,
        "path": str(configured_path) if configured_path else None,
        "sha256": None,
        "arm_verdict": None,
        "passes_required_activation_gate": not required,
        "errors": [],
    }
    if not required:
        return summary
    if configured_path is None:
        summary["errors"].append("activation evidence is required; pass --activation-json or set activation_json")
        return summary

    path = _relative_or_absolute(str(configured_path), repo_root).resolve()
    summary["path"] = str(path)
    if not path.is_file():
        summary["errors"].append(f"activation JSON does not exist: {path}")
        return summary
    try:
        evidence = _load_json(path)
    except (OSError, ValueError) as exc:
        summary["errors"].append(str(exc))
        return summary

    summary["sha256"] = _sha256_file(path)
    summary["arm_verdict"] = evidence.get("arm_verdict")
    if evidence.get("schema_version") != 2:
        summary["errors"].append("activation JSON schema_version must be 2 (hash-bound provenance contract)")
    if evidence.get("kind") != "m5_activation_classification":
        summary["errors"].append("activation JSON kind must be m5_activation_classification")
    if evidence.get("arm") != expected_arm:
        summary["errors"].append(f"activation JSON arm must be {expected_arm!r}, got {evidence.get('arm')!r}")
    expected_seed_keys = {str(seed) for seed in seeds}
    verdicts = evidence.get("seed_verdicts")
    if not isinstance(verdicts, dict) or set(verdicts) != expected_seed_keys:
        summary["errors"].append("activation JSON seed_verdicts must cover exactly the executed seeds")
        verdicts = {}

    detailed = evidence.get("seeds")
    if not isinstance(detailed, dict) or set(detailed) != expected_seed_keys:
        summary["errors"].append("activation JSON seeds must cover exactly the executed seeds")
        detailed = {}
    detailed_verdicts: dict[str, str] = {}
    for seed_key, record in detailed.items():
        if not isinstance(record, dict) or not isinstance(record.get("verdict"), str):
            summary["errors"].append(f"activation JSON seeds[{seed_key!r}] must contain a string verdict")
            continue
        detailed_verdicts[seed_key] = record["verdict"]
    if verdicts and detailed_verdicts and verdicts != detailed_verdicts:
        summary["errors"].append("activation JSON seed_verdicts do not match detailed seed verdicts")

    disqualifying = {
        key: verdict
        for key, verdict in detailed_verdicts.items()
        if verdict == "incomplete" or verdict.startswith("invalid-")
    }
    activated_count = sum(verdict == "activated" for verdict in detailed_verdicts.values())
    summary["recomputed_seeds_activated"] = activated_count
    summary["disqualifying_seed_verdicts"] = disqualifying
    if disqualifying:
        summary["errors"].append("activation JSON contains incomplete or invalid detailed seed verdicts")
    if activated_count < 2:
        summary["errors"].append("activation JSON must contain at least two detailed activated seed verdicts")
    if evidence.get("seeds_activated") != activated_count:
        summary["errors"].append("activation JSON seeds_activated does not match detailed seed verdicts")
    if evidence.get("seeds_total") != len(expected_seed_keys):
        summary["errors"].append("activation JSON seeds_total does not match the preregistered seed count")
    if evidence.get("missing_seeds") != [] or evidence.get("unexpected_seeds") != []:
        summary["errors"].append("activation JSON reports missing or unexpected seeds")
    if evidence.get("arm_verdict") != "activated":
        summary["errors"].append("activation arm_verdict must be 'activated' before the M5 result gates pass")
    summary["passes_required_activation_gate"] = not summary["errors"]
    return summary


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _resolved_path(value: str | Path, repo_root: Path) -> Path:
    return _relative_or_absolute(str(value), repo_root).resolve()


def _only_variant(plan: dict[str, Any], name: str, *, seed: int) -> dict[str, Any]:
    variants = [
        variant
        for variant in plan.get("variants", [])
        if isinstance(variant, dict) and variant.get("name") == name
    ]
    if len(variants) != 1:
        raise ValueError(f"seed {seed} run_plan must contain exactly one treatment variant {name!r}")
    return variants[0]


def _executed_activation_bindings(
    output_dir: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Load immutable executed plans and derive the expected activation inputs."""
    run_path = output_dir / "multiseed_run.json"
    aggregate_path = output_dir / "aggregate_comparison.json"
    if not run_path.is_file() or not aggregate_path.is_file():
        raise ValueError("finalization requires existing multiseed_run.json and aggregate_comparison.json")
    run = _load_json(run_path)
    aggregate = _load_json(aggregate_path)
    if run.get("schema_version") != 1 or run.get("kind") != "sonic_multiseed_run":
        raise ValueError("multiseed_run.json has an unsupported schema or kind")
    if run.get("dry_run") is not False:
        raise ValueError("dry-run artifacts can never be finalized as results")
    if run.get("execution_ok") is not True:
        raise ValueError("multiseed execution_ok must be true before finalization")
    if run.get("ok_for_causal_comparison") is not True:
        raise ValueError("multiseed causal-comparison gate did not pass")
    if run.get("passes_non_activation_preregistered_result_gates") is not True:
        raise ValueError("non-activation preregistered result gates did not pass")
    if run.get("missing_comparison_seeds") or run.get("failed_execution_seeds"):
        raise ValueError("multiseed run reports missing or failed seeds")

    seeds = run.get("seeds")
    if not isinstance(seeds, list) or set(seeds) != _EXPECTED_ACTIVATION_SEEDS:
        raise ValueError("finalization requires the exact preregistered seeds 0, 1, 2")
    treatment_name = run.get("variant_a")
    if not isinstance(treatment_name, str) or not treatment_name:
        raise ValueError("multiseed run has no treatment variant_a")

    bindings: dict[str, dict[str, Any]] = {}
    shared: dict[str, Any] | None = None
    for seed in sorted(seeds):
        seed_dir = output_dir / f"seed{seed}"
        plan_path = seed_dir / "run_plan.json"
        spec_path = seed_dir / "spec.json"
        if not plan_path.is_file() or not spec_path.is_file():
            raise ValueError(f"seed {seed} is missing run_plan.json or spec.json")
        plan = _load_json(plan_path)
        spec = _load_json(spec_path)
        if (
            plan.get("kind") != "sonic_paired_experiment_run"
            or plan.get("dry_run") is not False
            or plan.get("seed") != seed
            or plan.get("execution_ok") is not True
        ):
            raise ValueError(f"seed {seed} run_plan is not a successful executed plan")
        if plan.get("training_completeness_ok") is not True:
            raise ValueError(f"seed {seed} did not prove terminal training completeness")
        if spec.get("requires_activation_gate") is not True:
            raise ValueError(f"seed {seed} spec is not activation-gated")

        arm = spec.get("activation_arm")
        expected_iterations = spec.get("expected_training_iterations")
        if (
            isinstance(expected_iterations, bool)
            or not isinstance(expected_iterations, int)
            or expected_iterations <= 0
        ):
            raise ValueError(f"seed {seed} has invalid expected_training_iterations")
        treatment = _only_variant(plan, treatment_name, seed=seed)
        completion = treatment.get("training_completion")
        if not isinstance(completion, dict) or completion.get("complete") is not True:
            raise ValueError(f"seed {seed} treatment training is incomplete")
        if completion.get("expected_learning_iteration") != expected_iterations:
            raise ValueError(f"seed {seed} treatment expected iteration mismatch")
        if completion.get("observed_learning_iteration") != expected_iterations:
            raise ValueError(f"seed {seed} treatment observed iteration mismatch")
        if (
            treatment.get("execution_ok") is not True
            or treatment.get("checkpoint_source") != "trained_variant_checkpoint"
        ):
            raise ValueError(f"seed {seed} treatment was not executed from a fresh checkpoint")
        metrics_coverage = treatment.get("metrics_coverage")
        if not isinstance(metrics_coverage, dict) or metrics_coverage.get("exact_motion_key_coverage") is not True:
            raise ValueError(f"seed {seed} treatment eval coverage is not exact")

        checkpoint_provenance = treatment.get("checkpoint_provenance")
        checkpoint_sha256 = (
            checkpoint_provenance.get("sha256") if isinstance(checkpoint_provenance, dict) else None
        )
        if not _is_sha256(checkpoint_sha256):
            raise ValueError(f"seed {seed} treatment checkpoint SHA-256 is missing")
        checkpoint_path = _resolved_path(treatment["checkpoint"], repo_root)
        if not checkpoint_path.is_file():
            raise ValueError(f"seed {seed} treatment checkpoint no longer exists")
        if _sha256_file(checkpoint_path) != checkpoint_sha256:
            raise ValueError(f"seed {seed} treatment checkpoint changed after execution")

        command_results = treatment.get("command_results")
        train_results = [
            result
            for result in command_results or []
            if isinstance(result, dict) and result.get("kind") == "train"
        ]
        if len(train_results) != 1 or train_results[0].get("returncode") != 0:
            raise ValueError(f"seed {seed} has no unique successful treatment train command")
        train_log = _resolved_path(train_results[0]["log"], repo_root)
        if not train_log.is_file():
            raise ValueError(f"seed {seed} treatment training log no longer exists")

        input_provenance = plan.get("input_provenance")
        if not isinstance(input_provenance, dict):
            raise ValueError(f"seed {seed} input provenance is missing")
        manifest = input_provenance.get("dataset_manifest")
        ranking = input_provenance.get("difficulty_ranking")
        if not isinstance(manifest, dict) or not isinstance(ranking, dict):
            raise ValueError(f"seed {seed} dataset/ranking provenance is missing")
        static = {
            "arm": arm,
            "expected_iterations": expected_iterations,
            "dataset_manifest_sha256": manifest.get("sha256"),
            "dataset_manifest_sha256_kind": manifest.get("sha256_kind"),
            "paired_dataset_sha256": manifest.get("paired_dataset_sha256"),
            "difficulty_ranking_sha256": ranking.get("sha256"),
        }
        if not all(
            _is_sha256(static[field])
            for field in (
                "dataset_manifest_sha256",
                "paired_dataset_sha256",
                "difficulty_ranking_sha256",
            )
        ):
            raise ValueError(f"seed {seed} contains malformed hash-bound inputs")
        if static["dataset_manifest_sha256_kind"] not in {
            "file_bytes",
            "canonical_json_without_source_locations_v1",
        }:
            raise ValueError(f"seed {seed} has an unsupported manifest hash kind")
        if shared is None:
            shared = static
        elif static != shared:
            raise ValueError("executed seeds do not share one frozen activation contract")

        bindings[str(seed)] = {
            **static,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_global_step": expected_iterations,
            "train_log": str(train_log),
            "train_log_sha256": _sha256_file(train_log),
            "train_command": (treatment.get("commands") or {}).get("train"),
        }
    return run, aggregate, bindings


def _validate_sampler_command_binding(
    arm: str,
    command: str,
    sampler_config: dict[str, Any],
) -> list[str]:
    """Bind checkpoint sampler settings to explicit treatment command overrides."""
    if arm not in {"m5_l", "m5_a"}:
        return []
    errors: list[str] = []
    overrides = _command_overrides(command)
    prefix = "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."

    def one(field: str) -> str | None:
        values = overrides.get(prefix + field, [])
        if len(values) != 1:
            errors.append(f"treatment command must set adaptive_sampling.{field} exactly once")
            return None
        return values[0]

    if one("enable") != "true":
        errors.append("treatment command must enable adaptive sampling")
    expected_signal = "learnability" if arm == "m5_l" else "advantage_mass"
    if one("signal") != expected_signal:
        errors.append(f"treatment command must set signal={expected_signal}")
    paired_digest = one("paired_dataset_sha256")
    if paired_digest != sampler_config.get("paired_dataset_sha256"):
        errors.append(
            "treatment command adaptive_sampling.paired_dataset_sha256 does not match checkpoint sampler_config"
        )
    numeric_fields = (
        "optimism_k",
        "evidence_half_life",
        "advmass_n",
        "uniform_sampling_rate",
        "tripwire_max_prob_over_uniform",
        "bin_size",
    )
    for field in numeric_fields:
        command_value = one(field)
        configured_value = sampler_config.get(field)
        if command_value is None:
            continue
        try:
            matches = float(command_value) == float(configured_value)
        except (TypeError, ValueError):
            matches = False
        if not matches:
            errors.append(f"treatment command adaptive_sampling.{field} does not match checkpoint sampler_config")
    return errors


def _dump_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("checkpoints") if "checkpoints" in payload else [payload]
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("checkpoint dump must contain an object or a checkpoints list")
    return records


def finalize_multiseed_activation(
    *,
    output_dir: Path,
    activation_json: Path,
    telemetry_summary: Path,
    checkpoint_dumps: list[Path],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind activation evidence to an existing executed run without retraining."""
    output_dir = output_dir.resolve()
    run, aggregate, bindings = _executed_activation_bindings(output_dir, repo_root=repo_root)
    expected_arm = next(iter(bindings.values()))["arm"]
    summary = _activation_gate_summary(
        {"requires_activation_gate": True, "activation_arm": expected_arm},
        activation_json=activation_json,
        repo_root=repo_root,
        seeds=sorted(int(seed) for seed in bindings),
    )
    evidence_path = _resolved_path(activation_json, repo_root)
    evidence = _load_json(evidence_path)
    errors = list(summary["errors"])

    top_provenance = evidence.get("provenance")
    if not isinstance(top_provenance, dict):
        errors.append("activation JSON top-level provenance must be an object")
        top_provenance = {}
    shared = next(iter(bindings.values()))
    expected_top = {
        "dataset_manifest_sha256": shared["dataset_manifest_sha256"],
        "dataset_manifest_sha256_kind": shared["dataset_manifest_sha256_kind"],
        "paired_dataset_sha256": shared["paired_dataset_sha256"],
        "difficulty_ranking_sha256": shared["difficulty_ranking_sha256"],
    }
    for field, expected in expected_top.items():
        if top_provenance.get(field) != expected:
            errors.append(f"activation provenance {field} does not match executed run")
    if evidence.get("expected_iterations") != shared["expected_iterations"]:
        errors.append("activation expected_iterations does not match executed run")

    telemetry_path = _resolved_path(telemetry_summary, repo_root)
    if not telemetry_path.is_file():
        errors.append(f"telemetry summary does not exist: {telemetry_path}")
        telemetry = {}
    else:
        telemetry = _load_json(telemetry_path)
        telemetry_hash = _sha256_file(telemetry_path)
        if top_provenance.get("telemetry_summary_sha256") != telemetry_hash:
            errors.append("telemetry summary hash does not match activation provenance")
    telemetry_logs = telemetry.get("logs") if isinstance(telemetry, dict) else None
    if not isinstance(telemetry_logs, list):
        errors.append("telemetry summary logs must be a list")
        telemetry_logs = []
    expected_train_paths = {Path(binding["train_log"]).resolve(): seed for seed, binding in bindings.items()}
    matched_train_paths: set[Path] = set()
    for record in telemetry_logs:
        if not isinstance(record, dict) or not isinstance(record.get("log_path"), str):
            errors.append("telemetry summary contains a malformed log record")
            continue
        log_path = _resolved_path(record["log_path"], repo_root)
        if log_path not in expected_train_paths:
            errors.append(f"telemetry summary contains an unbound training log: {log_path}")
            continue
        matched_train_paths.add(log_path)
        reparsed = parse_sampler_telemetry(log_path.read_text(encoding="utf-8", errors="replace"))
        supplied = {key: value for key, value in record.items() if key != "log_path"}
        if reparsed != supplied:
            errors.append(f"telemetry summary does not reproduce training log {log_path}")
    if matched_train_paths != set(expected_train_paths):
        errors.append("telemetry summary does not cover exactly all treatment training logs")

    dump_by_sha: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for dump_path_value in checkpoint_dumps:
        dump_path = _resolved_path(dump_path_value, repo_root)
        if not dump_path.is_file():
            errors.append(f"checkpoint dump does not exist: {dump_path}")
            continue
        dump_by_sha[_sha256_file(dump_path)] = (
            dump_path,
            _dump_records(_load_json(dump_path)),
        )

    detailed = evidence.get("seeds")
    detailed = detailed if isinstance(detailed, dict) else {}
    bound_evidence: dict[str, Any] = {}
    required_provenance_keys = set(_BASE_SEED_PROVENANCE_KEYS)
    if expected_arm in _ZPD_ACTIVATION_ARMS:
        required_provenance_keys.update(_ZPD_SEED_PROVENANCE_KEYS)
    for seed, binding in bindings.items():
        seed_record = detailed.get(seed)
        provenance = seed_record.get("provenance") if isinstance(seed_record, dict) else None
        if not isinstance(provenance, dict):
            errors.append(f"activation seed {seed} provenance must be an object")
            continue
        if set(provenance) != required_provenance_keys:
            errors.append(
                f"activation seed {seed} provenance must contain exactly the "
                f"{expected_arm} schema-v2 fields"
            )
        expected_seed_fields = {
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "checkpoint_global_step": binding["checkpoint_global_step"],
            "difficulty_ranking_sha256": binding["difficulty_ranking_sha256"],
            "telemetry_summary_sha256": top_provenance.get("telemetry_summary_sha256"),
            "dataset_manifest_sha256": binding["dataset_manifest_sha256"],
            "dataset_manifest_sha256_kind": binding["dataset_manifest_sha256_kind"],
            "paired_dataset_sha256": binding["paired_dataset_sha256"],
        }
        for field, expected in expected_seed_fields.items():
            if provenance.get(field) != expected:
                errors.append(f"activation seed {seed} provenance {field} does not match executed run")
        dump_sha = provenance.get("checkpoint_dump_sha256")
        if not _is_sha256(dump_sha) or dump_sha not in dump_by_sha:
            errors.append(f"activation seed {seed} checkpoint dump is not supplied/hash-bound")
            continue
        dump_path, records = dump_by_sha[dump_sha]
        matching_records = [
            record
            for record in records
            if record.get("checkpoint_sha256") == binding["checkpoint_sha256"]
            and record.get("global_step") == binding["checkpoint_global_step"]
        ]
        if len(matching_records) != 1:
            errors.append(f"activation seed {seed} dump must contain one matching checkpoint record")
            continue
        dump_record = matching_records[0]
        if expected_arm in _ZPD_ACTIVATION_ARMS:
            if dump_record.get("sampler_schema") != provenance.get("sampler_schema"):
                errors.append(f"activation seed {seed} sampler_schema differs from dump")
            sampler_config = provenance.get("sampler_config")
            if dump_record.get("sampler_config") != sampler_config:
                errors.append(f"activation seed {seed} sampler_config differs from dump")
            if not isinstance(sampler_config, dict):
                errors.append(f"activation seed {seed} sampler_config must be an object")
                continue
            errors.extend(
                f"seed {seed}: {error}"
                for error in _validate_sampler_command_binding(
                    str(binding["arm"]),
                    str(binding["train_command"] or ""),
                    sampler_config,
                )
            )
        elif any(
            dump_record.get(field) is not None
            for field in ("sampler_schema", "sampler_config")
        ):
            errors.append(
                f"activation seed {seed} non-ZPD dump must not claim ZPD sampler provenance"
            )
        bound_evidence[seed] = {
            "checkpoint_dump": str(dump_path),
            "checkpoint_dump_sha256": dump_sha,
            "checkpoint_sha256": binding["checkpoint_sha256"],
            "checkpoint_global_step": binding["checkpoint_global_step"],
            "train_log": binding["train_log"],
            "train_log_sha256": binding["train_log_sha256"],
        }

    if errors:
        raise ValueError("activation finalization failed: " + "; ".join(errors))

    summary.update(
        {
            "finalized": True,
            "binding_kind": "executed_multiseed_checkpoint_and_log_v1",
            "telemetry_summary": str(telemetry_path),
            "telemetry_summary_sha256": _sha256_file(telemetry_path),
            "executed_seed_bindings": bound_evidence,
            "passes_required_activation_gate": True,
        }
    )
    result_ready = bool(aggregate.get("passes_non_activation_preregistered_result_gates") is True)
    aggregate["activation_summary"] = summary
    aggregate["passes_all_preregistered_result_gates"] = result_ready
    aggregate["result_gate_status"] = "result_ready" if result_ready else "result_gates_failed"
    run["activation_summary"] = summary
    run["passes_all_preregistered_result_gates"] = result_ready
    run["result_gate_status"] = aggregate["result_gate_status"]
    _write_json(output_dir / "aggregate_comparison.json", aggregate)
    write_aggregate_table_markdown(output_dir / "aggregate_table.md", aggregate)
    _write_json(output_dir / "multiseed_run.json", run)
    return run


def run_multiseed_experiment(
    template: dict[str, Any],
    *,
    seeds: list[int],
    output_dir: Path,
    dry_run: bool,
    repo_root: Path,
    variant_a: str | None,
    variant_b: str | None,
    effect_metric: str | None,
    a_minus_b_threshold: float,
    min_improved_seeds: int,
    retention_metric: str | None = None,
    retention_a_minus_b_threshold: float | None = None,
    activation_json: Path | None = None,
) -> dict[str, Any]:
    """Materialize every seed then aggregate the seed-level comparisons."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    # Validate result-gate configuration before writing plans or launching any
    # expensive train/eval subprocesses.
    validate_effect_metric(effect_metric)
    if template.get("requires_activation_gate") is True and set(seeds) != _EXPECTED_ACTIVATION_SEEDS:
        raise ValueError("activation-gated M5 templates require the preregistered seeds 0 1 2")
    if not _template_has_placeholder(template):
        raise ValueError(
            "spec template must contain a literal {seed} placeholder in at least one "
            "string value; otherwise all seeds would run identical commands"
        )
    contract_errors = validate_m5_multiseed_contract(template)
    if contract_errors:
        raise ValueError("invalid M5 multiseed contract: " + "; ".join(contract_errors))

    launch_input_provenance_by_seed: dict[str, Any] = {}
    preflight_json: str | None = None
    if not dry_run and template.get("requires_activation_gate") is True:
        preflight_report = preflight_multiseed_experiment(
            template,
            seeds=seeds,
            output_dir=output_dir,
            repo_root=repo_root,
            allow_identical_existing=True,
        )
        launch_input_provenance_by_seed = preflight_report["input_provenance_by_seed"]
        preflight_json = str(output_dir / "preflight.json")

    resolved_variant_a = str(variant_a or template.get("variant_a") or _DEFAULT_VARIANT_A)
    resolved_variant_b = str(variant_b or template.get("variant_b") or _DEFAULT_VARIANT_B)
    if resolved_variant_a == resolved_variant_b:
        raise ValueError("variant_a and variant_b must differ")

    template_variants = {
        str(variant.get("name"))
        for variant in template.get("variants", [])
        if isinstance(variant, dict) and variant.get("name")
    }
    unmatched = {resolved_variant_a, resolved_variant_b} - template_variants
    if template_variants and unmatched:
        raise ValueError(
            f"variant names {sorted(unmatched)} not found in spec template variants "
            f"{sorted(template_variants)}; pass matching --variant-a/--variant-b"
        )

    resolved_retention_metric = retention_metric or template.get("retention_metric")
    resolved_retention_threshold = (
        retention_a_minus_b_threshold
        if retention_a_minus_b_threshold is not None
        else float(template.get("retention_a_minus_b_threshold", 0.5))
    )

    seed_results: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    missing_comparison_seeds: list[int] = []
    failed_execution_seeds: list[int] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed{seed}"
        spec = render_spec_for_seed(template, seed)
        _write_json(seed_dir / "spec.json", spec)
        if launch_input_provenance_by_seed:
            expected_provenance = launch_input_provenance_by_seed[str(seed)]
            current_provenance = _verify_training_runtime_inputs(spec, repo_root=repo_root)
            if current_provenance != expected_provenance:
                raise ValueError(f"M5 seed {seed} runtime inputs changed after multiseed preflight")
        plan = materialize_paired_experiment(spec, output_dir=seed_dir, dry_run=dry_run, repo_root=repo_root)
        if launch_input_provenance_by_seed:
            if plan.get("input_provenance") != expected_provenance:
                raise ValueError(f"M5 seed {seed} runtime inputs changed after multiseed preflight")
        seed_results.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "comparison_json": plan.get("comparison_json"),
                "ok_for_causal_comparison": plan.get("ok_for_causal_comparison"),
                "execution_ok": plan.get("execution_ok"),
            }
        )
        if plan.get("execution_ok") is False:
            failed_execution_seeds.append(seed)
        if plan.get("comparison_json"):
            comparison_paths.append(Path(plan["comparison_json"]))
        else:
            missing_comparison_seeds.append(seed)

    aggregate: dict[str, Any] | None = None
    activation_summary = _activation_gate_summary(
        template,
        activation_json=activation_json,
        repo_root=repo_root,
        seeds=seeds,
    )
    if comparison_paths:
        aggregate = build_aggregate_comparison(
            comparison_paths,
            variant_a=resolved_variant_a,
            variant_b=resolved_variant_b,
            effect_metric=effect_metric,
            a_minus_b_threshold=a_minus_b_threshold,
            min_improved_seeds=min_improved_seeds,
            retention_metric=(str(resolved_retention_metric) if resolved_retention_metric else None),
            retention_a_minus_b_threshold=resolved_retention_threshold,
        )
        aggregate["dry_run"] = dry_run
        aggregate["passes_non_activation_preregistered_result_gates"] = bool(
            aggregate.get("passes_all_preregistered_result_gates") is True
        )
        if activation_summary["required"]:
            aggregate["activation_summary"] = activation_summary
        # A materialization-only run is never result-ready. Activation evidence
        # must also be bound to the executed checkpoints by the explicit
        # post-execution finalizer (added below), not merely present at launch.
        aggregate["passes_all_preregistered_result_gates"] = bool(
            not dry_run
            and not activation_summary["required"]
            and aggregate["passes_non_activation_preregistered_result_gates"]
        )
        if dry_run:
            aggregate["result_gate_status"] = "dry_run_not_result_ready"
        elif activation_summary["required"]:
            aggregate["result_gate_status"] = "awaiting_activation_finalization"
        elif aggregate["passes_all_preregistered_result_gates"]:
            aggregate["result_gate_status"] = "result_ready"
        else:
            aggregate["result_gate_status"] = "executed_result_gates_failed"
        write_aggregate_json(output_dir / "aggregate_comparison.json", aggregate)
        write_aggregate_table_markdown(output_dir / "aggregate_table.md", aggregate)

    run = {
        "schema_version": 1,
        "kind": "sonic_multiseed_run",
        "dry_run": dry_run,
        "seeds": seeds,
        "variant_a": resolved_variant_a,
        "variant_b": resolved_variant_b,
        "retention_metric": resolved_retention_metric,
        "retention_a_minus_b_threshold": resolved_retention_threshold,
        "output_dir": str(output_dir),
        "preflight_json": preflight_json,
        "input_provenance_by_seed": launch_input_provenance_by_seed,
        "seed_results": seed_results,
        "missing_comparison_seeds": missing_comparison_seeds,
        "failed_execution_seeds": failed_execution_seeds,
        "activation_summary": activation_summary,
        "aggregate_json": str(output_dir / "aggregate_comparison.json") if aggregate else None,
        # A seed that produced no comparison invalidates the whole run: the
        # aggregate (and its effect gate) would otherwise silently cover a
        # subset of the preregistered seeds.
        "ok_for_causal_comparison": (
            aggregate.get("ok_for_causal_comparison") is True
            and not missing_comparison_seeds
            and not failed_execution_seeds
            if aggregate
            else False
        ),
        "execution_ok": None if dry_run else not failed_execution_seeds,
        "passes_all_preregistered_result_gates": (
            aggregate.get("passes_all_preregistered_result_gates") is True
            and not missing_comparison_seeds
            and not failed_execution_seeds
            if aggregate
            else False
        ),
        "passes_non_activation_preregistered_result_gates": (
            aggregate.get("passes_non_activation_preregistered_result_gates") is True
            and not missing_comparison_seeds
            and not failed_execution_seeds
            if aggregate
            else False
        ),
        "result_gate_status": (aggregate.get("result_gate_status") if aggregate else "no_aggregate"),
    }
    _write_json(output_dir / "multiseed_run.json", run)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec-template",
        type=Path,
        help="Paired-experiment spec JSON with literal {seed} placeholders.",
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--preflight",
        action="store_true",
        help=(
            "Validate resolved M5 interpreter/checkpoint/D1/dataset inputs without "
            "launching training or evaluation."
        ),
    )
    mode.add_argument("--execute", action="store_true")
    mode.add_argument(
        "--finalize-activation",
        action="store_true",
        help="Bind activation evidence to an existing executed run without retraining.",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--variant-a",
        default=None,
        help="Treatment variant name; defaults to template variant_a or the legacy micro arm.",
    )
    parser.add_argument(
        "--variant-b",
        default=None,
        help="Control variant name; defaults to template variant_b or the legacy micro arm.",
    )
    parser.add_argument("--effect-metric", default="eval.all.mpjpe_g")
    parser.add_argument("--a-minus-b-threshold", type=float, default=-0.5)
    parser.add_argument("--min-improved-seeds", type=int, default=2)
    parser.add_argument(
        "--retention-metric",
        default=None,
        help="Frozen-ranking retention metric; defaults to the template value.",
    )
    parser.add_argument(
        "--retention-a-minus-b-threshold",
        type=float,
        default=None,
        help="Retention non-inferiority bound; defaults to template value or +0.5.",
    )
    parser.add_argument(
        "--activation-json",
        type=Path,
        default=None,
        help=(
            "M5 activation classifier output. Required before an activation-gated "
            "template can pass all preregistered result gates."
        ),
    )
    parser.add_argument(
        "--telemetry-summary",
        type=Path,
        default=None,
        help="Finalization only: telemetry JSON reproduced from treatment train logs.",
    )
    parser.add_argument(
        "--checkpoint-dump",
        type=Path,
        action="append",
        default=[],
        help="Finalization only: checkpoint dump; repeat for each file.",
    )
    args = parser.parse_args()

    if args.finalize_activation:
        if args.activation_json is None:
            parser.error("--finalize-activation requires --activation-json")
        if args.telemetry_summary is None:
            parser.error("--finalize-activation requires --telemetry-summary")
        if not args.checkpoint_dump:
            parser.error("--finalize-activation requires at least one --checkpoint-dump")
        try:
            run = finalize_multiseed_activation(
                output_dir=args.output_dir,
                activation_json=args.activation_json,
                telemetry_summary=args.telemetry_summary,
                checkpoint_dumps=args.checkpoint_dump,
                repo_root=args.repo_root,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"finalized activation-bound multiseed run at {args.output_dir / 'multiseed_run.json'}")
        return 0 if run["passes_all_preregistered_result_gates"] else 1

    if args.spec_template is None or args.seeds is None:
        parser.error("--dry-run/--preflight/--execute require --spec-template and --seeds")
    template = _load_json(args.spec_template)
    if args.preflight:
        try:
            preflight_multiseed_experiment(
                template,
                seeds=args.seeds,
                output_dir=args.output_dir,
                repo_root=args.repo_root,
            )
        except (OSError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        print(f"M5 preflight passed: {args.output_dir / 'preflight.json'}")
        return 0
    try:
        run = run_multiseed_experiment(
            template,
            seeds=args.seeds,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            repo_root=args.repo_root,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            effect_metric=args.effect_metric,
            a_minus_b_threshold=args.a_minus_b_threshold,
            min_improved_seeds=args.min_improved_seeds,
            retention_metric=args.retention_metric,
            retention_a_minus_b_threshold=args.retention_a_minus_b_threshold,
            activation_json=args.activation_json,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote multiseed run to {args.output_dir / 'multiseed_run.json'}")
    if run.get("aggregate_json"):
        print(f"wrote aggregate to {run['aggregate_json']}")
    if run.get("missing_comparison_seeds"):
        print(
            "WARNING: seed(s) "
            f"{run['missing_comparison_seeds']} produced no comparison.json; "
            "the aggregate covers a subset and ok_for_causal_comparison=false"
        )
    if args.execute and run.get("execution_ok") is not True:
        print(
            f"ERROR: seed execution failed for {run.get('failed_execution_seeds', [])}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
