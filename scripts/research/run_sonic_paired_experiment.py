#!/usr/bin/env python3
"""Run or plan paired SONIC experiments from an explicit paper-validation spec.

The launcher is intentionally dry-run friendly: use --dry-run to validate and materialize
commands/manifests/comparisons from existing summaries without starting IsaacLab jobs.
Use --execute only when the spec commands are ready for a real compute run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_repo_root = str(REPO_ROOT)
if _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from scripts.research.compare_sonic_manifests import (  # noqa: E402
    build_comparison,
    write_comparison_json,
    write_comparison_markdown,
)
from scripts.research.run_sim_d1_all_motion_eval import (  # noqa: E402
    _load_and_verify_dataset_manifest,
)
from scripts.research.sonic_experiment_manifest import (  # noqa: E402
    build_manifest,
    validate_manifest,
    write_manifest_markdown,
)
from scripts.research.summarize_sonic_logs import (  # noqa: E402
    load_difficulty_ranking,
    summarize_logs,
    write_markdown as write_summary_markdown,
)

_REQUIRED_TOP_LEVEL = (
    "experiment_group",
    "hypothesis",
    "seed",
    "dataset_robot",
    "dataset_smpl",
    "checkpoint",
    "variants",
)
_REQUIRED_VARIANT = ("name", "eval_command", "interpretation")

# Eval-strip guard (research_plan_zpd_teacher.md §3.2, expert-endorsed): eval
# re-applies trainer.schedule_dict at the checkpoint's global step
# (eval_agent_trl.py:466-470) and only strips train_only_events-scoped entries
# (:134-142) — a threshold-curriculum schedule inherited from the train exp
# config would silently corrupt eval comparisons. Hydra override forms that
# remove the schedule from an eval invocation:
_SCHEDULE_STRIP_PATTERNS = ("~trainer.schedule_dict", "trainer.schedule_dict=null")
# Trainer config-group overrides that carry a schedule_dict without the literal
# string appearing in the command. Any trainer config that gains a
# schedule_dict must be registered here or the guard cannot see it.
_SCHEDULE_BEARING_TRAINER_CONFIGS = ("trl_threshold_curriculum",)
_ACTIVATION_ARMS = {"m5_l", "m5_a", "m5_t", "failure_rate"}
_DATASET_MANIFEST_SHA256_KINDS = {
    "file_bytes",
    "canonical_json_without_source_locations_v1",
}
_NOMINAL_FULL_SEQUENCE_EVAL_OVERRIDES = {
    "seed": "0",
    "num_envs": "1",
    "callbacks.im_eval.max_eval_steps": "null",
    "eval_events": "nominal_d1",
    "manager_env/terminations": "tracking/eval",
    "manager_env.config.terrain_type": "plane",
    "manager_env.commands.motion.motion_lib_cfg.sort_motion_keys": "true",
    "manager_env.observations.policy.enable_corruption": "false",
    "manager_env.observations.tokenizer.enable_corruption": "false",
    "use_encoder": "g1",
}
_FULL_SEQUENCE_LIMITER_KEYS = {
    "algo.config.eval.num_eval_episodes",
    "manager_env.commands.motion.motion_lib_cfg.filter_motion_keys",
    "manager_env.commands.motion.motion_lib_cfg.max_unique_motions",
    "manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load",
    "manager_env.commands.motion.motion_lib_cfg.remove_motion_keys",
}
_ROBOT_MOTION_FILE_KEY = "manager_env.commands.motion.motion_lib_cfg.motion_file"
_SMPL_MOTION_FILE_KEY = "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file"
_ADAPTIVE_SAMPLING_PREFIX = "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling."
_ZPD_DUMP_BOUND_SAMPLER_FIELDS = (
    "enable",
    "signal",
    "optimism_k",
    "evidence_half_life",
    "advmass_n",
    "uniform_sampling_rate",
    "tripwire_max_prob_over_uniform",
    "bin_size",
    "paired_dataset_sha256",
)
_ALLOWED_COMMAND_ENVIRONMENT = frozenset({"HYDRA_FULL_ERROR", "LOGURU_LEVEL", "WANDB_MODE"})
_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_FORBIDDEN_SHELL_SYNTAX = re.compile(r"[\r\n;&|<>`#$]")
_TRAIN_ENTRYPOINT = "gear_sonic/train_agent_trl.py"
_EVAL_ENTRYPOINT = "gear_sonic/eval_agent_trl.py"
_OFFICIAL_ZPD_PAIR_CONTRACT = "official_failure_rate_vs_zpd_learnability_v1"
_OFFICIAL_SHARED_SAMPLER_VALUES = {
    "enable": "true",
    "bin_size": "50",
    "sequence_length_agnostic": "true",
    "init_num_failures": "1",
    "uniform_sampling_rate": "0.1",
    "pre_failure_sample_window": "200",
    "use_failure_rate_decay": "false",
    "decay_gamma": "0.8",
    "adp_samp_failure_rate_max_over_mean": "200",
}
_OFFICIAL_SHARED_MOTION_LIB_VALUES = {
    # Never let filesystem directory order change the motion/bin identity map
    # between otherwise paired launches.
    "sort_motion_keys": "true",
}
_OFFICIAL_FULL_SEQUENCE_EVAL_VALUES = {
    "seed": "0",
    "headless": "True",
    "eval_callbacks": "im_eval",
    "run_eval_loop": "False",
    "callbacks.im_eval.max_eval_steps": "null",
    "manager_env/terminations": "tracking/eval",
    "manager_env.observations.policy.enable_corruption": "False",
    "manager_env.observations.tokenizer.enable_corruption": "False",
}
_ZPD_LEARNABILITY_ONLY_SAMPLER_VALUES = {
    "optimism_k": "0.0",
    "evidence_half_life": "null",
    "advmass_n": "16",
    "tripwire_max_prob_over_uniform": "20.0",
    "family_kernel": "null",
}
_PAIR_TRAIN_OUTPUT_KEYS = frozenset({"exp_var", "experiment_dir"})
_PAIR_EVAL_OUTPUT_KEYS = frozenset({"checkpoint", "eval_output_dir"})


def _parse_command(command: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """Parse one command into an allowlisted environment and shell-free argv."""
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")
    forbidden = _FORBIDDEN_SHELL_SYNTAX.search(command)
    if forbidden is not None:
        raise ValueError(
            f"command contains forbidden shell syntax {forbidden.group(0)!r}; commands execute without a shell"
        )
    try:
        tokens = shlex.split(command, posix=True)
    except ValueError as exc:
        raise ValueError(f"command is not valid shell-style token syntax: {exc}") from exc
    if not tokens:
        raise ValueError("command has no executable")

    environment: dict[str, str] = {}
    argv_start = 0
    for index, token in enumerate(tokens):
        match = _ENV_ASSIGNMENT.fullmatch(token)
        if match is None:
            argv_start = index
            break
        name, value = match.groups()
        if name not in _ALLOWED_COMMAND_ENVIRONMENT:
            raise ValueError(
                f"leading environment assignment {name!r} is not allowlisted; "
                f"allowed={sorted(_ALLOWED_COMMAND_ENVIRONMENT)}"
            )
        if name in environment:
            raise ValueError(f"duplicate leading environment assignment: {name}")
        environment[name] = value
    else:
        raise ValueError("command has environment assignments but no executable")

    argv = tuple(tokens[argv_start:])
    if not argv:
        raise ValueError("command has no executable")
    return environment, argv


def _shell_tokens(command: str) -> list[str]:
    try:
        _, argv = _parse_command(command)
        return list(argv)
    except ValueError:
        return []


def _command_overrides(command: str) -> dict[str, list[str]]:
    """Parse Hydra-style ``[+~]key=value`` tokens without losing duplicates."""
    overrides: dict[str, list[str]] = {}
    for token in _shell_tokens(command):
        normalized = token.lstrip("+~")
        if "=" not in normalized:
            continue
        key, value = normalized.split("=", 1)
        overrides.setdefault(key, []).append(value)
    return overrides


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _has_unresolved_placeholder(value: Any) -> bool:
    text = str(value or "")
    return not text or "REPLACE_ME" in text or "{seed}" in text


def _root_override_tokens(argv: tuple[str, ...], key: str) -> list[str]:
    """Return exact root-level Hydra assignment tokens after the entrypoint."""
    result: list[str] = []
    for token in argv[2:]:
        normalized = token.lstrip("+~")
        if "=" not in normalized:
            continue
        token_key, _ = normalized.split("=", 1)
        if token_key == key:
            result.append(token)
    return result


def _command_parse_errors(index: int, variant: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("train_command", "eval_command"):
        command = variant.get(field)
        if command in (None, ""):
            continue
        try:
            _parse_command(str(command))
        except ValueError as exc:
            errors.append(f"variants[{index}].{field} is unsafe or unparseable: {exc}")
    return errors


def _uses_official_zpd_pair_contract(spec: dict[str, Any]) -> bool:
    return spec.get("sampler_pair_contract") == _OFFICIAL_ZPD_PAIR_CONTRACT


def _single_override(
    overrides: dict[str, list[str]],
    key: str,
    *,
    label: str,
    errors: list[str],
) -> str | None:
    values = overrides.get(key, [])
    if len(values) != 1:
        errors.append(f"{label} must set {key} exactly once; got {values}")
        return None
    return values[0]


def _paired_command_signature(
    command: str,
    *,
    ignored_override_keys: frozenset[str],
) -> tuple[Any, ...] | None:
    """Return order-insensitive Hydra semantics after removing declared pair axes."""
    try:
        environment, argv = _parse_command(command)
    except ValueError:
        return None
    assignments: list[tuple[str, str, str]] = []
    positional: list[str] = []
    seen: set[str] = set()
    for token in argv[2:]:
        normalized = token.lstrip("+~")
        if "=" not in normalized:
            positional.append(token)
            continue
        key, value = normalized.split("=", 1)
        if key in seen:
            # Duplicates are invalid independently; preserve them in the signature
            # so two identically ambiguous commands cannot appear paired.
            assignments.append(("<duplicate>", key, token))
            continue
        seen.add(key)
        if key not in ignored_override_keys:
            raw_key = token.split("=", 1)[0]
            assignments.append((raw_key, key, value))
    return (
        tuple(sorted(environment.items())),
        tuple(argv[:2]),
        tuple(positional),
        tuple(sorted(assignments)),
    )


def _official_zpd_pair_contract_errors(spec: dict[str, Any]) -> list[str]:
    """Validate the official failure-rate vs learnability single-axis comparison.

    Unlike the historical M5 activation contract, this contract deliberately has no
    SIM-D1 headroom/ranking prerequisite. It freezes the causal comparison directly:
    identical data, initialization, PPO/train/eval commands, and official sampler
    settings; only output routing plus the ZPD signal and its inert/diagnostic knobs
    may differ.
    """
    configured = spec.get("sampler_pair_contract")
    if configured in (None, ""):
        return []
    if configured != _OFFICIAL_ZPD_PAIR_CONTRACT:
        return [
            "sampler_pair_contract must be "
            f"{_OFFICIAL_ZPD_PAIR_CONTRACT!r}; got {configured!r}"
        ]

    errors: list[str] = []
    if spec.get("requires_activation_gate") is True:
        errors.append(
            "official sampler pair contract is independent of requires_activation_gate/SIM-D1"
        )
    for field in (
        "variant_a",
        "variant_b",
        "dataset_manifest_json",
        "dataset_manifest_sha256",
        "paired_dataset_sha256",
        "training_python_executable",
        "training_initialization_checkpoint",
        "training_initialization_checkpoint_sha256",
        "expected_training_iterations",
        "training_num_envs",
    ):
        if spec.get(field) in (None, ""):
            errors.append(f"official sampler pair contract requires {field}")

    for field in (
        "dataset_manifest_sha256",
        "paired_dataset_sha256",
        "training_initialization_checkpoint_sha256",
    ):
        value = spec.get(field)
        if value not in (None, "") and "REPLACE_ME" not in str(value) and not _is_sha256(value):
            errors.append(f"{field} must be a 64-character hexadecimal digest")
    sha_kind = spec.get("dataset_manifest_sha256_kind", "file_bytes")
    if sha_kind not in _DATASET_MANIFEST_SHA256_KINDS:
        errors.append(f"dataset_manifest_sha256_kind must be one of {sorted(_DATASET_MANIFEST_SHA256_KINDS)}")

    for field in ("expected_training_iterations", "training_num_envs"):
        value = spec.get(field)
        if value not in (None, "") and "REPLACE_ME" not in str(value):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, str))
                or not str(value).isdigit()
                or int(value) <= 0
            ):
                errors.append(f"{field} must be a positive integer")

    python_text = str(spec.get("training_python_executable") or "")
    if python_text in {"python", "python3", "py"}:
        errors.append("training_python_executable must be an explicit path, not a bare interpreter")
    if python_text and not _has_unresolved_placeholder(python_text) and not Path(python_text).is_absolute():
        errors.append("training_python_executable must be an absolute path")

    variants = spec.get("variants")
    if not isinstance(variants, list):
        return errors
    if len(variants) != 2:
        errors.append("official sampler pair contract requires exactly two variants")
        return errors
    by_name = {
        str(variant.get("name")): variant
        for variant in variants
        if isinstance(variant, dict) and variant.get("name") not in (None, "")
    }
    treatment_name = str(spec.get("variant_a") or "")
    baseline_name = str(spec.get("variant_b") or "")
    if treatment_name == baseline_name:
        errors.append("variant_a (learnability) and variant_b (failure_rate) must differ")
    if set(by_name) != {treatment_name, baseline_name}:
        errors.append(
            "variant_a/variant_b must identify exactly the two configured variants; "
            f"expected={[treatment_name, baseline_name]}, got={sorted(by_name)}"
        )
        return errors

    expected_iterations = str(spec.get("expected_training_iterations"))
    expected_num_envs = str(spec.get("training_num_envs"))
    initialization_text = str(spec.get("training_initialization_checkpoint") or "")
    output_checkpoints: list[str] = []
    train_signatures: dict[str, tuple[Any, ...] | None] = {}
    eval_signatures: dict[str, tuple[Any, ...] | None] = {}
    train_ignored = frozenset(
        set(_PAIR_TRAIN_OUTPUT_KEYS)
        | {_ADAPTIVE_SAMPLING_PREFIX + "signal"}
        | {
            _ADAPTIVE_SAMPLING_PREFIX + field
            for field in (*_ZPD_LEARNABILITY_ONLY_SAMPLER_VALUES, "paired_dataset_sha256")
        }
    )

    for name, variant in ((baseline_name, by_name[baseline_name]), (treatment_name, by_name[treatment_name])):
        index = variants.index(variant)
        label = f"variants[{index}] ({name})"
        if str(variant.get("checkpoint_source") or spec.get("checkpoint_source") or "") != (
            "trained_variant_checkpoint"
        ):
            errors.append(f"{label} must use checkpoint_source=trained_variant_checkpoint")
        checkpoint_text = str(variant.get("checkpoint") or "")
        if not checkpoint_text:
            errors.append(f"{label} must declare its trained checkpoint")
        else:
            output_checkpoints.append(checkpoint_text)
        if not variant.get("metrics_eval_json"):
            errors.append(f"{label} must declare metrics_eval_json from the official im_eval callback")

        parsed: dict[str, tuple[dict[str, str], tuple[str, ...]]] = {}
        for command_field, entrypoint in (
            ("train_command", _TRAIN_ENTRYPOINT),
            ("eval_command", _EVAL_ENTRYPOINT),
        ):
            command = str(variant.get(command_field) or "")
            try:
                environment, argv = _parse_command(command)
            except ValueError:
                continue  # The generic parser reports the detailed error once.
            parsed[command_field] = (environment, argv)
            if argv[0] != python_text:
                errors.append(
                    f"{label}.{command_field} must invoke training_python_executable={python_text!r}; "
                    f"got {argv[0]!r}"
                )
            if len(argv) < 2 or argv[1] != entrypoint:
                actual = argv[1] if len(argv) >= 2 else None
                errors.append(f"{label}.{command_field} must use exact entrypoint {entrypoint!r}; got {actual!r}")

        train_command = str(variant.get("train_command") or "")
        eval_command = str(variant.get("eval_command") or "")
        train_overrides = _command_overrides(train_command)
        eval_overrides = _command_overrides(eval_command)
        train_signatures[name] = _paired_command_signature(
            train_command, ignored_override_keys=train_ignored
        )
        eval_signatures[name] = _paired_command_signature(
            eval_command, ignored_override_keys=_PAIR_EVAL_OUTPUT_KEYS
        )

        if "train_command" in parsed:
            train_argv = parsed["train_command"][1]
            expected_checkpoint = f"+checkpoint={initialization_text}"
            checkpoint_tokens = _root_override_tokens(train_argv, "checkpoint")
            if checkpoint_tokens != [expected_checkpoint]:
                errors.append(
                    f"{label}.train_command must contain exactly {expected_checkpoint}; got {checkpoint_tokens}"
                )
            if _root_override_tokens(train_argv, "resume") != ["+resume=false"]:
                errors.append(f"{label}.train_command must contain exactly +resume=false")
            expected_seed = "{seed}" if "{seed}" in train_command else str(spec.get("seed"))
            if _root_override_tokens(train_argv, "seed") != [f"seed={expected_seed}"]:
                errors.append(f"{label}.train_command must contain exactly seed={expected_seed}")
        if "eval_command" in parsed:
            eval_argv = parsed["eval_command"][1]
            expected_eval_checkpoint = f"+checkpoint={checkpoint_text}"
            if _root_override_tokens(eval_argv, "checkpoint") != [expected_eval_checkpoint]:
                errors.append(
                    f"{label}.eval_command must contain exactly {expected_eval_checkpoint}"
                )

        if train_overrides.get("num_envs", []) != [expected_num_envs]:
            errors.append(f"{label}.train_command must set num_envs={expected_num_envs} exactly once")
        if train_overrides.get("algo.config.num_learning_iterations", []) != [expected_iterations]:
            errors.append(
                f"{label}.train_command must set algo.config.num_learning_iterations="
                f"{expected_iterations} exactly once"
            )
        for key, expected in _OFFICIAL_FULL_SEQUENCE_EVAL_VALUES.items():
            if eval_overrides.get(key, []) != [expected]:
                errors.append(f"{label}.eval_command must set {key}={expected} exactly once")
        if eval_overrides.get("num_envs", []) != [expected_num_envs]:
            errors.append(
                f"{label}.eval_command must set num_envs={expected_num_envs} exactly once"
            )
        configured_limiters = sorted(
            key for key in _FULL_SEQUENCE_LIMITER_KEYS if key in eval_overrides
        )
        if configured_limiters:
            errors.append(
                f"{label}.eval_command contains full-sequence limiter(s): {configured_limiters}"
            )

        experiment_dir = _single_override(
            train_overrides, "experiment_dir", label=f"{label}.train_command", errors=errors
        )
        _single_override(train_overrides, "exp_var", label=f"{label}.train_command", errors=errors)
        eval_output_dir = _single_override(
            eval_overrides, "eval_output_dir", label=f"{label}.eval_command", errors=errors
        )
        if experiment_dir and checkpoint_text and Path(experiment_dir) / "last.pt" != Path(checkpoint_text):
            errors.append(
                f"{label} checkpoint must be <train experiment_dir>/last.pt; "
                f"got experiment_dir={experiment_dir!r}, checkpoint={checkpoint_text!r}"
            )
        metrics_text = str(variant.get("metrics_eval_json") or "")
        if eval_output_dir and metrics_text and Path(eval_output_dir) / "metrics_eval.json" != Path(metrics_text):
            errors.append(
                f"{label} metrics_eval_json must be <eval_output_dir>/metrics_eval.json"
            )

        for field, expected in _OFFICIAL_SHARED_SAMPLER_VALUES.items():
            key = _ADAPTIVE_SAMPLING_PREFIX + field
            if train_overrides.get(key, []) != [expected]:
                errors.append(
                    f"{label}.train_command must set official adaptive_sampling.{field}={expected} exactly once"
                )
        for field, expected in _OFFICIAL_SHARED_MOTION_LIB_VALUES.items():
            key = f"manager_env.commands.motion.motion_lib_cfg.{field}"
            for command_label, overrides in (
                ("train_command", train_overrides),
                ("eval_command", eval_overrides),
            ):
                if overrides.get(key, []) != [expected]:
                    errors.append(
                        f"{label}.{command_label} must set shared motion_lib_cfg.{field}={expected} exactly once"
                    )
        expected_signal = "failure_rate" if name == baseline_name else "learnability"
        signal_key = _ADAPTIVE_SAMPLING_PREFIX + "signal"
        if train_overrides.get(signal_key, []) != [expected_signal]:
            errors.append(
                f"{label}.train_command must set adaptive_sampling.signal={expected_signal} exactly once"
            )

        zpd_expected = dict(_ZPD_LEARNABILITY_ONLY_SAMPLER_VALUES)
        zpd_expected["paired_dataset_sha256"] = str(spec.get("paired_dataset_sha256") or "")
        for field, expected in zpd_expected.items():
            key = _ADAPTIVE_SAMPLING_PREFIX + field
            values = train_overrides.get(key, [])
            if name == baseline_name:
                if values:
                    errors.append(
                        f"{label} official failure_rate control must not override "
                        f"ZPD-only adaptive_sampling.{field}"
                    )
            elif values != [expected]:
                errors.append(
                    f"{label}.train_command must set ZPD adaptive_sampling.{field}={expected} exactly once"
                )

    if len(output_checkpoints) != 2 or len(set(output_checkpoints)) != 2:
        errors.append("official sampler pair trained output checkpoints must be unique")
    if initialization_text and initialization_text in output_checkpoints:
        errors.append("trained output checkpoint must not collide with the training initialization checkpoint")
    if train_signatures.get(baseline_name) != train_signatures.get(treatment_name):
        errors.append(
            "paired train commands differ outside output routing and the declared ZPD sampler axis"
        )
    if eval_signatures.get(baseline_name) != eval_signatures.get(treatment_name):
        errors.append("paired eval commands differ outside checkpoint and eval_output_dir")
    return errors


def _activation_launch_contract_errors(spec: dict[str, Any]) -> list[str]:
    """Validate shell-free M5 interpreter, entrypoint, seed, and checkpoint bindings."""
    if spec.get("requires_activation_gate") is not True:
        return []

    errors: list[str] = []
    python_text = str(spec.get("training_python_executable") or "")
    initialization_text = str(spec.get("training_initialization_checkpoint") or "")
    for field, value in (
        ("training_python_executable", python_text),
        ("training_initialization_checkpoint", initialization_text),
    ):
        if not value:
            errors.append(f"requires_activation_gate=true requires {field}")
    if python_text in {"python", "python3", "py"}:
        errors.append("training_python_executable must be an explicit path, not a bare interpreter")
    if not _has_unresolved_placeholder(python_text) and not Path(python_text).is_absolute():
        errors.append("training_python_executable must be an absolute path")

    output_checkpoints: list[str] = []
    variants = spec.get("variants")
    if not isinstance(variants, list):
        return errors
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            continue
        checkpoint_text = str(variant.get("checkpoint") or "")
        if checkpoint_text:
            output_checkpoints.append(checkpoint_text)
        for command_field, entrypoint in (
            ("train_command", _TRAIN_ENTRYPOINT),
            ("eval_command", _EVAL_ENTRYPOINT),
        ):
            command = variant.get(command_field)
            if command in (None, ""):
                errors.append(f"variants[{index}] activation-gated run requires {command_field}")
                continue
            try:
                _, argv = _parse_command(str(command))
            except ValueError:
                # validate_spec records the detailed parser error once.
                continue
            if argv[0] != python_text:
                errors.append(
                    f"variants[{index}].{command_field} must invoke exactly "
                    f"training_python_executable={python_text!r}; got {argv[0]!r}"
                )
            if len(argv) < 2 or argv[1] != entrypoint:
                actual = argv[1] if len(argv) >= 2 else None
                errors.append(
                    f"variants[{index}].{command_field} must use exact entrypoint {entrypoint!r}; got {actual!r}"
                )

        try:
            _, train_argv = _parse_command(str(variant.get("train_command") or ""))
        except ValueError:
            train_argv = ()
        if train_argv:
            expected_checkpoint = f"+checkpoint={initialization_text}"
            checkpoint_tokens = _root_override_tokens(train_argv, "checkpoint")
            if checkpoint_tokens != [expected_checkpoint]:
                errors.append(
                    f"variants[{index}].train_command must contain exactly "
                    f"{expected_checkpoint}; got {checkpoint_tokens}"
                )
            resume_tokens = _root_override_tokens(train_argv, "resume")
            if resume_tokens != ["+resume=false"]:
                errors.append(
                    f"variants[{index}].train_command must contain exactly +resume=false; got {resume_tokens}"
                )
            expected_seed = (
                "{seed}" if "{seed}" in str(variant.get("train_command") or "") else str(spec.get("seed"))
            )
            seed_tokens = _root_override_tokens(train_argv, "seed")
            if seed_tokens != [f"seed={expected_seed}"]:
                errors.append(
                    f"variants[{index}].train_command must contain exactly seed={expected_seed}; got {seed_tokens}"
                )

        try:
            _, eval_argv = _parse_command(str(variant.get("eval_command") or ""))
        except ValueError:
            eval_argv = ()
        if eval_argv:
            expected_eval_checkpoint = f"+checkpoint={checkpoint_text}"
            eval_checkpoint_tokens = _root_override_tokens(eval_argv, "checkpoint")
            if eval_checkpoint_tokens != [expected_eval_checkpoint]:
                errors.append(
                    f"variants[{index}].eval_command must contain exactly "
                    f"{expected_eval_checkpoint}; got {eval_checkpoint_tokens}"
                )

    if len(output_checkpoints) != len(set(output_checkpoints)):
        errors.append("activation-gated variant checkpoints must be unique")
    if (
        initialization_text
        and not _has_unresolved_placeholder(initialization_text)
        and initialization_text in output_checkpoints
    ):
        errors.append("trained output checkpoint must not collide with the training initialization checkpoint")
    return errors


def _activation_contract_errors(spec: dict[str, Any]) -> list[str]:
    """Fail-closed static contract for M5 activation-screen templates."""
    if spec.get("requires_activation_gate") is not True:
        return []

    errors: list[str] = []
    for field in (
        "activation_arm",
        "dataset_manifest_json",
        "dataset_manifest_sha256",
        "difficulty_ranking_json",
        "difficulty_ranking_sha256",
        "sim_d1_classification_json",
        "sim_d1_classification_sha256",
        "sim_d1_source_checkpoint_sha256",
        "expected_training_iterations",
        "training_num_envs",
    ):
        if spec.get(field) in (None, ""):
            errors.append(f"requires_activation_gate=true requires {field}")

    activation_arm = spec.get("activation_arm")
    if activation_arm not in _ACTIVATION_ARMS:
        errors.append(
            f"activation_arm must be one of {sorted(_ACTIVATION_ARMS)} when requires_activation_gate=true"
        )
    sha_kind = spec.get("dataset_manifest_sha256_kind", "file_bytes")
    if sha_kind not in _DATASET_MANIFEST_SHA256_KINDS:
        errors.append(f"dataset_manifest_sha256_kind must be one of {sorted(_DATASET_MANIFEST_SHA256_KINDS)}")
    for field in (
        "dataset_manifest_sha256",
        "difficulty_ranking_sha256",
        "sim_d1_classification_sha256",
        "sim_d1_source_checkpoint_sha256",
    ):
        value = spec.get(field)
        if value not in (None, "") and "REPLACE_ME" not in str(value):
            if not _is_sha256(value):
                errors.append(f"{field} must be a 64-character hexadecimal digest")

    training_num_envs = spec.get("training_num_envs")
    unresolved_num_envs = "REPLACE_ME" in str(training_num_envs)
    if not unresolved_num_envs:
        if (
            isinstance(training_num_envs, bool)
            or not isinstance(training_num_envs, (int, str))
            or not str(training_num_envs).isdigit()
            or int(training_num_envs) <= 0
        ):
            errors.append("training_num_envs must be a positive integer")

    expected_training_iterations = spec.get("expected_training_iterations")
    unresolved_iterations = "REPLACE_ME" in str(expected_training_iterations)
    if not unresolved_iterations:
        if (
            isinstance(expected_training_iterations, bool)
            or not isinstance(expected_training_iterations, (int, str))
            or not str(expected_training_iterations).isdigit()
            or int(expected_training_iterations) <= 0
        ):
            errors.append("expected_training_iterations must be a positive integer")

    expected_num_envs = str(training_num_envs)
    for index, variant in enumerate(spec.get("variants", [])):
        train_tokens = _shell_tokens(str(variant.get("train_command") or ""))
        train_num_envs = [token.split("=", 1)[1] for token in train_tokens if token.startswith("num_envs=")]
        if train_num_envs != [expected_num_envs]:
            errors.append(
                f"variants[{index}] train_command must contain exactly "
                f"num_envs={expected_num_envs}; activation measurement requires one "
                "active environment per candidate motion"
            )
        expected_iterations = str(expected_training_iterations)
        train_iterations = _command_overrides(str(variant.get("train_command") or "")).get(
            "algo.config.num_learning_iterations", []
        )
        if train_iterations != [expected_iterations]:
            errors.append(
                f"variants[{index}] train_command must contain exactly "
                "algo.config.num_learning_iterations="
                f"{expected_iterations}; got {train_iterations}"
            )
        if activation_arm == "m5_t":
            telemetry_flag = _command_overrides(
                str(variant.get("train_command") or "")
            ).get("manager_env.config.log_m5t_height_termination", [])
            if telemetry_flag != ["true"]:
                errors.append(
                    f"variants[{index}] M5-T train_command must contain exactly "
                    "manager_env.config.log_m5t_height_termination=true; "
                    f"got {telemetry_flag}"
                )

        eval_overrides = _command_overrides(str(variant.get("eval_command") or ""))
        for key, expected_value in _NOMINAL_FULL_SEQUENCE_EVAL_OVERRIDES.items():
            actual_values = eval_overrides.get(key, [])
            if actual_values != [expected_value]:
                errors.append(
                    f"variants[{index}] eval_command must contain exactly one "
                    f"{key}={expected_value} override; got {actual_values}"
                )
        configured_limiters = sorted(key for key in _FULL_SEQUENCE_LIMITER_KEYS if key in eval_overrides)
        if configured_limiters:
            errors.append(
                f"variants[{index}] eval_command contains full-sequence limiter(s): {configured_limiters}"
            )
    return errors


def _uses_schedule(command: str) -> bool:
    if "schedule_dict" in command:
        return True
    return any(f"trainer={name}" in command for name in _SCHEDULE_BEARING_TRAINER_CONFIGS)


def _eval_strip_errors(index: int, variant: dict[str, Any]) -> list[str]:
    """Materialized-command checks for the schedule_dict eval-strip guard."""
    errors: list[str] = []
    eval_command = str(variant.get("eval_command") or "")
    train_command = str(variant.get("train_command") or "")
    eval_strips = any(pattern in eval_command for pattern in _SCHEDULE_STRIP_PATTERNS)
    if _uses_schedule(eval_command) and not eval_strips:
        errors.append(
            f"variants[{index}] eval_command sets schedule_dict without stripping it "
            f"(use one of {_SCHEDULE_STRIP_PATTERNS}); eval_agent_trl.py re-applies "
            "schedules at the checkpoint step and would corrupt the comparison"
        )
    if _uses_schedule(train_command) and not eval_strips:
        errors.append(
            f"variants[{index}] train_command uses schedule_dict (directly or via a "
            f"schedule-bearing trainer config) but eval_command does not strip it "
            f"(add one of {_SCHEDULE_STRIP_PATTERNS}); eval loads the checkpoint's "
            "saved training config, so the schedule would be re-applied at eval"
        )
    return errors


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
    effective_rankings: list[str] = []
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
        checkpoint_source = str(
            variant.get("checkpoint_source") or spec.get("checkpoint_source") or "configured_checkpoint"
        )
        if checkpoint_source == "trained_variant_checkpoint":
            checkpoint = variant.get("checkpoint")
            if not checkpoint:
                errors.append(
                    f"variants[{index}] with checkpoint_source=trained_variant_checkpoint "
                    "must provide its own checkpoint path"
                )
            elif checkpoint not in str(variant.get("eval_command") or ""):
                errors.append(
                    f"variants[{index}] eval_command does not reference its declared "
                    f"trained checkpoint {checkpoint!r}"
                )
        difficulty_ranking = variant.get("difficulty_ranking_json") or spec.get("difficulty_ranking_json")
        if difficulty_ranking:
            effective_rankings.append(str(difficulty_ranking))
        metrics_eval = variant.get("metrics_eval_json")
        if bool(difficulty_ranking) and not bool(metrics_eval):
            errors.append(
                f"variants[{index}] must provide metrics_eval_json together with "
                "difficulty_ranking_json (variant or top-level) for retention"
            )
        elif bool(metrics_eval) and not bool(difficulty_ranking) and not _uses_official_zpd_pair_contract(spec):
            errors.append(
                f"variants[{index}] must provide metrics_eval_json together with "
                "difficulty_ranking_json (variant or top-level) for retention"
            )
        errors.extend(_command_parse_errors(index, variant))
        errors.extend(_eval_strip_errors(index, variant))
    if len(set(effective_rankings)) > 1:
        errors.append("all retention variants must use the same frozen difficulty_ranking_json")
    errors.extend(_activation_contract_errors(spec))
    errors.extend(_activation_launch_contract_errors(spec))
    errors.extend(_official_zpd_pair_contract_errors(spec))
    return errors


def _command_environment(
    command_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Make bare ``python`` resolve to the orchestrator's environment first."""
    environment = os.environ.copy()
    interpreter_dir = str(Path(sys.executable).absolute().parent)
    inherited_path = environment.get("PATH", "")
    environment["PATH"] = interpreter_dir + os.pathsep + inherited_path if inherited_path else interpreter_dir
    environment.update(command_environment or {})
    return environment


def _command_environment_provenance() -> dict[str, str]:
    return {
        "orchestrator_python": sys.executable,
        "subprocess_path_prefix": str(Path(sys.executable).absolute().parent),
        "execution_mode": "shell_false_structured_argv",
        "allowed_leading_environment": ",".join(sorted(_ALLOWED_COMMAND_ENVIRONMENT)),
    }


def _run_command(command: str, log_path: Path, *, cwd: Path | None = None) -> int:
    command_environment, argv = _parse_command(command)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        process = subprocess.run(
            list(argv),
            cwd=str(cwd) if cwd else None,
            env=_command_environment(command_environment),
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_snapshot(path: Path) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _checkpoint_provenance(path: Path, *, repo_root: Path) -> dict[str, Any]:
    stat = path.stat()
    release_checkpoint = (repo_root / "sonic_release" / "last.pt").resolve()
    return {
        "sha256": _sha256_file(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "is_release_checkpoint": path.resolve() == release_checkpoint,
    }


def _verify_trained_checkpoint(
    checkpoint_text: str,
    *,
    repo_root: Path,
    before_train: tuple[int, int] | None,
    require_fresh: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    checkpoint_path = _relative_or_absolute(checkpoint_text, repo_root)
    after_train = _checkpoint_snapshot(checkpoint_path)
    if after_train is None:
        return None, f"trained checkpoint was not created: {checkpoint_path}"
    if require_fresh and before_train is not None and after_train == before_train:
        return None, f"trained checkpoint was not updated by this run: {checkpoint_path}"
    return _checkpoint_provenance(checkpoint_path, repo_root=repo_root), None


def _summary_is_stale(
    summary_json: Path, train_log: Path, eval_log: Path, metrics_eval_json: Path | None = None
) -> bool:
    if not summary_json.exists():
        return True
    summary_mtime = summary_json.stat().st_mtime
    source_paths = [train_log, eval_log]
    if metrics_eval_json is not None:
        source_paths.append(metrics_eval_json)
    return any(path.exists() and path.stat().st_mtime > summary_mtime for path in source_paths)


def _verify_variant_dataset_bindings(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    dataset_robot: Path,
    dataset_smpl: Path,
) -> list[dict[str, Any]]:
    """Require every train/eval command to use exactly the verified datasets."""
    expected_paths = {
        _ROBOT_MOTION_FILE_KEY: dataset_robot.resolve(),
        _SMPL_MOTION_FILE_KEY: dataset_smpl.resolve(),
    }
    bindings: list[dict[str, Any]] = []
    for index, variant in enumerate(spec.get("variants", [])):
        for command_kind in ("train_command", "eval_command"):
            overrides = _command_overrides(str(variant.get(command_kind) or ""))
            for key, expected_path in expected_paths.items():
                values = overrides.get(key, [])
                if len(values) != 1:
                    raise ValueError(
                        f"variants[{index}].{command_kind} must contain exactly one {key} override; got {values}"
                    )
                actual_path = _relative_or_absolute(values[0], repo_root).resolve()
                if actual_path != expected_path:
                    raise ValueError(
                        f"variants[{index}].{command_kind} {key} resolves to "
                        f"{actual_path}, expected verified path {expected_path}"
                    )
                bindings.append(
                    {
                        "variant": str(variant.get("name")),
                        "command": command_kind,
                        "key": key,
                        "path": str(actual_path),
                    }
                )
    return bindings


def _verify_zpd_treatment_sampler_command(spec: dict[str, Any], *, paired_dataset_sha256: str) -> dict[str, Any]:
    """Bind every dump-checked ZPD sampler knob to the treatment command."""
    arm = spec.get("activation_arm")
    if arm not in {"m5_l", "m5_a"}:
        return {}
    variant_a = spec.get("variant_a")
    treatment_variants = [
        variant
        for variant in spec.get("variants", [])
        if isinstance(variant, dict) and variant.get("name") == variant_a
    ]
    if len(treatment_variants) != 1:
        raise ValueError("ZPD activation preflight requires variant_a to identify exactly one treatment variant")
    treatment = treatment_variants[0]
    overrides = _command_overrides(str(treatment.get("train_command") or ""))
    raw: dict[str, str] = {}
    for field in _ZPD_DUMP_BOUND_SAMPLER_FIELDS:
        values = overrides.get(_ADAPTIVE_SAMPLING_PREFIX + field, [])
        if len(values) != 1:
            raise ValueError(
                f"ZPD treatment train_command must set adaptive_sampling.{field} exactly once; got {values}"
            )
        raw[field] = values[0]

    expected_signal = "learnability" if arm == "m5_l" else "advantage_mass"
    if raw["enable"] != "true":
        raise ValueError("ZPD treatment train_command must set adaptive_sampling.enable=true")
    if raw["signal"] != expected_signal:
        raise ValueError(f"ZPD treatment train_command must set adaptive_sampling.signal={expected_signal}")
    command_digest = raw["paired_dataset_sha256"].lower()
    if command_digest != paired_dataset_sha256:
        raise ValueError(
            "ZPD treatment adaptive_sampling.paired_dataset_sha256 does not match the verified dataset manifest"
        )

    def finite_float(field: str) -> float:
        try:
            value = float(raw[field])
        except ValueError as exc:
            raise ValueError(f"adaptive_sampling.{field} must be numeric, got {raw[field]!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"adaptive_sampling.{field} must be finite")
        return value

    def positive_int(field: str) -> int:
        value = raw[field]
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(f"adaptive_sampling.{field} must be a positive integer, got {value!r}")
        return int(value)

    optimism_k = finite_float("optimism_k")
    evidence_half_life = finite_float("evidence_half_life")
    if evidence_half_life <= 0:
        raise ValueError("adaptive_sampling.evidence_half_life must be positive")
    uniform_sampling_rate = finite_float("uniform_sampling_rate")
    if not 0.0 <= uniform_sampling_rate <= 1.0:
        raise ValueError("adaptive_sampling.uniform_sampling_rate must be within [0, 1]")
    tripwire = finite_float("tripwire_max_prob_over_uniform")
    if tripwire < 1.0:
        raise ValueError("adaptive_sampling.tripwire_max_prob_over_uniform must be >= 1")

    return {
        "variant": str(variant_a),
        "enable": True,
        "signal": expected_signal,
        "optimism_k": optimism_k,
        "evidence_half_life": evidence_half_life,
        "advmass_n": positive_int("advmass_n"),
        "uniform_sampling_rate": uniform_sampling_rate,
        "tripwire_max_prob_over_uniform": tripwire,
        "bin_size": positive_int("bin_size"),
        "paired_dataset_sha256": command_digest,
        "all_dump_bound_fields_explicit_once": True,
    }


def _verify_flat_paired_inventory(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    motion_keys: list[str],
) -> dict[str, Any]:
    """Prove both paired modalities are flat and exactly stem-matched."""
    expected_stems = set(motion_keys)
    if len(expected_stems) != len(motion_keys):
        raise ValueError("hash-bound dataset motion keys contain duplicates")

    result: dict[str, Any] = {
        "loader_glob": "*.pkl",
        "flat_direct_children": True,
        "exact_manifest_stems": True,
        "paired_stems_equal": True,
    }
    modality_stems: dict[str, set[str]] = {}
    for modality, field in (("robot", "dataset_robot"), ("smpl", "dataset_smpl")):
        root = _relative_or_absolute(str(spec[field]), repo_root).resolve()
        recursive_files = sorted(path for path in root.rglob("*.pkl") if path.is_file())
        nested_files = [str(path) for path in recursive_files if path.parent != root]
        if nested_files:
            raise ValueError(
                f"{modality} dataset must be flat because paired runtime lookup "
                f"requires direct-child *.pkl files; nested={nested_files}"
            )
        direct_files = sorted(path for path in root.glob("*.pkl") if path.is_file())
        stems = [path.stem for path in direct_files]
        if len(set(stems)) != len(stems):
            raise ValueError(f"{modality} dataset contains duplicate direct-child stems")
        actual_stems = set(stems)
        missing = sorted(expected_stems - actual_stems)
        extras = sorted(actual_stems - expected_stems)
        if missing or extras:
            raise ValueError(
                f"{modality} dataset direct-child stems do not exactly match the "
                f"hash-bound manifest; extras={extras}, missing={missing}"
            )
        modality_stems[modality] = actual_stems
        result[f"{modality}_file_count"] = len(direct_files)
    if modality_stems["robot"] != modality_stems["smpl"]:
        raise ValueError("robot and SMPL direct-child stems differ")
    return result


def _verify_exhaustive_dataset_inventory(
    manifest: dict[str, Any],
    *,
    dataset_robot: Path,
    dataset_smpl: Path,
) -> dict[str, Any]:
    """Match a flat manifest exactly to every direct-child ``*.pkl``."""
    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("variants"), list):
        raise ValueError("dataset manifest output.variants must be a list")
    roots = {"robot": dataset_robot.resolve(), "smpl": dataset_smpl.resolve()}
    manifest_dirs: dict[str, Path] = {}
    for modality in roots:
        directory = output.get(f"{modality}_dir")
        if not isinstance(directory, str) or not directory:
            raise ValueError(f"dataset manifest output.{modality}_dir must be non-empty")
        manifest_dirs[modality] = Path(directory)

    expected_files: dict[str, set[Path]] = {"robot": set(), "smpl": set()}
    for index, variant in enumerate(output["variants"]):
        if not isinstance(variant, dict):
            raise ValueError(f"dataset manifest output.variants[{index}] must be an object")
        motion_key = variant.get("motion_key")
        if not isinstance(motion_key, str) or not motion_key:
            raise ValueError(f"dataset manifest output.variants[{index}].motion_key must be non-empty")
        for modality, root in roots.items():
            record = variant.get(modality)
            relative_value = record.get("path") if isinstance(record, dict) else None
            if not isinstance(relative_value, str) or not relative_value:
                raise ValueError(f"dataset manifest output.variants[{index}].{modality}.path must be non-empty")
            relative_path = Path(relative_value)
            if relative_path.stem != motion_key:
                raise ValueError(
                    "dataset manifest motion_key/file-stem mismatch: "
                    f"{motion_key!r} != {relative_path.stem!r} ({relative_value})"
                )
            try:
                dataset_relative = relative_path.relative_to(manifest_dirs[modality])
            except ValueError as exc:
                raise ValueError(
                    f"dataset manifest {modality} path is outside its declared directory: {relative_value}"
                ) from exc
            if dataset_relative.parent != Path("."):
                raise ValueError(f"{modality} dataset manifest must be flat; got {relative_value}")
            expected_files[modality].add((root / dataset_relative).resolve())

    result: dict[str, Any] = {"loader_glob": "*.pkl"}
    for modality, root in roots.items():
        recursive_files = {path.resolve() for path in root.rglob("*.pkl") if path.is_file()}
        actual_files = {path.resolve() for path in root.glob("*.pkl") if path.is_file()}
        nested_files = sorted(str(path) for path in recursive_files - actual_files)
        if nested_files:
            raise ValueError(f"{modality} dataset must be flat; nested={nested_files}")
        stems: dict[str, list[str]] = {}
        for path in actual_files:
            stems.setdefault(path.stem, []).append(str(path))
        duplicate_stems = {stem: sorted(paths) for stem, paths in stems.items() if len(paths) > 1}
        if duplicate_stems:
            raise ValueError(f"{modality} dataset contains duplicate loader-visible stems: {duplicate_stems}")
        extras = sorted(str(path) for path in actual_files - expected_files[modality])
        missing = sorted(str(path) for path in expected_files[modality] - actual_files)
        if extras or missing:
            raise ValueError(f"{modality} dataset inventory is not exhaustive; extras={extras}, missing={missing}")
        result[f"{modality}_file_count"] = len(actual_files)
    result["flat_direct_children"] = True
    result["exact_direct_child_pkl_inventory"] = True
    return result


def _verify_sim_d1_classification(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    manifest_verification: dict[str, Any],
    ranking: list[str],
    dataset_robot: Path,
    dataset_smpl: Path,
) -> dict[str, Any]:
    classification_path = _relative_or_absolute(str(spec["sim_d1_classification_json"]), repo_root).resolve()
    if not classification_path.is_file():
        raise ValueError(f"sim_d1_classification_json does not exist: {classification_path}")
    expected_sha256 = str(spec["sim_d1_classification_sha256"]).lower()
    actual_sha256 = _sha256_file(classification_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"SIM-D1 classification SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    classification = _load_json(classification_path)
    if classification.get("schema_version") != 1:
        raise ValueError("SIM-D1 classification schema_version must be 1")
    if classification.get("kind") != "sim_d1_headroom_classification":
        raise ValueError("SIM-D1 classification kind must be sim_d1_headroom_classification")
    if (
        classification.get("verdict") != "PASS"
        or classification.get("pass") is not True
        or classification.get("eligible_for_effect_experiment") is not True
    ):
        raise ValueError(
            "SIM-D1 classification must have verdict PASS, pass=true, and eligible_for_effect_experiment=true"
        )

    motion_keys = list(manifest_verification["motion_keys"])
    motion_count = len(motion_keys)
    coverage = classification.get("coverage")
    if not isinstance(coverage, dict) or not (
        coverage.get("all_motion_coverage_independently_verified") is True
        and coverage.get("exact_key_set_verified") is True
        and coverage.get("evaluated_motion_count") == motion_count
        and coverage.get("expected_motion_count") == motion_count
    ):
        raise ValueError("SIM-D1 classification must independently verify exact all-motion coverage")
    if classification.get("ranking") != ranking:
        raise ValueError("hash-bound difficulty ranking must exactly match the SIM-D1 classification ranking")

    source = classification.get("source")
    if not isinstance(source, dict):
        raise ValueError("SIM-D1 classification source must be an object")
    expected_checkpoint_sha256 = str(spec["sim_d1_source_checkpoint_sha256"]).lower()
    if source.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("SIM-D1 source checkpoint SHA-256 does not match the preregistered value")
    if source.get("expected_motion_count") != motion_count:
        raise ValueError("SIM-D1 source expected_motion_count does not match the dataset manifest")
    for field, expected_path in (
        ("dataset_robot", dataset_robot.resolve()),
        ("dataset_smpl", dataset_smpl.resolve()),
    ):
        source_value = source.get(field)
        if not isinstance(source_value, str) or Path(source_value).resolve() != expected_path:
            raise ValueError(f"SIM-D1 source {field} does not match the verified dataset path")

    source_manifest = source.get("dataset_manifest")
    if not isinstance(source_manifest, dict):
        raise ValueError("SIM-D1 classification source.dataset_manifest must be an object")
    required_source_manifest = {
        "content_hashes_verified": True,
        "paired_dataset_sha256": manifest_verification["paired_dataset_sha256"],
        "sha256": manifest_verification["sha256"],
        "sha256_kind": manifest_verification["sha256_kind"],
        "motion_count": motion_count,
        "motion_keys": motion_keys,
        "verified_file_count": 2 * motion_count,
    }
    mismatches = {
        key: {"expected": expected, "actual": source_manifest.get(key)}
        for key, expected in required_source_manifest.items()
        if source_manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"SIM-D1 classification dataset-manifest provenance mismatch: {mismatches}")
    return {
        "path": str(classification_path),
        "sha256": actual_sha256,
        "verdict": "PASS",
        "eligible_for_effect_experiment": True,
        "source_checkpoint_sha256": expected_checkpoint_sha256,
        "exact_all_motion_coverage": True,
        "ranking_exactly_bound": True,
    }


def _verify_metrics_motion_coverage(metrics_path: Path, expected_motion_keys: list[str]) -> dict[str, Any]:
    metrics = _load_json(metrics_path)
    all_metrics = metrics.get("eval/all_metrics_dict")
    motion_keys = all_metrics.get("motion_keys") if isinstance(all_metrics, dict) else None
    if (
        not isinstance(motion_keys, list)
        or not all(isinstance(key, str) and key for key in motion_keys)
        or len(motion_keys) != len(set(motion_keys))
    ):
        raise ValueError(f"eval metrics {metrics_path} must contain unique non-empty motion_keys")
    expected = set(expected_motion_keys)
    actual = set(motion_keys)
    if len(motion_keys) != len(expected_motion_keys) or actual != expected:
        raise ValueError(
            "eval metrics do not exactly cover the hash-bound dataset motion keys: "
            f"missing={sorted(expected - actual)}, extras={sorted(actual - expected)}"
        )
    return {
        "path": str(metrics_path),
        "sha256": _sha256_file(metrics_path),
        "motion_count": len(motion_keys),
        "exact_motion_key_coverage": True,
    }


def _verify_activation_launch_runtime(spec: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Resolve and hash-bind activation-gated interpreter/init/output paths."""
    errors = _activation_launch_contract_errors(spec)
    for field in (
        "training_python_executable",
        "training_initialization_checkpoint",
    ):
        if _has_unresolved_placeholder(spec.get(field)):
            errors.append(f"{field} is unresolved: {spec.get(field)!r}")
    if errors:
        raise ValueError("activation launch contract failed: " + "; ".join(errors))

    python_text = str(spec["training_python_executable"])
    python_path = Path(python_text)
    if not python_path.is_absolute():
        raise ValueError("training_python_executable must be an absolute path")
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError(f"training_python_executable must be an existing executable file: {python_path}")

    initialization_path = _relative_or_absolute(
        str(spec["training_initialization_checkpoint"]), repo_root
    ).resolve()
    if not initialization_path.is_file():
        raise ValueError(f"training initialization checkpoint does not exist: {initialization_path}")
    initialization_sha256 = _sha256_file(initialization_path)
    expected_sha256 = str(spec["sim_d1_source_checkpoint_sha256"]).lower()
    if initialization_sha256 != expected_sha256:
        raise ValueError(
            "training initialization checkpoint SHA-256 does not match "
            "sim_d1_source_checkpoint_sha256: "
            f"expected {expected_sha256}, got {initialization_sha256}"
        )

    output_paths: list[Path] = []
    for variant in spec["variants"]:
        output_path = _relative_or_absolute(str(variant["checkpoint"]), repo_root).resolve()
        if output_path == initialization_path:
            raise ValueError(
                "trained output checkpoint must not collide with the training "
                f"initialization checkpoint: {output_path}"
            )
        output_paths.append(output_path)
    if len(output_paths) != len(set(output_paths)):
        raise ValueError("activation-gated trained output checkpoints must be unique")

    return {
        "training_python_executable": {
            "configured_path": python_text,
            "resolved_path": str(python_path.resolve()),
            "sha256": _sha256_file(python_path.resolve()),
            "executable": True,
            "all_train_and_eval_commands_exactly_bound": True,
            "entrypoints_exactly_bound": True,
        },
        "training_initialization_checkpoint": {
            "configured_path": str(spec["training_initialization_checkpoint"]),
            "resolved_path": str(initialization_path),
            "sha256": initialization_sha256,
            "sim_d1_source_checkpoint_sha256": expected_sha256,
            "hash_matches_sim_d1_source_checkpoint": True,
            "resume": False,
            "all_train_commands_exactly_bound": True,
        },
        "trained_output_checkpoints": {
            "paths": [str(path) for path in output_paths],
            "unique": True,
            "disjoint_from_initialization": True,
        },
    }


def _verify_initialization_checkpoint_unchanged(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    input_provenance: dict[str, Any],
) -> None:
    """Re-hash the frozen initialization immediately before each train launch."""
    recorded = input_provenance.get("training_initialization_checkpoint")
    if not isinstance(recorded, dict):
        raise ValueError("training initialization provenance is missing")
    path = _relative_or_absolute(str(spec["training_initialization_checkpoint"]), repo_root).resolve()
    if str(path) != recorded.get("resolved_path") or not path.is_file():
        raise ValueError("training initialization checkpoint path changed after preflight")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != recorded.get("sha256"):
        raise ValueError(
            "training initialization checkpoint changed after preflight: "
            f"expected {recorded.get('sha256')}, got {actual_sha256}"
        )


def _verify_activation_preflight(spec: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Hash-bind M5 inputs and prove every candidate is active in training."""
    launch_provenance = _verify_activation_launch_runtime(spec, repo_root=repo_root)
    manifest_path = _relative_or_absolute(str(spec["dataset_manifest_json"]), repo_root).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"dataset_manifest_json does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("dataset manifest output must be an object")
    motion_keys = output.get("motion_keys")
    motion_count = output.get("motion_count")
    if (
        not isinstance(motion_keys, list)
        or not motion_keys
        or not all(isinstance(key, str) and key for key in motion_keys)
        or len(set(motion_keys)) != len(motion_keys)
    ):
        raise ValueError("dataset manifest output.motion_keys must contain unique non-empty strings")
    if isinstance(motion_count, bool) or not isinstance(motion_count, int) or motion_count != len(motion_keys):
        raise ValueError("dataset manifest output.motion_count must equal the number of motion_keys")
    paired_dataset_sha256 = output.get("paired_dataset_sha256")
    if not _is_sha256(paired_dataset_sha256):
        raise ValueError("dataset manifest output.paired_dataset_sha256 must be a SHA-256 digest")

    training_num_envs = int(spec["training_num_envs"])
    if training_num_envs != motion_count:
        raise ValueError(
            "training_num_envs must equal the hash-bound dataset motion count: "
            f"configured {training_num_envs}, manifest has {motion_count}"
        )

    dataset_robot = _relative_or_absolute(str(spec["dataset_robot"]), repo_root).resolve()
    dataset_smpl = _relative_or_absolute(str(spec["dataset_smpl"]), repo_root).resolve()
    command_bindings = _verify_variant_dataset_bindings(
        spec,
        repo_root=repo_root,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
    )
    exhaustive_inventory = _verify_exhaustive_dataset_inventory(
        manifest,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
    )
    flat_inventory = _verify_flat_paired_inventory(spec, repo_root=repo_root, motion_keys=list(motion_keys))
    manifest_verification = _load_and_verify_dataset_manifest(
        manifest_path,
        expected_manifest_sha256=str(spec["dataset_manifest_sha256"]),
        manifest_sha256_kind=str(spec.get("dataset_manifest_sha256_kind", "file_bytes")),
        expected_paired_dataset_sha256=str(paired_dataset_sha256),
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
        coverage={
            "motion_keys": list(motion_keys),
            "expected_motion_count": motion_count,
        },
    )
    zpd_treatment_sampler_config = _verify_zpd_treatment_sampler_command(
        spec,
        paired_dataset_sha256=manifest_verification["paired_dataset_sha256"],
    )

    ranking_path = _relative_or_absolute(str(spec["difficulty_ranking_json"]), repo_root).resolve()
    if not ranking_path.is_file():
        raise ValueError(f"difficulty_ranking_json does not exist: {ranking_path}")
    expected_ranking_sha256 = str(spec["difficulty_ranking_sha256"]).lower()
    actual_ranking_sha256 = _sha256_file(ranking_path)
    if actual_ranking_sha256 != expected_ranking_sha256:
        raise ValueError(
            f"difficulty ranking SHA-256 mismatch: expected {expected_ranking_sha256}, got {actual_ranking_sha256}"
        )
    ranking = load_difficulty_ranking(ranking_path)
    if len(ranking) != motion_count or set(ranking) != set(motion_keys):
        raise ValueError("difficulty ranking must cover exactly the hash-bound dataset motion keys")
    classification_verification = _verify_sim_d1_classification(
        spec,
        repo_root=repo_root,
        manifest_verification=manifest_verification,
        ranking=ranking,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
    )

    return {
        **launch_provenance,
        "dataset_manifest": manifest_verification,
        "dataset_inventory": {**exhaustive_inventory, **flat_inventory},
        "command_dataset_bindings": command_bindings,
        "sim_d1_classification": classification_verification,
        "zpd_treatment_sampler_config": zpd_treatment_sampler_config,
        "difficulty_ranking": {
            "path": str(ranking_path),
            "sha256": actual_ranking_sha256,
            "motion_count": len(ranking),
            "exact_dataset_coverage": True,
        },
        "expected_training_iterations": int(spec["expected_training_iterations"]),
        "training_num_envs": training_num_envs,
        "all_candidate_motions_active": True,
    }


def _verify_official_zpd_pair_preflight(spec: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Hash-bind a frozen paired dataset without imposing a SIM-D1 verdict."""
    contract_errors = _official_zpd_pair_contract_errors(spec)
    if contract_errors:
        raise ValueError("official sampler pair contract failed: " + "; ".join(contract_errors))

    for field in (
        "training_python_executable",
        "training_initialization_checkpoint",
        "training_initialization_checkpoint_sha256",
        "dataset_manifest_json",
        "dataset_manifest_sha256",
        "paired_dataset_sha256",
    ):
        if _has_unresolved_placeholder(spec.get(field)):
            raise ValueError(f"{field} is unresolved: {spec.get(field)!r}")

    python_path = Path(str(spec["training_python_executable"]))
    if not python_path.is_absolute() or not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise ValueError(
            "training_python_executable must be an existing absolute executable file: "
            f"{python_path}"
        )
    initialization_path = _relative_or_absolute(
        str(spec["training_initialization_checkpoint"]), repo_root
    ).resolve()
    if not initialization_path.is_file():
        raise ValueError(f"training initialization checkpoint does not exist: {initialization_path}")
    initialization_sha256 = _sha256_file(initialization_path)
    expected_initialization_sha256 = str(spec["training_initialization_checkpoint_sha256"]).lower()
    if initialization_sha256 != expected_initialization_sha256:
        raise ValueError(
            "training initialization checkpoint SHA-256 mismatch: "
            f"expected {expected_initialization_sha256}, got {initialization_sha256}"
        )

    dataset_robot = _relative_or_absolute(str(spec["dataset_robot"]), repo_root).resolve()
    dataset_smpl = _relative_or_absolute(str(spec["dataset_smpl"]), repo_root).resolve()
    command_bindings = _verify_variant_dataset_bindings(
        spec,
        repo_root=repo_root,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
    )
    manifest_path = _relative_or_absolute(str(spec["dataset_manifest_json"]), repo_root).resolve()
    if not manifest_path.is_file():
        raise ValueError(f"dataset_manifest_json does not exist: {manifest_path}")
    manifest = _load_json(manifest_path)
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise ValueError("dataset manifest output must be an object")
    motion_keys = output.get("motion_keys")
    motion_count = output.get("motion_count")
    if (
        not isinstance(motion_keys, list)
        or not motion_keys
        or not all(isinstance(key, str) and key for key in motion_keys)
        or len(set(motion_keys)) != len(motion_keys)
        or isinstance(motion_count, bool)
        or not isinstance(motion_count, int)
        or motion_count != len(motion_keys)
    ):
        raise ValueError(
            "dataset manifest output.motion_keys must be unique non-empty strings and "
            "output.motion_count must match"
        )
    manifest_paired_sha256 = output.get("paired_dataset_sha256")
    expected_paired_sha256 = str(spec["paired_dataset_sha256"]).lower()
    if manifest_paired_sha256 != expected_paired_sha256:
        raise ValueError(
            "paired_dataset_sha256 does not match dataset manifest output: "
            f"expected {expected_paired_sha256}, got {manifest_paired_sha256}"
        )

    exhaustive_inventory = _verify_exhaustive_dataset_inventory(
        manifest,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
    )
    flat_inventory = _verify_flat_paired_inventory(
        spec, repo_root=repo_root, motion_keys=list(motion_keys)
    )
    manifest_verification = _load_and_verify_dataset_manifest(
        manifest_path,
        expected_manifest_sha256=str(spec["dataset_manifest_sha256"]),
        manifest_sha256_kind=str(spec.get("dataset_manifest_sha256_kind", "file_bytes")),
        expected_paired_dataset_sha256=expected_paired_sha256,
        dataset_robot=dataset_robot,
        dataset_smpl=dataset_smpl,
        coverage={
            "motion_keys": list(motion_keys),
            "expected_motion_count": motion_count,
        },
    )

    output_paths = [
        _relative_or_absolute(str(variant["checkpoint"]), repo_root).resolve()
        for variant in spec["variants"]
    ]
    if len(set(output_paths)) != 2 or initialization_path in output_paths:
        raise ValueError(
            "paired trained checkpoints must be unique and disjoint from initialization"
        )

    return {
        "training_python_executable": {
            "configured_path": str(spec["training_python_executable"]),
            "resolved_path": str(python_path.resolve()),
            "sha256": _sha256_file(python_path.resolve()),
            "executable": True,
            "all_train_and_eval_commands_exactly_bound": True,
            "entrypoints_exactly_bound": True,
        },
        "training_initialization_checkpoint": {
            "configured_path": str(spec["training_initialization_checkpoint"]),
            "resolved_path": str(initialization_path),
            "sha256": initialization_sha256,
            "expected_sha256": expected_initialization_sha256,
            "hash_matches_preregistered_checkpoint": True,
            "resume": False,
            "all_train_commands_exactly_bound": True,
        },
        "trained_output_checkpoints": {
            "paths": [str(path) for path in output_paths],
            "unique": True,
            "disjoint_from_initialization": True,
        },
        "dataset_manifest": manifest_verification,
        "dataset_inventory": {**exhaustive_inventory, **flat_inventory},
        "command_dataset_bindings": command_bindings,
        "sampler_pair_contract": {
            "kind": _OFFICIAL_ZPD_PAIR_CONTRACT,
            "baseline_variant": str(spec["variant_b"]),
            "baseline_signal": "failure_rate",
            "treatment_variant": str(spec["variant_a"]),
            "treatment_signal": "learnability",
            "shared_official_sampler_values": dict(_OFFICIAL_SHARED_SAMPLER_VALUES),
            "shared_motion_lib_values": dict(_OFFICIAL_SHARED_MOTION_LIB_VALUES),
            "zpd_only_values": {
                **_ZPD_LEARNABILITY_ONLY_SAMPLER_VALUES,
                "paired_dataset_sha256": expected_paired_sha256,
            },
            "train_commands_equal_outside_declared_axis": True,
            "eval_commands_equal_outside_output_routing": True,
            "sim_d1_headroom_required": False,
        },
        "expected_training_iterations": int(spec["expected_training_iterations"]),
        "training_num_envs": int(spec["training_num_envs"]),
        "dataset_motion_count": motion_count,
    }


def _execute_preflight_errors(spec: dict[str, Any], *, repo_root: Path) -> list[str]:
    errors: list[str] = []
    for field in ("dataset_robot", "dataset_smpl"):
        value = str(spec.get(field) or "")
        if "REPLACE_ME" in value or not value:
            errors.append(f"{field} is unresolved: {value!r}")
            continue
        path = _relative_or_absolute(value, repo_root)
        if not path.exists():
            errors.append(f"{field} does not exist: {path}")

    ranking_locations: list[tuple[str, Any]] = []
    if spec.get("difficulty_ranking_json"):
        ranking_locations.append(("difficulty_ranking_json", spec["difficulty_ranking_json"]))
    for index, variant in enumerate(spec.get("variants", [])):
        if variant.get("difficulty_ranking_json"):
            ranking_locations.append(
                (
                    f"variants[{index}].difficulty_ranking_json",
                    variant["difficulty_ranking_json"],
                )
            )
    for label, ranking in ranking_locations:
        ranking_text = str(ranking)
        if "REPLACE_ME" in ranking_text:
            errors.append(f"{label} is unresolved: {ranking_text!r}")
        elif not _relative_or_absolute(ranking_text, repo_root).is_file():
            errors.append(f"{label} does not exist: {_relative_or_absolute(ranking_text, repo_root)}")

    for index, variant in enumerate(spec.get("variants", [])):
        for field in (
            "train_command",
            "eval_command",
            "checkpoint",
            "metrics_eval_json",
        ):
            value = str(variant.get(field) or "")
            if "REPLACE_ME" in value or "{seed}" in value:
                errors.append(f"variants[{index}].{field} contains an unresolved placeholder")
    if spec.get("requires_activation_gate") is True:
        for field in (
            "dataset_manifest_json",
            "dataset_manifest_sha256",
            "difficulty_ranking_sha256",
            "sim_d1_classification_json",
            "sim_d1_classification_sha256",
            "sim_d1_source_checkpoint_sha256",
            "training_python_executable",
            "training_initialization_checkpoint",
            "expected_training_iterations",
            "training_num_envs",
        ):
            value = str(spec.get(field) or "")
            if "REPLACE_ME" in value or not value:
                errors.append(f"{field} is unresolved: {value!r}")
    if _uses_official_zpd_pair_contract(spec):
        for field in (
            "dataset_manifest_json",
            "dataset_manifest_sha256",
            "paired_dataset_sha256",
            "training_initialization_checkpoint_sha256",
            "training_python_executable",
            "training_initialization_checkpoint",
            "expected_training_iterations",
            "training_num_envs",
        ):
            value = str(spec.get(field) or "")
            if "REPLACE_ME" in value or "{seed}" in value or not value:
                errors.append(f"{field} is unresolved: {value!r}")
    return errors


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
    input_provenance: dict[str, Any] = {}
    if not dry_run:
        preflight_errors = _execute_preflight_errors(spec, repo_root=repo_root)
        if not preflight_errors and spec.get("requires_activation_gate") is True:
            try:
                input_provenance = _verify_activation_preflight(spec, repo_root=repo_root)
            except (OSError, ValueError) as exc:
                preflight_errors.append(str(exc))
        elif not preflight_errors and _uses_official_zpd_pair_contract(spec):
            try:
                input_provenance = _verify_official_zpd_pair_preflight(spec, repo_root=repo_root)
            except (OSError, ValueError) as exc:
                preflight_errors.append(str(exc))
        if preflight_errors:
            raise ValueError("execute preflight failed: " + "; ".join(preflight_errors))

    output_dir.mkdir(parents=True, exist_ok=True)
    group = str(spec["experiment_group"])
    seed = int(spec["seed"])
    activation_gated = spec.get("requires_activation_gate") is True
    strict_training_contract = activation_gated or _uses_official_zpd_pair_contract(spec)
    expected_training_iterations = (
        int(spec["expected_training_iterations"]) if strict_training_contract else None
    )
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
        metrics_eval_text = variant.get("metrics_eval_json")
        metrics_eval_json = _relative_or_absolute(str(metrics_eval_text), repo_root) if metrics_eval_text else None
        ranking_text = variant.get("difficulty_ranking_json") or spec.get("difficulty_ranking_json")
        difficulty_ranking_json = _relative_or_absolute(str(ranking_text), repo_root) if ranking_text else None

        command_results: list[dict[str, Any]] = []
        execution_errors: list[str] = []
        metrics_coverage: dict[str, Any] = {}
        training_completion: dict[str, Any] = {
            "required": strict_training_contract,
            "expected_learning_iteration": expected_training_iterations,
            "observed_learning_iteration": None,
            "complete": None if dry_run or not strict_training_contract else False,
            "parser_semantics": "one_based_terminal_learning_iteration",
        }
        checkpoint_source = str(
            variant.get("checkpoint_source") or spec.get("checkpoint_source") or "configured_checkpoint"
        )
        checkpoint = str(variant.get("checkpoint") or spec["checkpoint"])
        checkpoint_provenance = variant.get("checkpoint_provenance")
        if not dry_run:
            trained_checkpoint_before: tuple[int, int] | None = None
            if checkpoint_source == "trained_variant_checkpoint":
                trained_checkpoint_before = _checkpoint_snapshot(_relative_or_absolute(checkpoint, repo_root))

            if variant.get("train_command"):
                if strict_training_contract:
                    try:
                        _verify_initialization_checkpoint_unchanged(
                            spec,
                            repo_root=repo_root,
                            input_provenance=input_provenance,
                        )
                    except ValueError as exc:
                        execution_errors.append(str(exc))
                if not execution_errors:
                    rc = _run_command(str(variant["train_command"]), train_log, cwd=repo_root)
                    command_results.append({"kind": "train", "returncode": rc, "log": str(train_log)})
                    if rc != 0:
                        execution_errors.append(f"train command failed with return code {rc}")
                    if strict_training_contract:
                        train_summary = summarize_logs(train_log).get("train", {})
                        observed_iteration = train_summary.get("learning_iteration")
                        training_completion["observed_learning_iteration"] = observed_iteration
                        training_completion["complete"] = observed_iteration == expected_training_iterations
                        if rc == 0 and not training_completion["complete"]:
                            execution_errors.append(
                                "training did not reach the exact preregistered terminal "
                                "learning iteration: expected "
                                f"{expected_training_iterations}, observed "
                                f"{observed_iteration}"
                            )

            if checkpoint_source == "trained_variant_checkpoint" and not execution_errors:
                checkpoint_provenance, checkpoint_error = _verify_trained_checkpoint(
                    checkpoint,
                    repo_root=repo_root,
                    before_train=trained_checkpoint_before,
                    require_fresh=bool(variant.get("train_command")),
                )
                if checkpoint_error:
                    execution_errors.append(checkpoint_error)

            if variant.get("eval_command") and not execution_errors:
                metrics_eval_before = (
                    _checkpoint_snapshot(metrics_eval_json) if metrics_eval_json is not None else None
                )
                rc = _run_command(str(variant["eval_command"]), eval_log, cwd=repo_root)
                command_results.append({"kind": "eval", "returncode": rc, "log": str(eval_log)})
                if rc != 0:
                    execution_errors.append(f"eval command failed with return code {rc}")
                elif metrics_eval_json is not None:
                    metrics_eval_after = _checkpoint_snapshot(metrics_eval_json)
                    if metrics_eval_after is None:
                        execution_errors.append(
                            f"eval did not produce required per-motion metrics: {metrics_eval_json}"
                        )
                    elif metrics_eval_before is not None and metrics_eval_after == metrics_eval_before:
                        execution_errors.append(
                            f"eval did not update required per-motion metrics: {metrics_eval_json}"
                        )

            if not execution_errors and strict_training_contract:
                if metrics_eval_json is None:
                    execution_errors.append("strict paired eval requires per-motion metrics")
                else:
                    try:
                        metrics_coverage = _verify_metrics_motion_coverage(
                            metrics_eval_json,
                            list(input_provenance["dataset_manifest"]["motion_keys"]),
                        )
                    except (OSError, ValueError) as exc:
                        execution_errors.append(str(exc))

        if not _summary_is_stale(summary_json, train_log, eval_log, metrics_eval_json):
            # Existing summary is newer than logs; keep it immutable and use it as source of truth.
            summary_source = summary_json
        elif train_log.exists() or eval_log.exists():
            summary = summarize_logs(
                train_log if train_log.exists() else None,
                eval_log if eval_log.exists() else None,
                metrics_eval_json=(
                    metrics_eval_json if metrics_eval_json is not None and metrics_eval_json.is_file() else None
                ),
                difficulty_ranking_json=(
                    difficulty_ranking_json
                    if metrics_eval_json is not None and metrics_eval_json.is_file()
                    else None
                ),
            )
            _write_json(summary_json, summary)
            write_summary_markdown(summary_md, summary)
            summary_source = summary_json
        else:
            summary_source = None

        manifest_errors: list[str] = []
        if summary_source is not None and not execution_errors:
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
                checkpoint_source=checkpoint_source,
                checkpoint_provenance=checkpoint_provenance,
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
                "execution_ok": None if dry_run else not execution_errors,
                "execution_errors": execution_errors,
                "checkpoint": checkpoint,
                "checkpoint_source": checkpoint_source,
                "checkpoint_provenance": checkpoint_provenance or {},
                "metrics_eval_json": str(metrics_eval_json) if metrics_eval_json else None,
                "metrics_coverage": metrics_coverage,
                "training_completion": training_completion,
                "difficulty_ranking_json": (str(difficulty_ranking_json) if difficulty_ranking_json else None),
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
        "sampler_pair_contract": spec.get("sampler_pair_contract"),
        "expected_training_iterations": expected_training_iterations,
        "training_completeness_ok": (
            None
            if dry_run or not strict_training_contract
            else all(result["training_completion"]["complete"] is True for result in variant_results)
        ),
        "command_environment": _command_environment_provenance(),
        "input_provenance": input_provenance,
        "output_dir": str(output_dir),
        "variants": variant_results,
        "execution_ok": (
            None if dry_run else all(result.get("execution_ok") is True for result in variant_results)
        ),
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
        for key in [
            "experiment_group",
            "dry_run",
            "seed",
            "dataset_robot",
            "dataset_smpl",
            "checkpoint",
            "sampler_pair_contract",
            "expected_training_iterations",
            "training_completeness_ok",
            "ok_for_causal_comparison",
        ]:
            f.write(f"| `{key}` | {_format_value(plan.get(key))} |\n")
        f.write("\n## Variants\n\n")
        for variant in plan.get("variants", []):
            f.write(f"### {variant['name']}\n\n")
            f.write("| Field | Value |\n|---|---|\n")
            for key in [
                "execution_ok",
                "execution_errors",
                "checkpoint",
                "checkpoint_source",
                "checkpoint_provenance",
                "training_completion",
                "metrics_coverage",
                "summary_json",
                "manifest_json",
                "manifest_errors",
            ]:
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
    if args.execute and plan.get("execution_ok") is not True:
        print("ERROR: one or more variant executions failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
