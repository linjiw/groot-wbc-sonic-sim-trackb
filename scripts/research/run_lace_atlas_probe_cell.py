#!/usr/bin/env python3
"""Prepare or launch one locked LACE atlas-probe schedule cell."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.instrument_runtime import (  # noqa: E402
    SCIENTIFIC_CACHE_ENVIRONMENT_KEYS,
    SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
    build_plan_binding,
    build_runtime_handshake,
    build_source_bundle,
    discover_scientific_source_paths,
    validate_scientific_cache_environment,
    write_new_json,
)
from gear_sonic.research.lace.schedule_batch_loader import (  # noqa: E402
    build_launch_plan,
    load_locked_rollout_schedule,
    select_locked_atlas_probe_cell_for_filtered_library,
    sha256_file,
)

_PROTECTED_HYDRA_KEYS = (
    "checkpoint",
    "num_envs",
    "seed",
    "run_eval_loop",
    "run_once",
    "headless",
    "max_render_steps",
    "use_encoder",
    "eval_callbacks",
    "manager_env.config.terrain_type",
    "manager_env.config.render_results",
    "manager_env.observations.policy.enable_corruption",
    "manager_env.observations.tokenizer.enable_corruption",
    "manager_env.commands.motion.filter_motion_keys",
    "manager_env.commands.motion.motion_lib_cfg.motion_file",
    "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file",
    "manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable",
    "manager_env.commands.motion.motion_lib_cfg.filter_motion_keys",
    "manager_env.commands.motion.atlas_probe_mode",
    "manager_env.commands.motion.atlas_probe_schedule_sha256",
    "manager_env.commands.motion.atlas_probe_assignments",
    "manager_env.recorders.failure_atlas",
    "lace_scientific_instrument_required",
    "lace_instrument_handshake_path",
    "lace_instrument_handshake_file_sha256",
    "lace_launch_plan_path",
    "lace_analysis_protocol_path",
    "lace_analysis_protocol_file_sha256",
    "lace_analysis_protocol_sha256",
    "lace_analysis_protocol_lock_path",
    "lace_analysis_protocol_lock_file_sha256",
    "lace_analysis_protocol_lock_sha256",
    "lace_protocol_preflight_request_path",
    "lace_protocol_preflight_request_file_sha256",
    "protocol_preflight_request_sha256",
)

_PYTHONPATH_SEMANTICS = "repository_root_first_preserve_inherited_unique_entries_v1"


def _repo_first_pythonpath(inherited: str | None) -> str:
    entries = [str(REPO_ROOT)]
    if inherited:
        entries.extend(entry for entry in inherited.split(os.pathsep) if entry)
    unique_entries: list[str] = []
    for entry in entries:
        if entry not in unique_entries:
            unique_entries.append(entry)
    return os.pathsep.join(unique_entries)


def _launch_cache_environment(
    *,
    storage_root: Path,
    schedule_sha256: str,
    cell: Mapping[str, Any],
    plan_output: Path,
    rollout_output: Path,
) -> tuple[str, dict[str, str]]:
    from gear_sonic.research.lace.schema import canonical_sha256

    root = storage_root.expanduser()
    root = root if root.is_absolute() else Path.cwd() / root
    if root.is_symlink():
        raise ValueError(f"runtime storage root may not be a symlink: {root}")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    token = canonical_sha256(
        {
            "schedule_sha256": schedule_sha256,
            "cell": dict(cell),
            "plan_output": str(plan_output),
            "rollout_output": str(rollout_output),
        }
    )
    launch_root = root / token
    names = {
        "TMPDIR": "tmp",
        "XDG_CACHE_HOME": "xdg-cache",
        "ISAACLAB_USD_CACHE_DIR": "isaaclab-usd-cache",
        "CUDA_CACHE_PATH": "cuda-cache",
        "TORCH_HOME": "torch-home",
        "OMNI_USER_CACHE_DIR": "omni-user-cache",
    }
    if tuple(names) != SCIENTIFIC_CACHE_ENVIRONMENT_KEYS:
        raise RuntimeError("runner cache environment keys drifted from runtime validation")
    environment = {key: str(launch_root / suffix) for key, suffix in names.items()}
    for path in environment.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    validate_scientific_cache_environment(environment, launch_cache_token=token)
    return token, environment


def _write_canonical_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    if not force:
        try:
            write_new_json(path, payload)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite launch plan {path}; pass --force explicitly"
            ) from error
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w" if force else "x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite launch plan {path}; pass --force explicitly"
        ) from error


def _reject_protected_overrides(arguments: Sequence[str]) -> None:
    for argument in arguments:
        normalized = argument.lstrip("+~")
        key = normalized.split("=", 1)[0]
        if any(
            key == protected or key.startswith(f"{protected}.")
            for protected in _PROTECTED_HYDRA_KEYS
        ):
            raise ValueError(
                f"pass-through Hydra argument {argument!r} conflicts with a frozen atlas field"
            )
        if key == "manager_env/recorders":
            raise ValueError(
                "pass-through arguments cannot replace the frozen lace_atlas recorder group"
            )


def _checkpoint_sha256(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"checkpoint is not a file: {path}")
    return sha256_file(path)


def _checkpoint_bundle(checkpoint: Path) -> dict[str, str]:
    config = checkpoint.parent / "config.yaml"
    if not config.is_file():
        raise ValueError(
            "checkpoint parent must contain SONIC config.yaml for architecture reconstruction: "
            f"{config}"
        )
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _checkpoint_sha256(checkpoint),
        "config_path": str(config),
        "config_sha256": sha256_file(config),
    }


def _current_git_commit() -> str:
    completed = subprocess.run(  # noqa: S603
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot resolve repository commit: {completed.stderr.strip()}")
    commit = completed.stdout.strip()
    if len(commit) != 40 or commit != commit.lower():
        raise RuntimeError("git rev-parse did not return a full lowercase commit id")
    try:
        int(commit, 16)
    except ValueError as error:
        raise RuntimeError("git rev-parse returned a non-hexadecimal commit id") from error
    return commit


def _instrument_runtime_paths(instrument_output: Path) -> dict[str, Path]:
    output = instrument_output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--instrument-output must use the .json suffix")
    return {
        "instrument": output,
        "handshake": output.with_name(f"{output.stem}.runtime-handshake.json"),
        "binding": output.with_name(f"{output.stem}.runtime-binding.json"),
        "rollout_binding": output.with_name(f"{output.stem}.rollout-binding.json"),
    }


def _instrumented_rollout_path(rollout_output: Path) -> Path:
    output = rollout_output.expanduser().resolve()
    if output.suffix != ".jsonl":
        raise ValueError("--rollout-output must use the .jsonl suffix")
    return output.with_name(f"{output.stem}.instrumented.jsonl")


def _validated_eval_entrypoint(path: Path, *, scientific_use: bool) -> Path:
    entrypoint = path.expanduser().resolve()
    if not entrypoint.is_file():
        raise ValueError(f"eval entrypoint is not a file: {entrypoint}")
    if scientific_use:
        canonical_entrypoint = (REPO_ROOT / "gear_sonic/eval_agent_trl.py").resolve()
        if entrypoint != canonical_entrypoint:
            raise ValueError(
                "scientific schedules require the source-bound canonical eval entrypoint: "
                f"{canonical_entrypoint}"
            )
    return entrypoint


def _validate_new_scientific_paths(
    *,
    plan_output: Path,
    rollout_output: Path,
    runtime_paths: Mapping[str, Path],
    force: bool,
) -> None:
    if force:
        raise ValueError("scientific launch artifacts are immutable and may not use --force")
    named_paths = {
        "plan output": plan_output,
        "rollout output": rollout_output,
        **{name.replace("_", " "): path for name, path in runtime_paths.items()},
    }
    resolved_paths = [path.expanduser().resolve() for path in named_paths.values()]
    if len(resolved_paths) != len(set(resolved_paths)):
        raise ValueError("scientific launch, rollout, and instrument paths must be distinct")
    for name, path in named_paths.items():
        resolved = path.expanduser().resolve()
        if resolved.exists():
            raise FileExistsError(f"scientific {name} already exists: {resolved}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schedule-lock",
        type=Path,
        required=True,
        help="Frozen schema-v2 schedule lock JSON",
    )
    parser.add_argument("--probe-policy-id", required=True)
    parser.add_argument("--domain-randomization-seed", type=int, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--repeat-index", type=int, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Exact checkpoint file; its bytes must match the policy's frozen SHA-256",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        required=True,
        help="Must equal the full motion count of the selected schedule cell",
    )
    parser.add_argument(
        "--robot-motion-root",
        type=Path,
        required=True,
        help="Robot motion directory; selected files are checked against the frozen inventory",
    )
    smpl_group = parser.add_mutually_exclusive_group(required=True)
    smpl_group.add_argument(
        "--smpl-motion-root",
        type=Path,
        help="Real SMPL directory with a direct <motion_key>.pkl for every selected motion",
    )
    smpl_group.add_argument(
        "--use-dummy-smpl",
        action="store_true",
        help="Explicit G1-only pilot mode; rejected by scientific schedules",
    )
    parser.add_argument(
        "--rollout-output",
        type=Path,
        required=True,
        help="New recorder JSONL path (normally below /data)",
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        required=True,
        help="New canonical launch-plan JSON path (normally below /data)",
    )
    parser.add_argument(
        "--instrument-output",
        type=Path,
        help=(
            "Required new instrument JSON path for scientific schedules; forbidden for pilots. "
            "Runtime handshake/binding paths are derived beside it."
        ),
    )
    parser.add_argument(
        "--analysis-protocol-lock",
        type=Path,
        help=(
            "Required exclusive-create pre-outcome protocol lock for scientific schedules; "
            "forbidden for non-scientific pilots. Preserve its digest in an append-only registry."
        ),
    )
    parser.add_argument(
        "--protocol-preflight-output",
        type=Path,
        help=(
            "New analysis-protocol candidate JSON. This outcome-free bootstrap mode "
            "forbids a protocol lock/instrument, derives request+receipt siblings, "
            "and exits before the eval entrypoint's first explicit reset."
        ),
    )
    parser.add_argument(
        "--expected-analysis-protocol-lock-sha256",
        help=(
            "Independently recorded preregistration digest. Required for scientific launch; "
            "do not copy it from a post-outcome lock file."
        ),
    )
    parser.add_argument(
        "--instrument-source",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional repository source to byte-bind. Scientific runs already bind every "
            "regular gear_sonic source/robot-asset file (excluding generated caches) plus the "
            "launcher/build scripts."
        ),
    )
    parser.add_argument(
        "--runtime-storage-root",
        type=Path,
        help=(
            "Root for per-launch temp/Isaac/Omni/CUDA/Torch caches. Defaults beside "
            "--rollout-output so /data rollouts never fall back to a root-filesystem cache."
        ),
    )
    parser.add_argument(
        "--eval-entrypoint",
        type=Path,
        default=REPO_ROOT / "gear_sonic/eval_agent_trl.py",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing plan only")
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Run the generated eval command after writing the plan; default is print-only",
    )
    parser.add_argument(
        "hydra_args",
        nargs=argparse.REMAINDER,
        help="Additional non-conflicting Hydra arguments after '--'",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preflight_mode = args.protocol_preflight_output is not None
    passthrough = list(args.hydra_args)
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    _reject_protected_overrides(passthrough)

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_bundle = _checkpoint_bundle(checkpoint)
    locked = load_locked_rollout_schedule(args.schedule_lock, repo_root=REPO_ROOT)
    selection = select_locked_atlas_probe_cell_for_filtered_library(
        locked,
        probe_policy_id=args.probe_policy_id,
        domain_randomization_seed=args.domain_randomization_seed,
        phase_id=args.phase_id,
        repeat_index=args.repeat_index,
        expected_checkpoint_sha256=checkpoint_bundle["checkpoint_sha256"],
        num_envs=args.num_envs,
    )
    rollout_output = args.rollout_output.expanduser().resolve()
    if not bool(selection.locked_schedule.manifest["scientific_use"]):
        rollout_output.parent.mkdir(parents=True, exist_ok=True)
    plan_output = args.plan_output.expanduser().resolve()
    plan = build_launch_plan(
        selection,
        checkpoint_path=checkpoint,
        rollout_output_path=rollout_output,
        robot_motion_root=args.robot_motion_root,
        smpl_motion_root=args.smpl_motion_root,
        use_dummy_smpl=args.use_dummy_smpl,
    )
    if plan["scientific_use"] and passthrough:
        raise ValueError(
            "scientific schedules forbid passthrough Hydra arguments; freeze every "
            "measurement knob in the analysis protocol instead"
        )
    if plan["scientific_use"] and args.launch and not preflight_mode:
        raise ValueError(
            "scientific launch is NO-GO until an external append-only protocol-lock "
            "anchor and outcome-free live protocol preflight are verified; local lock "
            "self-attestation is procedural evidence only"
        )
    runtime_storage_root = (
        args.runtime_storage_root
        if args.runtime_storage_root is not None
        else (
            Path("/data/robotixx/groot-wbc-sonic-research/lace/runtime-cache")
            if plan["scientific_use"]
            else rollout_output.parent / ".lace-runtime"
        )
    )
    if plan["scientific_use"]:
        resolved_storage_root = runtime_storage_root.expanduser().resolve()
        if not resolved_storage_root.is_relative_to(Path("/data")):
            raise ValueError("scientific runtime storage root must be under /data")
    entrypoint = _validated_eval_entrypoint(
        args.eval_entrypoint,
        scientific_use=bool(plan["scientific_use"]),
    )
    if preflight_mode:
        if not plan["scientific_use"]:
            raise ValueError("protocol preflight requires a frozen scientific schedule")
        if (
            args.instrument_output is not None
            or args.analysis_protocol_lock is not None
            or args.expected_analysis_protocol_lock_sha256 is not None
        ):
            raise ValueError(
                "protocol preflight forbids instrument/protocol-lock inputs; its candidate "
                "must be locked and externally anchored only after zero-outcome capture"
            )
        if args.force:
            raise ValueError("protocol preflight artifacts are immutable and forbid --force")
        protocol_output = args.protocol_preflight_output.expanduser().resolve()
        if protocol_output.suffix != ".json":
            raise ValueError("--protocol-preflight-output must use the .json suffix")
        request_output = protocol_output.with_name(f"{protocol_output.stem}.preflight-request.json")
        receipt_output = protocol_output.with_name(f"{protocol_output.stem}.preflight-receipt.json")
        preflight_paths = {
            "plan": plan_output,
            "recorder": rollout_output,
            "protocol": protocol_output,
            "request": request_output,
            "receipt": receipt_output,
        }
        if len(set(preflight_paths.values())) != len(preflight_paths):
            raise ValueError("protocol preflight paths must be distinct")
        for name, path in preflight_paths.items():
            if path.exists():
                raise FileExistsError(f"protocol preflight {name} already exists: {path}")
            if not path.is_relative_to(Path("/data")):
                raise ValueError(f"protocol preflight {name} must be stored under /data")

        from gear_sonic.research.lace.protocol_preflight import (
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD,
            build_protocol_preflight_request,
        )
        from gear_sonic.research.lace.schema import canonical_sha256

        source_paths = discover_scientific_source_paths(
            REPO_ROOT,
            extra_paths=args.instrument_source,
        )
        source_bundle = build_source_bundle(REPO_ROOT, source_paths)
        request = build_protocol_preflight_request(
            schedule_lock_path=selection.locked_schedule.lock_path,
            schedule_path=selection.locked_schedule.schedule_path,
            schedule_manifest=selection.locked_schedule.manifest,
            split_path=selection.locked_schedule.split_path,
            split_manifest=selection.locked_schedule.split_manifest,
            reference_length_inventory_path=(
                selection.locked_schedule.reference_length_inventory_path
            ),
            reference_length_inventory=(selection.locked_schedule.reference_length_inventory),
            cell=plan["cell"],
            rollout_ids=plan["rollout_ids"],
            checkpoint_bundle=checkpoint_bundle,
            dataset_binding_sha256=canonical_sha256(plan["dataset_binding"]),
            source_bundle=source_bundle,
            git_commit=_current_git_commit(),
            launch_plan_path=plan_output,
            recorder_output_path=rollout_output,
            analysis_protocol_output_path=protocol_output,
            preflight_receipt_output_path=receipt_output,
            repo_root=REPO_ROOT,
        )
        request_path = write_new_json(request_output, request)
        cache_token, cache_environment = _launch_cache_environment(
            storage_root=runtime_storage_root,
            schedule_sha256=plan["schedule_sha256"],
            cell=plan["cell"],
            plan_output=plan_output,
            rollout_output=rollout_output,
        )
        request_binding = {
            "request_path": str(request_path),
            "request_file_sha256": sha256_file(request_path),
            PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD: request[
                PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD
            ],
        }
        plan["protocol_preflight"] = request_binding
        plan["instrument_runtime"] = {
            "required": False,
            "state": "outcome_free_protocol_preflight_only_no_scientific_claims",
            "instrument_output_path": None,
        }
        plan["analysis_protocol_path"] = None
        plan["analysis_protocol_file_sha256"] = None
        plan["analysis_protocol_sha256"] = None
        plan["analysis_protocol_lock_path"] = None
        plan["analysis_protocol_lock_file_sha256"] = None
        plan["analysis_protocol_lock_sha256"] = None
        preflight_overrides = [
            "++lace_scientific_instrument_required=false",
            "++lace_protocol_preflight_request_path=" + json.dumps(str(request_path)),
            "++lace_protocol_preflight_request_file_sha256="
            + request_binding["request_file_sha256"],
            "++protocol_preflight_request_sha256="
            + request[PROTOCOL_PREFLIGHT_REQUEST_DIGEST_FIELD],
            "++lace_launch_plan_path=" + json.dumps(str(plan_output)),
        ]
        plan["hydra_overrides"] = [*plan["hydra_overrides"], *preflight_overrides]
        plan["eval_entrypoint"] = str(entrypoint)
        plan["passthrough_hydra_args"] = passthrough
        plan["checkpoint_bundle"] = checkpoint_bundle
        plan["command"] = [
            str(Path(sys.executable).resolve()),
            str(entrypoint),
            *passthrough,
            *plan["hydra_overrides"],
        ]
        pythonpath = _repo_first_pythonpath(os.environ.get("PYTHONPATH"))
        plan["launch_environment"] = {
            "PYTHONPATH": pythonpath,
            "semantics": _PYTHONPATH_SEMANTICS,
            "scientific_cache_environment": cache_environment,
            "scientific_cache_environment_semantics": (SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS),
            "launch_cache_token": cache_token,
        }
        plan["launch_plan_sha256"] = canonical_sha256(
            plan,
            digest_field="launch_plan_sha256",
        )
        _write_canonical_json(plan_output, plan, force=False)
        print(
            json.dumps(
                {
                    "launch": args.launch,
                    "mode": "outcome_free_protocol_preflight",
                    "plan_output": str(plan_output),
                    "protocol_output": str(protocol_output),
                    "request_output": str(request_output),
                    "receipt_output": str(receipt_output),
                    "rollout_output_must_remain_absent": str(rollout_output),
                    "shell_command": shlex.join(
                        [
                            "env",
                            f"PYTHONPATH={pythonpath}",
                            *(f"{key}={cache_environment[key]}" for key in cache_environment),
                            *plan["command"],
                        ]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        if args.launch:
            launch_environment = dict(os.environ)
            launch_environment["PYTHONPATH"] = pythonpath
            launch_environment.update(cache_environment)
            completed = subprocess.run(  # noqa: S603
                plan["command"],
                cwd=REPO_ROOT,
                check=False,
                env=launch_environment,
            )
            return int(completed.returncode)
        return 0
    if plan["scientific_use"]:
        if args.instrument_output is None:
            raise ValueError("scientific schedules require --instrument-output")
        if args.analysis_protocol_lock is None:
            raise ValueError("scientific schedules require --analysis-protocol-lock before launch")
        if args.expected_analysis_protocol_lock_sha256 is None:
            raise ValueError(
                "scientific schedules require an independently supplied "
                "--expected-analysis-protocol-lock-sha256"
            )
        from gear_sonic.research.lace.analysis_protocol_lock import (
            ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD,
            checkpoint_config_binding,
            execution_cell_binding,
            load_analysis_protocol_lock,
        )

        protocol_lock_path, protocol_lock, protocol_path, _ = load_analysis_protocol_lock(
            args.analysis_protocol_lock.expanduser().absolute(),
            repo_root=REPO_ROOT,
        )
        if (
            args.expected_analysis_protocol_lock_sha256
            != protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
        ):
            raise ValueError(
                "analysis protocol lock differs from the independently supplied "
                "preregistration digest"
            )
        expected_checkpoint_config = checkpoint_config_binding(
            protocol_lock,
            probe_policy_id=plan["cell"]["probe_policy_id"],
            checkpoint_sha256=plan["cell"]["checkpoint_sha256"],
        )
        if (
            checkpoint_bundle["config_path"] != expected_checkpoint_config["config_path"]
            or checkpoint_bundle["config_sha256"] != expected_checkpoint_config["config_sha256"]
        ):
            raise ValueError(
                "checkpoint companion config differs from the pre-outcome "
                "analysis-protocol-lock binding"
            )
        runtime_paths = _instrument_runtime_paths(args.instrument_output)
        runtime_paths["instrumented_rollout"] = _instrumented_rollout_path(rollout_output)
        registered_execution = execution_cell_binding(
            protocol_lock,
            cell=plan["cell"],
        )
        expected_execution_paths = {
            "plan_output_path": str(plan_output),
            "rollout_output_path": str(rollout_output),
            "instrumented_rollout_output_path": str(runtime_paths["instrumented_rollout"]),
            "instrument_output_path": str(runtime_paths["instrument"]),
            "runtime_handshake_output_path": str(runtime_paths["handshake"]),
            "instrument_binding_output_path": str(runtime_paths["binding"]),
            "rollout_binding_output_path": str(runtime_paths["rollout_binding"]),
        }
        if any(
            registered_execution[field] != path for field, path in expected_execution_paths.items()
        ):
            raise ValueError(
                "scientific output paths differ from the preregistered one-attempt cell layout"
            )
        if (
            str(resolved_storage_root)
            != protocol_lock["execution_registry"]["runtime_storage_root"]
        ):
            raise ValueError(
                "scientific runtime storage root differs from the preregistered layout"
            )
        _validate_new_scientific_paths(
            plan_output=plan_output,
            rollout_output=rollout_output,
            runtime_paths=runtime_paths,
            force=args.force,
        )
        # This is the first mutation below either preregistered root.  All lock,
        # checkpoint, cell, output, and cache-root coordinates are proven above.
        cache_token, cache_environment = _launch_cache_environment(
            storage_root=runtime_storage_root,
            schedule_sha256=plan["schedule_sha256"],
            cell=plan["cell"],
            plan_output=plan_output,
            rollout_output=rollout_output,
        )
        source_paths = discover_scientific_source_paths(
            REPO_ROOT,
            extra_paths=args.instrument_source,
        )
        handshake = build_runtime_handshake(
            schedule_sha256=plan["schedule_sha256"],
            checkpoint_sha256=plan["cell"]["checkpoint_sha256"],
            probe_policy_id=plan["cell"]["probe_policy_id"],
            rollout_ids=plan["rollout_ids"],
            rollout_output_path=rollout_output,
            instrumented_rollout_output_path=runtime_paths["instrumented_rollout"],
            launch_plan_path=plan_output,
            instrument_output_path=runtime_paths["instrument"],
            instrument_binding_output_path=runtime_paths["binding"],
            rollout_binding_output_path=runtime_paths["rollout_binding"],
            analysis_protocol_path=protocol_path,
            git_commit=_current_git_commit(),
            repo_root=REPO_ROOT,
            source_paths=source_paths,
        )
        handshake_path = write_new_json(runtime_paths["handshake"], handshake)
        instrument_runtime = build_plan_binding(handshake_path, handshake)
        runtime_overrides = [
            "++lace_scientific_instrument_required=true",
            "++lace_instrument_handshake_path=" + json.dumps(str(handshake_path)),
            "++lace_instrument_handshake_file_sha256="
            + instrument_runtime["handshake_file_sha256"],
            "++lace_launch_plan_path=" + json.dumps(str(plan_output)),
            "++lace_analysis_protocol_path=" + json.dumps(handshake["analysis_protocol_path"]),
            "++lace_analysis_protocol_file_sha256=" + handshake["analysis_protocol_file_sha256"],
            "++lace_analysis_protocol_sha256=" + handshake["analysis_protocol_sha256"],
            "++lace_analysis_protocol_lock_path=" + json.dumps(str(protocol_lock_path)),
            "++lace_analysis_protocol_lock_file_sha256=" + sha256_file(protocol_lock_path),
            "++lace_analysis_protocol_lock_sha256="
            + protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD],
        ]
        plan["analysis_protocol_path"] = handshake["analysis_protocol_path"]
        plan["analysis_protocol_file_sha256"] = handshake["analysis_protocol_file_sha256"]
        plan["analysis_protocol_sha256"] = handshake["analysis_protocol_sha256"]
        plan["analysis_protocol_lock_path"] = str(protocol_lock_path)
        plan["analysis_protocol_lock_file_sha256"] = sha256_file(protocol_lock_path)
        plan["analysis_protocol_lock_sha256"] = protocol_lock[ANALYSIS_PROTOCOL_LOCK_DIGEST_FIELD]
    else:
        if (
            args.instrument_output is not None
            or args.instrument_source
            or args.analysis_protocol_lock is not None
            or args.expected_analysis_protocol_lock_sha256 is not None
        ):
            raise ValueError(
                "non-scientific pilot schedules may not claim an instrument output or sources"
            )
        instrument_runtime = {
            "required": False,
            "state": "forbidden_for_non_scientific_contract_smoke",
            "instrument_output_path": None,
        }
        runtime_overrides = ["++lace_scientific_instrument_required=false"]
        plan["analysis_protocol_path"] = None
        plan["analysis_protocol_file_sha256"] = None
        plan["analysis_protocol_sha256"] = None
        plan["analysis_protocol_lock_path"] = None
        plan["analysis_protocol_lock_file_sha256"] = None
        plan["analysis_protocol_lock_sha256"] = None
        cache_token, cache_environment = _launch_cache_environment(
            storage_root=runtime_storage_root,
            schedule_sha256=plan["schedule_sha256"],
            cell=plan["cell"],
            plan_output=plan_output,
            rollout_output=rollout_output,
        )
    plan["instrument_runtime"] = instrument_runtime
    plan["hydra_overrides"] = [*plan["hydra_overrides"], *runtime_overrides]
    command = [
        str(Path(sys.executable).resolve()),
        str(entrypoint),
        *passthrough,
        *plan["hydra_overrides"],
    ]
    plan["eval_entrypoint"] = str(entrypoint)
    plan["passthrough_hydra_args"] = passthrough
    plan["command"] = command
    plan["checkpoint_bundle"] = checkpoint_bundle
    pythonpath = _repo_first_pythonpath(os.environ.get("PYTHONPATH"))
    plan["launch_environment"] = {
        "PYTHONPATH": pythonpath,
        "semantics": _PYTHONPATH_SEMANTICS,
        "scientific_cache_environment": cache_environment,
        "scientific_cache_environment_semantics": SCIENTIFIC_CACHE_ENVIRONMENT_SEMANTICS,
        "launch_cache_token": cache_token,
    }
    # Bind the final entrypoint and command, not merely the reusable override set.
    from gear_sonic.research.lace.schema import canonical_sha256

    plan["launch_plan_sha256"] = canonical_sha256(
        plan,
        digest_field="launch_plan_sha256",
    )
    _write_canonical_json(plan_output, plan, force=args.force)

    print(
        json.dumps(
            {
                "launch": args.launch,
                "launch_plan_sha256": plan["launch_plan_sha256"],
                "instrument_runtime": plan["instrument_runtime"],
                "num_envs": plan["num_envs"],
                "plan_output": str(plan_output),
                "rollout_output": str(rollout_output),
                "schedule_sha256": plan["schedule_sha256"],
                "shell_command": shlex.join(
                    [
                        "env",
                        f"PYTHONPATH={pythonpath}",
                        *(f"{key}={cache_environment[key]}" for key in cache_environment),
                        *command,
                    ]
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.launch:
        launch_environment = dict(os.environ)
        launch_environment["PYTHONPATH"] = pythonpath
        launch_environment.update(cache_environment)
        completed = subprocess.run(  # noqa: S603
            command,
            cwd=REPO_ROOT,
            check=False,
            env=launch_environment,
        )
        return int(completed.returncode)
    return 0


if __name__ == "__main__":
    os.environ.setdefault(
        "TMPDIR",
        "/data/robotixx/groot-wbc-sonic-research/lace/tmp",
    )
    sys.exit(main())
