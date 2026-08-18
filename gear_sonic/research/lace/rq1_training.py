"""Fail-closed planning and accounting for one LACE RQ1 training arm.

The fixed sampler proves which distribution is installed.  This module closes
the next boundary: it freezes one arm into one fresh training process, records
the motion that occupied every control transition, and emits a receipt only
when the complete PPO schedule was attempted and every synchronized update
succeeded.

The planning and validation functions are CPU-only.  Simulator integration is
limited to :class:`RQ1RuntimeAccounting`, whose tensor inputs are converted to
plain integer vectors immediately.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any
import uuid

from gear_sonic.research.lace.fixed_distribution import (
    FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
    FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
    FIXED_DISTRIBUTION_DRAW_REPORT_KIND,
    FIXED_DISTRIBUTION_DRAW_REPORT_SCHEMA_VERSION,
    resolve_fixed_distribution,
)
from gear_sonic.research.lace.intervention_plan import (
    INTERVENTION_PLAN_DIGEST_FIELD,
)
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.throughput import (
    expected_optimizer_schedule,
    verify_partition_subset,
)

PROTOCOL_KIND = "lace_rq1_training_protocol"
PROTOCOL_SCHEMA_VERSION = 2
PROTOCOL_DIGEST_FIELD = "protocol_sha256"
PLAN_KIND = "lace_rq1_training_plan"
PLAN_SCHEMA_VERSION = 2
PLAN_DIGEST_FIELD = "plan_sha256"
RUNTIME_METRICS_KIND = "lace_rq1_training_runtime_metrics"
RUNTIME_METRICS_SCHEMA_VERSION = 2
RUNTIME_METRICS_DIGEST_FIELD = "runtime_metrics_sha256"
RECEIPT_KIND = "lace_rq1_training_receipt"
RECEIPT_SCHEMA_VERSION = 2
RECEIPT_DIGEST_FIELD = "receipt_sha256"
ATTEMPT_CLAIM_KIND = "lace_rq1_training_attempt_claim"
ATTEMPT_CLAIM_SCHEMA_VERSION = 1
ATTEMPT_CLAIM_DIGEST_FIELD = "attempt_claim_sha256"
EXECUTION_COMPLETION_KIND = "lace_rq1_training_execution_completion"
EXECUTION_COMPLETION_SCHEMA_VERSION = 1
EXECUTION_COMPLETION_DIGEST_FIELD = "execution_completion_sha256"
EXPERIMENT = "manager/universal_token/g1_only/lace_lite_s"
PARTITION = "D_curriculum"
BASE_ARM_ID = "base"
RUNTIME_METRICS_FILENAME = "rq1_runtime_metrics.json"
RECEIPT_FILENAME = "rq1_training_receipt.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")
_UTC_RE = re.compile(
    r"(?P<seconds>[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,9}))?Z"
)
_RUNTIME_ENV_NAMES = {
    "ACCEPT_EULA",
    "CARB_APP_PATH",
    "CONDA_DEFAULT_ENV",
    "CONDA_PREFIX",
    "CUDA_HOME",
    "CUDA_PATH",
    "DISPLAY",
    "EXP_PATH",
    "HOME",
    "ISAACLAB_PATH",
    "ISAAC_PATH",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "SHELL",
    "TERM",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
    "XAUTHORITY",
}
_RUNTIME_ENV_PREFIXES = (
    "CUDA_",
    "CUBLAS_",
    "CUDNN_",
    "HYDRA_",
    "ISAAC_",
    "MKL_",
    "NCCL_",
    "NVIDIA_",
    "OMP_",
    "OMNI_",
    "PYTORCH_",
    "TORCH_",
)
_PROTOCOL_FIELDS = {
    "kind",
    "schema_version",
    "frozen",
    "scientific_use",
    "declared_before_transfer_outcomes",
    "run_id",
    "arm_id",
    "repo_root",
    "storage_root",
    "output_root",
    "tmp_root",
    "python_executable",
    "train_entrypoint",
    "experiment",
    "seed",
    "fixed_distribution",
    "dataset",
    "initialization",
    "training",
    "gpu_isolation",
    PROTOCOL_DIGEST_FIELD,
}
_FIXED_FIELDS = {"plan_lock_path", "plan_lock_file_sha256"}
_DATASET_FIELDS = {
    "partition",
    "split_manifest",
    "subset_root",
    "expected_motion_count",
}
_INITIALIZATION_FIELDS = {
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_config_path",
    "checkpoint_config_sha256",
    "mode",
    "resume",
}
_TRAINING_FIELDS = {
    "num_envs_per_rank",
    "world_size",
    "rollout_steps_per_iteration",
    "iterations",
    "ppo_epochs",
    "minibatches_per_epoch",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "decimation",
    "terrain_type",
}
_GPU_ISOLATION_FIELDS = {
    "device_index",
    "device_uuid",
    "device_name",
    "driver_version",
    "total_memory_mib",
    "minimum_free_preflight_mib",
    "monitor_poll_seconds",
}


class RQ1TrainingError(ValueError):
    """Raised when an RQ1 plan, runtime record, or receipt fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RQ1TrainingError(message)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: set[str], name: str) -> None:
    _require(set(value) == fields, f"{name} fields are invalid")


def _require_sha256(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _positive_int(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{name} must be a positive integer",
    )
    return int(value)


def _nonnegative_int(value: Any, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return int(value)


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    _require(isinstance(value, str) and value, f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RQ1TrainingError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> Any:
    raise RQ1TrainingError(f"non-finite JSON constant: {token}")


def _finite_json_float(token: str) -> float:
    value = float(token)
    _require(math.isfinite(value), f"non-finite JSON number: {token}")
    return value


def _read_regular_bytes(path: Path, name: str) -> bytes:
    """Read one stable regular-file buffer without following a final symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RQ1TrainingError(f"cannot open {name}: {path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{name} must be a regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(before_identity == after_identity, f"{name} changed while being read")
    _require(len(payload) == before.st_size, f"{name} byte count changed while reading")
    return payload


def file_sha256(path: str | Path) -> str:
    candidate = Path(path).resolve()
    return hashlib.sha256(_read_regular_bytes(candidate, "file")).hexdigest()


def load_json_object(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).resolve()
    payload = _read_regular_bytes(candidate, "JSON artifact")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_duplicate_rejector,
            parse_constant=_reject_constant,
            parse_float=_finite_json_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RQ1TrainingError(f"invalid JSON artifact {candidate}: {error}") from error
    _require(isinstance(value, dict), f"JSON root must be an object: {candidate}")
    return value


def write_new_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically publish canonical JSON while refusing overwrite or resume."""

    declared = Path(path).expanduser()
    if os.path.lexists(declared):
        raise FileExistsError(f"refusing to overwrite existing artifact: {declared}")
    destination = declared.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing artifact: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)
    directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    payload = _read_regular_bytes(path, "bound source")
    record_path = str(path.relative_to(root)) if root is not None else str(path)
    return {
        "path": record_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _source_inventory(repo_root: Path) -> list[dict[str, Any]]:
    """Conservatively bind repository-owned Python and Hydra source bytes."""

    paths = set((repo_root / "gear_sonic").rglob("*.py"))
    paths.update((repo_root / "gear_sonic/config").rglob("*.yaml"))
    paths.update((repo_root / "gear_sonic/config").rglob("*.yml"))
    for candidate in (
        repo_root / "gear_sonic/pyproject.toml",
        repo_root / "pyproject.toml",
        repo_root / "scripts/research/run_lace_rq1_training.py",
    ):
        if candidate.is_file():
            paths.add(candidate)
    resolved = sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    _require(resolved, "RQ1 executable source inventory is empty")
    return [_file_record(path, root=repo_root) for path in resolved]


def _verify_source_inventory(repo_root: Path, records: Any) -> None:
    _require(isinstance(records, list) and records, "source inventory must be non-empty")
    expected_paths: list[str] = []
    for index, raw in enumerate(records):
        record = _require_mapping(raw, f"source_inventory[{index}]")
        _require_exact_fields(record, {"path", "bytes", "sha256"}, "source record")
        relative = record.get("path")
        _require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "source path must be canonical and repository-relative",
        )
        path = (repo_root / relative).resolve()
        _require(_is_within(path, repo_root), f"source path escapes repository: {relative}")
        actual = _file_record(path, root=repo_root)
        _require(actual == dict(record), f"bound source bytes drifted: {relative}")
        expected_paths.append(relative)
    _require(
        expected_paths == sorted(set(expected_paths)), "source inventory order/identity drifted"
    )
    _require(records == _source_inventory(repo_root), "source inventory closure drifted")


def _robot_asset_inventory(repo_root: Path) -> list[dict[str, Any]]:
    """Bind the conservative G1 robot-description closure used by Lite-S."""

    asset_root = repo_root / "gear_sonic/data/assets/robot_description"
    paths: set[Path] = {asset_root / "mjcf/g1_29dof_rev_1_0.xml"}
    for relative_root in ("meshes/g1", "urdf/g1"):
        root = asset_root / relative_root
        _require(root.is_dir(), f"G1 asset directory is missing: {root}")
        paths.update(path for path in root.rglob("*") if path.is_file())
    _require(all(path.is_file() for path in paths), "G1 robot-description closure is incomplete")
    resolved = sorted(
        (path.resolve() for path in paths),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )
    return [_file_record(path, root=repo_root) for path in resolved]


def _verify_robot_asset_inventory(repo_root: Path, records: Any) -> None:
    _require(isinstance(records, list) and records, "robot asset inventory must be non-empty")
    _require(
        records == _robot_asset_inventory(repo_root), "G1 robot-description asset bytes drifted"
    )


def _allowed_runtime_environment() -> dict[str, str]:
    """Return the exact non-secret environment inherited by the future child."""

    selected: dict[str, str] = {}
    for name, value in os.environ.items():
        if any(
            token in name.upper()
            for token in ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
        ):
            continue
        if name in _RUNTIME_ENV_NAMES or name.startswith(_RUNTIME_ENV_PREFIXES):
            selected[name] = value
    return selected


def _runtime_versions(python_executable: Path) -> dict[str, Any]:
    probe = """
import importlib.metadata
import json
import platform
import sys
packages = {}
for distribution in importlib.metadata.distributions():
    name = (distribution.metadata.get('Name') or '').strip().lower()
    if name:
        packages[name] = distribution.version
try:
    import torch
    torch_runtime = {
        'cuda': torch.version.cuda,
        'debug': torch.version.debug,
        'git_version': torch.version.git_version,
    }
except Exception as error:
    torch_runtime = {'import_error': type(error).__name__}
print(json.dumps({
    'executable': sys.executable,
    'implementation': platform.python_implementation(),
    'packages': dict(sorted(packages.items())),
    'platform': platform.platform(),
    'python': platform.python_version(),
    'torch_runtime': torch_runtime,
}, sort_keys=True))
"""
    try:
        result = subprocess.run(  # noqa: S603
            [str(python_executable), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RQ1TrainingError("could not resolve the exact training runtime") from error
    _require(
        isinstance(value, dict)
        and Path(str(value.get("executable"))).resolve() == python_executable,
        "runtime version probe used an unexpected interpreter",
    )
    return value


def _validate_protocol(protocol: Mapping[str, Any], *, protocol_path: Path) -> dict[str, Any]:
    _require_exact_fields(protocol, _PROTOCOL_FIELDS, "RQ1 protocol")
    _require(protocol.get("kind") == PROTOCOL_KIND, "RQ1 protocol kind mismatch")
    _require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA_VERSION,
        "RQ1 protocol schema version mismatch",
    )
    for field in ("frozen", "scientific_use", "declared_before_transfer_outcomes"):
        _require(protocol.get(field) is True, f"RQ1 protocol {field} must be true")
    digest = _require_sha256(protocol.get(PROTOCOL_DIGEST_FIELD), PROTOCOL_DIGEST_FIELD)
    _require(
        digest == canonical_sha256(protocol, digest_field=PROTOCOL_DIGEST_FIELD),
        "RQ1 protocol self digest mismatch",
    )
    run_id = protocol.get("run_id")
    arm_id = protocol.get("arm_id")
    _require(isinstance(run_id, str) and _SAFE_ID_RE.fullmatch(run_id), "run_id is unsafe")
    _require(isinstance(arm_id, str) and _SAFE_ID_RE.fullmatch(arm_id), "arm_id is unsafe")
    seed = _nonnegative_int(protocol.get("seed"), "seed")
    base = protocol_path.parent
    repo_root = _resolve_path(protocol.get("repo_root"), base=base, name="repo_root")
    storage_root = _resolve_path(protocol.get("storage_root"), base=base, name="storage_root")
    output_root = _resolve_path(protocol.get("output_root"), base=base, name="output_root")
    tmp_root = _resolve_path(protocol.get("tmp_root"), base=base, name="tmp_root")
    _require(repo_root.is_dir(), f"repo_root is not a directory: {repo_root}")
    _require(storage_root.is_dir(), f"storage_root is not a directory: {storage_root}")
    for name, path in (("output_root", output_root), ("tmp_root", tmp_root)):
        _require(_is_within(path, storage_root), f"{name} must be beneath storage_root")
        _require(path != storage_root, f"{name} may not equal storage_root")
    output_dir = output_root / run_id
    tmp_dir = tmp_root / run_id
    _require(output_dir != tmp_dir, "RQ1 output and temporary directories must differ")
    experiment = protocol.get("experiment")
    _require(experiment == EXPERIMENT, f"experiment must be {EXPERIMENT!r}")
    python_executable = _resolve_path(
        protocol.get("python_executable"), base=base, name="python_executable"
    )
    train_entrypoint = _resolve_path(
        protocol.get("train_entrypoint"), base=repo_root, name="train_entrypoint"
    )
    _require(python_executable.is_file(), "python_executable is missing")
    _require(train_entrypoint.is_file(), "train_entrypoint is missing")
    _require(_is_within(train_entrypoint, repo_root), "train_entrypoint must be in repo_root")

    fixed = _require_mapping(protocol.get("fixed_distribution"), "fixed_distribution")
    _require_exact_fields(fixed, _FIXED_FIELDS, "fixed_distribution")
    lock_path = _resolve_path(
        fixed.get("plan_lock_path"), base=base, name="fixed_distribution.plan_lock_path"
    )
    lock_sha = _require_sha256(
        fixed.get("plan_lock_file_sha256"), "fixed_distribution.plan_lock_file_sha256"
    )
    _require(file_sha256(lock_path) == lock_sha, "intervention-plan lock bytes drifted")

    dataset_config = _require_mapping(protocol.get("dataset"), "dataset")
    _require_exact_fields(dataset_config, _DATASET_FIELDS, "dataset")
    _require(dataset_config.get("partition") == PARTITION, "RQ1 partition must be D_curriculum")
    expected_motion_count = _positive_int(
        dataset_config.get("expected_motion_count"), "dataset.expected_motion_count"
    )
    split_path = _resolve_path(
        dataset_config.get("split_manifest"), base=base, name="dataset.split_manifest"
    )
    subset_root = _resolve_path(
        dataset_config.get("subset_root"), base=base, name="dataset.subset_root"
    )
    _require(_is_within(subset_root, storage_root), "dataset subset must be beneath storage_root")

    initialization = _require_mapping(protocol.get("initialization"), "initialization")
    _require_exact_fields(initialization, _INITIALIZATION_FIELDS, "initialization")
    _require(initialization.get("mode") == "model_only", "initialization mode must be model_only")
    _require(initialization.get("resume") is False, "RQ1 full resume is forbidden")
    checkpoint_path = _resolve_path(
        initialization.get("checkpoint_path"), base=base, name="checkpoint_path"
    )
    checkpoint_config_path = _resolve_path(
        initialization.get("checkpoint_config_path"), base=base, name="checkpoint_config_path"
    )
    checkpoint_sha = _require_sha256(
        initialization.get("checkpoint_sha256"), "initialization.checkpoint_sha256"
    )
    checkpoint_config_sha = _require_sha256(
        initialization.get("checkpoint_config_sha256"),
        "initialization.checkpoint_config_sha256",
    )
    _require(file_sha256(checkpoint_path) == checkpoint_sha, "initial checkpoint bytes drifted")
    _require(
        file_sha256(checkpoint_config_path) == checkpoint_config_sha,
        "initial checkpoint config bytes drifted",
    )
    _require(
        checkpoint_path.parent == checkpoint_config_path.parent,
        "checkpoint and companion config must share a directory",
    )
    _require(
        not _is_within(checkpoint_path, output_dir),
        "model-only initialization checkpoint may not be inside the fresh output directory",
    )

    training = _require_mapping(protocol.get("training"), "training")
    _require_exact_fields(training, _TRAINING_FIELDS, "training")
    values = {
        name: _positive_int(training.get(name), f"training.{name}")
        for name in (
            "num_envs_per_rank",
            "world_size",
            "rollout_steps_per_iteration",
            "iterations",
            "ppo_epochs",
            "minibatches_per_epoch",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "decimation",
        )
    }
    _require(values["world_size"] == 1, "single-5090 RQ1 runs require world_size=1")
    _require(training.get("terrain_type") == "plane", "RQ1 terrain_type must be plane")
    num_envs = values["num_envs_per_rank"]
    minibatches = values["minibatches_per_epoch"]
    _require(num_envs % minibatches == 0, "num_envs must divide exactly into minibatches")
    local_minibatch = num_envs // minibatches
    per_device = values["per_device_train_batch_size"]
    _require(
        local_minibatch % per_device == 0,
        "local minibatch must divide exactly into per-device microbatches",
    )
    num_microbatches = local_minibatch // per_device
    schedule_per_iteration = expected_optimizer_schedule(
        num_ppo_epochs=values["ppo_epochs"],
        num_mini_batches=minibatches,
        num_micro_batches=num_microbatches,
        gradient_accumulation_steps=values["gradient_accumulation_steps"],
    )
    gpu_isolation = _require_mapping(protocol.get("gpu_isolation"), "gpu_isolation")
    _require_exact_fields(gpu_isolation, _GPU_ISOLATION_FIELDS, "gpu_isolation")
    _require(gpu_isolation.get("device_index") == 0, "RQ1 GPU device_index must be zero")
    device_uuid = gpu_isolation.get("device_uuid")
    _require(
        isinstance(device_uuid, str) and re.fullmatch(r"GPU-[A-Za-z0-9-]+", device_uuid),
        "RQ1 GPU UUID is invalid",
    )
    _require(
        gpu_isolation.get("device_name") == "NVIDIA GeForce RTX 5090",
        "RQ1 GPU must be the frozen NVIDIA GeForce RTX 5090",
    )
    driver_version = gpu_isolation.get("driver_version")
    _require(
        isinstance(driver_version, str) and bool(driver_version),
        "RQ1 GPU driver version is missing",
    )
    total_memory_mib = _positive_int(
        gpu_isolation.get("total_memory_mib"),
        "gpu_isolation.total_memory_mib",
    )
    _require(total_memory_mib >= 32000, "RQ1 RTX 5090 total memory is unexpectedly small")
    minimum_free_mib = _positive_int(
        gpu_isolation.get("minimum_free_preflight_mib"),
        "gpu_isolation.minimum_free_preflight_mib",
    )
    _require(
        minimum_free_mib == 28672 and minimum_free_mib < total_memory_mib,
        "RQ1 minimum free-memory preflight must be exactly 28672 MiB",
    )
    poll_seconds = gpu_isolation.get("monitor_poll_seconds")
    _require(
        isinstance(poll_seconds, (int, float))
        and not isinstance(poll_seconds, bool)
        and float(poll_seconds) == 1.0,
        "RQ1 GPU monitor poll cadence must be exactly 1.0 second",
    )
    gpu_contract = {
        "device_index": 0,
        "device_uuid": device_uuid,
        "device_name": "NVIDIA GeForce RTX 5090",
        "driver_version": driver_version,
        "total_memory_mib": total_memory_mib,
        "minimum_free_preflight_mib": minimum_free_mib,
        "monitor_poll_seconds": 1.0,
        "authorized_process_scope": "child_process_group_only",
        "foreign_compute_process_limit": 0,
        "foreign_compute_memory_mib_limit": 0,
        "preflight_required": True,
        "continuous_samples_required": True,
        "postflight_required": True,
        "raw_sample_hash_chain_required": True,
    }
    gpu_contract["gpu_isolation_contract_sha256"] = canonical_sha256(gpu_contract)
    return {
        "protocol_path": protocol_path,
        "protocol_file_sha256": file_sha256(protocol_path),
        "protocol_sha256": digest,
        "run_id": run_id,
        "arm_id": arm_id,
        "seed": seed,
        "repo_root": repo_root,
        "storage_root": storage_root,
        "output_dir": output_dir,
        "tmp_dir": tmp_dir,
        "python_executable": python_executable,
        "python_executable_sha256": file_sha256(python_executable),
        "train_entrypoint": train_entrypoint,
        "experiment": experiment,
        "lock_path": lock_path,
        "lock_sha256": lock_sha,
        "split_path": split_path,
        "subset_root": subset_root,
        "expected_motion_count": expected_motion_count,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_config_path": checkpoint_config_path,
        "checkpoint_config_sha256": checkpoint_config_sha,
        "training": {**values, "terrain_type": "plane"},
        "local_minibatch_size": local_minibatch,
        "num_microbatches": num_microbatches,
        "schedule_per_iteration": schedule_per_iteration,
        "gpu_isolation_contract": gpu_contract,
    }


def _panel_contract(plan_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = plan_artifact.get("interventions")
    _require(isinstance(rows, list) and rows, "intervention plan panels are missing")
    panels: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        row = _require_mapping(raw, f"interventions[{index}]")
        panel_id = row.get("panel_id")
        motion_keys = row.get("panel_motion_keys")
        _require(
            isinstance(panel_id, str) and _SAFE_ID_RE.fullmatch(panel_id),
            "intervention panel_id is unsafe",
        )
        _require(panel_id not in seen, f"duplicate intervention panel: {panel_id}")
        seen.add(panel_id)
        _require(
            isinstance(motion_keys, list)
            and motion_keys
            and motion_keys == sorted(set(motion_keys)),
            f"{panel_id} motion keys must be sorted and unique",
        )
        panels.append({"panel_id": panel_id, "motion_keys": list(motion_keys)})
    _require(
        [panel["panel_id"] for panel in panels] == sorted(seen),
        "intervention panels must be canonically ordered",
    )
    return panels


def _training_semantics(
    *,
    values: Mapping[str, Any],
    dataset: Mapping[str, Any],
    fixed_binding: Mapping[str, Any],
) -> dict[str, Any]:
    training = values["training"]
    attempts_per_iteration = values["schedule_per_iteration"]["optimizer_expected_attempts"]
    updates_per_iteration = values["schedule_per_iteration"]["synchronized_expected_updates"]
    iterations = int(training["iterations"])
    num_envs = int(training["num_envs_per_rank"])
    rollout_steps = int(training["rollout_steps_per_iteration"])
    decimation = int(training["decimation"])
    return {
        "experiment": values["experiment"],
        "seed": values["seed"],
        "dataset_subset_sha256": dataset["subset_sha256"],
        "resident_motion_count": dataset["motion_count"],
        "resident_motion_order_sha256": dataset["resident_motion_order_sha256"],
        "resident_motion_set_sha256": dataset["resident_motion_set_sha256"],
        "model_only_initial_checkpoint_sha256": values["checkpoint_sha256"],
        "model_only_initial_checkpoint_config_sha256": values["checkpoint_config_sha256"],
        INTERVENTION_PLAN_DIGEST_FIELD: fixed_binding[INTERVENTION_PLAN_DIGEST_FIELD],
        "plan_lock_file_sha256": fixed_binding["plan_lock_file_sha256"],
        "num_envs_per_rank": num_envs,
        "world_size": int(training["world_size"]),
        "rollout_steps_per_iteration": rollout_steps,
        "iterations": iterations,
        "ppo_epochs": int(training["ppo_epochs"]),
        "minibatches_per_epoch": int(training["minibatches_per_epoch"]),
        "local_minibatch_size": values["local_minibatch_size"],
        "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
        "num_microbatches_per_minibatch": values["num_microbatches"],
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "decimation": decimation,
        "terrain_type": training["terrain_type"],
        "planned_control_transitions": num_envs * rollout_steps * iterations,
        "planned_physics_substeps": num_envs * rollout_steps * iterations * decimation,
        "physics_substeps_semantics": (
            "derived_control_transitions_times_live_verified_decimation_not_directly_observed"
        ),
        "planned_optimizer_opportunities": attempts_per_iteration * iterations,
        "planned_synchronized_parameter_updates": updates_per_iteration * iterations,
        "sampler_modes": {
            "adaptive_sampling": False,
            "all_motions_loaded": True,
            "motion_order": "lexicographic_motion_key",
            "use_paired_motions": False,
            "sample_unique_motions": False,
            "is_evaluating": False,
            "atlas_probe_mode": False,
            "multi_object_mode": False,
            "im_resample_callback": False,
        },
        "initialization_mode": "model_only_resume_false",
    }


def _repo_first_pythonpath(repo_root: Path) -> str:
    entries = [str(repo_root)]
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    return os.pathsep.join(entries)


def _launch_environment(values: Mapping[str, Any]) -> dict[str, str]:
    tmp_dir = values["tmp_dir"]
    environment = _allowed_runtime_environment()
    environment.update(
        {
            "CUDA_CACHE_PATH": str(tmp_dir / "cuda-cache"),
            "CUDA_VISIBLE_DEVICES": "0",
            "HYDRA_FULL_ERROR": "1",
            "ISAACLAB_USD_CACHE_DIR": str(tmp_dir / "isaaclab-usd-cache"),
            "LOGURU_LEVEL": "INFO",
            "PYTHONPATH": _repo_first_pythonpath(values["repo_root"]),
            "PYTHONHASHSEED": str(values["seed"]),
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(tmp_dir),
            "TORCH_EXTENSIONS_DIR": str(tmp_dir / "torch-extensions"),
            "WANDB_MODE": "disabled",
            "XDG_CACHE_HOME": str(tmp_dir / "xdg-cache"),
        }
    )
    return dict(sorted(environment.items()))


def _matched_launch_environment_contract(
    environment: Mapping[str, str],
    values: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve scientific env settings while removing per-run path names."""

    replacements = {
        str(values["tmp_dir"]): "${RQ1_TMP_DIR}",
        str(values["output_dir"]): "${RQ1_OUTPUT_DIR}",
        str(values["repo_root"]): "${RQ1_REPO_ROOT}",
        str(values["storage_root"]): "${RQ1_STORAGE_ROOT}",
    }
    ordered_replacements = sorted(replacements.items(), key=lambda row: len(row[0]), reverse=True)
    normalized: dict[str, str] = {}
    for name, raw_value in sorted(environment.items()):
        value = raw_value
        for path, token in ordered_replacements:
            value = value.replace(path, token)
        normalized[name] = value
    contract: dict[str, Any] = {
        "normalization": "exact_effective_environment_with_run_paths_tokenized_v1",
        "path_tokens": sorted(set(replacements.values())),
        "variables": normalized,
    }
    contract["matched_launch_environment_sha256"] = canonical_sha256(contract)
    return contract


def _launch_command(
    *,
    values: Mapping[str, Any],
    dataset: Mapping[str, Any],
    plan_path: Path,
    plan_sha256: str,
) -> tuple[str, dict[str, str], list[str]]:
    training = values["training"]
    output_dir = values["output_dir"]
    lock_path = values["lock_path"]
    arm_id = values["arm_id"]
    environment = _launch_environment(values)
    overrides = [
        f"+exp={values['experiment']}",
        "+callbacks=rq1_training",
        "~callbacks.im_resample",
        "~callbacks.read_eval",
        f"checkpoint={values['checkpoint_path']}",
        "+resume=false",
        "auto_load_latest=false",
        f"num_envs={training['num_envs_per_rank']}",
        "headless=true",
        "use_wandb=false",
        f"seed={values['seed']}",
        f"exp_var=rq1_{arm_id}",
        f"experiment_dir={output_dir}",
        f"++manager_env.config.decimation={training['decimation']}",
        f"++manager_env.config.terrain_type={training['terrain_type']}",
        "++manager_env.config.multi_object_per_env=false",
        f"++algo.config.num_learning_iterations={training['iterations']}",
        f"++algo.config.num_steps_per_env={training['rollout_steps_per_iteration']}",
        f"++algo.config.num_learning_epochs={training['ppo_epochs']}",
        f"++algo.config.num_mini_batches={training['minibatches_per_epoch']}",
        "++algo.config.rq1_training_metrics=true",
        f"++algo.trl.gradient_accumulation_steps={training['gradient_accumulation_steps']}",
        f"++algo.trl.per_device_train_batch_size={training['per_device_train_batch_size']}",
        "++algo.config.save_interval=1000000000",
        "++callbacks.model_save.save_frequency=1000000000",
        f"++callbacks.model_save.save_last_frequency={training['iterations']}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={dataset['robot_dir']}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={dataset['smpl_dir']}",
        f"++manager_env.commands.motion.motion_lib_cfg.override_num_motions_to_load={dataset['motion_count']}",
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        "++manager_env.commands.motion.use_paired_motions=false",
        "++manager_env.commands.motion.sample_unique_motions=false",
        "++manager_env.commands.motion.atlas_probe_mode=false",
        "++manager_env.commands.motion.atlas_probe_assignments=null",
        "++manager_env.commands.motion.atlas_probe_schedule_sha256=null",
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution.enable=true",
        f"++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution.plan_lock_path={lock_path}",
        "++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution."
        f"plan_lock_file_sha256={values['lock_sha256']}",
        f"++manager_env.commands.motion.motion_lib_cfg.lace_fixed_distribution.distribution_id={arm_id}",
        f"++rq1_training.plan_path={plan_path}",
        f"++rq1_training.plan_sha256={plan_sha256}",
        f"++rq1_training.runtime_metrics_path={output_dir / RUNTIME_METRICS_FILENAME}",
    ]
    return (
        str(values["repo_root"]),
        environment,
        [
            str(values["python_executable"]),
            str(values["train_entrypoint"]),
            *overrides,
        ],
    )


def _compose_expected_saved_config(
    *,
    repo_root: Path,
    overrides: Sequence[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Reproduce the exact unresolved snapshot saved by ``train_agent_trl``."""

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RQ1TrainingError("Hydra/OmegaConf are required for RQ1 planning") from error
    with initialize_config_dir(config_dir=str(repo_root / "gear_sonic/config"), version_base="1.1"):
        config = compose(config_name="base", overrides=list(overrides))
    # These are the only deterministic mutations before train_agent_trl.py
    # captures ``unresolved_conf``.  world_size=1 fixes multi_gpu to false and
    # therefore prevents the conditional seed/global-rank mutations.
    config.algo.trl.output_dir = str(output_dir)
    config.multi_gpu = False
    container = OmegaConf.to_container(config, resolve=False)
    _require(isinstance(container, dict), "expected saved Hydra config is not an object")
    return container


def _build_rq1_training_plan(
    protocol: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    plan_path: str | Path,
    require_pristine: bool,
) -> dict[str, Any]:
    """Deep-resolve one external intervention lock into one fresh run plan."""

    protocol_file = Path(protocol_path).resolve()
    values = _validate_protocol(protocol, protocol_path=protocol_file)
    intended_plan_path = Path(plan_path).resolve()
    _require(
        _is_within(intended_plan_path, values["storage_root"]),
        "RQ1 plan must be stored beneath storage_root",
    )
    if require_pristine:
        for name, path in (
            ("output directory", values["output_dir"]),
            ("temporary directory", values["tmp_dir"]),
            ("plan target", intended_plan_path),
        ):
            _require(
                not os.path.lexists(path),
                f"RQ1 {name} must not exist before planning: {path}",
            )
    dataset = verify_partition_subset(
        values["subset_root"],
        split_manifest_path=values["split_path"],
        partition=PARTITION,
    )
    _require(
        dataset["motion_count"] == values["expected_motion_count"],
        "verified D_curriculum motion count differs from the protocol",
    )
    fixed_config = {
        "enable": True,
        "plan_lock_path": str(values["lock_path"]),
        "plan_lock_file_sha256": values["lock_sha256"],
        "distribution_id": values["arm_id"],
    }
    probabilities, binding = resolve_fixed_distribution(
        fixed_config,
        loaded_motion_keys=dataset["resident_motion_keys"],
    )
    _require(
        binding["motion_count"] == dataset["motion_count"],
        "intervention plan and D_curriculum subset motion counts differ",
    )
    _require(
        Path(dataset["split_manifest"]).resolve() == Path(binding["split_manifest_path"]).resolve(),
        "training split path differs from the intervention-plan lock",
    )
    _require(
        file_sha256(dataset["split_manifest"]) == binding["split_manifest_file_sha256"],
        "training split bytes differ from the intervention-plan lock",
    )
    subset_manifest = load_json_object(dataset["manifest"])
    _require(
        subset_manifest.get("split_sha256") == binding["split_sha256"]
        and subset_manifest.get("selection_sha256") == binding["split_selection_sha256"],
        "training split identities differ from the intervention-plan lock",
    )
    plan_artifact = load_json_object(binding["plan_path"])
    panels = _panel_contract(plan_artifact)
    all_panel_keys = [key for panel in panels for key in panel["motion_keys"]]
    _require(
        sorted(all_panel_keys) == binding["motion_keys"]
        and len(all_panel_keys) == len(set(all_panel_keys)),
        "source panels must exactly partition the resident motion universe",
    )
    valid_arm_ids = [BASE_ARM_ID, *[panel["panel_id"] for panel in panels]]
    _require(values["arm_id"] in valid_arm_ids, "protocol selects an unknown RQ1 arm")
    _require(
        valid_arm_ids.count(values["arm_id"]) == 1,
        "protocol must select exactly one unique RQ1 arm",
    )
    source_inventory = _source_inventory(values["repo_root"])
    source_inventory_sha256 = canonical_sha256({"sources": source_inventory})
    robot_asset_inventory = _robot_asset_inventory(values["repo_root"])
    robot_asset_inventory_sha256 = canonical_sha256({"assets": robot_asset_inventory})
    runtime_versions = _runtime_versions(values["python_executable"])
    runtime_versions["executable_sha256"] = values["python_executable_sha256"]
    training_semantics = _training_semantics(values=values, dataset=dataset, fixed_binding=binding)
    launch_environment = _launch_environment(values)
    matched_launch_environment = _matched_launch_environment_contract(
        launch_environment,
        values,
    )
    matched_training_config_sha256 = canonical_sha256(
        {
            "training_semantics": training_semantics,
            "dataset": {
                "subset_sha256": dataset["subset_sha256"],
                "manifest_file_sha256": dataset["manifest_file_sha256"],
                "split_manifest_file_sha256": binding["split_manifest_file_sha256"],
                "split_sha256": binding["split_sha256"],
                "split_selection_sha256": binding["split_selection_sha256"],
            },
            "initialization": {
                "checkpoint_sha256": values["checkpoint_sha256"],
                "checkpoint_config_sha256": values["checkpoint_config_sha256"],
            },
            "source_inventory_sha256": source_inventory_sha256,
            "robot_asset_inventory_sha256": robot_asset_inventory_sha256,
            "runtime_versions": runtime_versions,
            "gpu_isolation_contract": values["gpu_isolation_contract"],
            "matched_launch_environment": matched_launch_environment,
        }
    )
    cell_digest = canonical_sha256(
        {
            "experiment": values["experiment"],
            "arm_id": values["arm_id"],
            "seed": values["seed"],
            "matched_training_config_sha256": matched_training_config_sha256,
            "plan_lock_file_sha256": binding["plan_lock_file_sha256"],
            INTERVENTION_PLAN_DIGEST_FIELD: binding[INTERVENTION_PLAN_DIGEST_FIELD],
            "distribution_sha256": binding["distribution_sha256"],
            "motion_order_sha256": binding["motion_order_sha256"],
        }
    )
    cell_id = f"rq1-{cell_digest[:32]}"
    attempt_claim_path = values["storage_root"] / "rq1_attempt_claims" / f"{cell_id}.json"
    if require_pristine:
        _require(
            not os.path.lexists(attempt_claim_path),
            f"RQ1 attempt claim must not exist before planning: {attempt_claim_path}",
        )
    arm_contract = {
        "arm_id": values["arm_id"],
        "distribution_kind": binding["distribution_kind"],
        "distribution_sha256": binding["distribution_sha256"],
        "fixed_distribution_config_sha256": binding["fixed_distribution_config_sha256"],
        "accounting_scope": binding["accounting_scope"],
        "probabilities": list(probabilities),
        "base_probabilities": list(plan_artifact["base_probabilities"]),
        "motion_keys": list(binding["motion_keys"]),
        "motion_count": int(binding["motion_count"]),
        "motion_order_sha256": binding["motion_order_sha256"],
        "panels": panels,
        FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD: binding[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD],
        INTERVENTION_PLAN_DIGEST_FIELD: binding[INTERVENTION_PLAN_DIGEST_FIELD],
        "plan_lock_path": binding["plan_lock_path"],
        "plan_lock_file_sha256": binding["plan_lock_file_sha256"],
        "split_manifest_path": binding["split_manifest_path"],
        "split_manifest_file_sha256": binding["split_manifest_file_sha256"],
        "split_sha256": binding["split_sha256"],
        "split_selection_sha256": binding["split_selection_sha256"],
    }
    base_plan: dict[str, Any] = {
        "kind": PLAN_KIND,
        "schema_version": PLAN_SCHEMA_VERSION,
        "frozen": True,
        "scientific_use": True,
        "declared_before_transfer_outcomes": True,
        "run_id": values["run_id"],
        "cell_id": cell_id,
        "arm_id": values["arm_id"],
        "protocol": {
            "path": str(protocol_file),
            "file_sha256": values["protocol_file_sha256"],
            PROTOCOL_DIGEST_FIELD: values["protocol_sha256"],
        },
        "plan_path": str(intended_plan_path),
        "repo_root": str(values["repo_root"]),
        "storage_root": str(values["storage_root"]),
        "output_dir": str(values["output_dir"]),
        "tmp_dir": str(values["tmp_dir"]),
        "dataset": deepcopy(dataset),
        "initialization": {
            "mode": "model_only",
            "resume": False,
            "checkpoint_path": str(values["checkpoint_path"]),
            "checkpoint_bytes": values["checkpoint_path"].stat().st_size,
            "checkpoint_sha256": values["checkpoint_sha256"],
            "checkpoint_config_path": str(values["checkpoint_config_path"]),
            "checkpoint_config_bytes": values["checkpoint_config_path"].stat().st_size,
            "checkpoint_config_sha256": values["checkpoint_config_sha256"],
        },
        "arm_contract": arm_contract,
        "training_semantics": training_semantics,
        "matched_training_config_sha256": matched_training_config_sha256,
        "source_inventory": source_inventory,
        "source_inventory_sha256": source_inventory_sha256,
        "robot_asset_inventory": robot_asset_inventory,
        "robot_asset_inventory_sha256": robot_asset_inventory_sha256,
        "runtime_versions": runtime_versions,
        "matched_launch_environment": matched_launch_environment,
        "gpu_isolation_contract": deepcopy(values["gpu_isolation_contract"]),
        "runtime_artifacts": {
            "attempt_claim": str(attempt_claim_path),
            "resolved_config": str(values["output_dir"] / "config.yaml"),
            "final_checkpoint": str(values["output_dir"] / "last.pt"),
            "hydra_log": str(values["output_dir"] / ".hydra/train.log"),
            "process_log": str(values["output_dir"] / "process.log"),
            "runtime_metrics": str(values["output_dir"] / RUNTIME_METRICS_FILENAME),
            "gpu_isolation_monitor": str(values["output_dir"] / "gpu_isolation_monitor.json"),
            "execution_completion": str(values["output_dir"] / "execution_completion.json"),
            "receipt": str(values["output_dir"] / RECEIPT_FILENAME),
        },
        "launch_policy": {
            "default": "plan_and_cpu_dry_compose_only",
            "scientific_launch_ready": False,
            "path_independent_cell_identity": True,
            "one_process_one_arm": True,
            "resume": "forbidden",
            "overwrite": "forbidden",
            "fresh_output_directory": True,
            "local_attempt_claim_scope": (
                "advisory_atomic_no_clobber_within_declared_storage_root_only"
            ),
            "canonical_attempt_authority": ("absent_external_append_only_registry_required"),
            "external_gpu_exclusivity_preflight_required": True,
            "external_append_only_attempt_registry_required_for_scientific_use": True,
            "child_environment": "exact_sanitized_allowlist_no_unrecorded_inheritance",
            "canonical_working_directory": str(values["repo_root"]),
        },
    }
    # The command binds the canonical plan identity.  It intentionally cannot
    # contain the plan file hash, which would create a self-referential digest;
    # the completed receipt binds the exact plan bytes instead.
    provisional_sha = canonical_sha256(base_plan)
    cwd, environment, argv = _launch_command(
        values=values,
        dataset=dataset,
        plan_path=intended_plan_path,
        plan_sha256=provisional_sha,
    )
    _require(
        environment == launch_environment,
        "effective launch environment changed during RQ1 planning",
    )
    base_plan["launch_cwd"] = cwd
    base_plan["launch_environment"] = environment
    base_plan["argv"] = argv
    base_plan["command_sha256"] = canonical_sha256(
        {"cwd": cwd, "environment": environment, "argv": argv}
    )
    expected_saved_config = _compose_expected_saved_config(
        repo_root=values["repo_root"],
        overrides=argv[2:],
        output_dir=values["output_dir"],
    )
    base_plan["expected_saved_config"] = expected_saved_config
    base_plan["expected_saved_config_sha256"] = canonical_sha256(expected_saved_config)
    # Replacing the provisional identity in argv with the final identity is a
    # circular construction.  Freeze a launch-contract digest that excludes
    # argv, then self-hash the complete artifact.  The callback verifies both.
    base_plan["launch_contract_sha256"] = provisional_sha
    base_plan[PLAN_DIGEST_FIELD] = canonical_sha256(base_plan, digest_field=PLAN_DIGEST_FIELD)
    return base_plan


def build_rq1_training_plan(
    protocol: Mapping[str, Any],
    *,
    protocol_path: str | Path,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Create one plan only when every local attempt target is pristine."""

    return _build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=plan_path,
        require_pristine=True,
    )


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    _require(plan.get("kind") == PLAN_KIND, "RQ1 plan kind mismatch")
    _require(plan.get("schema_version") == PLAN_SCHEMA_VERSION, "RQ1 plan schema mismatch")
    _require(plan.get("frozen") is True, "RQ1 plan must be frozen")
    _require(plan.get("scientific_use") is True, "RQ1 plan must be scientific")
    _require(
        plan.get("declared_before_transfer_outcomes") is True,
        "RQ1 plan must predate transfer outcomes",
    )
    digest = _require_sha256(plan.get(PLAN_DIGEST_FIELD), PLAN_DIGEST_FIELD)
    _require(
        digest == canonical_sha256(plan, digest_field=PLAN_DIGEST_FIELD),
        "RQ1 plan self digest mismatch",
    )
    command = {
        "cwd": plan.get("launch_cwd"),
        "environment": plan.get("launch_environment"),
        "argv": plan.get("argv"),
    }
    _require(plan.get("command_sha256") == canonical_sha256(command), "command digest mismatch")
    expected_config = _require_mapping(plan.get("expected_saved_config"), "expected_saved_config")
    _require(
        plan.get("expected_saved_config_sha256") == canonical_sha256(expected_config),
        "expected saved-config digest mismatch",
    )


def validate_rq1_training_plan(plan: Mapping[str, Any], *, verify_sources: bool = True) -> None:
    """Deep-rebuild a plan from its frozen protocol and compare every field."""

    _validate_plan_shape(plan)
    plan_path = Path(str(plan.get("plan_path"))).resolve()
    protocol_record = _require_mapping(plan.get("protocol"), "plan.protocol")
    protocol_path = Path(str(protocol_record.get("path"))).resolve()
    _require(
        file_sha256(protocol_path) == protocol_record.get("file_sha256"),
        "RQ1 protocol file bytes drifted",
    )
    protocol = load_json_object(protocol_path)
    _require(
        protocol.get(PROTOCOL_DIGEST_FIELD) == protocol_record.get(PROTOCOL_DIGEST_FIELD),
        "RQ1 protocol identity drifted",
    )
    rebuilt = _build_rq1_training_plan(
        protocol,
        protocol_path=protocol_path,
        plan_path=plan_path,
        require_pristine=False,
    )
    _require(dict(plan) == rebuilt, "RQ1 plan differs from deterministic deep reconstruction")
    if verify_sources:
        repo_root = Path(str(plan.get("repo_root"))).resolve()
        _verify_source_inventory(repo_root, plan.get("source_inventory"))
        _verify_robot_asset_inventory(repo_root, plan.get("robot_asset_inventory"))


def _hydra_overrides(plan: Mapping[str, Any]) -> list[str]:
    argv = plan.get("argv")
    _require(isinstance(argv, list) and len(argv) >= 3, "RQ1 argv is invalid")
    _require(all(isinstance(value, str) for value in argv), "RQ1 argv entries must be strings")
    return list(argv[2:])


def _audit_rq1_config_container(
    container: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit executable fields in either a dry-composed or saved config."""

    expected_container = _require_mapping(
        plan.get("expected_saved_config"),
        "plan.expected_saved_config",
    )
    _require(
        canonical_sha256(expected_container) == plan.get("expected_saved_config_sha256"),
        "plan expected saved-config identity drifted",
    )
    _require(
        dict(container) == dict(expected_container),
        "saved RQ1 config differs from the exact composed config",
    )

    callbacks = _require_mapping(container.get("callbacks"), "callbacks")
    _require("rq1_training" in callbacks, "RQ1 accounting callback is absent")
    _require("im_resample" not in callbacks, "im_resample callback must be absent")
    _require("read_eval" not in callbacks, "read_eval callback must be absent")
    _require(
        sum(
            isinstance(value, Mapping)
            and value.get("_target_")
            == "gear_sonic.trl.callbacks.rq1_training_callback.RQ1TrainingCallback"
            for value in callbacks.values()
        )
        == 1,
        "resolved config must contain exactly one RQ1 callback",
    )
    manager_env = _require_mapping(container.get("manager_env"), "manager_env")
    manager_config = _require_mapping(manager_env.get("config"), "manager_env.config")
    command = _require_mapping(
        _require_mapping(manager_env.get("commands"), "manager_env.commands").get("motion"),
        "manager_env.commands.motion",
    )
    motion_lib = _require_mapping(command.get("motion_lib_cfg"), "motion_lib_cfg")
    fixed = _require_mapping(motion_lib.get("lace_fixed_distribution"), "lace_fixed_distribution")
    algo = _require_mapping(container.get("algo"), "algo")
    algo_config = _require_mapping(algo.get("config"), "algo.config")
    trl = _require_mapping(algo.get("trl"), "algo.trl")
    callback_inputs = _require_mapping(container.get("rq1_training"), "rq1_training")
    arm = _require_mapping(plan.get("arm_contract"), "arm_contract")
    semantics = _require_mapping(plan.get("training_semantics"), "training_semantics")
    dataset = _require_mapping(plan.get("dataset"), "dataset")

    expected = {
        "adaptive_sampling_disabled": _require_mapping(
            motion_lib.get("adaptive_sampling"), "adaptive_sampling"
        ).get("enable")
        is False,
        "all_motion_count": motion_lib.get("override_num_motions_to_load")
        == arm.get("motion_count"),
        "sort_motion_keys": motion_lib.get("sort_motion_keys") is True,
        "fixed_distribution_enabled": fixed.get("enable") is True,
        "one_arm": fixed.get("distribution_id") == plan.get("arm_id"),
        "plan_lock_path": Path(str(fixed.get("plan_lock_path"))).resolve()
        == Path(str(arm.get("plan_lock_path"))).resolve(),
        "plan_lock_sha256": fixed.get("plan_lock_file_sha256") == arm.get("plan_lock_file_sha256"),
        "use_paired_motions": command.get("use_paired_motions") is False,
        "sample_unique_motions": command.get("sample_unique_motions") is False,
        "atlas_probe_mode": command.get("atlas_probe_mode") is False,
        "atlas_assignments_absent": command.get("atlas_probe_assignments") is None,
        "atlas_schedule_absent": command.get("atlas_probe_schedule_sha256") is None,
        "resume_false": container.get("resume") is False,
        "auto_load_latest_false": container.get("auto_load_latest") is False,
        "rq1_metrics_enabled": algo_config.get("rq1_training_metrics") is True,
        "multi_object_disabled": manager_config.get("multi_object_per_env") is False,
        "num_envs": container.get("num_envs") == semantics.get("num_envs_per_rank"),
        "iterations": algo_config.get("num_learning_iterations") == semantics.get("iterations"),
        "rollout_steps": algo_config.get("num_steps_per_env")
        == semantics.get("rollout_steps_per_iteration"),
        "ppo_epochs": algo_config.get("num_learning_epochs") == semantics.get("ppo_epochs"),
        "minibatches": algo_config.get("num_mini_batches")
        == semantics.get("minibatches_per_epoch"),
        "per_device_batch": trl.get("per_device_train_batch_size")
        == semantics.get("per_device_train_batch_size"),
        "gradient_accumulation": trl.get("gradient_accumulation_steps")
        == semantics.get("gradient_accumulation_steps"),
        "decimation": manager_config.get("decimation") == semantics.get("decimation"),
        "terrain": manager_config.get("terrain_type") == semantics.get("terrain_type"),
        "robot_dataset": Path(str(motion_lib.get("motion_file"))).resolve()
        == Path(str(dataset.get("robot_dir"))).resolve(),
        "smpl_dataset": Path(str(motion_lib.get("smpl_motion_file"))).resolve()
        == Path(str(dataset.get("smpl_dir"))).resolve(),
        "checkpoint": Path(str(container.get("checkpoint"))).resolve()
        == Path(str(plan["initialization"]["checkpoint_path"])).resolve(),
        "experiment_dir": Path(str(container.get("experiment_dir"))).resolve()
        == Path(str(plan["output_dir"])).resolve(),
        "callback_plan_path": Path(str(callback_inputs.get("plan_path"))).resolve()
        == Path(str(plan["plan_path"])).resolve(),
        "callback_launch_contract": callback_inputs.get("plan_sha256")
        == plan.get("launch_contract_sha256"),
        "callback_metrics_path": Path(str(callback_inputs.get("runtime_metrics_path"))).resolve()
        == Path(str(plan["runtime_artifacts"]["runtime_metrics"])).resolve(),
    }
    failed = [name for name, passed in expected.items() if not passed]
    _require(not failed, "resolved RQ1 config violates mode contract: " + ", ".join(failed))
    readback = {
        "arm_id": fixed.get("distribution_id"),
        "callback_names": sorted(callbacks),
        "motion_count": motion_lib.get("override_num_motions_to_load"),
        "motion_file": motion_lib.get("motion_file"),
        "smpl_motion_file": motion_lib.get("smpl_motion_file"),
        "num_envs": container.get("num_envs"),
        "resume": container.get("resume"),
        "checkpoint": container.get("checkpoint"),
        "experiment_dir": container.get("experiment_dir"),
        "training_modes": expected,
    }
    readback["readback_sha256"] = canonical_sha256(readback)
    return readback


def dry_compose_rq1_training_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Compose Hydra on CPU and assert every sampler/update mode is executable."""

    validate_rq1_training_plan(plan)
    repo_root = Path(str(plan["repo_root"]))
    container = _compose_expected_saved_config(
        repo_root=repo_root,
        overrides=_hydra_overrides(plan),
        output_dir=Path(str(plan["output_dir"])),
    )
    return _audit_rq1_config_container(container, plan)


def audit_saved_rq1_training_config(
    path: str | Path,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the exact config emitted by ``train_agent_trl.py``."""

    try:
        from omegaconf import OmegaConf
    except ImportError as error:  # pragma: no cover - dependency gate
        raise RQ1TrainingError("OmegaConf is required to audit the saved RQ1 config") from error
    try:
        config = OmegaConf.load(Path(path).resolve())
        container = OmegaConf.to_container(config, resolve=False)
    except (OSError, ValueError) as error:
        raise RQ1TrainingError("could not parse the saved RQ1 training config") from error
    _require(isinstance(container, dict), "saved RQ1 config is not an object")
    return _audit_rq1_config_container(container, plan)


def _require_utc(value: Any, name: str) -> str:
    _require(
        isinstance(value, str) and _UTC_RE.fullmatch(value) is not None,
        f"{name} must be an RFC3339 UTC timestamp",
    )
    return value


def _utc_unix_ns(value: Any, name: str) -> int:
    timestamp = _require_utc(value, name)
    match = _UTC_RE.fullmatch(timestamp)
    assert match is not None  # established by _require_utc
    try:
        parsed = datetime.strptime(match.group("seconds"), "%Y-%m-%dT%H:%M:%S")
    except ValueError as error:
        raise RQ1TrainingError(f"{name} is not a valid calendar timestamp") from error
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return calendar.timegm(parsed.timetuple()) * 1_000_000_000 + int(fraction or "0")


def validate_attempt_claim(
    claim: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    require_files: bool = True,
) -> None:
    """Validate the external launcher's atomic, local one-attempt tombstone."""

    fields = {
        "kind",
        "schema_version",
        "frozen",
        "cell_id",
        "attempt_id",
        "plan_path",
        "plan_file_sha256",
        PLAN_DIGEST_FIELD,
        "command_sha256",
        "cwd",
        "argv",
        "environment",
        "launcher_pid",
        "launcher_process_group_id",
        "preflight_completed_at_unix_ns",
        "deep_plan_reconstruction_pass",
        "dataset_bytes_preflight_pass",
        "source_asset_bytes_preflight_pass",
        "claimed_at_utc",
        "claimed_at_unix_ns",
        "authority_scope",
        ATTEMPT_CLAIM_DIGEST_FIELD,
    }
    _require_exact_fields(claim, fields, "RQ1 attempt claim")
    _require(claim.get("kind") == ATTEMPT_CLAIM_KIND, "attempt claim kind mismatch")
    _require(
        claim.get("schema_version") == ATTEMPT_CLAIM_SCHEMA_VERSION,
        "attempt claim schema mismatch",
    )
    _require(claim.get("frozen") is True, "attempt claim must be frozen")
    _require(claim.get("cell_id") == plan.get("cell_id"), "attempt claim cell mismatch")
    attempt_id = claim.get("attempt_id")
    _require(
        isinstance(attempt_id, str) and _SAFE_ID_RE.fullmatch(attempt_id) is not None,
        "attempt_id is unsafe",
    )
    runtime_artifacts = plan.get("runtime_artifacts")
    declared_claim_path = (
        Path(str(runtime_artifacts["attempt_claim"]))
        if isinstance(runtime_artifacts, Mapping)
        else None
    )
    claim_path = declared_claim_path.resolve() if declared_claim_path is not None else None
    _require(
        Path(str(claim.get("plan_path"))).resolve() == Path(str(plan["plan_path"])).resolve(),
        "attempt claim plan path mismatch",
    )
    _require_sha256(claim.get("plan_file_sha256"), "attempt claim plan_file_sha256")
    if require_files:
        _require(
            claim.get("plan_file_sha256") == file_sha256(plan["plan_path"]),
            "attempt claim plan-file bytes mismatch",
        )
    _require(
        claim.get(PLAN_DIGEST_FIELD) == plan.get(PLAN_DIGEST_FIELD), "attempt claim plan mismatch"
    )
    _require(
        claim.get("command_sha256") == plan.get("command_sha256"),
        "attempt claim command mismatch",
    )
    _require(claim.get("cwd") == plan.get("launch_cwd"), "attempt claim cwd mismatch")
    _require(claim.get("argv") == plan.get("argv"), "attempt claim argv mismatch")
    _require(
        claim.get("environment") == plan.get("launch_environment"),
        "attempt claim effective environment mismatch",
    )
    _positive_int(claim.get("launcher_pid"), "attempt claim launcher_pid")
    _positive_int(
        claim.get("launcher_process_group_id"),
        "attempt claim launcher_process_group_id",
    )
    preflight_at = _positive_int(
        claim.get("preflight_completed_at_unix_ns"),
        "attempt claim preflight_completed_at_unix_ns",
    )
    for field in (
        "deep_plan_reconstruction_pass",
        "dataset_bytes_preflight_pass",
        "source_asset_bytes_preflight_pass",
    ):
        _require(claim.get(field) is True, f"attempt claim reports failed {field}")
    claimed_at = _positive_int(claim.get("claimed_at_unix_ns"), "attempt claim claimed_at_unix_ns")
    _require(
        _utc_unix_ns(claim.get("claimed_at_utc"), "attempt claim claimed_at_utc") == claimed_at,
        "attempt claim UTC and unix-nanosecond timestamps differ",
    )
    _require(preflight_at <= claimed_at, "attempt claim predates its deep preflight")
    _require(
        claim.get("authority_scope") == "local_atomic_no_clobber_not_external_worm",
        "attempt claim overstates its authority",
    )
    digest = _require_sha256(claim.get(ATTEMPT_CLAIM_DIGEST_FIELD), ATTEMPT_CLAIM_DIGEST_FIELD)
    _require(
        digest == canonical_sha256(claim, digest_field=ATTEMPT_CLAIM_DIGEST_FIELD),
        "attempt claim self digest mismatch",
    )
    if require_files:
        _require(
            declared_claim_path is not None
            and not declared_claim_path.is_symlink()
            and claim_path is not None
            and claim_path.is_file(),
            "attempt claim must be a non-symlink regular file",
        )
        _require(load_json_object(claim_path) == dict(claim), "attempt claim file content drifted")
        config_path = Path(str(runtime_artifacts["resolved_config"]))
        if config_path.is_file():
            _require(
                claim_path.stat().st_mtime_ns <= config_path.stat().st_mtime_ns,
                "attempt claim was acquired after output creation",
            )


def validate_gpu_isolation_monitor(
    monitor: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    attempt_id: str,
    child_pid: int,
    child_process_group_id: int,
) -> None:
    fields = {
        "kind",
        "schema_version",
        "frozen",
        "cell_id",
        "attempt_id",
        "gpu_isolation_contract_sha256",
        "device_index",
        "device_uuid",
        "device_name",
        "driver_version",
        "total_memory_mib",
        "minimum_free_preflight_mib",
        "monitor_poll_seconds",
        "authorized_child_pid",
        "authorized_process_group_id",
        "raw_samples",
        "exclusive_preflight_pass",
        "continuous_monitor_pass",
        "postflight_pass",
        "foreign_compute_process_observations",
        "foreign_memory_peak_mib",
        "sample_count",
        "minimum_observed_free_mib",
        "peak_used_mib",
        "maximum_observed_poll_gap_seconds",
        "started_at_unix_ns",
        "ended_at_unix_ns",
        "raw_sample_chain_sha256",
        "gpu_monitor_sha256",
    }
    _require_exact_fields(monitor, fields, "GPU isolation monitor")
    _require(monitor.get("kind") == "lace_rq1_gpu_isolation_monitor", "GPU monitor kind mismatch")
    _require(
        monitor.get("schema_version") == 2 and monitor.get("frozen") is True,
        "GPU monitor schema mismatch",
    )
    _require(monitor.get("cell_id") == plan.get("cell_id"), "GPU monitor cell mismatch")
    _require(monitor.get("attempt_id") == attempt_id, "GPU monitor attempt mismatch")
    contract = _require_mapping(plan.get("gpu_isolation_contract"), "GPU isolation contract")
    _require(
        monitor.get("gpu_isolation_contract_sha256")
        == contract.get("gpu_isolation_contract_sha256")
        == canonical_sha256(contract, digest_field="gpu_isolation_contract_sha256"),
        "GPU monitor contract identity mismatch",
    )
    identity_map = {
        "device_index": "device_index",
        "device_uuid": "device_uuid",
        "device_name": "device_name",
        "driver_version": "driver_version",
        "total_memory_mib": "total_memory_mib",
        "minimum_free_preflight_mib": "minimum_free_preflight_mib",
        "monitor_poll_seconds": "monitor_poll_seconds",
    }
    for monitor_field, contract_field in identity_map.items():
        _require(
            monitor.get(monitor_field) == contract.get(contract_field),
            f"GPU monitor {monitor_field} differs from the frozen contract",
        )
    _require(
        monitor.get("authorized_child_pid") == child_pid
        and monitor.get("authorized_process_group_id") == child_process_group_id,
        "GPU monitor authorized child process identity mismatch",
    )
    samples = monitor.get("raw_samples")
    _require(
        isinstance(samples, list) and len(samples) >= 3, "GPU monitor raw samples are incomplete"
    )
    _require(monitor.get("sample_count") == len(samples), "GPU monitor sample count mismatch")
    phases: list[str] = []
    observed_times: list[int] = []
    free_values: list[int] = []
    used_values: list[int] = []
    sample_digests: list[str] = []
    previous_digest: str | None = None
    foreign_process_count = 0
    foreign_memory_peak = 0
    sample_fields = {
        "sequence_index",
        "phase",
        "observed_at_unix_ns",
        "device_index",
        "device_uuid",
        "device_name",
        "driver_version",
        "total_memory_mib",
        "used_mib",
        "free_mib",
        "compute_processes",
        "raw_query",
        "raw_query_sha256",
        "previous_sample_sha256",
        "sample_sha256",
    }
    for index, raw in enumerate(samples):
        sample = _require_mapping(raw, f"GPU raw_samples[{index}]")
        _require_exact_fields(sample, sample_fields, "GPU raw sample")
        _require(sample.get("sequence_index") == index, "GPU sample sequence is not contiguous")
        phase = sample.get("phase")
        _require(phase in {"preflight", "continuous", "postflight"}, "GPU sample phase is invalid")
        phases.append(str(phase))
        observed_at = _positive_int(sample.get("observed_at_unix_ns"), "GPU sample timestamp")
        _require(
            not observed_times or observed_at > observed_times[-1],
            "GPU sample timestamps are not increasing",
        )
        observed_times.append(observed_at)
        for sample_field, contract_field in (
            ("device_index", "device_index"),
            ("device_uuid", "device_uuid"),
            ("device_name", "device_name"),
            ("driver_version", "driver_version"),
            ("total_memory_mib", "total_memory_mib"),
        ):
            _require(
                sample.get(sample_field) == contract.get(contract_field),
                f"GPU sample {sample_field} drifted",
            )
        used = _nonnegative_int(sample.get("used_mib"), "GPU sample used_mib")
        free = _nonnegative_int(sample.get("free_mib"), "GPU sample free_mib")
        _require(
            used <= contract["total_memory_mib"] and free <= contract["total_memory_mib"],
            "GPU memory sample is impossible",
        )
        used_values.append(used)
        free_values.append(free)
        processes = sample.get("compute_processes")
        _require(isinstance(processes, list), "GPU sample compute_processes must be a list")
        unexpected_memory = 0
        for process_index, raw_process in enumerate(processes):
            process = _require_mapping(raw_process, f"GPU process[{process_index}]")
            _require_exact_fields(
                process,
                {"pid", "process_group_id", "used_memory_mib"},
                "GPU process sample",
            )
            _positive_int(process.get("pid"), "GPU process pid")
            process_group_id = _positive_int(
                process.get("process_group_id"),
                "GPU process process_group_id",
            )
            process_memory = _nonnegative_int(
                process.get("used_memory_mib"),
                "GPU process used_memory_mib",
            )
            if process_group_id != child_process_group_id:
                foreign_process_count += 1
                unexpected_memory += process_memory
        foreign_memory_peak = max(foreign_memory_peak, unexpected_memory)
        raw_query = sample.get("raw_query")
        _require(
            isinstance(raw_query, str) and raw_query, "GPU sample raw query evidence is missing"
        )
        _require(
            sample.get("raw_query_sha256") == hashlib.sha256(raw_query.encode("utf-8")).hexdigest(),
            "GPU sample raw query digest mismatch",
        )
        _require(
            sample.get("previous_sample_sha256") == previous_digest,
            "GPU sample hash chain is broken",
        )
        sample_digest = _require_sha256(sample.get("sample_sha256"), "GPU sample_sha256")
        _require(
            sample_digest == canonical_sha256(sample, digest_field="sample_sha256"),
            "GPU sample self digest mismatch",
        )
        previous_digest = sample_digest
        sample_digests.append(sample_digest)
    _require(
        phases[0] == "preflight" and phases[-1] == "postflight",
        "GPU pre/postflight samples are missing",
    )
    _require(
        all(phase == "continuous" for phase in phases[1:-1]),
        "GPU continuous sample phases are invalid",
    )
    _require(
        samples[0]["compute_processes"] == [] and samples[-1]["compute_processes"] == [],
        "GPU pre/postflight observed compute processes",
    )
    threshold = int(contract["minimum_free_preflight_mib"])
    _require(
        free_values[0] >= threshold and free_values[-1] >= threshold,
        "GPU pre/postflight free memory is below the frozen threshold",
    )
    gaps = [
        (right - left) / 1_000_000_000
        for left, right in zip(observed_times[:-1], observed_times[1:], strict=True)
    ]
    maximum_gap = max(gaps)
    _require(
        maximum_gap <= float(contract["monitor_poll_seconds"]) * 2.5,
        "GPU monitor poll cadence has an unexplained gap",
    )
    _require(
        monitor.get("minimum_observed_free_mib") == min(free_values),
        "GPU minimum free memory drifted",
    )
    _require(monitor.get("peak_used_mib") == max(used_values), "GPU peak used memory drifted")
    _require(
        monitor.get("maximum_observed_poll_gap_seconds") == maximum_gap,
        "GPU maximum poll gap drifted",
    )
    _require(
        monitor.get("raw_sample_chain_sha256")
        == canonical_sha256({"sample_sha256": sample_digests}),
        "GPU raw-sample chain identity mismatch",
    )
    _require(monitor.get("exclusive_preflight_pass") is True, "exclusive GPU preflight failed")
    _require(monitor.get("continuous_monitor_pass") is True, "continuous GPU isolation failed")
    _require(monitor.get("postflight_pass") is True, "GPU postflight failed")
    _require(
        foreign_process_count == 0
        and foreign_memory_peak == 0
        and monitor.get("foreign_compute_process_observations") == 0
        and monitor.get("foreign_memory_peak_mib") == 0,
        "foreign GPU contamination was observed",
    )
    started = _positive_int(monitor.get("started_at_unix_ns"), "GPU monitor started_at_unix_ns")
    ended = _positive_int(monitor.get("ended_at_unix_ns"), "GPU monitor ended_at_unix_ns")
    _require(
        started == observed_times[0] and ended == observed_times[-1],
        "GPU monitor time bounds drifted",
    )
    digest = _require_sha256(monitor.get("gpu_monitor_sha256"), "gpu_monitor_sha256")
    _require(
        digest == canonical_sha256(monitor, digest_field="gpu_monitor_sha256"),
        "GPU monitor self digest mismatch",
    )


def validate_execution_completion(
    completion: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    """Validate launcher-owned success evidence after the child has exited."""

    fields = {
        "kind",
        "schema_version",
        "frozen",
        "cell_id",
        "attempt_id",
        "attempt_claim",
        PLAN_DIGEST_FIELD,
        "plan_file_sha256",
        "command_sha256",
        "cwd",
        "argv",
        "environment",
        "child_pid",
        "child_process_group_id",
        "started_at_utc",
        "ended_at_utc",
        "started_at_unix_ns",
        "ended_at_unix_ns",
        "exit_code",
        "term_signal",
        "oom_killed",
        "foreign_process_contamination",
        "process_log",
        "runtime_metrics",
        "final_checkpoint",
        "gpu_isolation_monitor",
        EXECUTION_COMPLETION_DIGEST_FIELD,
    }
    _require_exact_fields(completion, fields, "execution completion")
    _require(
        completion.get("kind") == EXECUTION_COMPLETION_KIND, "execution completion kind mismatch"
    )
    _require(
        completion.get("schema_version") == EXECUTION_COMPLETION_SCHEMA_VERSION,
        "execution completion schema mismatch",
    )
    _require(completion.get("frozen") is True, "execution completion must be frozen")
    _require(completion.get("cell_id") == plan.get("cell_id"), "execution completion cell mismatch")
    claim_record = _require_mapping(completion.get("attempt_claim"), "completion attempt_claim")
    _require_exact_fields(
        claim_record,
        {"path", "bytes", "sha256", ATTEMPT_CLAIM_DIGEST_FIELD},
        "completion attempt-claim record",
    )
    claim_path = Path(str(plan["runtime_artifacts"]["attempt_claim"])).resolve()
    actual_claim_artifact = _required_artifact(claim_path, "attempt claim")
    _require(
        {key: claim_record[key] for key in ("path", "bytes", "sha256")} == actual_claim_artifact,
        "execution completion attempt-claim bytes mismatch",
    )
    claim = load_json_object(claim_path)
    validate_attempt_claim(claim, plan)
    _require(
        claim_record.get(ATTEMPT_CLAIM_DIGEST_FIELD) == claim.get(ATTEMPT_CLAIM_DIGEST_FIELD),
        "execution completion attempt-claim identity mismatch",
    )
    _require(completion.get("attempt_id") == claim.get("attempt_id"), "completion attempt mismatch")
    _require(
        completion.get(PLAN_DIGEST_FIELD) == plan.get(PLAN_DIGEST_FIELD), "completion plan mismatch"
    )
    _require(
        completion.get("plan_file_sha256") == file_sha256(plan["plan_path"]),
        "completion plan-file bytes mismatch",
    )
    _require(
        completion.get("command_sha256") == plan.get("command_sha256"),
        "completion command mismatch",
    )
    _require(completion.get("cwd") == plan.get("launch_cwd"), "completion cwd mismatch")
    _require(completion.get("argv") == plan.get("argv"), "completion argv mismatch")
    _require(
        completion.get("environment") == plan.get("launch_environment"),
        "completion effective environment mismatch",
    )
    child_pid = _positive_int(completion.get("child_pid"), "completion child_pid")
    child_process_group_id = _positive_int(
        completion.get("child_process_group_id"),
        "completion child_process_group_id",
    )
    _require(
        child_process_group_id == child_pid and child_pid != claim["launcher_pid"],
        "RQ1 child did not run in a dedicated process group",
    )
    started = _positive_int(completion.get("started_at_unix_ns"), "completion started_at_unix_ns")
    ended = _positive_int(completion.get("ended_at_unix_ns"), "completion ended_at_unix_ns")
    _require(
        _utc_unix_ns(completion.get("started_at_utc"), "completion started_at_utc") == started
        and _utc_unix_ns(completion.get("ended_at_utc"), "completion ended_at_utc") == ended,
        "completion UTC and unix-nanosecond timestamps differ",
    )
    _require(
        ended >= started >= claim["claimed_at_unix_ns"], "completion timing precedes its claim"
    )
    _require(completion.get("exit_code") == 0, "RQ1 child process did not exit zero")
    _require(completion.get("term_signal") is None, "RQ1 child process received a signal")
    _require(completion.get("oom_killed") is False, "RQ1 child process was OOM-killed")
    _require(
        completion.get("foreign_process_contamination") is False,
        "foreign process contamination was reported",
    )
    expected_artifacts = {
        "process_log": plan["runtime_artifacts"]["process_log"],
        "runtime_metrics": plan["runtime_artifacts"]["runtime_metrics"],
        "final_checkpoint": plan["runtime_artifacts"]["final_checkpoint"],
        "gpu_isolation_monitor": plan["runtime_artifacts"]["gpu_isolation_monitor"],
    }
    for name, path in expected_artifacts.items():
        record = _require_mapping(completion.get(name), f"completion {name}")
        _require_exact_fields(record, {"path", "bytes", "sha256"}, f"completion {name}")
        _require(
            dict(record) == _required_artifact(path, f"completion {name}"),
            f"execution completion {name} bytes mismatch",
        )
    monitor = load_json_object(expected_artifacts["gpu_isolation_monitor"])
    validate_gpu_isolation_monitor(
        monitor,
        plan,
        attempt_id=str(claim["attempt_id"]),
        child_pid=child_pid,
        child_process_group_id=child_process_group_id,
    )
    _require(
        monitor["started_at_unix_ns"] <= started and monitor["ended_at_unix_ns"] >= ended,
        "GPU isolation monitor did not cover the full child lifetime",
    )
    digest = _require_sha256(
        completion.get(EXECUTION_COMPLETION_DIGEST_FIELD),
        EXECUTION_COMPLETION_DIGEST_FIELD,
    )
    _require(
        digest
        == canonical_sha256(
            completion,
            digest_field=EXECUTION_COMPLETION_DIGEST_FIELD,
        ),
        "execution completion self digest mismatch",
    )


def torch_state_dict_record(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Hash an exact tensor state dictionary without dtype conversion."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency gate
        raise RQ1TrainingError("PyTorch is required for state-dict provenance") from error
    _require(isinstance(state_dict, Mapping) and state_dict, "state_dict must be non-empty")
    entries: list[dict[str, Any]] = []
    for key in sorted(state_dict):
        tensor = state_dict[key]
        _require(isinstance(key, str) and key, "state_dict keys must be non-empty strings")
        _require(isinstance(tensor, torch.Tensor), f"state_dict entry is not a tensor: {key}")
        cpu = tensor.detach().cpu().contiguous()
        payload = cpu.view(torch.uint8).numpy().tobytes()
        entries.append(
            {
                "key": key,
                "shape": list(cpu.shape),
                "dtype": str(cpu.dtype),
                "numel": int(cpu.numel()),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    record = {
        "key_count": len(entries),
        "total_numel": sum(entry["numel"] for entry in entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }
    record["state_dict_sha256"] = canonical_sha256(record)
    return record


def strict_load_rq1_state_dict(
    module: Any,
    state_dict: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly load one RQ1 module and prove post-load tensor identity."""

    checkpoint_record = torch_state_dict_record(state_dict)
    try:
        incompatible = module.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RQ1TrainingError(f"strict RQ1 {label} checkpoint load failed: {error}") from error
    _require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        f"strict RQ1 {label} checkpoint load reported incompatible keys",
    )
    post_load_record = torch_state_dict_record(module.state_dict())
    _require(
        checkpoint_record == post_load_record,
        f"strict RQ1 {label} post-load state differs from checkpoint tensors",
    )
    return checkpoint_record, post_load_record


def _validate_state_dict_record(record: Mapping[str, Any], name: str) -> None:
    _require_exact_fields(
        record,
        {"key_count", "total_numel", "total_bytes", "entries", "state_dict_sha256"},
        name,
    )
    entries = record.get("entries")
    _require(isinstance(entries, list) and entries, f"{name} entries are invalid")
    keys: list[str] = []
    total_numel = 0
    total_bytes = 0
    for index, raw in enumerate(entries):
        entry = _require_mapping(raw, f"{name}.entries[{index}]")
        _require_exact_fields(
            entry,
            {"key", "shape", "dtype", "numel", "bytes", "sha256"},
            f"{name} entry",
        )
        key = entry.get("key")
        _require(isinstance(key, str) and key, f"{name} entry key is invalid")
        keys.append(key)
        shape = entry.get("shape")
        _require(
            isinstance(shape, list)
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in shape
            ),
            f"{name} entry shape is invalid",
        )
        _require(isinstance(entry.get("dtype"), str) and entry["dtype"], f"{name} dtype is invalid")
        total_numel += _nonnegative_int(entry.get("numel"), f"{name} numel")
        total_bytes += _nonnegative_int(entry.get("bytes"), f"{name} bytes")
        _require_sha256(entry.get("sha256"), f"{name} tensor sha256")
    _require(keys == sorted(set(keys)), f"{name} keys are not canonical")
    _require(record.get("key_count") == len(entries), f"{name} key count mismatch")
    _require(record.get("total_numel") == total_numel, f"{name} numel total mismatch")
    _require(record.get("total_bytes") == total_bytes, f"{name} byte total mismatch")
    _require(
        record.get("state_dict_sha256")
        == canonical_sha256(record, digest_field="state_dict_sha256"),
        f"{name} self digest mismatch",
    )


def validate_initialization_report(report: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    fields = {
        "kind",
        "schema_version",
        "checkpoint_path",
        "checkpoint_sha256",
        "resume",
        "policy_source_key",
        "policy_strict",
        "policy_checkpoint_state",
        "policy_post_load_state",
        "value_source_key",
        "value_strict",
        "value_checkpoint_state",
        "value_post_load_state",
        "optimizer_state_restored",
        "lr_scheduler_state_restored",
        "environment_state_restored",
        "trainer_state_restored",
        "optimizer_state_entry_count_before_training",
        "initialization_report_sha256",
    }
    _require_exact_fields(report, fields, "RQ1 initialization report")
    _require(
        report.get("kind") == "lace_rq1_model_only_initialization_report",
        "initialization kind mismatch",
    )
    _require(report.get("schema_version") == 1, "initialization schema mismatch")
    initialization = _require_mapping(plan.get("initialization"), "plan initialization")
    _require(
        Path(str(report.get("checkpoint_path"))).resolve()
        == Path(str(initialization.get("checkpoint_path"))).resolve(),
        "initialization checkpoint path mismatch",
    )
    _require(
        report.get("checkpoint_sha256") == initialization.get("checkpoint_sha256"),
        "initialization checkpoint bytes mismatch",
    )
    _require(report.get("resume") is False, "RQ1 initialization resumed training state")
    _require(
        report.get("policy_source_key") in {"actor_model_state_dict", "policy_state_dict"}
        and report.get("policy_strict") is True,
        "RQ1 policy was not loaded strictly",
    )
    policy_checkpoint = _require_mapping(
        report.get("policy_checkpoint_state"), "policy checkpoint state"
    )
    policy_post = _require_mapping(report.get("policy_post_load_state"), "policy post-load state")
    _validate_state_dict_record(policy_checkpoint, "policy checkpoint state")
    _validate_state_dict_record(policy_post, "policy post-load state")
    _require(policy_checkpoint == policy_post, "policy post-load state identity mismatch")
    _require(
        report.get("value_source_key") == "value_state_dict" and report.get("value_strict") is True,
        "RQ1 value model was not loaded strictly",
    )
    value_checkpoint = _require_mapping(
        report.get("value_checkpoint_state"), "value checkpoint state"
    )
    value_post = _require_mapping(report.get("value_post_load_state"), "value post-load state")
    _validate_state_dict_record(value_checkpoint, "value checkpoint state")
    _validate_state_dict_record(value_post, "value post-load state")
    _require(value_checkpoint == value_post, "value post-load state identity mismatch")
    for field in (
        "optimizer_state_restored",
        "lr_scheduler_state_restored",
        "environment_state_restored",
        "trainer_state_restored",
    ):
        _require(report.get(field) is False, f"RQ1 initialization unexpectedly restored {field}")
    _require(
        report.get("optimizer_state_entry_count_before_training") == 0,
        "RQ1 optimizer was not freshly initialized",
    )
    digest = _require_sha256(
        report.get("initialization_report_sha256"),
        "initialization_report_sha256",
    )
    _require(
        digest == canonical_sha256(report, digest_field="initialization_report_sha256"),
        "initialization report self digest mismatch",
    )


def _int_vector(value: Any, name: str) -> list[int]:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().reshape(-1).tolist()
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a vector",
    )
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            result.append(int(item))
            continue
        _require(
            isinstance(item, (int, float))
            and math.isfinite(float(item))
            and float(item).is_integer(),
            f"{name}[{index}] must be integral",
        )
        result.append(int(item))
    return result


class RQ1RuntimeAccounting:
    """Exact per-motion occupancy, reset, and PPO-update accumulator."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        _validate_plan_shape(plan)
        self.plan = deepcopy(dict(plan))
        arm = _require_mapping(plan.get("arm_contract"), "arm_contract")
        semantics = _require_mapping(plan.get("training_semantics"), "training_semantics")
        motion_keys = arm.get("motion_keys")
        _require(
            isinstance(motion_keys, list)
            and motion_keys
            and motion_keys == sorted(set(motion_keys)),
            "runtime motion keys must be sorted and unique",
        )
        self.plan_sha256 = str(plan[PLAN_DIGEST_FIELD])
        self.launch_contract_sha256 = str(plan["launch_contract_sha256"])
        self.arm_id = str(plan["arm_id"])
        self.runtime_binding_sha256 = str(arm[FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD])
        self.motion_keys = list(motion_keys)
        self.motion_count = len(self.motion_keys)
        self.base_probabilities = [float(value) for value in arm["base_probabilities"]]
        self.arm_probabilities = [float(value) for value in arm["probabilities"]]
        _require(
            len(self.base_probabilities) == self.motion_count
            and len(self.arm_probabilities) == self.motion_count,
            "runtime probability length mismatch",
        )
        _require(
            abs(math.fsum(self.base_probabilities) - 1.0) <= 1e-12
            and abs(math.fsum(self.arm_probabilities) - 1.0) <= 1e-12,
            "runtime probabilities must sum to one",
        )
        self.panels = deepcopy(list(arm["panels"]))
        key_to_index = {key: index for index, key in enumerate(self.motion_keys)}
        self.panel_motion_indices: dict[str, list[int]] = {}
        covered: set[int] = set()
        for panel in self.panels:
            indices = [key_to_index[key] for key in panel["motion_keys"]]
            _require(not (covered & set(indices)), "motion belongs to multiple runtime panels")
            covered.update(indices)
            self.panel_motion_indices[str(panel["panel_id"])] = indices
        _require(covered == set(range(self.motion_count)), "runtime panels do not cover motions")
        self.num_envs = int(semantics["num_envs_per_rank"])
        self.expected_iterations = int(semantics["iterations"])
        self.rollout_steps = int(semantics["rollout_steps_per_iteration"])
        self.expected_control_transitions = int(semantics["planned_control_transitions"])
        self.expected_optimizer_opportunities = int(semantics["planned_optimizer_opportunities"])
        self.expected_synchronized_updates = int(
            semantics["planned_synchronized_parameter_updates"]
        )
        self.training_factorization = {
            key: deepcopy(semantics[key])
            for key in (
                "num_envs_per_rank",
                "world_size",
                "rollout_steps_per_iteration",
                "iterations",
                "ppo_epochs",
                "minibatches_per_epoch",
                "local_minibatch_size",
                "per_device_train_batch_size",
                "num_microbatches_per_minibatch",
                "gradient_accumulation_steps",
                "decimation",
                "planned_control_transitions",
                "planned_physics_substeps",
                "physics_substeps_semantics",
                "planned_optimizer_opportunities",
                "planned_synchronized_parameter_updates",
            )
        }
        self.optimizer_opportunities_per_iteration = (
            self.expected_optimizer_opportunities // self.expected_iterations
        )
        self.synchronized_updates_per_iteration = (
            self.expected_synchronized_updates // self.expected_iterations
        )
        self.control_step_calls = 0
        self.control_step_occupancy_counts = [0] * self.motion_count
        self.episode_completion_counts = [0] * self.motion_count
        self.termination_counts = [0] * self.motion_count
        self.timeout_counts = [0] * self.motion_count
        self.optimizer_iterations = 0
        self.optimizer_opportunities = 0
        self.optimizer_step_attempts = 0
        self.optimizer_nonfinite_skips = 0
        self.optimizer_sync_boundaries = 0
        self.successful_parameter_updates = 0
        self.optimizer_accelerator_skips = 0
        self.synchronized_update_skips = 0

    def record_control_step(self, motion_ids: Any, dones: Any, timeouts: Any) -> None:
        """Attribute one transition to each motion active before ``env.step``."""

        ids = _int_vector(motion_ids, "motion_ids")
        done_values = _int_vector(dones, "dones")
        timeout_values = _int_vector(timeouts, "timeouts")
        _require(
            len(ids) == len(done_values) == len(timeout_values) == self.num_envs,
            "RQ1 control-step vectors must exactly match num_envs",
        )
        for env_index, (motion_id, done, timeout) in enumerate(
            zip(ids, done_values, timeout_values, strict=True)
        ):
            _require(0 <= motion_id < self.motion_count, "motion ID is outside the resident set")
            _require(done in (0, 1), f"dones[{env_index}] must be boolean")
            _require(timeout in (0, 1), f"timeouts[{env_index}] must be boolean")
            _require(not timeout or done, "timeout may only be set on a completed episode")
            self.control_step_occupancy_counts[motion_id] += 1
            if done:
                self.episode_completion_counts[motion_id] += 1
                if timeout:
                    self.timeout_counts[motion_id] += 1
                else:
                    self.termination_counts[motion_id] += 1
        self.control_step_calls += 1

    def record_optimizer_iteration(
        self,
        *,
        optimizer_step_attempts: int,
        optimizer_nonfinite_skips: int,
        optimizer_accelerator_skips: int,
        sync_boundaries: int,
        successful_parameter_updates: int,
        synchronized_update_skips: int,
    ) -> None:
        values = {
            "optimizer_step_attempts": optimizer_step_attempts,
            "optimizer_nonfinite_skips": optimizer_nonfinite_skips,
            "optimizer_accelerator_skips": optimizer_accelerator_skips,
            "sync_boundaries": sync_boundaries,
            "successful_parameter_updates": successful_parameter_updates,
            "synchronized_update_skips": synchronized_update_skips,
        }
        parsed = {name: _nonnegative_int(value, name) for name, value in values.items()}
        _require(
            parsed["optimizer_step_attempts"] + parsed["optimizer_nonfinite_skips"]
            == self.optimizer_opportunities_per_iteration,
            "optimizer attempts/nonfinite skips violate the PPO factorization",
        )
        _require(
            parsed["sync_boundaries"] == self.synchronized_updates_per_iteration,
            "optimizer sync boundaries violate gradient accumulation",
        )
        _require(
            parsed["successful_parameter_updates"] + parsed["synchronized_update_skips"]
            == self.synchronized_updates_per_iteration,
            "successful/skipped parameter updates violate the PPO factorization",
        )
        _require(
            parsed["optimizer_accelerator_skips"] <= parsed["synchronized_update_skips"],
            "accelerator skips exceed synchronized update skips",
        )
        self.optimizer_iterations += 1
        self.optimizer_opportunities += self.optimizer_opportunities_per_iteration
        self.optimizer_step_attempts += parsed["optimizer_step_attempts"]
        self.optimizer_nonfinite_skips += parsed["optimizer_nonfinite_skips"]
        self.optimizer_accelerator_skips += parsed["optimizer_accelerator_skips"]
        self.optimizer_sync_boundaries += parsed["sync_boundaries"]
        self.successful_parameter_updates += parsed["successful_parameter_updates"]
        self.synchronized_update_skips += parsed["synchronized_update_skips"]

    def _panel_rows(self, draw_counts: Sequence[int]) -> list[dict[str, Any]]:
        total = sum(self.control_step_occupancy_counts)
        _require(total > 0, "cannot summarize an RQ1 run without control transitions")
        _require(
            len(draw_counts) == self.motion_count,
            "panel draw counts differ from the resident motion universe",
        )
        rows: list[dict[str, Any]] = []
        for panel in self.panels:
            panel_id = str(panel["panel_id"])
            indices = self.panel_motion_indices[panel_id]
            occupancy = sum(self.control_step_occupancy_counts[index] for index in indices)
            base_mass = math.fsum(self.base_probabilities[index] for index in indices)
            arm_mass = math.fsum(self.arm_probabilities[index] for index in indices)
            realized_mass = occupancy / total
            rows.append(
                {
                    "panel_id": panel_id,
                    "motion_count": len(indices),
                    "draw_count": sum(int(draw_counts[index]) for index in indices),
                    "control_step_occupancy": occupancy,
                    "realized_exposure": realized_mass,
                    "base_exposure": base_mass,
                    "planned_arm_exposure": arm_mass,
                    "planned_signed_exposure_delta_vs_p0": arm_mass - base_mass,
                    "realized_signed_exposure_delta_vs_p0": realized_mass - base_mass,
                    "episode_completion_count": sum(
                        self.episode_completion_counts[index] for index in indices
                    ),
                    "termination_count": sum(self.termination_counts[index] for index in indices),
                    "timeout_count": sum(self.timeout_counts[index] for index in indices),
                }
            )
        return rows

    def build_runtime_metrics(
        self,
        draw_report: Mapping[str, Any],
        *,
        final_checkpoint: Mapping[str, Any],
        attempt_claim: Mapping[str, Any],
        initialization_report: Mapping[str, Any],
    ) -> dict[str, Any]:
        checkpoint_record = _require_mapping(final_checkpoint, "final_checkpoint")
        _require_exact_fields(
            checkpoint_record,
            {"path", "bytes", "sha256"},
            "final_checkpoint",
        )
        _require(
            isinstance(checkpoint_record.get("path"), str)
            and checkpoint_record["path"]
            and _positive_int(checkpoint_record.get("bytes"), "final_checkpoint.bytes") > 0,
            "final checkpoint record is invalid",
        )
        _require_sha256(checkpoint_record.get("sha256"), "final_checkpoint.sha256")
        claim_record = _require_mapping(attempt_claim, "attempt_claim")
        validate_attempt_claim(claim_record, self.plan, require_files=False)
        init_record = _require_mapping(initialization_report, "initialization_report")
        validate_initialization_report(init_record, self.plan)
        expected_draw_fields = {
            "kind",
            "schema_version",
            FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD,
            "fixed_distribution_config_sha256",
            "arm_id",
            INTERVENTION_PLAN_DIGEST_FIELD,
            "plan_lock_file_sha256",
            "distribution_sha256",
            "motion_keys",
            "motion_count",
            "draw_counts",
            "total_draw_count",
            "counter_sum",
            "exact_invariants_pass",
            "accounting_scope",
            FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
        }
        _require_exact_fields(draw_report, expected_draw_fields, "fixed-distribution draw report")
        _require(
            draw_report.get("kind") == FIXED_DISTRIBUTION_DRAW_REPORT_KIND
            and draw_report.get("schema_version") == FIXED_DISTRIBUTION_DRAW_REPORT_SCHEMA_VERSION,
            "fixed-distribution draw report kind/schema mismatch",
        )
        _require(
            draw_report.get("exact_invariants_pass") is True, "draw-report exact invariants failed"
        )
        _require(
            draw_report.get(FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD) == self.runtime_binding_sha256,
            "fixed-distribution draw report binding differs from the runtime arm",
        )
        _require(draw_report.get("arm_id") == self.arm_id, "draw report arm mismatch")
        arm = _require_mapping(self.plan.get("arm_contract"), "arm_contract")
        for report_field, arm_field in (
            ("fixed_distribution_config_sha256", "fixed_distribution_config_sha256"),
            (INTERVENTION_PLAN_DIGEST_FIELD, INTERVENTION_PLAN_DIGEST_FIELD),
            ("plan_lock_file_sha256", "plan_lock_file_sha256"),
            ("distribution_sha256", "distribution_sha256"),
            ("accounting_scope", "accounting_scope"),
        ):
            _require(
                draw_report.get(report_field) == arm.get(arm_field),
                f"draw report {report_field} differs from the arm",
            )
        _require(
            draw_report.get("motion_keys") == self.motion_keys,
            "draw report motion identity mismatch",
        )
        _require(
            draw_report.get("motion_count") == self.motion_count,
            "draw report motion count mismatch",
        )
        raw_draw_counts = draw_report.get("draw_counts")
        _require(
            isinstance(raw_draw_counts, list)
            and len(raw_draw_counts) == self.motion_count
            and all(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
                for value in raw_draw_counts
            ),
            "draw report counts are invalid",
        )
        total_draw_count = _nonnegative_int(
            draw_report.get("total_draw_count"), "draw report total_draw_count"
        )
        _require(
            draw_report.get("counter_sum") == total_draw_count
            and sum(raw_draw_counts) == total_draw_count,
            "draw report count invariant failed",
        )
        _require(
            total_draw_count >= self.num_envs,
            "fixed sampler did not record the initial environment assignments",
        )
        _require(
            draw_report.get(FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD)
            == canonical_sha256(
                draw_report,
                digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
            ),
            "draw report self digest mismatch",
        )
        total_control = sum(self.control_step_occupancy_counts)
        total_completions = sum(self.episode_completion_counts)
        total_terminations = sum(self.termination_counts)
        total_timeouts = sum(self.timeout_counts)
        _require(
            total_control == self.expected_control_transitions,
            "realized control transitions differ from the fixed budget",
        )
        _require(
            self.control_step_calls == self.expected_iterations * self.rollout_steps,
            "realized control-step calls differ from the rollout factorization",
        )
        _require(
            total_completions == total_terminations + total_timeouts,
            "episode completion causes are not exhaustive and disjoint",
        )
        _require(
            total_completions <= total_control, "episode completions exceed occupied transitions"
        )
        for index in range(self.motion_count):
            _require(
                self.episode_completion_counts[index] <= self.control_step_occupancy_counts[index],
                "per-motion episode completions exceed occupied transitions",
            )
            _require(
                self.episode_completion_counts[index] <= raw_draw_counts[index],
                "per-motion episode completions exceed sampler draws",
            )
        _require(
            total_draw_count >= max(self.num_envs, total_completions),
            "sampler draws are infeasible for the observed resets",
        )
        _require(
            self.optimizer_iterations == self.expected_iterations,
            "optimizer iteration count differs from the fixed budget",
        )
        _require(
            self.optimizer_opportunities == self.expected_optimizer_opportunities,
            "optimizer opportunity count differs from the fixed budget",
        )
        _require(
            self.optimizer_step_attempts == self.expected_optimizer_opportunities,
            "not every planned optimizer step was attempted",
        )
        _require(self.optimizer_nonfinite_skips == 0, "non-finite optimizer skip observed")
        _require(self.optimizer_accelerator_skips == 0, "accelerator optimizer skip observed")
        _require(self.synchronized_update_skips == 0, "synchronized update skip observed")
        _require(
            self.optimizer_sync_boundaries == self.expected_synchronized_updates,
            "optimizer sync boundaries differ from the PPO factorization",
        )
        _require(
            self.successful_parameter_updates == self.expected_synchronized_updates,
            "successful parameter updates differ from the fixed budget",
        )
        per_motion = []
        for index, key in enumerate(self.motion_keys):
            realized = self.control_step_occupancy_counts[index] / total_control
            per_motion.append(
                {
                    "motion_key": key,
                    "motion_id": index,
                    "draw_count": int(draw_report["draw_counts"][index]),
                    "control_step_occupancy": self.control_step_occupancy_counts[index],
                    "realized_exposure": realized,
                    "base_exposure": self.base_probabilities[index],
                    "planned_arm_exposure": self.arm_probabilities[index],
                    "planned_signed_exposure_delta_vs_p0": (
                        self.arm_probabilities[index] - self.base_probabilities[index]
                    ),
                    "realized_signed_exposure_delta_vs_p0": (
                        realized - self.base_probabilities[index]
                    ),
                    "episode_completion_count": self.episode_completion_counts[index],
                    "termination_count": self.termination_counts[index],
                    "timeout_count": self.timeout_counts[index],
                }
            )
        metrics: dict[str, Any] = {
            "kind": RUNTIME_METRICS_KIND,
            "schema_version": RUNTIME_METRICS_SCHEMA_VERSION,
            PLAN_DIGEST_FIELD: self.plan_sha256,
            "launch_contract_sha256": self.launch_contract_sha256,
            "arm_id": self.arm_id,
            "motion_count": self.motion_count,
            "motion_keys": list(self.motion_keys),
            "draw_report": deepcopy(dict(draw_report)),
            "final_checkpoint": deepcopy(dict(checkpoint_record)),
            "attempt_claim": deepcopy(dict(claim_record)),
            "initialization_report": deepcopy(dict(init_record)),
            "training_factorization": deepcopy(self.training_factorization),
            "control_accounting": {
                "control_step_calls": self.control_step_calls,
                "num_envs_per_rank": self.num_envs,
                "total_control_transitions": total_control,
                "expected_control_transitions": self.expected_control_transitions,
                "episode_completion_count": total_completions,
                "termination_count": total_terminations,
                "timeout_count": total_timeouts,
                "completion_definition": "done; timeout_precedence; attributed_to_pre_step_motion",
                "per_motion": per_motion,
                "per_panel": self._panel_rows(raw_draw_counts),
            },
            "optimizer_accounting": {
                "iterations": self.optimizer_iterations,
                "expected_iterations": self.expected_iterations,
                "optimizer_opportunities": self.optimizer_opportunities,
                "expected_optimizer_opportunities": self.expected_optimizer_opportunities,
                "optimizer_step_attempts": self.optimizer_step_attempts,
                "optimizer_nonfinite_skips": self.optimizer_nonfinite_skips,
                "optimizer_accelerator_skips": self.optimizer_accelerator_skips,
                "sync_boundaries": self.optimizer_sync_boundaries,
                "successful_parameter_updates": self.successful_parameter_updates,
                "expected_successful_parameter_updates": self.expected_synchronized_updates,
                "synchronized_update_skips": self.synchronized_update_skips,
            },
            "exact_budget_pass": True,
        }
        metrics[RUNTIME_METRICS_DIGEST_FIELD] = canonical_sha256(
            metrics,
            digest_field=RUNTIME_METRICS_DIGEST_FIELD,
        )
        return metrics


def validate_runtime_metrics(metrics: Mapping[str, Any], plan: Mapping[str, Any]) -> None:
    """Recompute a completed aggregate from its per-motion sufficient statistics."""

    _validate_plan_shape(plan)
    _require(metrics.get("kind") == RUNTIME_METRICS_KIND, "runtime metrics kind mismatch")
    _require(
        metrics.get("schema_version") == RUNTIME_METRICS_SCHEMA_VERSION,
        "runtime metrics schema mismatch",
    )
    digest = _require_sha256(
        metrics.get(RUNTIME_METRICS_DIGEST_FIELD), RUNTIME_METRICS_DIGEST_FIELD
    )
    _require(
        digest == canonical_sha256(metrics, digest_field=RUNTIME_METRICS_DIGEST_FIELD),
        "runtime metrics self digest mismatch",
    )
    _require(metrics.get(PLAN_DIGEST_FIELD) == plan.get(PLAN_DIGEST_FIELD), "metrics plan mismatch")
    _require(metrics.get("arm_id") == plan.get("arm_id"), "metrics arm mismatch")
    _require(metrics.get("exact_budget_pass") is True, "runtime exact-budget gate failed")
    arm = _require_mapping(plan.get("arm_contract"), "arm_contract")
    _require(metrics.get("motion_keys") == arm.get("motion_keys"), "metrics motion order mismatch")
    control = _require_mapping(metrics.get("control_accounting"), "control_accounting")
    optimizer = _require_mapping(metrics.get("optimizer_accounting"), "optimizer_accounting")
    per_motion = control.get("per_motion")
    per_panel = control.get("per_panel")
    _require(
        isinstance(per_motion, list) and len(per_motion) == arm["motion_count"],
        "per-motion rows invalid",
    )
    _require(
        isinstance(per_panel, list) and len(per_panel) == len(arm["panels"]),
        "per-panel rows invalid",
    )
    _require(
        sum(_nonnegative_int(row.get("control_step_occupancy"), "occupancy") for row in per_motion)
        == control.get("total_control_transitions")
        == control.get("expected_control_transitions")
        == plan["training_semantics"]["planned_control_transitions"],
        "per-motion occupancy does not equal the control budget",
    )
    completion = sum(
        _nonnegative_int(row.get("episode_completion_count"), "completion") for row in per_motion
    )
    termination = sum(
        _nonnegative_int(row.get("termination_count"), "termination") for row in per_motion
    )
    timeout = sum(_nonnegative_int(row.get("timeout_count"), "timeout") for row in per_motion)
    _require(completion == termination + timeout, "per-motion completion causes drifted")
    _require(
        completion <= control.get("total_control_transitions"), "completions exceed transitions"
    )
    for index, raw in enumerate(per_motion):
        row = _require_mapping(raw, f"per_motion[{index}]")
        occupancy = _nonnegative_int(row.get("control_step_occupancy"), "occupancy")
        row_completion = _nonnegative_int(row.get("episode_completion_count"), "completion")
        row_termination = _nonnegative_int(row.get("termination_count"), "termination")
        row_timeout = _nonnegative_int(row.get("timeout_count"), "timeout")
        _require(
            row_completion == row_termination + row_timeout, "per-motion causes are not disjoint"
        )
        _require(row_completion <= occupancy, "per-motion completions exceed occupancy")
    _require(
        optimizer.get("optimizer_step_attempts")
        == optimizer.get("expected_optimizer_opportunities")
        == plan["training_semantics"]["planned_optimizer_opportunities"],
        "optimizer attempts differ from the plan",
    )
    _require(
        optimizer.get("successful_parameter_updates")
        == optimizer.get("expected_successful_parameter_updates")
        == plan["training_semantics"]["planned_synchronized_parameter_updates"],
        "successful parameter updates differ from the plan",
    )
    _require(
        optimizer.get("sync_boundaries")
        == plan["training_semantics"]["planned_synchronized_parameter_updates"],
        "optimizer sync boundaries differ from the plan",
    )
    for field in (
        "optimizer_nonfinite_skips",
        "optimizer_accelerator_skips",
        "synchronized_update_skips",
    ):
        _require(optimizer.get(field) == 0, f"runtime metrics report {field}")
    draw_report = _require_mapping(metrics.get("draw_report"), "draw_report")
    _require(
        draw_report.get(FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD)
        == arm.get(FIXED_DISTRIBUTION_BINDING_DIGEST_FIELD),
        "runtime draw-report binding differs from the arm",
    )
    _require(
        draw_report.get(FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD)
        == canonical_sha256(
            draw_report,
            digest_field=FIXED_DISTRIBUTION_DRAW_REPORT_DIGEST_FIELD,
        ),
        "runtime draw-report self digest mismatch",
    )
    draw_counts = draw_report.get("draw_counts")
    _require(
        isinstance(draw_counts, list) and len(draw_counts) == arm["motion_count"],
        "runtime draw counts are invalid",
    )
    for index, row in enumerate(per_motion):
        _require(
            _nonnegative_int(row.get("episode_completion_count"), "completion")
            <= _nonnegative_int(draw_counts[index], "draw_count"),
            "per-motion completions exceed sampler draws",
        )
    claim_record = _require_mapping(metrics.get("attempt_claim"), "attempt_claim")
    runtime_artifacts = plan.get("runtime_artifacts")
    validate_attempt_claim(
        claim_record,
        plan,
        require_files=isinstance(runtime_artifacts, Mapping),
    )
    initialization_report = _require_mapping(
        metrics.get("initialization_report"),
        "initialization_report",
    )
    validate_initialization_report(initialization_report, plan)

    # Treat only integer sufficient statistics as observations.  Every rate,
    # probability, signed delta, panel aggregation, and headline total is
    # deterministically rebuilt so a fully rehashed derived-field edit fails.
    reconstructed = RQ1RuntimeAccounting(plan)
    for index, raw in enumerate(per_motion):
        row = _require_mapping(raw, f"per_motion[{index}]")
        reconstructed.control_step_occupancy_counts[index] = _nonnegative_int(
            row.get("control_step_occupancy"), "control_step_occupancy"
        )
        reconstructed.episode_completion_counts[index] = _nonnegative_int(
            row.get("episode_completion_count"), "episode_completion_count"
        )
        reconstructed.termination_counts[index] = _nonnegative_int(
            row.get("termination_count"), "termination_count"
        )
        reconstructed.timeout_counts[index] = _nonnegative_int(
            row.get("timeout_count"), "timeout_count"
        )
    reconstructed.control_step_calls = _nonnegative_int(
        control.get("control_step_calls"), "control_step_calls"
    )
    reconstructed.optimizer_iterations = _nonnegative_int(
        optimizer.get("iterations"), "optimizer iterations"
    )
    reconstructed.optimizer_opportunities = _nonnegative_int(
        optimizer.get("optimizer_opportunities"), "optimizer_opportunities"
    )
    reconstructed.optimizer_step_attempts = _nonnegative_int(
        optimizer.get("optimizer_step_attempts"), "optimizer_step_attempts"
    )
    reconstructed.optimizer_nonfinite_skips = _nonnegative_int(
        optimizer.get("optimizer_nonfinite_skips"), "optimizer_nonfinite_skips"
    )
    reconstructed.optimizer_accelerator_skips = _nonnegative_int(
        optimizer.get("optimizer_accelerator_skips"), "optimizer_accelerator_skips"
    )
    reconstructed.optimizer_sync_boundaries = _nonnegative_int(
        optimizer.get("sync_boundaries"), "sync_boundaries"
    )
    reconstructed.successful_parameter_updates = _nonnegative_int(
        optimizer.get("successful_parameter_updates"), "successful_parameter_updates"
    )
    reconstructed.synchronized_update_skips = _nonnegative_int(
        optimizer.get("synchronized_update_skips"), "synchronized_update_skips"
    )
    checkpoint_record = _require_mapping(metrics.get("final_checkpoint"), "final_checkpoint")
    if isinstance(runtime_artifacts, Mapping):
        expected_path = Path(str(runtime_artifacts.get("final_checkpoint"))).resolve()
        _require(
            Path(str(checkpoint_record.get("path"))).resolve() == expected_path,
            "runtime final-checkpoint path differs from the plan",
        )
        _require(
            _required_artifact(expected_path, "runtime final checkpoint")
            == dict(checkpoint_record),
            "runtime final-checkpoint bytes drifted",
        )
    expected = reconstructed.build_runtime_metrics(
        draw_report,
        final_checkpoint=checkpoint_record,
        attempt_claim=claim_record,
        initialization_report=initialization_report,
    )
    _require(
        dict(metrics) == expected,
        "runtime metrics differ from deterministic sufficient-statistic reconstruction",
    )


def _required_artifact(path: str | Path, name: str) -> dict[str, Any]:
    candidate = Path(path).resolve()
    payload = _read_regular_bytes(candidate, name)
    _require(payload, f"{name} must be non-empty")
    return {
        "path": str(candidate),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_rq1_training_receipt(
    plan: Mapping[str, Any],
    *,
    plan_path: str | Path,
) -> dict[str, Any]:
    """Bind completed logs, metrics, configs, checkpoints, data, and sources."""

    validate_rq1_training_plan(plan)
    plan_file = Path(plan_path).resolve()
    _require(plan_file == Path(str(plan["plan_path"])).resolve(), "receipt plan path mismatch")
    on_disk_plan = load_json_object(plan_file)
    _require(on_disk_plan == dict(plan), "receipt plan bytes decode to different content")
    artifacts = _require_mapping(plan.get("runtime_artifacts"), "runtime_artifacts")
    metrics_path = Path(str(artifacts.get("runtime_metrics"))).resolve()
    metrics = load_json_object(metrics_path)
    validate_runtime_metrics(metrics, plan)
    completion_path = Path(str(artifacts.get("execution_completion"))).resolve()
    completion = load_json_object(completion_path)
    validate_execution_completion(completion, plan)
    gpu_monitor_path = Path(str(artifacts.get("gpu_isolation_monitor"))).resolve()
    gpu_monitor = load_json_object(gpu_monitor_path)
    artifact_records = {
        "plan": _required_artifact(plan_file, "RQ1 plan"),
        "protocol": _required_artifact(plan["protocol"]["path"], "RQ1 protocol"),
        "attempt_claim": _required_artifact(artifacts.get("attempt_claim"), "attempt claim"),
        "resolved_config": _required_artifact(
            artifacts.get("resolved_config"), "resolved training config"
        ),
        "initial_checkpoint": _required_artifact(
            plan["initialization"]["checkpoint_path"], "initial checkpoint"
        ),
        "initial_checkpoint_config": _required_artifact(
            plan["initialization"]["checkpoint_config_path"],
            "initial checkpoint config",
        ),
        "final_checkpoint": _required_artifact(
            artifacts.get("final_checkpoint"), "final checkpoint"
        ),
        "hydra_log": _required_artifact(artifacts.get("hydra_log"), "Hydra train log"),
        "process_log": _required_artifact(artifacts.get("process_log"), "process log"),
        "runtime_metrics": _required_artifact(metrics_path, "runtime metrics"),
        "gpu_isolation_monitor": _required_artifact(
            artifacts.get("gpu_isolation_monitor"),
            "GPU isolation monitor",
        ),
        "execution_completion": _required_artifact(completion_path, "execution completion"),
        "dataset_manifest": _required_artifact(
            plan["dataset"]["manifest"], "dataset subset manifest"
        ),
        "split_manifest": _required_artifact(plan["dataset"]["split_manifest"], "split manifest"),
    }
    saved_config_readback = audit_saved_rq1_training_config(
        artifact_records["resolved_config"]["path"],
        plan,
    )
    _require(
        artifact_records["initial_checkpoint"]["sha256"]
        == plan["initialization"]["checkpoint_sha256"],
        "receipt initial checkpoint differs from the plan",
    )
    _require(
        artifact_records["initial_checkpoint_config"]["sha256"]
        == plan["initialization"]["checkpoint_config_sha256"],
        "receipt initial checkpoint config differs from the plan",
    )
    _require(
        artifact_records["runtime_metrics"]["sha256"] == file_sha256(metrics_path),
        "runtime metrics bytes changed during receipt construction",
    )
    receipt: dict[str, Any] = {
        "kind": RECEIPT_KIND,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        PLAN_DIGEST_FIELD: plan[PLAN_DIGEST_FIELD],
        "launch_contract_sha256": plan["launch_contract_sha256"],
        "run_id": plan["run_id"],
        "arm_id": plan["arm_id"],
        "matched_training_config_sha256": plan["matched_training_config_sha256"],
        "dataset_subset_sha256": plan["dataset"]["subset_sha256"],
        "source_inventory_sha256": plan["source_inventory_sha256"],
        "robot_asset_inventory_sha256": plan["robot_asset_inventory_sha256"],
        "source_bytes_reverified": True,
        "model_only_initialization_observed": True,
        "resume": False,
        "exact_budget_pass": True,
        "execution_exit_code": completion["exit_code"],
        "attempt_id": completion["attempt_id"],
        "local_attempt_claim_verified": True,
        "external_worm_authority_verified": False,
        "scientific_launch_ready": False,
        "runtime_metrics_sha256": metrics[RUNTIME_METRICS_DIGEST_FIELD],
        "execution_completion_sha256": completion[EXECUTION_COMPLETION_DIGEST_FIELD],
        "gpu_isolation_contract_sha256": plan["gpu_isolation_contract"][
            "gpu_isolation_contract_sha256"
        ],
        "gpu_isolation_monitor_sha256": gpu_monitor["gpu_monitor_sha256"],
        "saved_config_readback": saved_config_readback,
        "artifacts": artifact_records,
    }
    receipt[RECEIPT_DIGEST_FIELD] = canonical_sha256(
        receipt,
        digest_field=RECEIPT_DIGEST_FIELD,
    )
    return receipt


def validate_rq1_training_receipt(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> None:
    validate_rq1_training_plan(plan)
    _require(receipt.get("kind") == RECEIPT_KIND, "RQ1 receipt kind mismatch")
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION, "receipt schema mismatch")
    digest = _require_sha256(receipt.get(RECEIPT_DIGEST_FIELD), RECEIPT_DIGEST_FIELD)
    _require(
        digest == canonical_sha256(receipt, digest_field=RECEIPT_DIGEST_FIELD),
        "RQ1 receipt self digest mismatch",
    )
    _require(receipt.get(PLAN_DIGEST_FIELD) == plan.get(PLAN_DIGEST_FIELD), "receipt plan mismatch")
    _require(receipt.get("arm_id") == plan.get("arm_id"), "receipt arm mismatch")
    _require(receipt.get("exact_budget_pass") is True, "receipt exact-budget gate failed")
    artifacts = _require_mapping(receipt.get("artifacts"), "receipt.artifacts")
    for name, raw in artifacts.items():
        record = _require_mapping(raw, f"receipt artifact {name}")
        _require_exact_fields(record, {"path", "bytes", "sha256"}, "receipt artifact")
        actual = _required_artifact(record.get("path"), f"receipt artifact {name}")
        _require(actual == dict(record), f"receipt artifact bytes drifted: {name}")
    metrics = load_json_object(artifacts["runtime_metrics"]["path"])
    validate_runtime_metrics(metrics, plan)
    _require(
        receipt.get("runtime_metrics_sha256") == metrics[RUNTIME_METRICS_DIGEST_FIELD],
        "receipt runtime-metrics identity mismatch",
    )
    completion = load_json_object(artifacts["execution_completion"]["path"])
    validate_execution_completion(completion, plan)
    _require(
        receipt.get("execution_completion_sha256") == completion[EXECUTION_COMPLETION_DIGEST_FIELD],
        "receipt execution-completion identity mismatch",
    )
    monitor = load_json_object(artifacts["gpu_isolation_monitor"]["path"])
    _require(
        receipt.get("gpu_isolation_contract_sha256")
        == plan["gpu_isolation_contract"]["gpu_isolation_contract_sha256"]
        and receipt.get("gpu_isolation_monitor_sha256") == monitor.get("gpu_monitor_sha256"),
        "receipt GPU-isolation identity mismatch",
    )
    _require(
        receipt.get("execution_exit_code") == 0
        and receipt.get("local_attempt_claim_verified") is True
        and receipt.get("external_worm_authority_verified") is False
        and receipt.get("scientific_launch_ready") is False,
        "receipt overstates scientific launch authority",
    )
    saved_config_readback = audit_saved_rq1_training_config(
        artifacts["resolved_config"]["path"],
        plan,
    )
    _require(
        receipt.get("saved_config_readback") == saved_config_readback,
        "receipt saved-config readback mismatch",
    )
    # Rebuild the entire receipt from the frozen plan and live artifact bytes.
    # This closes omissions and fully rehashed edits to fields which individual
    # artifact checks do not otherwise need to interpret.
    expected = build_rq1_training_receipt(
        plan,
        plan_path=plan["plan_path"],
    )
    _require(
        dict(receipt) == expected,
        "RQ1 receipt differs from deterministic reconstruction",
    )


def assert_rq1_trainer_callbacks(callbacks: Sequence[Any], *, resume: bool) -> None:
    """Runtime guard used by the trainer before any rollout is collected."""

    _require(resume is False, "RQ1 training requires model-only initialization with resume=False")
    names = [callback.__class__.__name__ for callback in callbacks]
    _require("ImResampleCallback" not in names, "RQ1 training forbids ImResampleCallback")
    _require(
        names.count("RQ1TrainingCallback") == 1,
        "RQ1 training requires exactly one RQ1TrainingCallback",
    )
    _require(
        names.count("ModelSaveCallback") == 1,
        "RQ1 training requires exactly one ModelSaveCallback",
    )
