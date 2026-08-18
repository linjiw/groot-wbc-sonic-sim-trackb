"""Fail-closed planning and aggregation for the SONIC-Lite env-count sweep.

The module is CPU-only: importing it never imports Isaac Lab or PyTorch.  GPU
execution is an explicit operation in ``scripts/research/run_lace_throughput.py``.
Each environment count is an isolated process and a fresh output directory;
the default operation only writes a hash-bound plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
from typing import Any
import uuid

from gear_sonic.research.lace.sonic_lite import audit_lite_s_profile

PROTOCOL_KIND = "lace_sonic_lite_throughput_protocol"
PLAN_KIND = "lace_sonic_lite_throughput_plan"
REPORT_KIND = "lace_sonic_lite_throughput_report"
SUBSET_KIND = "lace_partition_directory"
EXPECTED_ENV_COUNTS = (128, 256, 512, 1024)
PLAN_SCHEMA_VERSION = 1
METRICS_FILENAME = "throughput_metrics.jsonl"
LAUNCH_RESULT_FILENAME = "launch_result.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_OOM_PATTERNS = (
    "cuda out of memory",
    "out of memory",
    "cudaerrormemoryallocation",
    "cuda_error_out_of_memory",
)
_NUMERICAL_PATTERNS = (
    "nan in gradient",
    "nan/inf grad",
    "non-finite",
    "floatingpointerror",
)
_SIMULATOR_PATTERNS = (
    "physx error",
    "physx fatal",
    "simulation failed",
    "failed to create simulation",
)


class ThroughputProtocolError(ValueError):
    """Raised when a frozen protocol, plan, or result violates its contract."""


class ThroughputLaunchError(RuntimeError):
    """Raised when a benchmark cell cannot be launched safely."""


def expected_optimizer_schedule(
    *,
    num_ppo_epochs: int,
    num_mini_batches: int,
    num_micro_batches: int,
    gradient_accumulation_steps: int,
) -> dict[str, int]:
    """Resolve attempted calls and synchronized updates for one PPO iteration."""

    values = {
        "num_ppo_epochs": num_ppo_epochs,
        "num_mini_batches": num_mini_batches,
        "num_micro_batches": num_micro_batches,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    for label, value in values.items():
        _require_positive_int(value, label=label)
    attempts = num_ppo_epochs * num_mini_batches * num_micro_batches
    if attempts % gradient_accumulation_steps:
        raise ThroughputProtocolError(
            "optimizer attempts must divide exactly by gradient_accumulation_steps"
        )
    return {
        "optimizer_expected_attempts": attempts,
        "synchronized_expected_updates": attempts // gradient_accumulation_steps,
    }


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using one canonical encoding."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def resident_motion_identity(motion_keys: Sequence[str]) -> dict[str, Any]:
    """Return the count/order/set identity for a resident motion-key sequence."""

    keys = list(motion_keys)
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ThroughputProtocolError("resident motion keys must be non-empty strings")
    return {
        "resident_motion_count": len(keys),
        "resident_unique_motion_count": len(set(keys)),
        "resident_motion_order_sha256": canonical_sha256(keys),
        "resident_motion_set_sha256": canonical_sha256(sorted(set(keys))),
    }


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ThroughputProtocolError(f"required file is missing: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ThroughputProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object while rejecting duplicate keys and non-finite values."""

    candidate = Path(path)
    try:
        value = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_json_pairs_no_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ThroughputProtocolError(f"non-finite JSON constant: {token}")
            ),
        )
    except OSError as error:
        raise ThroughputProtocolError(f"cannot read JSON file: {candidate}") from error
    except json.JSONDecodeError as error:
        raise ThroughputProtocolError(f"invalid JSON file: {candidate}: {error}") from error
    if not isinstance(value, dict):
        raise ThroughputProtocolError(f"JSON root must be an object: {candidate}")
    return value


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ThroughputProtocolError(f"{label} must be an object")
    return value


def _require_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ThroughputProtocolError(f"{label} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ThroughputProtocolError(f"{label} must be a non-negative integer")
    return value


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ThroughputProtocolError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _resolved(path: str | Path, *, base: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _source_tree(repo_root: Path) -> list[dict[str, Any]]:
    """Bind every YAML fragment that can participate in Hydra composition."""

    config_root = repo_root / "gear_sonic/config"
    paths = sorted(config_root.rglob("*.yaml"))
    if not paths:
        raise ThroughputProtocolError(f"no Hydra YAML files found beneath {config_root}")
    return [
        {
            "path": str(path.relative_to(repo_root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]


def _python_source_tree(repo_root: Path) -> list[dict[str, Any]]:
    """Bind the executable repository-owned Python closure conservatively."""

    paths = list((repo_root / "gear_sonic").rglob("*.py"))
    paths.append(repo_root / "scripts/research/run_lace_throughput.py")
    unique_paths = sorted(set(path.resolve() for path in paths))
    if not unique_paths or any(not path.is_file() for path in unique_paths):
        raise ThroughputProtocolError("executable Python source closure is incomplete")
    return [
        {
            "path": str(path.relative_to(repo_root)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in unique_paths
    ]


def _runtime_versions(python_executable: Path) -> dict[str, Any]:
    """Query package metadata from the exact interpreter used by cells."""

    probe = """
import importlib.metadata
import json
import platform
import sys
names = ('accelerate', 'hydra-core', 'numpy', 'torch', 'transformers', 'trl')
packages = {}
for name in names:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
print(json.dumps({
    'python': platform.python_version(),
    'implementation': platform.python_implementation(),
    'platform': platform.platform(),
    'executable': sys.executable,
    'packages': packages,
}, sort_keys=True))
"""
    try:
        result = subprocess.run(  # noqa: S603
            [str(python_executable), "-c", probe],
            capture_output=True,
            text=True,
            check=True,
        )
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise ThroughputProtocolError(
            f"cannot query runtime package versions from {python_executable}"
        ) from error
    if not isinstance(value, dict) or Path(str(value.get("executable"))).resolve() != (
        python_executable.resolve()
    ):
        raise ThroughputProtocolError("runtime version probe used an unexpected interpreter")
    return value


def build_partition_subset_manifest(
    split_manifest_path: str | Path,
    *,
    partition: str,
) -> dict[str, Any]:
    """Build the immutable inventory for one source-disjoint split partition."""

    split_path = Path(split_manifest_path).resolve()
    split = load_json_object(split_path)
    from gear_sonic.research.lace.schema import validate_split_manifest

    try:
        validate_split_manifest(split, verify_digest=True)
    except ValueError as error:
        raise ThroughputProtocolError(f"invalid source-disjoint split: {error}") from error
    motions = split.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ThroughputProtocolError("split manifest must contain non-empty motions")
    selected = [record for record in motions if record.get("partition") == partition]
    if not selected:
        raise ThroughputProtocolError(f"split contains no motions for {partition}")
    records: list[dict[str, Any]] = []
    keys: set[str] = set()
    for raw in sorted(selected, key=lambda value: str(value.get("motion_key"))):
        if not isinstance(raw, Mapping):
            raise ThroughputProtocolError("split motion records must be objects")
        key = raw.get("motion_key")
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise ThroughputProtocolError(f"unsafe motion key: {key!r}")
        if key in keys:
            raise ThroughputProtocolError(f"duplicate motion key: {key}")
        keys.add(key)
        record: dict[str, Any] = {"motion_key": key}
        for modality, field in (("robot", "robot_path"), ("smpl", "smpl_path")):
            source = Path(str(raw.get(field, ""))).resolve()
            if not source.is_file() or source.suffix != ".pkl":
                raise ThroughputProtocolError(f"{key}: invalid {modality} source: {source}")
            if source.stem != key:
                raise ThroughputProtocolError(f"{key}: {modality} source stem differs")
            record[modality] = {
                "source_path": str(source),
                "bytes": source.stat().st_size,
                "sha256": file_sha256(source),
            }
        records.append(record)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": SUBSET_KIND,
        "partition": partition,
        "split_manifest": str(split_path),
        "split_manifest_file_sha256": file_sha256(split_path),
        "split_sha256": split.get("split_sha256"),
        "selection_sha256": split.get("selection_sha256"),
        "motion_count": len(records),
        "motions": records,
    }
    manifest["subset_sha256"] = canonical_sha256(manifest)
    return manifest


def materialize_partition_subset(
    split_manifest_path: str | Path,
    *,
    partition: str,
    destination: str | Path,
) -> dict[str, Any]:
    """Atomically materialize a paired subset as absolute symlinks.

    Existing destinations are never reused or overwritten.  The immutable
    manifest records source bytes and content digests so launch preflight can
    detect source or symlink drift.
    """

    target = Path(destination).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"subset destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    manifest = build_partition_subset_manifest(split_manifest_path, partition=partition)
    try:
        (stage / "robot_filtered").mkdir(parents=True)
        (stage / "smpl_filtered").mkdir()
        for record in manifest["motions"]:
            key = record["motion_key"]
            for modality, directory in (("robot", "robot_filtered"), ("smpl", "smpl_filtered")):
                destination_path = stage / directory / f"{key}.pkl"
                destination_path.symlink_to(record[modality]["source_path"])
        manifest_path = stage / "subset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def verify_partition_subset(
    subset_root: str | Path,
    *,
    split_manifest_path: str | Path,
    partition: str,
) -> dict[str, Any]:
    """Verify exact paired inventories, source targets, bytes, and hashes."""

    root = Path(subset_root).resolve()
    manifest_path = root / "subset_manifest.json"
    observed = load_json_object(manifest_path)
    stored_digest = observed.get("subset_sha256")
    unsigned = dict(observed)
    unsigned.pop("subset_sha256", None)
    if stored_digest != canonical_sha256(unsigned):
        raise ThroughputProtocolError("subset manifest self-hash mismatch")
    expected = build_partition_subset_manifest(split_manifest_path, partition=partition)
    if observed != expected:
        raise ThroughputProtocolError("materialized subset manifest differs from frozen split")

    expected_names = {f"{record['motion_key']}.pkl" for record in expected["motions"]}
    for modality, directory in (("robot", "robot_filtered"), ("smpl", "smpl_filtered")):
        inventory = root / directory
        if not inventory.is_dir():
            raise ThroughputProtocolError(f"missing subset directory: {inventory}")
        observed_names = {path.name for path in inventory.iterdir()}
        if observed_names != expected_names:
            raise ThroughputProtocolError(f"{modality} subset inventory mismatch")
        for record in expected["motions"]:
            path = inventory / f"{record['motion_key']}.pkl"
            if not path.is_file():
                raise ThroughputProtocolError(f"subset entry is not a file: {path}")
            source = Path(record[modality]["source_path"])
            if path.resolve() != source.resolve():
                raise ThroughputProtocolError(f"subset entry target drifted: {path}")
            if path.stat().st_size != record[modality]["bytes"]:
                raise ThroughputProtocolError(f"subset entry byte count drifted: {path}")
            if file_sha256(path) != record[modality]["sha256"]:
                raise ThroughputProtocolError(f"subset entry digest drifted: {path}")
    resident_motion_keys = [str(record["motion_key"]) for record in expected["motions"]]
    resident_identity = resident_motion_identity(resident_motion_keys)
    return {
        "root": str(root),
        "manifest": str(manifest_path),
        "manifest_file_sha256": file_sha256(manifest_path),
        "subset_manifest_file_sha256": file_sha256(manifest_path),
        "subset_sha256": stored_digest,
        "split_manifest_file_sha256": expected["split_manifest_file_sha256"],
        "split_sha256": expected["split_sha256"],
        "selection_sha256": expected["selection_sha256"],
        "motion_count": expected["motion_count"],
        **resident_identity,
        "resident_motion_keys": resident_motion_keys,
        "partition": partition,
        "split_manifest": str(Path(split_manifest_path).resolve()),
        "robot_dir": str(root / "robot_filtered"),
        "smpl_dir": str(root / "smpl_filtered"),
    }


def _validate_protocol(protocol: Mapping[str, Any], *, protocol_path: Path) -> dict[str, Any]:
    if protocol.get("schema_version") != 1 or protocol.get("kind") != PROTOCOL_KIND:
        raise ThroughputProtocolError("unsupported throughput protocol schema/kind")
    if tuple(protocol.get("env_counts", ())) != EXPECTED_ENV_COUNTS:
        raise ThroughputProtocolError(f"env_counts must be exactly {list(EXPECTED_ENV_COUNTS)}")
    benchmark = _require_mapping(protocol.get("benchmark"), label="benchmark")
    warmup = _require_positive_int(
        benchmark.get("warmup_iterations"), label="benchmark.warmup_iterations"
    )
    timed = _require_positive_int(
        benchmark.get("timed_iterations"), label="benchmark.timed_iterations"
    )
    if timed < 20:
        raise ThroughputProtocolError("benchmark.timed_iterations must be at least 20")
    expected_schedule = {
        "num_steps_per_env": 24,
        "decimation": 4,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gradient_accumulation_steps": 1,
    }
    for key, expected in expected_schedule.items():
        if benchmark.get(key) != expected:
            raise ThroughputProtocolError(f"benchmark.{key} must be frozen to {expected}")

    selection = _require_mapping(protocol.get("selection_rule"), label="selection_rule")
    expected_selection = {
        "eligibility": "all_gates_pass",
        "metric": "sustained_end_to_end_control_transitions_per_second",
        "direction": "maximize",
        "tie_breaker": "smaller_num_envs",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            raise ThroughputProtocolError(f"selection_rule.{key} must be {expected!r}")
    safety_margin = _require_positive_int(
        selection.get("cuda_memory_safety_margin_mib"),
        label="selection_rule.cuda_memory_safety_margin_mib",
    )

    launch = _require_mapping(protocol.get("launch"), label="launch")
    gpu_index = _require_nonnegative_int(launch.get("gpu_index"), label="launch.gpu_index")
    max_external = _require_nonnegative_int(
        launch.get("max_external_gpu_memory_mib"),
        label="launch.max_external_gpu_memory_mib",
    )
    if max_external != 0:
        raise ThroughputProtocolError(
            "launch.max_external_gpu_memory_mib must be zero for isolated measurements"
        )
    min_free = _require_positive_int(
        launch.get("minimum_free_gpu_memory_mib"),
        label="launch.minimum_free_gpu_memory_mib",
    )
    timeout_seconds = _require_positive_int(
        launch.get("cell_timeout_seconds"), label="launch.cell_timeout_seconds"
    )
    poll = launch.get("monitor_poll_seconds")
    if isinstance(poll, bool) or not isinstance(poll, (int, float)) or not 0.25 <= poll <= 10.0:
        raise ThroughputProtocolError("launch.monitor_poll_seconds must be in [0.25, 10]")

    base = protocol_path.parent
    repo_root = _resolved(protocol.get("repo_root", "."), base=base)
    storage_root = _resolved(protocol.get("storage_root", "."), base=base)
    output_root = _resolved(protocol.get("output_root", "."), base=base)
    tmp_root = _resolved(protocol.get("tmp_root", "."), base=base)
    if not repo_root.is_dir():
        raise ThroughputProtocolError(f"repo_root is not a directory: {repo_root}")
    if not storage_root.is_dir():
        raise ThroughputProtocolError(f"storage_root is not a directory: {storage_root}")
    for label, path in (("output_root", output_root), ("tmp_root", tmp_root)):
        if not _is_within(path, storage_root):
            raise ThroughputProtocolError(f"{label} must be beneath storage_root")
    if output_root == storage_root or tmp_root == storage_root:
        raise ThroughputProtocolError("output_root/tmp_root cannot equal storage_root")

    exp = protocol.get("experiment")
    if exp != "manager/universal_token/g1_only/lace_lite_s":
        raise ThroughputProtocolError("experiment must be the frozen Lite-S profile")
    seed = _require_nonnegative_int(protocol.get("seed"), label="seed")
    dataset = _require_mapping(protocol.get("dataset"), label="dataset")
    partition = dataset.get("partition")
    if partition != "D_curriculum":
        raise ThroughputProtocolError("throughput data partition must be D_curriculum")
    split_path = _resolved(dataset.get("split_manifest", ""), base=base)
    subset_root = _resolved(dataset.get("subset_root", ""), base=base)
    if not _is_within(subset_root, storage_root):
        raise ThroughputProtocolError("dataset.subset_root must be beneath storage_root")
    expected_dataset_identity = dataset.get("expected_identity")
    if expected_dataset_identity is not None:
        expected_dataset_identity = _require_mapping(
            expected_dataset_identity,
            label="dataset.expected_identity",
        )
        expected_keys = {
            "split_manifest_file_sha256",
            "split_sha256",
            "selection_sha256",
            "subset_manifest_file_sha256",
            "subset_sha256",
            "motion_count",
            "resident_motion_count",
            "resident_unique_motion_count",
            "resident_motion_order_sha256",
            "resident_motion_set_sha256",
        }
        if set(expected_dataset_identity) != expected_keys:
            raise ThroughputProtocolError(
                "dataset.expected_identity must contain exactly the frozen identity fields"
            )
        for key in expected_keys - {
            "motion_count",
            "resident_motion_count",
            "resident_unique_motion_count",
        }:
            _require_sha256(
                expected_dataset_identity.get(key),
                label=f"dataset.expected_identity.{key}",
            )
        for key in (
            "motion_count",
            "resident_motion_count",
            "resident_unique_motion_count",
        ):
            _require_positive_int(
                expected_dataset_identity.get(key),
                label=f"dataset.expected_identity.{key}",
            )
    workload = _require_mapping(protocol.get("workload"), label="workload")
    _require_positive_int(
        workload.get("resident_motion_count"), label="workload.resident_motion_count"
    )
    expected_workload = {
        "resident_motion_order": "lexicographic_motion_key",
        "all_motions_loaded_required": True,
        "terrain_type": "plane",
    }
    for key, expected in expected_workload.items():
        if workload.get(key) != expected:
            raise ThroughputProtocolError(f"workload.{key} must be {expected!r}")

    checkpoint = _require_mapping(protocol.get("checkpoint"), label="checkpoint")
    checkpoint_path = _resolved(checkpoint.get("path", ""), base=base)
    checkpoint_config = _resolved(checkpoint.get("config_path", ""), base=base)
    if checkpoint_path.name != "last.pt" or checkpoint_config.name != "config.yaml":
        raise ThroughputProtocolError(
            "checkpoint paths must end in one Lite-S bundle's last.pt and config.yaml"
        )
    if checkpoint_path.parent != checkpoint_config.parent:
        raise ThroughputProtocolError("checkpoint and config must share one bundle directory")
    if (
        not _is_within(checkpoint_path.parent, storage_root)
        or checkpoint_path.parent == storage_root
    ):
        raise ThroughputProtocolError(
            "checkpoint initialization bundle must be strictly beneath storage_root"
        )
    python_executable = _resolved(protocol.get("python_executable", ""), base=base)
    train_entrypoint = _resolved(protocol.get("train_entrypoint", ""), base=repo_root)
    run_id = protocol.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", run_id):
        raise ThroughputProtocolError("run_id must be a filesystem-safe lowercase identifier")

    return {
        "warmup_iterations": warmup,
        "timed_iterations": timed,
        "total_iterations": warmup + timed,
        "safety_margin_mib": safety_margin,
        "gpu_index": gpu_index,
        "max_external_mib": max_external,
        "min_free_mib": min_free,
        "timeout_seconds": timeout_seconds,
        "poll_seconds": float(poll),
        "repo_root": repo_root,
        "storage_root": storage_root,
        "output_root": output_root / run_id,
        "tmp_root": tmp_root / run_id,
        "experiment": exp,
        "seed": seed,
        "partition": partition,
        "workload": dict(workload),
        "split_path": split_path,
        "subset_root": subset_root,
        "expected_dataset_identity": (
            None if expected_dataset_identity is None else dict(expected_dataset_identity)
        ),
        "checkpoint_path": checkpoint_path,
        "checkpoint_config": checkpoint_config,
        "initialization_dir": checkpoint_path.parent,
        "python_executable": python_executable,
        "train_entrypoint": train_entrypoint,
        "run_id": run_id,
        "benchmark": dict(benchmark),
        "selection_rule": dict(selection),
    }


def resolve_materialization_targets(
    protocol: Mapping[str, Any], *, protocol_path: str | Path
) -> dict[str, Any]:
    """Validate a protocol before exposing its subset/init write targets."""

    values = _validate_protocol(protocol, protocol_path=Path(protocol_path).resolve())
    return {
        "repo_root": values["repo_root"],
        "split_path": values["split_path"],
        "partition": values["partition"],
        "subset_root": values["subset_root"],
        "initialization_dir": values["initialization_dir"],
        "seed": values["seed"],
    }


def _repo_first_pythonpath(repo_root: Path) -> str:
    entries = [str(repo_root)]
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    return os.pathsep.join(entries)


def _command_for_cell(
    *,
    values: Mapping[str, Any],
    contract_sha256: str,
    cell_contract_sha256: str,
    cell_id: str,
    num_envs: int,
    output_dir: Path,
    dataset: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    benchmark = values["benchmark"]
    total_iterations = int(values["total_iterations"])
    cell_tmp = values["tmp_root"] / cell_id
    environment = {
        "CUDA_CACHE_PATH": str(cell_tmp / "cuda-cache"),
        "CUDA_VISIBLE_DEVICES": str(values["gpu_index"]),
        "HYDRA_FULL_ERROR": "1",
        "ISAACLAB_USD_CACHE_DIR": str(cell_tmp / "isaaclab-usd-cache"),
        "LOGURU_LEVEL": "INFO",
        "PYTHONPATH": _repo_first_pythonpath(values["repo_root"]),
        "TMPDIR": str(cell_tmp),
        "TORCH_EXTENSIONS_DIR": str(cell_tmp / "torch-extensions"),
        "WANDB_MODE": "disabled",
        "XDG_CACHE_HOME": str(cell_tmp / "xdg-cache"),
    }
    argv = [
        str(values["python_executable"]),
        str(values["train_entrypoint"]),
        f"+exp={values['experiment']}",
        "+callbacks=throughput_benchmark",
        f"checkpoint={values['checkpoint_path']}",
        "+resume=false",
        f"num_envs={num_envs}",
        f"++manager_env.config.decimation={benchmark['decimation']}",
        f"++manager_env.config.terrain_type={values['workload']['terrain_type']}",
        "headless=true",
        "use_wandb=false",
        f"seed={values['seed']}",
        f"exp_var=throughput_{cell_id}",
        f"experiment_dir={output_dir}",
        f"++algo.config.num_learning_iterations={total_iterations}",
        f"++algo.config.num_steps_per_env={benchmark['num_steps_per_env']}",
        f"++algo.config.num_learning_epochs={benchmark['num_learning_epochs']}",
        f"++algo.config.num_mini_batches={benchmark['num_mini_batches']}",
        "++algo.config.throughput_benchmark_metrics=true",
        f"++algo.trl.gradient_accumulation_steps={benchmark['gradient_accumulation_steps']}",
        f"++algo.trl.per_device_train_batch_size={num_envs // int(benchmark['num_mini_batches'])}",
        "++algo.config.save_interval=1000000000",
        "++callbacks.model_save.save_frequency=1000000000",
        "++callbacks.model_save.save_last_frequency=1000000000",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={dataset['robot_dir']}",
        f"++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file={dataset['smpl_dir']}",
        "++manager_env.commands.motion.motion_lib_cfg."
        f"override_num_motions_to_load={dataset['resident_motion_count']}",
        "++manager_env.commands.motion.motion_lib_cfg.sort_motion_keys=true",
        "++manager_env.commands.motion.motion_lib_cfg.adaptive_sampling.enable=false",
        f"++throughput_benchmark.contract_sha256={contract_sha256}",
        f"++throughput_benchmark.cell_contract_sha256={cell_contract_sha256}",
        f"++throughput_benchmark.cell_id={cell_id}",
        f"++throughput_benchmark.warmup_iterations={values['warmup_iterations']}",
        f"++throughput_benchmark.timed_iterations={values['timed_iterations']}",
        f"++throughput_benchmark.dataset_subset_sha256={dataset['subset_sha256']}",
        "++throughput_benchmark.expected_resident_motion_count="
        f"{dataset['resident_motion_count']}",
        "++throughput_benchmark.expected_resident_motion_order_sha256="
        f"{dataset['resident_motion_order_sha256']}",
        "++throughput_benchmark.expected_resident_motion_set_sha256="
        f"{dataset['resident_motion_set_sha256']}",
        f"++throughput_benchmark.expected_terrain_type={values['workload']['terrain_type']}",
    ]
    return environment, argv


def build_throughput_plan(
    protocol: Mapping[str, Any],
    *,
    protocol_path: str | Path,
) -> dict[str, Any]:
    """Build a hash-bound plan; missing initialization files make it unlaunchable."""

    path = Path(protocol_path).resolve()
    values = _validate_protocol(protocol, protocol_path=path)
    if not values["python_executable"].is_file():
        raise ThroughputProtocolError(
            f"python_executable is not a file: {values['python_executable']}"
        )
    if not values["train_entrypoint"].is_file():
        raise ThroughputProtocolError(
            f"train_entrypoint is not a file: {values['train_entrypoint']}"
        )
    dataset = verify_partition_subset(
        values["subset_root"],
        split_manifest_path=values["split_path"],
        partition=values["partition"],
    )
    expected_dataset_identity = values["expected_dataset_identity"]
    if expected_dataset_identity is not None:
        observed_dataset_identity = {key: dataset[key] for key in expected_dataset_identity}
        if observed_dataset_identity != expected_dataset_identity:
            raise ThroughputProtocolError(
                "verified data subset differs from dataset.expected_identity"
            )
    if dataset["resident_motion_count"] != values["workload"]["resident_motion_count"]:
        raise ThroughputProtocolError(
            "verified resident motion count differs from the frozen workload"
        )
    profile = audit_lite_s_profile(values["repo_root"]).to_dict()
    source_tree = _source_tree(values["repo_root"])
    python_source_tree = _python_source_tree(values["repo_root"])
    runtime_versions = _runtime_versions(values["python_executable"])

    readiness_errors: list[str] = []
    receipt_path = values["initialization_dir"] / "initialization_receipt.json"
    checkpoint_record: dict[str, Any] = {
        "path": str(values["checkpoint_path"]),
        "config_path": str(values["checkpoint_config"]),
        "initialization_receipt_path": str(receipt_path),
    }
    for label, checkpoint_path in (
        ("checkpoint", values["checkpoint_path"]),
        ("checkpoint config", values["checkpoint_config"]),
        ("initialization receipt", receipt_path),
    ):
        if not checkpoint_path.is_file():
            readiness_errors.append(f"missing {label}: {checkpoint_path}")
        else:
            checkpoint_record[f"{label.replace(' ', '_')}_bytes"] = checkpoint_path.stat().st_size
            checkpoint_record[f"{label.replace(' ', '_')}_sha256"] = file_sha256(checkpoint_path)

    if not readiness_errors:
        from gear_sonic.research.lace.lite_init import (
            LiteInitializationError,
            verify_lite_initialization,
        )

        try:
            receipt = verify_lite_initialization(
                values["initialization_dir"],
                repo_root=values["repo_root"],
                expected_seed=values["seed"],
            )
        except (LiteInitializationError, OSError, ValueError) as error:
            readiness_errors.append(f"invalid Lite-S initialization: {error}")
        else:
            checkpoint_record["initialization"] = {
                "receipt_sha256": receipt["receipt_sha256"],
                "model_spec_sha256": receipt["model_spec_sha256"],
                "policy_state_sha256": receipt["policy"]["state_sha256"],
                "value_state_sha256": receipt["value"]["state_sha256"],
                "profile_config_sha256": receipt["profile_config_sha256"],
                "construction_order": receipt["construction_order"],
                "environment_constructed": receipt["environment_constructed"],
                "seed": receipt["seed"],
            }

    protocol_contract = {
        "protocol_file_sha256": file_sha256(path),
        "protocol_canonical_sha256": canonical_sha256(protocol),
        "profile": profile,
        "hydra_config_sources": source_tree,
        "executable_python_sources": python_source_tree,
        "python_executable": {
            "path": str(values["python_executable"]),
            "bytes": values["python_executable"].stat().st_size,
            "sha256": file_sha256(values["python_executable"]),
        },
        "runtime_versions": runtime_versions,
        "dataset": dataset,
        "workload": values["workload"],
        "checkpoint": checkpoint_record,
        "experiment": values["experiment"],
        "seed": values["seed"],
        "benchmark": values["benchmark"],
        "selection_rule": values["selection_rule"],
        "env_counts": list(EXPECTED_ENV_COUNTS),
    }
    benchmark_contract_sha256 = canonical_sha256(protocol_contract)
    cells: list[dict[str, Any]] = []
    for num_envs in EXPECTED_ENV_COUNTS:
        cell_id = f"env_{num_envs:04d}"
        output_dir = values["output_root"] / cell_id
        local_mini_batch_size = num_envs // int(values["benchmark"]["num_mini_batches"])
        cell_training_contract = {
            **protocol_contract,
            "num_envs": num_envs,
            "local_batch_size": num_envs,
            "local_mini_batch_size": local_mini_batch_size,
            "per_device_train_batch_size": local_mini_batch_size,
            "num_micro_batches": 1,
            "optimizer_schedule": expected_optimizer_schedule(
                num_ppo_epochs=int(values["benchmark"]["num_learning_epochs"]),
                num_mini_batches=int(values["benchmark"]["num_mini_batches"]),
                num_micro_batches=1,
                gradient_accumulation_steps=int(values["benchmark"]["gradient_accumulation_steps"]),
            ),
            "output_dir": str(output_dir),
        }
        cell_contract_sha256 = canonical_sha256(cell_training_contract)
        environment, argv = _command_for_cell(
            values=values,
            contract_sha256=benchmark_contract_sha256,
            cell_contract_sha256=cell_contract_sha256,
            cell_id=cell_id,
            num_envs=num_envs,
            output_dir=output_dir,
            dataset=dataset,
        )
        cells.append(
            {
                "cell_id": cell_id,
                "num_envs": num_envs,
                "output_dir": str(output_dir),
                "metrics_jsonl": str(output_dir / METRICS_FILENAME),
                "process_log": str(output_dir / "process.log"),
                "launch_result": str(output_dir / LAUNCH_RESULT_FILENAME),
                "cell_training_contract": cell_training_contract,
                "cell_contract_sha256": cell_contract_sha256,
                "launch_environment": environment,
                "argv": argv,
                "command_sha256": canonical_sha256({"environment": environment, "argv": argv}),
            }
        )
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "run_id": values["run_id"],
        "protocol_path": str(path),
        "repo_root": str(values["repo_root"]),
        "storage_root": str(values["storage_root"]),
        "output_root": str(values["output_root"]),
        "tmp_root": str(values["tmp_root"]),
        "benchmark_contract": protocol_contract,
        "benchmark_contract_sha256": benchmark_contract_sha256,
        "launch_ready": not readiness_errors,
        "readiness_errors": readiness_errors,
        "launch_policy": {
            "default": "plan_only",
            "explicit_single_cell_only": True,
            "order": "strictly_ascending_num_envs",
            "larger_cells_after_oom": "forbidden",
            "resume": "forbidden",
            "overwrite": "forbidden",
            "gpu_index": values["gpu_index"],
            "max_external_gpu_memory_mib": values["max_external_mib"],
            "minimum_free_gpu_memory_mib": values["min_free_mib"],
            "cell_timeout_seconds": values["timeout_seconds"],
            "monitor_poll_seconds": values["poll_seconds"],
        },
        "cells": cells,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate plan self-hash, contracts, cell order, and command hashes."""

    if plan.get("schema_version") != PLAN_SCHEMA_VERSION or plan.get("kind") != PLAN_KIND:
        raise ThroughputProtocolError("unsupported throughput plan")
    unsigned = dict(plan)
    stored_plan_sha = unsigned.pop("plan_sha256", None)
    if stored_plan_sha != canonical_sha256(unsigned):
        raise ThroughputProtocolError("throughput plan self-hash mismatch")
    contract = _require_mapping(plan.get("benchmark_contract"), label="benchmark_contract")
    if plan.get("benchmark_contract_sha256") != canonical_sha256(contract):
        raise ThroughputProtocolError("benchmark contract hash mismatch")
    cells = plan.get("cells")
    if not isinstance(cells, list) or [cell.get("num_envs") for cell in cells] != list(
        EXPECTED_ENV_COUNTS
    ):
        raise ThroughputProtocolError("plan cells must exactly follow the frozen env-count order")
    for cell in cells:
        contract_value = _require_mapping(
            cell.get("cell_training_contract"), label="cell_training_contract"
        )
        if cell.get("cell_contract_sha256") != canonical_sha256(contract_value):
            raise ThroughputProtocolError(f"{cell.get('cell_id')}: cell contract hash mismatch")
        command = {"environment": cell.get("launch_environment"), "argv": cell.get("argv")}
        if cell.get("command_sha256") != canonical_sha256(command):
            raise ThroughputProtocolError(f"{cell.get('cell_id')}: command hash mismatch")


def write_new_json(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create a canonical artifact while refusing resume/overwrite."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.parent / (f".{destination.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is an atomic no-replace publication.  An
        # existing destination fails with EEXIST instead of being overwritten.
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


def _parse_gpu_inventory(text: str, *, gpu_index: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not line.strip():
            continue
        if len(parts) != 11:
            raise ThroughputLaunchError(f"unexpected nvidia-smi GPU row: {line!r}")
        try:
            row = {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "driver_version": parts[3],
                "total_mib": int(parts[4]),
                "used_mib": int(parts[5]),
                "free_mib": int(parts[6]),
                "pstate": parts[7],
                "temperature_c": None if parts[8] == "N/A" else int(parts[8]),
                "graphics_clock_mhz": None if parts[9] == "N/A" else int(parts[9]),
                "memory_clock_mhz": None if parts[10] == "N/A" else int(parts[10]),
            }
        except ValueError as error:
            raise ThroughputLaunchError(f"non-numeric nvidia-smi GPU row: {line!r}") from error
        rows.append(row)
    matches = [row for row in rows if row["index"] == gpu_index]
    if len(matches) != 1:
        raise ThroughputLaunchError(f"GPU index {gpu_index} was not reported exactly once")
    return matches[0]


def _parse_compute_apps(text: str, *, gpu_uuid: str) -> list[dict[str, int]]:
    apps: list[dict[str, int]] = []
    for line in text.splitlines():
        if not line.strip() or line.strip() == "No running processes found":
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            raise ThroughputLaunchError(f"unexpected nvidia-smi process row: {line!r}")
        if parts[0] != gpu_uuid:
            continue
        try:
            apps.append({"pid": int(parts[1]), "used_memory_mib": int(parts[2])})
        except ValueError as error:
            raise ThroughputLaunchError(f"non-numeric nvidia-smi process row: {line!r}") from error
    return sorted(apps, key=lambda value: value["pid"])


def query_gpu_preflight(
    gpu_index: int,
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Return physical GPU inventory and external compute-process usage."""

    gpu_result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,"
            "memory.free,pstate,temperature.gpu,clocks.current.graphics,"
            "clocks.current.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    gpu = _parse_gpu_inventory(gpu_result.stdout, gpu_index=gpu_index)
    process_result = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    apps = _parse_compute_apps(process_result.stdout, gpu_uuid=gpu["uuid"])
    return {
        **gpu,
        "compute_processes": apps,
        "external_compute_memory_mib": sum(app["used_memory_mib"] for app in apps),
    }


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _process_tree_snapshot(root_pid: int) -> dict[str, Any]:
    """Return current descendant PIDs and aggregate RSS for one child tree."""

    parent_by_pid: dict[int, int] = {}
    rss_by_pid: dict[int, int] = {}
    for status_path in Path("/proc").glob("[0-9]*/status"):
        try:
            pid = int(status_path.parent.name)
            parent = None
            rss = 0
            for line in status_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("PPid:"):
                    parent = int(line.split()[1])
                elif line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
            if parent is not None:
                parent_by_pid[pid] = parent
                rss_by_pid[pid] = rss
        except (FileNotFoundError, OSError, IndexError, ValueError):
            continue
    tree = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parent_by_pid.items():
            if parent in tree and pid not in tree:
                tree.add(pid)
                changed = True
    live = sorted(pid for pid in tree if pid in parent_by_pid or pid == root_pid)
    return {
        "pids": live,
        "process_count": len(live),
        "rss_bytes": sum(rss_by_pid.get(pid, 0) for pid in live),
    }


def _pid_in_process_group(pid: int, process_group_id: int) -> bool:
    try:
        return os.getpgid(pid) == process_group_id
    except OSError:
        return False


def _finite_range(values: Sequence[int | None]) -> list[int] | None:
    present = [int(value) for value in values if value is not None]
    return [min(present), max(present)] if present else None


def _classify_failure(returncode: int, log_text: str, *, timed_out: bool) -> str | None:
    lowered = log_text.lower()
    if timed_out:
        return "timeout"
    if any(pattern in lowered for pattern in _OOM_PATTERNS):
        return "oom"
    if any(pattern in lowered for pattern in _NUMERICAL_PATTERNS):
        return "numerical"
    if any(pattern in lowered for pattern in _SIMULATOR_PATTERNS):
        return "simulator"
    if returncode != 0:
        return "process_error"
    return None


def _verify_bound_files(plan: Mapping[str, Any]) -> None:
    # A valid self-hash alone is not authoritative: an editor could alter an
    # outer path/command and recompute it.  Rebuild from the byte-bound protocol
    # and require the complete plan (including every command) to be identical.
    try:
        protocol = load_json_object(plan["protocol_path"])
        rebuilt = build_throughput_plan(protocol, protocol_path=plan["protocol_path"])
    except (ThroughputProtocolError, OSError, ValueError) as error:
        raise ThroughputLaunchError(f"cannot rebuild bound throughput plan: {error}") from error
    if dict(plan) != rebuilt:
        raise ThroughputLaunchError("throughput plan differs from a fresh protocol rebuild")

    contract = _require_mapping(plan["benchmark_contract"], label="benchmark_contract")
    protocol_path = Path(str(plan["protocol_path"]))
    if file_sha256(protocol_path) != contract["protocol_file_sha256"]:
        raise ThroughputLaunchError("protocol bytes changed after plan creation")
    checkpoint = _require_mapping(contract["checkpoint"], label="checkpoint")
    for prefix in ("checkpoint", "checkpoint_config", "initialization_receipt"):
        path_key = {
            "checkpoint": "path",
            "checkpoint_config": "config_path",
            "initialization_receipt": "initialization_receipt_path",
        }[prefix]
        digest_key = f"{prefix}_sha256"
        if digest_key not in checkpoint:
            raise ThroughputLaunchError("plan has no complete initialization checkpoint binding")
        if file_sha256(Path(str(checkpoint[path_key]))) != checkpoint[digest_key]:
            raise ThroughputLaunchError(f"{prefix} bytes changed after plan creation")
    initialization = _require_mapping(
        checkpoint.get("initialization"), label="checkpoint.initialization"
    )
    from gear_sonic.research.lace.lite_init import (
        LiteInitializationError,
        verify_lite_initialization,
    )

    try:
        receipt = verify_lite_initialization(
            Path(str(checkpoint["path"])).parent,
            repo_root=plan["repo_root"],
            expected_seed=initialization["seed"],
        )
    except (LiteInitializationError, OSError, ValueError) as error:
        raise ThroughputLaunchError(f"Lite-S initialization no longer verifies: {error}") from error
    expected_initialization = {
        "receipt_sha256": receipt["receipt_sha256"],
        "model_spec_sha256": receipt["model_spec_sha256"],
        "policy_state_sha256": receipt["policy"]["state_sha256"],
        "value_state_sha256": receipt["value"]["state_sha256"],
        "profile_config_sha256": receipt["profile_config_sha256"],
        "construction_order": receipt["construction_order"],
        "environment_constructed": receipt["environment_constructed"],
        "seed": receipt["seed"],
    }
    if dict(initialization) != expected_initialization:
        raise ThroughputLaunchError("Lite-S initialization identity differs from plan")
    dataset = _require_mapping(contract["dataset"], label="dataset")
    if file_sha256(Path(str(dataset["manifest"]))) != dataset["manifest_file_sha256"]:
        raise ThroughputLaunchError("data subset manifest bytes changed after plan creation")
    try:
        verified_dataset = verify_partition_subset(
            dataset["root"],
            split_manifest_path=dataset["split_manifest"],
            partition=dataset["partition"],
        )
    except ThroughputProtocolError as error:
        raise ThroughputLaunchError(f"data subset no longer verifies: {error}") from error
    if dict(dataset) != verified_dataset:
        raise ThroughputLaunchError("data subset identity differs from plan")
    repo_root = Path(str(plan["repo_root"]))
    for record in contract["hydra_config_sources"]:
        if file_sha256(repo_root / record["path"]) != record["sha256"]:
            raise ThroughputLaunchError(
                f"Hydra config changed after plan creation: {record['path']}"
            )
    for record in contract["executable_python_sources"]:
        if file_sha256(repo_root / record["path"]) != record["sha256"]:
            raise ThroughputLaunchError(
                f"Python source changed after plan creation: {record['path']}"
            )
    python_executable = _require_mapping(contract["python_executable"], label="python_executable")
    if file_sha256(python_executable["path"]) != python_executable["sha256"]:
        raise ThroughputLaunchError("Python executable changed after plan creation")


def _cell_by_env(plan: Mapping[str, Any], num_envs: int) -> tuple[int, Mapping[str, Any]]:
    cells = plan["cells"]
    matches = [(index, cell) for index, cell in enumerate(cells) if cell["num_envs"] == num_envs]
    if len(matches) != 1:
        raise ThroughputLaunchError(f"plan does not contain exactly one N_env={num_envs} cell")
    return matches[0]


def _verify_launch_order(plan: Mapping[str, Any], cell_index: int) -> None:
    for previous in plan["cells"][:cell_index]:
        result_path = Path(previous["launch_result"])
        if not result_path.is_file():
            raise ThroughputLaunchError(
                f"strict ascending isolation requires completed {previous['cell_id']} first"
            )
        result = load_json_object(result_path)
        validate_launch_result(plan, previous, result)
        if result.get("failure_class") == "oom":
            raise ThroughputLaunchError(
                f"larger cells are forbidden after OOM in {previous['cell_id']}"
            )
        if result.get("returncode") != 0 or result.get("failure_class") is not None:
            raise ThroughputLaunchError(
                f"resolve failed predecessor {previous['cell_id']} before launching a larger cell"
            )


def launch_throughput_cell(plan: Mapping[str, Any], *, num_envs: int) -> dict[str, Any]:
    """Explicitly launch one isolated cell after all safety preflights pass."""

    validate_plan(plan)
    if not plan.get("launch_ready"):
        raise ThroughputLaunchError(
            "plan is not launch-ready: " + "; ".join(plan.get("readiness_errors", []))
        )
    _verify_bound_files(plan)
    cell_index, cell = _cell_by_env(plan, num_envs)
    _verify_launch_order(plan, cell_index)
    output_dir = Path(cell["output_dir"])
    tmpdir = Path(cell["launch_environment"]["TMPDIR"])
    if output_dir.exists() or output_dir.is_symlink():
        raise ThroughputLaunchError(
            f"cell output exists; resume/overwrite is forbidden: {output_dir}"
        )
    if tmpdir.exists() or tmpdir.is_symlink():
        raise ThroughputLaunchError(
            f"cell temporary directory exists; resume/overwrite is forbidden: {tmpdir}"
        )
    launch_policy = plan["launch_policy"]
    try:
        preflight = query_gpu_preflight(int(launch_policy["gpu_index"]))
    except (OSError, subprocess.CalledProcessError) as error:
        raise ThroughputLaunchError(f"cannot query GPU preflight: {error}") from error
    if preflight["external_compute_memory_mib"] > launch_policy["max_external_gpu_memory_mib"]:
        raise ThroughputLaunchError(
            "external GPU memory exceeds frozen limit: "
            f"{preflight['external_compute_memory_mib']} MiB > "
            f"{launch_policy['max_external_gpu_memory_mib']} MiB"
        )
    if preflight["free_mib"] < launch_policy["minimum_free_gpu_memory_mib"]:
        raise ThroughputLaunchError(
            f"GPU free memory {preflight['free_mib']} MiB is below frozen minimum "
            f"{launch_policy['minimum_free_gpu_memory_mib']} MiB"
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    tmpdir.mkdir(parents=True, exist_ok=False)
    process_log = Path(cell["process_log"])
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in cell["launch_environment"].items()})
    start = time.monotonic()
    peak_process_tree_rss = 0
    peak_process_tree_count = 0
    timed_out = False
    launch_error: str | None = None
    gpu_monitor_failure: str | None = None
    gpu_monitor_samples = 0
    minimum_gpu_free_mib = int(preflight["free_mib"])
    peak_gpu_used_mib = int(preflight["used_mib"])
    peak_unexpected_gpu_memory_mib = 0
    first_unexpected_gpu_processes: list[dict[str, int]] | None = None
    observed_pstates = {str(preflight["pstate"])}
    observed_temperatures = [preflight["temperature_c"]]
    observed_graphics_clocks = [preflight["graphics_clock_mhz"]]
    observed_memory_clocks = [preflight["memory_clock_mhz"]]
    gpu_identity_fields = ("index", "uuid", "name", "driver_version", "total_mib")
    gpu_identity = {field: preflight[field] for field in gpu_identity_fields}
    postflight: dict[str, Any] | None = None
    returncode = -1

    def terminate_process_group(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)

    def record_gpu_sample(snapshot: Mapping[str, Any], process_group_id: int) -> None:
        nonlocal gpu_monitor_samples
        nonlocal minimum_gpu_free_mib
        nonlocal peak_gpu_used_mib
        nonlocal peak_unexpected_gpu_memory_mib
        nonlocal first_unexpected_gpu_processes
        nonlocal gpu_monitor_failure
        gpu_monitor_samples += 1
        if any(snapshot.get(field) != gpu_identity[field] for field in gpu_identity_fields):
            gpu_monitor_failure = "gpu_identity_drift"
            return
        minimum_gpu_free_mib = min(minimum_gpu_free_mib, int(snapshot["free_mib"]))
        peak_gpu_used_mib = max(peak_gpu_used_mib, int(snapshot["used_mib"]))
        observed_pstates.add(str(snapshot["pstate"]))
        observed_temperatures.append(snapshot["temperature_c"])
        observed_graphics_clocks.append(snapshot["graphics_clock_mhz"])
        observed_memory_clocks.append(snapshot["memory_clock_mhz"])
        unexpected = [
            dict(app)
            for app in snapshot["compute_processes"]
            if not _pid_in_process_group(int(app["pid"]), process_group_id)
        ]
        unexpected_memory = sum(int(app["used_memory_mib"]) for app in unexpected)
        peak_unexpected_gpu_memory_mib = max(peak_unexpected_gpu_memory_mib, unexpected_memory)
        if unexpected:
            if first_unexpected_gpu_processes is None:
                first_unexpected_gpu_processes = unexpected
            gpu_monitor_failure = "gpu_contamination"

    with process_log.open("x", encoding="utf-8") as stream:
        try:
            process = subprocess.Popen(  # noqa: S603
                [str(token) for token in cell["argv"]],
                cwd=plan["repo_root"],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as error:
            launch_error = f"cannot start benchmark child: {error}"
            stream.write(launch_error + "\n")
            stream.flush()
        else:
            while process.poll() is None:
                process_tree = _process_tree_snapshot(process.pid)
                peak_process_tree_rss = max(peak_process_tree_rss, int(process_tree["rss_bytes"]))
                peak_process_tree_count = max(
                    peak_process_tree_count, int(process_tree["process_count"])
                )
                try:
                    gpu_sample = query_gpu_preflight(int(launch_policy["gpu_index"]))
                    record_gpu_sample(gpu_sample, process.pid)
                except (
                    OSError,
                    subprocess.CalledProcessError,
                    ThroughputLaunchError,
                ) as error:
                    gpu_monitor_failure = "gpu_monitor_error"
                    launch_error = f"cannot query GPU during benchmark: {error}"
                if gpu_monitor_failure is not None:
                    stream.write(
                        f"LACE GPU isolation failure: {gpu_monitor_failure}"
                        + (f" ({launch_error})" if launch_error else "")
                        + "\n"
                    )
                    stream.flush()
                    terminate_process_group(process)
                    break
                if time.monotonic() - start > launch_policy["cell_timeout_seconds"]:
                    timed_out = True
                    terminate_process_group(process)
                    break
                time.sleep(float(launch_policy["monitor_poll_seconds"]))
            returncode = process.wait()
    elapsed = time.monotonic() - start
    try:
        postflight = query_gpu_preflight(int(launch_policy["gpu_index"]))
    except (OSError, subprocess.CalledProcessError, ThroughputLaunchError) as error:
        if gpu_monitor_failure is None:
            gpu_monitor_failure = "gpu_monitor_error"
            launch_error = f"cannot query GPU postflight: {error}"
    else:
        if any(postflight.get(field) != gpu_identity[field] for field in gpu_identity_fields):
            gpu_monitor_failure = gpu_monitor_failure or "gpu_identity_drift"
        if postflight["compute_processes"]:
            unexpected_memory = sum(
                int(app["used_memory_mib"]) for app in postflight["compute_processes"]
            )
            peak_unexpected_gpu_memory_mib = max(peak_unexpected_gpu_memory_mib, unexpected_memory)
            if first_unexpected_gpu_processes is None:
                first_unexpected_gpu_processes = [
                    dict(app) for app in postflight["compute_processes"]
                ]
            gpu_monitor_failure = gpu_monitor_failure or "gpu_process_leak"
    log_bytes = process_log.read_bytes()
    tail = log_bytes[-2_000_000:].decode("utf-8", errors="replace")
    failure_class = gpu_monitor_failure
    if failure_class is None:
        failure_class = (
            "launch_error"
            if launch_error is not None
            else _classify_failure(returncode, tail, timed_out=timed_out)
        )
    metrics_path = Path(cell["metrics_jsonl"])
    metrics_record: dict[str, Any] | None = None
    metrics_error: str | None = None
    if metrics_path.is_file():
        metrics_record = {
            "path": str(metrics_path),
            "bytes": metrics_path.stat().st_size,
            "sha256": file_sha256(metrics_path),
            "row_count": None,
        }
        try:
            metric_rows = load_iteration_rows(metrics_path)
        except ThroughputProtocolError as error:
            metrics_error = str(error)
        else:
            metrics_record["row_count"] = len(metric_rows)
            expected_rows = int(plan["benchmark_contract"]["benchmark"]["warmup_iterations"]) + int(
                plan["benchmark_contract"]["benchmark"]["timed_iterations"]
            )
            if len(metric_rows) != expected_rows:
                metrics_error = f"metrics row count {len(metric_rows)} differs from {expected_rows}"
    elif failure_class is None:
        metrics_error = "successful process did not create throughput metrics"
    if failure_class is None and metrics_error is not None:
        failure_class = "metrics_contract"
    result = {
        "schema_version": 1,
        "kind": "lace_sonic_lite_throughput_launch_result",
        "plan_sha256": plan["plan_sha256"],
        "benchmark_contract_sha256": plan["benchmark_contract_sha256"],
        "cell_contract_sha256": cell["cell_contract_sha256"],
        "command_sha256": cell["command_sha256"],
        "cell_id": cell["cell_id"],
        "num_envs": num_envs,
        "returncode": returncode,
        "failure_class": failure_class,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "external_process_peak_rss_bytes": peak_process_tree_rss,
        "process_tree_peak_rss_bytes": peak_process_tree_rss,
        "process_tree_peak_process_count": peak_process_tree_count,
        "final_output_dir_bytes": _directory_bytes(output_dir),
        "final_tmp_dir_bytes": _directory_bytes(tmpdir),
        "process_log": {
            "path": str(process_log),
            "bytes": len(log_bytes),
            "sha256": file_sha256(process_log),
        },
        "metrics": metrics_record,
        "metrics_error": metrics_error,
        "launch_error": launch_error,
        "gpu_preflight": preflight,
        "gpu_monitor": {
            "sample_count": gpu_monitor_samples,
            "minimum_free_mib": minimum_gpu_free_mib,
            "peak_used_mib": peak_gpu_used_mib,
            "peak_unexpected_memory_mib": peak_unexpected_gpu_memory_mib,
            "first_unexpected_processes": first_unexpected_gpu_processes,
            "observed_pstates": sorted(observed_pstates),
            "maximum_temperature_c": max(
                (value for value in observed_temperatures if value is not None),
                default=None,
            ),
            "graphics_clock_range_mhz": _finite_range(observed_graphics_clocks),
            "memory_clock_range_mhz": _finite_range(observed_memory_clocks),
            "failure": gpu_monitor_failure,
        },
        "gpu_postflight": postflight,
        "runtime_versions": plan["benchmark_contract"]["runtime_versions"],
    }
    result["launch_result_sha256"] = canonical_sha256(result)
    write_new_json(cell["launch_result"], result)
    return result


def load_iteration_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load canonical JSONL iteration records with duplicate-key rejection."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ThroughputProtocolError(f"missing throughput metrics: {candidate}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ThroughputProtocolError(f"blank JSONL row at {candidate}:{line_number}")
        try:
            row = json.loads(
                line,
                object_pairs_hook=_json_pairs_no_duplicates,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ThroughputProtocolError(
                        f"non-finite JSON constant at {candidate}:{line_number}: {token}"
                    )
                ),
            )
        except json.JSONDecodeError as error:
            raise ThroughputProtocolError(
                f"invalid JSONL row at {candidate}:{line_number}"
            ) from error
        if not isinstance(row, dict):
            raise ThroughputProtocolError(f"JSONL row must be an object: {candidate}:{line_number}")
        rows.append(row)
    return rows


def validate_launch_result(
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    launch_result: Mapping[str, Any],
) -> None:
    """Validate a launch receipt and its immutable log/metrics bindings."""

    unsigned = dict(launch_result)
    stored_digest = unsigned.pop("launch_result_sha256", None)
    if stored_digest != canonical_sha256(unsigned):
        raise ThroughputProtocolError("launch result self-hash mismatch")
    expected_fields = {
        "schema_version": 1,
        "kind": "lace_sonic_lite_throughput_launch_result",
        "plan_sha256": plan["plan_sha256"],
        "benchmark_contract_sha256": plan["benchmark_contract_sha256"],
        "cell_contract_sha256": cell["cell_contract_sha256"],
        "command_sha256": cell["command_sha256"],
        "cell_id": cell["cell_id"],
        "num_envs": cell["num_envs"],
    }
    for field, expected in expected_fields.items():
        if launch_result.get(field) != expected:
            raise ThroughputProtocolError(f"launch result {field} differs from plan")
    returncode = launch_result.get("returncode")
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        raise ThroughputProtocolError("launch result returncode must be an integer")
    allowed_failures = {
        None,
        "timeout",
        "oom",
        "numerical",
        "simulator",
        "process_error",
        "metrics_contract",
        "launch_error",
        "gpu_contamination",
        "gpu_monitor_error",
        "gpu_identity_drift",
        "gpu_process_leak",
    }
    if launch_result.get("failure_class") not in allowed_failures:
        raise ThroughputProtocolError("launch result has an unknown failure class")
    if launch_result.get("failure_class") is None and returncode != 0:
        raise ThroughputProtocolError("nonzero launch result omitted failure class")
    if not isinstance(launch_result.get("timed_out"), bool):
        raise ThroughputProtocolError("launch result timed_out must be boolean")
    elapsed = launch_result.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ThroughputProtocolError("launch result elapsed_seconds must be non-negative")
    for field in (
        "external_process_peak_rss_bytes",
        "final_output_dir_bytes",
        "final_tmp_dir_bytes",
        "process_tree_peak_rss_bytes",
        "process_tree_peak_process_count",
    ):
        value = launch_result.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ThroughputProtocolError(f"launch result {field} must be non-negative")
    gpu_preflight = _require_mapping(
        launch_result.get("gpu_preflight"), label="launch_result.gpu_preflight"
    )
    if gpu_preflight.get("index") != plan["launch_policy"]["gpu_index"]:
        raise ThroughputProtocolError("launch GPU preflight index differs from plan")
    for field in ("uuid", "name", "driver_version", "total_mib"):
        if gpu_preflight.get(field) in (None, ""):
            raise ThroughputProtocolError(f"launch GPU preflight omitted {field}")
    if gpu_preflight.get("compute_processes") != []:
        raise ThroughputProtocolError("launch GPU preflight was not process-isolated")
    if launch_result.get("runtime_versions") != plan["benchmark_contract"]["runtime_versions"]:
        raise ThroughputProtocolError("launch runtime versions differ from plan")
    gpu_monitor = _require_mapping(
        launch_result.get("gpu_monitor"), label="launch_result.gpu_monitor"
    )
    sample_count = gpu_monitor.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ThroughputProtocolError("launch GPU monitor sample count is invalid")
    if launch_result.get("failure_class") is None and sample_count == 0:
        raise ThroughputProtocolError("successful launch has no continuous GPU samples")
    monitor_failure = gpu_monitor.get("failure")
    if monitor_failure is not None and launch_result.get("failure_class") != monitor_failure:
        raise ThroughputProtocolError("GPU monitor failure differs from launch failure class")
    postflight = launch_result.get("gpu_postflight")
    if postflight is None:
        if launch_result.get("failure_class") != "gpu_monitor_error":
            raise ThroughputProtocolError("launch omitted GPU postflight without monitor failure")
    else:
        postflight_record = _require_mapping(postflight, label="launch_result.gpu_postflight")
        for field in ("index", "uuid", "name", "driver_version", "total_mib"):
            if postflight_record.get(field) != gpu_preflight.get(field):
                raise ThroughputProtocolError("launch GPU identity drifted by postflight")
        if (
            launch_result.get("failure_class") is None
            and postflight_record.get("compute_processes") != []
        ):
            raise ThroughputProtocolError("successful launch left a GPU process behind")

    process_log = _require_mapping(
        launch_result.get("process_log"), label="launch_result.process_log"
    )
    expected_log = Path(str(cell["process_log"]))
    if Path(str(process_log.get("path"))) != expected_log:
        raise ThroughputProtocolError("launch process-log path differs from plan")
    if not expected_log.is_file() or expected_log.stat().st_size != process_log.get("bytes"):
        raise ThroughputProtocolError("launch process-log byte count differs")
    if file_sha256(expected_log) != process_log.get("sha256"):
        raise ThroughputProtocolError("launch process-log digest differs")

    metrics = launch_result.get("metrics")
    expected_metrics = Path(str(cell["metrics_jsonl"]))
    if metrics is None:
        if expected_metrics.exists():
            raise ThroughputProtocolError("unbound throughput metrics file exists")
        if launch_result.get("failure_class") is None:
            raise ThroughputProtocolError("successful launch has no metrics binding")
        return
    metrics_record = _require_mapping(metrics, label="launch_result.metrics")
    if Path(str(metrics_record.get("path"))) != expected_metrics:
        raise ThroughputProtocolError("launch metrics path differs from plan")
    if not expected_metrics.is_file() or expected_metrics.stat().st_size != metrics_record.get(
        "bytes"
    ):
        raise ThroughputProtocolError("launch metrics byte count differs")
    if file_sha256(expected_metrics) != metrics_record.get("sha256"):
        raise ThroughputProtocolError("launch metrics digest differs")
    row_count = metrics_record.get("row_count")
    if row_count is None:
        if launch_result.get("failure_class") is None or not isinstance(
            launch_result.get("metrics_error"), str
        ):
            raise ThroughputProtocolError("unparseable metrics require a failed launch receipt")
        return
    if isinstance(row_count, bool) or not isinstance(row_count, int) or row_count < 0:
        raise ThroughputProtocolError("launch metrics row count is invalid")
    rows = load_iteration_rows(expected_metrics)
    if len(rows) != row_count:
        raise ThroughputProtocolError("launch metrics row count differs")
    expected_rows = int(plan["benchmark_contract"]["benchmark"]["warmup_iterations"]) + int(
        plan["benchmark_contract"]["benchmark"]["timed_iterations"]
    )
    if launch_result.get("failure_class") is None and (
        row_count != expected_rows or launch_result.get("metrics_error") is not None
    ):
        raise ThroughputProtocolError("successful launch metrics are not exact and complete")


def summarize_cell(
    plan: Mapping[str, Any],
    cell: Mapping[str, Any],
    *,
    launch_result: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply all preregistered eligibility gates and summarize one cell."""

    validate_launch_result(plan, cell, launch_result)
    if (
        launch_result.get("metrics") is not None
        and launch_result["metrics"].get("row_count") is not None
    ):
        bound_rows = load_iteration_rows(cell["metrics_jsonl"])
        if list(rows) != bound_rows:
            raise ThroughputProtocolError("provided metric rows differ from launch binding")
    benchmark = plan["benchmark_contract"]["benchmark"]
    warmup = int(benchmark["warmup_iterations"])
    timed_count = int(benchmark["timed_iterations"])
    total = warmup + timed_count
    reasons: list[str] = []
    if launch_result.get("plan_sha256") != plan["plan_sha256"]:
        reasons.append("launch_plan_hash_mismatch")
    if launch_result.get("cell_contract_sha256") != cell["cell_contract_sha256"]:
        reasons.append("launch_cell_hash_mismatch")
    if launch_result.get("returncode") != 0:
        reasons.append("nonzero_process_exit")
    if launch_result.get("failure_class") is not None:
        reasons.append(f"failure_class:{launch_result.get('failure_class')}")
    if len(rows) != total:
        reasons.append("iteration_count_mismatch")

    timed: list[Mapping[str, Any]] = []
    expected_iterations = list(range(1, total + 1))
    if [row.get("iteration") for row in rows] != expected_iterations:
        reasons.append("iteration_order_mismatch")
    transitions_per_iteration = int(cell["num_envs"]) * int(benchmark["num_steps_per_env"])
    substeps_per_iteration = transitions_per_iteration * int(benchmark["decimation"])
    optimizer_schedule = cell["cell_training_contract"]["optimizer_schedule"]
    dataset_contract = plan["benchmark_contract"]["dataset"]
    workload_contract = plan["benchmark_contract"]["workload"]
    for row in rows:
        iteration = row.get("iteration")
        expected_phase = "warmup" if isinstance(iteration, int) and iteration <= warmup else "timed"
        if row.get("phase") != expected_phase:
            reasons.append("phase_mismatch")
        if row.get("benchmark_contract_sha256") != plan["benchmark_contract_sha256"]:
            reasons.append("metric_benchmark_hash_mismatch")
        if row.get("cell_contract_sha256") != cell["cell_contract_sha256"]:
            reasons.append("metric_cell_hash_mismatch")
        if row.get("schema_version") != 1 or row.get("kind") != (
            "lace_sonic_lite_throughput_iteration"
        ):
            reasons.append("metric_schema_mismatch")
        if row.get("cell_id") != cell["cell_id"]:
            reasons.append("metric_cell_id_mismatch")
        if row.get("num_envs") != cell["num_envs"]:
            reasons.append("metric_num_envs_mismatch")
        expected_row_fields = {
            "num_steps_per_env": benchmark["num_steps_per_env"],
            "decimation": benchmark["decimation"],
            "control_transitions": transitions_per_iteration,
            "physics_substeps": substeps_per_iteration,
            "optimizer_expected_attempts": optimizer_schedule["optimizer_expected_attempts"],
            "synchronized_expected_updates": optimizer_schedule["synchronized_expected_updates"],
            "termination_count_definition": ("done_and_not_timeout_timeout_precedence"),
            "dataset_subset_sha256": dataset_contract["subset_sha256"],
            "resident_motion_count": dataset_contract["resident_motion_count"],
            "resident_unique_motion_count": dataset_contract["resident_motion_count"],
            "resident_universe_motion_count": dataset_contract["resident_motion_count"],
            "resident_all_motions_loaded": True,
            "resident_motion_order_sha256": dataset_contract["resident_motion_order_sha256"],
            "resident_motion_set_sha256": dataset_contract["resident_motion_set_sha256"],
            "terrain_type": workload_contract["terrain_type"],
            "resident_dataset_identity_attested": True,
        }
        for field, expected in expected_row_fields.items():
            if row.get(field) != expected:
                reasons.append(f"metric_contract_mismatch:{field}")
        if row.get("optimizer_nonfinite_skips") != 0:
            reasons.append("optimizer_nonfinite_skip")
        if row.get("optimizer_accelerator_skips") != 0:
            reasons.append("optimizer_accelerator_skip")
        if row.get("optimizer_step_attempts", 0) + row.get(
            "optimizer_nonfinite_skips", 0
        ) != row.get("optimizer_expected_attempts"):
            reasons.append("optimizer_attempt_mismatch")
        if row.get("sync_boundaries") != row.get("synchronized_expected_updates"):
            reasons.append("sync_boundary_mismatch")
        if row.get("synchronized_parameter_updates", 0) + row.get(
            "synchronized_update_skips", 0
        ) != row.get("synchronized_expected_updates"):
            reasons.append("synchronized_update_mismatch")
        if row.get("numerical_failure_count") != 0:
            reasons.append("numerical_failure")
        if row.get("reset_count") != row.get("termination_count", 0) + row.get("timeout_count", 0):
            reasons.append("reset_counter_mismatch")
        if row.get("phase") == "timed":
            timed.append(row)
    if len(timed) != timed_count:
        reasons.append("timed_iteration_count_mismatch")

    def finite_positive(name: str) -> list[float]:
        values: list[float] = []
        for row in timed:
            value = row.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                reasons.append(f"invalid_metric:{name}")
                continue
            number = float(value)
            if not math.isfinite(number) or number <= 0.0:
                reasons.append(f"invalid_metric:{name}")
                continue
            values.append(number)
        return values

    end_to_end_seconds = finite_positive("end_to_end_iteration_seconds")
    collection_seconds = finite_positive("collection_seconds")
    learn_seconds = finite_positive("learn_seconds")
    peak_reserved = max((int(row.get("cuda_peak_reserved_bytes", 0)) for row in rows), default=0)
    total_cuda = max((int(row.get("cuda_total_bytes", 0)) for row in rows), default=0)
    global_free_values = [
        int(row.get("cuda_global_free_bytes", 0))
        for row in rows
        if isinstance(row.get("cuda_global_free_bytes"), int)
        and not isinstance(row.get("cuda_global_free_bytes"), bool)
    ]
    minimum_global_free = min(global_free_values, default=0)
    monitor_minimum_free = int(launch_result["gpu_monitor"]["minimum_free_mib"]) * 1024**2
    minimum_observed_free = min(minimum_global_free, monitor_minimum_free)
    safety_bytes = (
        int(plan["benchmark_contract"]["selection_rule"]["cuda_memory_safety_margin_mib"]) * 1024**2
    )
    if len(global_free_values) != total or minimum_observed_free < safety_bytes:
        reasons.append("cuda_memory_safety_margin_failed")
    sustained_iter_per_hour = (
        3600.0 * len(end_to_end_seconds) / sum(end_to_end_seconds)
        if len(end_to_end_seconds) == timed_count
        else None
    )
    transitions = transitions_per_iteration
    reset_count = sum(int(row.get("reset_count", 0)) for row in timed)
    termination_count = sum(int(row.get("termination_count", 0)) for row in timed)
    timeout_count = sum(int(row.get("timeout_count", 0)) for row in timed)
    timed_transition_count = len(timed) * transitions
    return {
        "cell_id": cell["cell_id"],
        "num_envs": cell["num_envs"],
        "eligible": not reasons,
        "ineligibility_reasons": sorted(set(reasons)),
        "timed_iterations": len(timed),
        "sustained_end_to_end_iterations_per_hour": sustained_iter_per_hour,
        "sustained_end_to_end_control_transitions_per_second": (
            transitions * len(end_to_end_seconds) / sum(end_to_end_seconds)
            if len(end_to_end_seconds) == timed_count
            else None
        ),
        "sustained_end_to_end_physics_substeps_per_second": (
            transitions
            * int(benchmark["decimation"])
            * len(end_to_end_seconds)
            / sum(end_to_end_seconds)
            if len(end_to_end_seconds) == timed_count
            else None
        ),
        "sustained_collection_physics_substeps_per_second": (
            transitions
            * int(benchmark["decimation"])
            * len(collection_seconds)
            / sum(collection_seconds)
            if len(collection_seconds) == timed_count
            else None
        ),
        "mean_collection_seconds": (
            sum(collection_seconds) / len(collection_seconds) if collection_seconds else None
        ),
        "mean_learn_seconds": sum(learn_seconds) / len(learn_seconds) if learn_seconds else None,
        "cuda_peak_allocated_bytes": max(
            (int(row.get("cuda_peak_allocated_bytes", 0)) for row in rows), default=0
        ),
        "cuda_peak_reserved_bytes": peak_reserved,
        "cuda_total_bytes": total_cuda,
        "cuda_allocator_unreserved_bytes": total_cuda - peak_reserved,
        "minimum_cuda_global_free_bytes": minimum_global_free,
        "minimum_continuous_gpu_free_bytes": monitor_minimum_free,
        "minimum_observed_gpu_free_bytes": minimum_observed_free,
        "process_peak_rss_bytes": max(
            (int(row.get("process_peak_rss_bytes", 0)) for row in rows), default=0
        ),
        "external_process_peak_rss_bytes": launch_result.get("external_process_peak_rss_bytes"),
        "process_tree_peak_rss_bytes": launch_result.get("process_tree_peak_rss_bytes"),
        "process_tree_peak_process_count": launch_result.get("process_tree_peak_process_count"),
        "final_tmp_dir_bytes": launch_result.get("final_tmp_dir_bytes"),
        "output_dir_growth_bytes": max(
            (int(row.get("output_dir_growth_bytes", 0)) for row in rows), default=0
        ),
        "reset_count": reset_count,
        "termination_count": termination_count,
        "timeout_count": timeout_count,
        "reset_rate_per_control_transition": (
            reset_count / timed_transition_count if timed_transition_count else None
        ),
        "termination_rate_per_control_transition": (
            termination_count / timed_transition_count if timed_transition_count else None
        ),
        "timeout_rate_per_control_transition": (
            timeout_count / timed_transition_count if timed_transition_count else None
        ),
        "termination_count_definition": "done_and_not_timeout_timeout_precedence",
        "dataset_subset_sha256": dataset_contract["subset_sha256"],
        "resident_motion_count": dataset_contract["resident_motion_count"],
        "resident_unique_motion_count": dataset_contract["resident_unique_motion_count"],
        "resident_motion_keys": dataset_contract["resident_motion_keys"],
        "resident_motion_order_sha256": dataset_contract["resident_motion_order_sha256"],
        "resident_motion_set_sha256": dataset_contract["resident_motion_set_sha256"],
        "resident_all_motions_loaded": True,
        "terrain_type": workload_contract["terrain_type"],
        "optimizer_step_attempts": sum(int(row.get("optimizer_step_attempts", 0)) for row in timed),
        "optimizer_nonfinite_skips": sum(
            int(row.get("optimizer_nonfinite_skips", 0)) for row in timed
        ),
        "optimizer_accelerator_skips": sum(
            int(row.get("optimizer_accelerator_skips", 0)) for row in timed
        ),
        "synchronized_parameter_updates": sum(
            int(row.get("synchronized_parameter_updates", 0)) for row in timed
        ),
        "synchronized_update_skips": sum(
            int(row.get("synchronized_update_skips", 0)) for row in timed
        ),
        "failure_class": launch_result.get("failure_class"),
    }


def aggregate_throughput_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate completed cells and select N* by the frozen rule."""

    validate_plan(plan)
    summaries: list[dict[str, Any]] = []
    prior_oom_cell: str | None = None
    stable_gpu_identity: dict[str, Any] | None = None
    for cell in plan["cells"]:
        result_path = Path(cell["launch_result"])
        metrics_path = Path(cell["metrics_jsonl"])
        if not result_path.is_file():
            if metrics_path.exists():
                raise ThroughputProtocolError(
                    f"{cell['cell_id']}: metrics exist without a launch result"
                )
            reason = (
                f"not_run_after_prior_oom:{prior_oom_cell}"
                if prior_oom_cell is not None
                else "incomplete_cell"
            )
            summaries.append(
                {
                    "cell_id": cell["cell_id"],
                    "num_envs": cell["num_envs"],
                    "eligible": False,
                    "terminal": prior_oom_cell is not None,
                    "ineligibility_reasons": [reason],
                }
            )
            continue
        if prior_oom_cell is not None:
            raise ThroughputProtocolError(
                f"{cell['cell_id']}: launch result exists after OOM in {prior_oom_cell}"
            )
        launch_result = load_json_object(result_path)
        metrics_record = launch_result.get("metrics")
        rows = (
            load_iteration_rows(metrics_path)
            if isinstance(metrics_record, Mapping) and metrics_record.get("row_count") is not None
            else []
        )
        summary = summarize_cell(
            plan,
            cell,
            launch_result=launch_result,
            rows=rows,
        )
        current_gpu_identity = {
            field: launch_result["gpu_preflight"][field]
            for field in ("uuid", "name", "driver_version", "total_mib")
        }
        if stable_gpu_identity is None:
            stable_gpu_identity = current_gpu_identity
        elif current_gpu_identity != stable_gpu_identity:
            summary["eligible"] = False
            summary["ineligibility_reasons"] = sorted(
                set(summary["ineligibility_reasons"] + ["cross_cell_gpu_identity_mismatch"])
            )
        summary["terminal"] = True
        summary["launch_result_file_sha256"] = file_sha256(result_path)
        summary["metrics_file_sha256"] = (
            metrics_record.get("sha256") if isinstance(metrics_record, Mapping) else None
        )
        summaries.append(summary)
        if launch_result.get("failure_class") == "oom":
            prior_oom_cell = str(cell["cell_id"])
    eligible = [summary for summary in summaries if summary["eligible"]]
    selected = (
        min(
            eligible,
            key=lambda value: (
                -float(value["sustained_end_to_end_control_transitions_per_second"]),
                int(value["num_envs"]),
            ),
        )
        if eligible
        else None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": REPORT_KIND,
        "plan_sha256": plan["plan_sha256"],
        "benchmark_contract_sha256": plan["benchmark_contract_sha256"],
        "selection_rule": plan["benchmark_contract"]["selection_rule"],
        "runtime_versions": plan["benchmark_contract"]["runtime_versions"],
        "complete": all(bool(item.get("terminal")) for item in summaries),
        "stopped_after_oom_cell_id": prior_oom_cell,
        "stable_gpu_invariants": stable_gpu_identity,
        "selected_num_envs": None if selected is None else selected["num_envs"],
        "selected_cell_id": None if selected is None else selected["cell_id"],
        "cells": summaries,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
