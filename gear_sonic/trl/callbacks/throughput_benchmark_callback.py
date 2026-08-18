"""Opt-in JSONL telemetry for the LACE SONIC-Lite throughput sweep.

This callback is intentionally inert unless explicitly composed by the
research launcher.  It records process-local resource counters that cannot be
recovered reliably from the trainer's formatted console output.  Scientific
selection is performed later by :mod:`gear_sonic.research.lace.throughput`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import re
import resource
import time
from typing import Any

import torch
from transformers import TrainerCallback

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_REQUIRED_BENCHMARK_LOGS = (
    "benchmark/optimizer_step_attempts",
    "benchmark/optimizer_nonfinite_skips",
    "benchmark/optimizer_accelerator_skips",
    "benchmark/optimizer_expected_attempts",
    "benchmark/sync_boundaries",
    "benchmark/synchronized_parameter_updates",
    "benchmark/synchronized_update_skips",
    "benchmark/synchronized_expected_updates",
    "benchmark/reset_count",
    "benchmark/termination_count",
    "benchmark/timeout_count",
    "benchmark/resident_motion_count",
    "benchmark/resident_unique_motion_count",
    "benchmark/resident_universe_motion_count",
    "benchmark/resident_all_motions_loaded",
    "benchmark/resident_motion_order_sha256",
    "benchmark/resident_motion_set_sha256",
    "benchmark/terrain_type",
)


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _directory_bytes(root: Path) -> int:
    """Return allocated output bytes without following symlinks."""

    if not root.exists():
        return 0
    total = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            try:
                total += path.lstat().st_size
            except FileNotFoundError:
                # A logger may atomically replace a file while this snapshot is
                # taken.  The next iteration observes the replacement.
                continue
    return total


def _current_rss_bytes() -> int:
    """Read current resident memory from procfs, falling back to peak RSS."""

    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, IndexError, ValueError):
        pass
    # Linux reports ru_maxrss in KiB.  The benchmark is Linux/Isaac-only.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _cuda_snapshot() -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("throughput telemetry requires CUDA")
    device = torch.cuda.current_device()
    global_free, global_total = torch.cuda.mem_get_info(device)
    return {
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        # CUDA's global device view includes Isaac/PhysX/renderer allocations
        # that are invisible to PyTorch allocator counters.
        "cuda_global_free_bytes": int(global_free),
        "cuda_global_total_bytes": int(global_total),
    }


class ThroughputBenchmarkCallback(TrainerCallback):
    """Write one fail-closed resource record for every benchmark iteration."""

    def __init__(
        self,
        output_path: str,
        benchmark_contract_sha256: str,
        cell_contract_sha256: str,
        cell_id: str,
        num_envs: int,
        num_steps_per_env: int,
        decimation: int,
        warmup_iterations: int,
        timed_iterations: int,
        dataset_subset_sha256: str,
        expected_resident_motion_count: int,
        expected_resident_motion_order_sha256: str,
        expected_resident_motion_set_sha256: str,
        expected_terrain_type: str,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_root = self.output_path.parent
        for label, digest in (
            ("benchmark_contract_sha256", benchmark_contract_sha256),
            ("cell_contract_sha256", cell_contract_sha256),
            ("dataset_subset_sha256", dataset_subset_sha256),
            ("expected_resident_motion_order_sha256", expected_resident_motion_order_sha256),
            ("expected_resident_motion_set_sha256", expected_resident_motion_set_sha256),
        ):
            if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("cell_id must be a non-empty string")
        self.benchmark_contract_sha256 = benchmark_contract_sha256
        self.cell_contract_sha256 = cell_contract_sha256
        self.cell_id = cell_id
        self.dataset_subset_sha256 = dataset_subset_sha256
        self.expected_resident_motion_order_sha256 = expected_resident_motion_order_sha256
        self.expected_resident_motion_set_sha256 = expected_resident_motion_set_sha256
        self.expected_resident_motion_count = _positive_int(
            expected_resident_motion_count, label="expected_resident_motion_count"
        )
        if expected_terrain_type != "plane":
            raise ValueError("throughput benchmark terrain must be plane")
        self.expected_terrain_type = expected_terrain_type
        self.num_envs = _positive_int(num_envs, label="num_envs")
        self.num_steps_per_env = _positive_int(num_steps_per_env, label="num_steps_per_env")
        self.decimation = _positive_int(decimation, label="decimation")
        self.warmup_iterations = _positive_int(warmup_iterations, label="warmup_iterations")
        self.timed_iterations = _positive_int(timed_iterations, label="timed_iterations")
        if self.timed_iterations < 20:
            raise ValueError("timed_iterations must be at least 20")
        self.total_iterations = self.warmup_iterations + self.timed_iterations
        self._previous_log_time: float | None = None
        self._initial_output_bytes = 0
        self._last_iteration = 0

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if not state.is_world_process_zero:
            return control
        if self.output_path.exists():
            raise FileExistsError(
                f"throughput metrics already exist; resume/overwrite is forbidden: "
                f"{self.output_path}"
            )
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._initial_output_bytes = _directory_bytes(self.output_root)
        if not torch.cuda.is_available():
            raise RuntimeError("throughput benchmark requires a CUDA device")
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        self._previous_log_time = time.monotonic()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if not state.is_world_process_zero:
            return control
        if not isinstance(logs, dict):
            raise RuntimeError("throughput callback requires trainer log metrics")
        iteration = int(state.global_step)
        if iteration != self._last_iteration + 1:
            raise RuntimeError(
                "throughput iterations must be contiguous: "
                f"expected {self._last_iteration + 1}, got {iteration}"
            )
        if iteration > self.total_iterations:
            raise RuntimeError(f"trainer exceeded frozen benchmark length {self.total_iterations}")
        missing = [key for key in _REQUIRED_BENCHMARK_LOGS if key not in logs]
        if missing:
            raise RuntimeError(f"trainer omitted required throughput metrics: {missing}")

        now = time.monotonic()
        if self._previous_log_time is None:
            raise RuntimeError("throughput callback was not initialized at train begin")
        wall_seconds = now - self._previous_log_time
        self._previous_log_time = now
        collection_seconds = float(logs["collection_time"])
        learn_seconds = float(logs["learn_time"])
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (wall_seconds, collection_seconds, learn_seconds)
        ):
            raise RuntimeError("non-finite or non-positive throughput timing observed")

        reset_count = int(logs["benchmark/reset_count"])
        termination_count = int(logs["benchmark/termination_count"])
        timeout_count = int(logs["benchmark/timeout_count"])
        optimizer_step_attempts = int(logs["benchmark/optimizer_step_attempts"])
        optimizer_nonfinite_skips = int(logs["benchmark/optimizer_nonfinite_skips"])
        optimizer_accelerator_skips = int(logs["benchmark/optimizer_accelerator_skips"])
        optimizer_expected_attempts = int(logs["benchmark/optimizer_expected_attempts"])
        sync_boundaries = int(logs["benchmark/sync_boundaries"])
        synchronized_parameter_updates = int(logs["benchmark/synchronized_parameter_updates"])
        synchronized_update_skips = int(logs["benchmark/synchronized_update_skips"])
        synchronized_expected_updates = int(logs["benchmark/synchronized_expected_updates"])
        resident_motion_count = int(logs["benchmark/resident_motion_count"])
        resident_unique_motion_count = int(logs["benchmark/resident_unique_motion_count"])
        resident_universe_motion_count = int(logs["benchmark/resident_universe_motion_count"])
        resident_all_motions_loaded = logs["benchmark/resident_all_motions_loaded"]
        if not isinstance(resident_all_motions_loaded, bool):
            raise RuntimeError("resident all-loaded attestation must be boolean")
        resident_motion_order_sha256 = logs["benchmark/resident_motion_order_sha256"]
        resident_motion_set_sha256 = logs["benchmark/resident_motion_set_sha256"]
        terrain_type = logs["benchmark/terrain_type"]
        transitions = self.num_envs * self.num_steps_per_env
        counters = (
            reset_count,
            termination_count,
            timeout_count,
            optimizer_step_attempts,
            optimizer_nonfinite_skips,
            optimizer_accelerator_skips,
            sync_boundaries,
            synchronized_parameter_updates,
            synchronized_update_skips,
        )
        if min(counters) < 0:
            raise RuntimeError("throughput counters cannot be negative")
        if reset_count != termination_count + timeout_count or reset_count > transitions:
            raise RuntimeError("reset/termination/timeout counters violate the rollout contract")
        if optimizer_step_attempts + optimizer_nonfinite_skips != optimizer_expected_attempts:
            raise RuntimeError("optimizer attempt/skip counters violate the PPO schedule")
        if sync_boundaries != synchronized_expected_updates:
            raise RuntimeError("observed synchronization boundaries violate the PPO schedule")
        if (
            synchronized_parameter_updates + synchronized_update_skips
            != synchronized_expected_updates
        ):
            raise RuntimeError("synchronized update/skip counters violate the PPO schedule")
        expected_resident_count = self.expected_resident_motion_count
        if (
            resident_motion_count != expected_resident_count
            or resident_unique_motion_count != expected_resident_count
            or resident_universe_motion_count != expected_resident_count
            or not resident_all_motions_loaded
        ):
            raise RuntimeError("resident motion count/all-loaded attestation failed")
        if resident_motion_order_sha256 != self.expected_resident_motion_order_sha256:
            raise RuntimeError("resident motion order differs from the bound dataset")
        if resident_motion_set_sha256 != self.expected_resident_motion_set_sha256:
            raise RuntimeError("resident motion set differs from the bound dataset")
        if terrain_type != self.expected_terrain_type:
            raise RuntimeError("runtime terrain differs from the frozen plane workload")

        output_bytes = _directory_bytes(self.output_root)
        row = {
            "schema_version": 1,
            "kind": "lace_sonic_lite_throughput_iteration",
            "benchmark_contract_sha256": self.benchmark_contract_sha256,
            "cell_contract_sha256": self.cell_contract_sha256,
            "cell_id": self.cell_id,
            "iteration": iteration,
            "phase": "warmup" if iteration <= self.warmup_iterations else "timed",
            "num_envs": self.num_envs,
            "num_steps_per_env": self.num_steps_per_env,
            "decimation": self.decimation,
            "control_transitions": transitions,
            "physics_substeps": transitions * self.decimation,
            "collection_seconds": collection_seconds,
            "learn_seconds": learn_seconds,
            "trainer_iteration_seconds": collection_seconds + learn_seconds,
            "end_to_end_iteration_seconds": wall_seconds,
            "collection_control_transitions_per_second": transitions / collection_seconds,
            "collection_physics_substeps_per_second": (
                transitions * self.decimation / collection_seconds
            ),
            "trainer_control_transitions_per_second": (
                transitions / (collection_seconds + learn_seconds)
            ),
            "end_to_end_control_transitions_per_second": transitions / wall_seconds,
            "end_to_end_physics_substeps_per_second": (
                transitions * self.decimation / wall_seconds
            ),
            "end_to_end_iterations_per_hour": 3600.0 / wall_seconds,
            "dataset_subset_sha256": self.dataset_subset_sha256,
            "resident_motion_count": resident_motion_count,
            "resident_unique_motion_count": resident_unique_motion_count,
            "resident_universe_motion_count": resident_universe_motion_count,
            "resident_all_motions_loaded": resident_all_motions_loaded,
            "resident_motion_order_sha256": resident_motion_order_sha256,
            "resident_motion_set_sha256": resident_motion_set_sha256,
            "terrain_type": terrain_type,
            "resident_dataset_identity_attested": True,
            "optimizer_step_attempts": optimizer_step_attempts,
            "optimizer_nonfinite_skips": optimizer_nonfinite_skips,
            "optimizer_accelerator_skips": optimizer_accelerator_skips,
            "optimizer_expected_attempts": optimizer_expected_attempts,
            "sync_boundaries": sync_boundaries,
            "synchronized_parameter_updates": synchronized_parameter_updates,
            "synchronized_update_skips": synchronized_update_skips,
            "synchronized_expected_updates": synchronized_expected_updates,
            "reset_count": reset_count,
            "termination_count": termination_count,
            "timeout_count": timeout_count,
            "reset_rate_per_control_transition": reset_count / transitions,
            "termination_rate_per_control_transition": termination_count / transitions,
            "timeout_rate_per_control_transition": timeout_count / transitions,
            "termination_count_definition": "done_and_not_timeout_timeout_precedence",
            "numerical_failure_count": (optimizer_nonfinite_skips + optimizer_accelerator_skips),
            "process_rss_bytes": _current_rss_bytes(),
            "process_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            ),
            "output_dir_bytes": output_bytes,
            "output_dir_growth_bytes": output_bytes - self._initial_output_bytes,
            **_cuda_snapshot(),
        }
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.write("\n")
        self._last_iteration = iteration
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN201, ARG002
        if state.is_world_process_zero and self._last_iteration != self.total_iterations:
            raise RuntimeError(
                "throughput training ended before all frozen iterations: "
                f"{self._last_iteration}/{self.total_iterations}"
            )
        return control
