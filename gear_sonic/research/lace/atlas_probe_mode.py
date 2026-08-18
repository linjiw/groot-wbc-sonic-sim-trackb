"""CPU-pure resolution of a vectorized LACE atlas-probe schedule batch."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from gear_sonic.research.lace.schedule import derive_runtime_rng_seed, quantize_start_step
from gear_sonic.research.lace.schema import canonical_sha256

ATLAS_PROBE_ROW_FIELDS = {
    "rollout_id",
    "partition",
    "split_sha256",
    "split_selection_sha256",
    "motion_key",
    "probe_policy_id",
    "checkpoint_sha256",
    "domain_randomization_seed",
    "runtime_rng_seed",
    "phase_id",
    "target_fraction",
    "reference_num_steps",
    "start_step",
    "realized_fraction",
    "repeat_index",
}


@dataclass(frozen=True)
class AtlasProbeBatch:
    """Resolved fixed assignments for one homogeneous CRN schedule cell."""

    rollout_ids: tuple[str, ...]
    motion_keys: tuple[str, ...]
    motion_ids: tuple[int, ...]
    start_steps: tuple[int, ...]
    reference_num_steps: tuple[int, ...]
    probe_policy_id: str
    checkpoint_sha256: str
    domain_randomization_seed: int
    runtime_rng_seed: int
    phase_id: str
    target_fraction: float
    repeat_index: int
    schedule_sha256: str
    split_sha256: str
    split_selection_sha256: str

    @property
    def num_envs(self) -> int:
        return len(self.motion_ids)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    _require(value == value.lower(), f"{name} must use lowercase hexadecimal")
    return value


def _exact_rollout_id(row: Mapping[str, Any]) -> str:
    rollout_id = row["rollout_id"]
    _require(isinstance(rollout_id, str) and ":" in rollout_id, "rollout_id is invalid")
    prefix, declared_digest = rollout_id.split(":", 1)
    _require(prefix and len(declared_digest) == 64, "rollout_id is invalid")
    identity = {
        "identity_version": 1,
        "split_sha256": row["split_sha256"],
        "split_selection_sha256": row["split_selection_sha256"],
        "motion_key": row["motion_key"],
        "probe_policy_id": row["probe_policy_id"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "domain_randomization_seed": row["domain_randomization_seed"],
        "runtime_rng_seed": row["runtime_rng_seed"],
        "phase_id": row["phase_id"],
        "target_fraction": row["target_fraction"],
        "reference_num_steps": row["reference_num_steps"],
        "start_step": row["start_step"],
        "realized_fraction": row["realized_fraction"],
        "repeat_index": row["repeat_index"],
    }
    _require(
        declared_digest == canonical_sha256(identity),
        "rollout_id does not bind the exact atlas-probe row",
    )
    return rollout_id


def resolve_atlas_probe_batch(
    assignments: Sequence[Mapping[str, Any]],
    *,
    schedule_sha256: str,
    loaded_motion_keys: Sequence[str],
    loaded_reference_num_steps: Sequence[int],
    num_envs: int,
) -> AtlasProbeBatch:
    """Resolve schedule motion keys to live IDs and validate exact clip lengths.

    A batch must be one policy/DR/phase/repeat cell across distinct motions. This
    is the batching shape that preserves common random numbers: the runner can
    seed the vectorized reset once with ``runtime_rng_seed``, while every policy
    comparison uses the same row set and reset seed.
    """

    _require(
        isinstance(num_envs, int) and not isinstance(num_envs, bool) and num_envs > 0,
        "num_envs must be a positive integer",
    )
    frozen_schedule_sha256 = _sha256(schedule_sha256, "schedule_sha256")
    _require(
        isinstance(assignments, Sequence) and not isinstance(assignments, (str, bytes)),
        "atlas_probe_assignments must be a sequence",
    )
    rows = list(assignments)
    _require(
        len(rows) == num_envs,
        f"atlas_probe_assignments length {len(rows)} does not match num_envs {num_envs}",
    )
    _require(
        isinstance(loaded_motion_keys, Sequence)
        and not isinstance(loaded_motion_keys, (str, bytes))
        and bool(loaded_motion_keys),
        "loaded_motion_keys must be a non-empty sequence",
    )
    loaded_keys = list(loaded_motion_keys)
    _require(
        all(isinstance(key, str) and key for key in loaded_keys),
        "loaded_motion_keys must contain non-empty strings",
    )
    _require(len(loaded_keys) == len(set(loaded_keys)), "loaded_motion_keys must be unique")
    _require(
        isinstance(loaded_reference_num_steps, Sequence)
        and not isinstance(loaded_reference_num_steps, (str, bytes)),
        "loaded_reference_num_steps must be a sequence",
    )
    loaded_steps = list(loaded_reference_num_steps)
    _require(
        len(loaded_steps) == len(loaded_keys),
        "loaded motion keys and reference lengths must have equal length",
    )
    _require(
        all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 2
            for value in loaded_steps
        ),
        "loaded_reference_num_steps must contain integers >= 2",
    )
    motion_id_by_key = {key: index for index, key in enumerate(loaded_keys)}

    rollout_ids: list[str] = []
    motion_keys: list[str] = []
    motion_ids: list[int] = []
    start_steps: list[int] = []
    reference_steps: list[int] = []
    common_conditions: set[tuple[Any, ...]] = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"atlas_probe_assignments[{index}] must be a mapping")
        _require(
            set(row) == ATLAS_PROBE_ROW_FIELDS,
            f"atlas_probe_assignments[{index}] does not match the exact schedule-row schema",
        )
        _require(row["partition"] == "D_atlas", f"atlas_probe_assignments[{index}] is not D_atlas")
        split_sha256 = _sha256(row["split_sha256"], f"assignments[{index}].split_sha256")
        selection_sha256 = _sha256(
            row["split_selection_sha256"],
            f"assignments[{index}].split_selection_sha256",
        )
        policy_id = row["probe_policy_id"]
        _require(
            isinstance(policy_id, str) and policy_id, f"assignments[{index}] policy is invalid"
        )
        checkpoint = _sha256(row["checkpoint_sha256"], f"assignments[{index}].checkpoint")
        seed = row["domain_randomization_seed"]
        repeat_index = row["repeat_index"]
        runtime_seed = row["runtime_rng_seed"]
        phase_id = row["phase_id"]
        _require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"assignments[{index}] DR seed must be an integer",
        )
        _require(
            isinstance(repeat_index, int)
            and not isinstance(repeat_index, bool)
            and repeat_index >= 0,
            f"assignments[{index}] repeat_index is invalid",
        )
        _require(
            isinstance(phase_id, str) and phase_id,
            f"assignments[{index}] phase_id is invalid",
        )
        _require(
            isinstance(runtime_seed, int)
            and not isinstance(runtime_seed, bool)
            and 0 <= runtime_seed <= 0xFFFFFFFF,
            f"assignments[{index}] runtime_rng_seed is invalid",
        )
        _require(
            runtime_seed == derive_runtime_rng_seed(seed, phase_id, repeat_index),
            f"assignments[{index}] runtime_rng_seed violates the frozen derivation",
        )
        target = row["target_fraction"]
        _require(
            isinstance(target, (int, float))
            and not isinstance(target, bool)
            and math.isfinite(float(target))
            and 0.0 <= float(target) <= 1.0,
            f"assignments[{index}] target_fraction is invalid",
        )
        motion_key = row["motion_key"]
        _require(
            isinstance(motion_key, str) and motion_key in motion_id_by_key,
            f"assignments[{index}] motion_key is absent from the loaded motion library",
        )
        motion_id = motion_id_by_key[motion_key]
        declared_steps = row["reference_num_steps"]
        _require(
            isinstance(declared_steps, int)
            and not isinstance(declared_steps, bool)
            and declared_steps >= 2,
            f"assignments[{index}] reference_num_steps is invalid",
        )
        actual_steps = loaded_steps[motion_id]
        _require(
            declared_steps == actual_steps,
            f"assignments[{index}] reference_num_steps={declared_steps} does not match "
            f"loaded length {actual_steps} for {motion_key!r}",
        )
        start_step = row["start_step"]
        _require(
            isinstance(start_step, int)
            and not isinstance(start_step, bool)
            and start_step == quantize_start_step(float(target), declared_steps),
            f"assignments[{index}] start_step violates the frozen quantization",
        )
        realized = row["realized_fraction"]
        _require(
            isinstance(realized, (int, float))
            and not isinstance(realized, bool)
            and math.isfinite(float(realized))
            and math.isclose(
                float(realized),
                start_step / (declared_steps - 1),
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"assignments[{index}] realized_fraction is inconsistent",
        )
        rollout_ids.append(_exact_rollout_id(row))
        motion_keys.append(motion_key)
        motion_ids.append(motion_id)
        start_steps.append(start_step)
        reference_steps.append(declared_steps)
        common_conditions.add(
            (
                split_sha256,
                selection_sha256,
                policy_id,
                checkpoint,
                seed,
                runtime_seed,
                phase_id,
                float(target),
                repeat_index,
            )
        )

    _require(len(set(rollout_ids)) == num_envs, "atlas-probe rollout_ids must be unique")
    _require(len(set(motion_keys)) == num_envs, "atlas-probe motion_keys must be unique per batch")
    _require(
        len(common_conditions) == 1,
        "atlas-probe batch must share policy, DR seed, runtime seed, phase, and repeat",
    )
    condition = next(iter(common_conditions))
    return AtlasProbeBatch(
        rollout_ids=tuple(rollout_ids),
        motion_keys=tuple(motion_keys),
        motion_ids=tuple(motion_ids),
        start_steps=tuple(start_steps),
        reference_num_steps=tuple(reference_steps),
        split_sha256=str(condition[0]),
        split_selection_sha256=str(condition[1]),
        probe_policy_id=str(condition[2]),
        checkpoint_sha256=str(condition[3]),
        domain_randomization_seed=int(condition[4]),
        runtime_rng_seed=int(condition[5]),
        phase_id=str(condition[6]),
        target_fraction=float(condition[7]),
        repeat_index=int(condition[8]),
        schedule_sha256=frozen_schedule_sha256,
    )


def require_runtime_rng_seed_readback(
    batch: AtlasProbeBatch,
    process_seed_readback: Any,
    *,
    source: str = "env.cfg.seed",
) -> int:
    """Verify the runner's applied process seed against the frozen CRN seed.

    SONIC's evaluation entrypoint seeds Python/NumPy/Torch, forwards that same
    value to ``AppLauncher``, and stores it in the live environment config.  A
    recorder can therefore use ``env.cfg.seed`` as the explicit runner
    readback.  This proves which seed the runner applied; it deliberately does
    *not* claim that realized domain-randomization parameters were captured.
    """

    _require(isinstance(batch, AtlasProbeBatch), "batch must be an AtlasProbeBatch")
    _require(isinstance(source, str) and source, "seed readback source must be non-empty")
    _require(
        isinstance(process_seed_readback, int) and not isinstance(process_seed_readback, bool),
        f"{source} must be an integer runtime seed readback",
    )
    applied = int(process_seed_readback)
    _require(
        applied == batch.runtime_rng_seed,
        f"{source}={applied} does not match atlas runtime_rng_seed={batch.runtime_rng_seed}",
    )
    return applied
