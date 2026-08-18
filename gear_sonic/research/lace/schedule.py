"""Deterministic, simulator-independent rollout schedules for the LACE atlas.

The schedule is the scientific unit of intended atlas collection.  It binds a
frozen split, exact motion selection, ordered policy checkpoints, common
random-number seeds, named phase targets, and repeat indices before a simulator
is launched.  Integer start steps are materialized per motion because nominal
fractions do not identify the same discrete frame on clips of different length.
"""

from __future__ import annotations

from copy import deepcopy
from decimal import ROUND_FLOOR, Decimal
import math
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.reference_lengths import (
    REFERENCE_LENGTH_DIGEST_FIELD,
    REFERENCE_LENGTH_KIND,
    REFERENCE_LENGTH_SCHEMA_VERSION,
    reference_num_steps_by_motion,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schema import canonical_sha256, validate_split_manifest

SCHEDULE_KIND = "lace_probe_rollout_schedule"
LEGACY_SCHEDULE_SCHEMA_VERSION = 1
SCHEDULE_SCHEMA_VERSION = 2
SUPPORTED_SCHEDULE_SCHEMA_VERSIONS = (
    LEGACY_SCHEDULE_SCHEMA_VERSION,
    SCHEDULE_SCHEMA_VERSION,
)
SCHEDULE_MODES = ("scientific", "pilot")
SCHEDULE_DIGEST_FIELD = "schedule_sha256"
QUANTIZATION_RULE_ID = "nearest_integer_ties_to_lower_v1"
RUNTIME_RNG_SEED_DERIVATION_ID = "sha256_canonical_json_uint32_prefix_v1"
DOMAIN_RANDOMIZATION_SEED_SEMANTICS = (
    "declared_experimental_condition; runtime_rng_seed is the per-tuple reseed value"
)

QUANTIZATION_SPEC = {
    "id": QUANTIZATION_RULE_ID,
    "continuous_index": "target_fraction * (reference_num_steps - 1)",
    "rounding": "nearest integer; exact half-integer ties choose the lower integer",
    "valid_start_step": "0 <= start_step <= reference_num_steps - 1",
    "realized_fraction": "start_step / (reference_num_steps - 1)",
}

RUNTIME_RNG_SEED_DERIVATION_SPEC = {
    "id": RUNTIME_RNG_SEED_DERIVATION_ID,
    "namespace": "lace-runtime-rng-v1",
    "inputs": ["domain_randomization_seed", "phase_id", "repeat_index"],
    "canonicalization": (
        "UTF-8 JSON with allow_nan=false, ensure_ascii=false, separators=(',', ':'), "
        "sort_keys=true"
    ),
    "hash": "SHA-256",
    "extraction": "first 8 hexadecimal digest digits interpreted as unsigned uint32",
    "range": "0 <= runtime_rng_seed <= 4294967295",
    "pairing": "shared across all motions and probe policies for the same input tuple",
    "collision_policy": "fail schedule construction or validation",
}

_CARTESIAN_ORDER = [
    "motion_key",
    "probe_policy_id",
    "domain_randomization_seed",
    "phase_id",
    "repeat_index",
]

_ROLLOUT_FIELDS = {
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

_RUNTIME_SEED_FIELDS = {
    "domain_randomization_seed",
    "phase_id",
    "repeat_index",
    "runtime_rng_seed",
}

_REFERENCE_LENGTH_BINDING_FIELDS = {
    "kind",
    "schema_version",
    REFERENCE_LENGTH_DIGEST_FIELD,
    "artifact_mode",
    "scientific_use",
    "split_sha256",
    "split_selection_sha256",
    "partition",
    "selected_motion_keys",
    "selection_complete_for_d_atlas",
    "source_file_set_sha256",
    "target_fps",
    "runtime_contract",
    "dataset_provenance",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_sha256(value: Any, name: str) -> str:
    _require(isinstance(value, str) and len(value) == 64, f"{name} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be hexadecimal") from error
    return value


def _require_sequence(value: Any, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a sequence",
    )
    return value


def quantize_start_step(target_fraction: float, reference_num_steps: int) -> int:
    """Quantize a phase target to a valid frame, choosing lower on exact ties.

    Decimal arithmetic over the JSON-style decimal spelling avoids letting
    binary floating-point noise decide which side of a half-integer wins.
    Clips require at least two frames so both the target and realized fractions
    have an unambiguous ``[0, 1]`` coordinate.
    """

    _require(
        isinstance(reference_num_steps, int)
        and not isinstance(reference_num_steps, bool)
        and reference_num_steps >= 2,
        "reference_num_steps must be an integer >= 2",
    )
    _require(
        not isinstance(target_fraction, bool),
        "target_fraction must be numeric, not boolean",
    )
    try:
        fraction = float(target_fraction)
    except (TypeError, ValueError) as error:
        raise ValueError("target_fraction must be numeric") from error
    _require(
        math.isfinite(fraction) and 0.0 <= fraction <= 1.0,
        "target_fraction must be finite and in [0, 1]",
    )

    continuous_index = Decimal(str(fraction)) * Decimal(reference_num_steps - 1)
    lower = int(continuous_index.to_integral_value(rounding=ROUND_FLOOR))
    if continuous_index - Decimal(lower) > Decimal("0.5"):
        lower += 1
    return min(max(lower, 0), reference_num_steps - 1)


def derive_runtime_rng_seed(
    domain_randomization_seed: int,
    phase_id: str,
    repeat_index: int,
) -> int:
    """Derive the runner seed while preserving common random numbers.

    The declared domain-randomization seed remains an experimental condition.
    This derived value is what a runner must use when it reseeds a concrete
    tuple.  Including ``repeat_index`` prevents nominal repeats from replaying
    identical randomness, while omitting motion and policy deliberately pairs
    those axes.
    """

    _require(
        isinstance(domain_randomization_seed, int)
        and not isinstance(domain_randomization_seed, bool),
        "domain_randomization_seed must be an integer",
    )
    _require(isinstance(phase_id, str) and phase_id, "phase_id must be a non-empty string")
    _require(
        isinstance(repeat_index, int) and not isinstance(repeat_index, bool) and repeat_index >= 0,
        "repeat_index must be a nonnegative integer",
    )
    payload = {
        "namespace": RUNTIME_RNG_SEED_DERIVATION_SPEC["namespace"],
        "domain_randomization_seed": domain_randomization_seed,
        "phase_id": phase_id,
        "repeat_index": repeat_index,
    }
    return int(canonical_sha256(payload)[:8], 16)


def _validate_policies(raw: Any) -> list[dict[str, str]]:
    policies = _require_sequence(raw, "probe_policies")
    _require(bool(policies), "probe_policies must be non-empty")
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(policies):
        _require(isinstance(item, Mapping), f"probe_policies[{index}] must be a mapping")
        policy_id = item.get("id")
        _require(
            isinstance(policy_id, str) and policy_id,
            f"probe_policies[{index}].id must be non-empty",
        )
        _require(policy_id not in seen, f"duplicate probe policy id: {policy_id}")
        seen.add(policy_id)
        checkpoint = _require_sha256(
            item.get("checkpoint_sha256"),
            f"probe_policies[{index}].checkpoint_sha256",
        )
        result.append({"id": policy_id, "checkpoint_sha256": checkpoint})
    return result


def _validate_seeds(raw: Any) -> list[int]:
    seeds = _require_sequence(raw, "domain_randomization_seeds")
    _require(bool(seeds), "domain_randomization_seeds must be non-empty")
    result = list(seeds)
    _require(
        all(isinstance(seed, int) and not isinstance(seed, bool) for seed in result),
        "domain_randomization_seeds must contain integers",
    )
    _require(len(result) == len(set(result)), "domain_randomization_seeds must be unique")
    return result


def _validate_phase_targets(raw: Any) -> list[dict[str, str | float]]:
    phases = _require_sequence(raw, "phase_targets")
    _require(bool(phases), "phase_targets must be non-empty")
    result: list[dict[str, str | float]] = []
    seen: set[str] = set()
    for index, item in enumerate(phases):
        _require(isinstance(item, Mapping), f"phase_targets[{index}] must be a mapping")
        phase_id = item.get("phase_id")
        _require(
            isinstance(phase_id, str) and phase_id,
            f"phase_targets[{index}].phase_id must be non-empty",
        )
        _require(phase_id not in seen, f"duplicate phase_id: {phase_id}")
        seen.add(phase_id)
        target = item.get("target_fraction")
        _require(
            not isinstance(target, bool),
            f"phase_targets[{index}].target_fraction must not be boolean",
        )
        try:
            fraction = float(target)
        except (TypeError, ValueError) as error:
            raise ValueError(f"phase_targets[{index}].target_fraction must be numeric") from error
        _require(
            math.isfinite(fraction) and 0.0 <= fraction <= 1.0,
            f"phase_targets[{index}].target_fraction must be in [0, 1]",
        )
        result.append({"phase_id": phase_id, "target_fraction": fraction})
    return result


def _validate_repeat_count(value: Any) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        "repeats must be a positive integer",
    )
    return int(value)


def _build_runtime_seed_schedule(
    seeds: Sequence[int],
    phases: Sequence[Mapping[str, str | float]],
    repeat_count: int,
) -> list[dict[str, int | str]]:
    records: list[dict[str, int | str]] = []
    identities_by_runtime_seed: dict[int, tuple[int, str, int]] = {}
    for seed in seeds:
        for phase in phases:
            phase_id = str(phase["phase_id"])
            for repeat_index in range(repeat_count):
                identity = (seed, phase_id, repeat_index)
                runtime_seed = derive_runtime_rng_seed(*identity)
                previous = identities_by_runtime_seed.setdefault(runtime_seed, identity)
                _require(
                    previous == identity,
                    "runtime RNG seed collision between "
                    f"{previous} and {identity}: {runtime_seed}",
                )
                records.append(
                    {
                        "domain_randomization_seed": seed,
                        "phase_id": phase_id,
                        "repeat_index": repeat_index,
                        "runtime_rng_seed": runtime_seed,
                    }
                )
    return records


def _validate_reference_steps(
    raw: Mapping[str, int],
    selected_motion_keys: Sequence[str],
) -> dict[str, int]:
    _require(isinstance(raw, Mapping), "reference_num_steps must be a mapping")
    selected = set(selected_motion_keys)
    _require(
        set(raw) == selected,
        "reference_num_steps keys must exactly match selected_motion_keys",
    )
    result: dict[str, int] = {}
    for motion_key in selected_motion_keys:
        value = raw[motion_key]
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 2,
            f"reference_num_steps[{motion_key!r}] must be an integer >= 2",
        )
        result[motion_key] = int(value)
    return result


def _reference_length_binding(
    inventory: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact inventory coordinates frozen into schema-v2 schedules."""

    validate_reference_length_inventory(
        inventory,
        split_manifest=split_manifest,
        verify_digest=True,
    )
    binding = {
        "kind": inventory["kind"],
        "schema_version": inventory["schema_version"],
        REFERENCE_LENGTH_DIGEST_FIELD: inventory[REFERENCE_LENGTH_DIGEST_FIELD],
        "artifact_mode": inventory["artifact_mode"],
        "scientific_use": inventory["scientific_use"],
        "split_sha256": inventory["split_sha256"],
        "split_selection_sha256": inventory["split_selection_sha256"],
        "partition": inventory["partition"],
        "selected_motion_keys": list(inventory["selected_motion_keys"]),
        "selection_complete_for_d_atlas": inventory["selection_complete_for_d_atlas"],
        "source_file_set_sha256": inventory["source_file_set_sha256"],
        "target_fps": inventory["target_fps"],
        "runtime_contract": deepcopy(inventory["runtime_contract"]),
        "dataset_provenance": deepcopy(inventory["dataset_provenance"]),
    }
    _require(
        binding["kind"] == REFERENCE_LENGTH_KIND
        and binding["schema_version"] == REFERENCE_LENGTH_SCHEMA_VERSION,
        "reference-length inventory kind/schema mismatch",
    )
    return binding


def _validate_reference_length_binding(
    raw: Any,
    *,
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any] | None,
    split_manifest: Mapping[str, Any] | None,
) -> None:
    _require(
        isinstance(raw, Mapping),
        "schema-v2 schedule requires reference_length_inventory_binding",
    )
    _require(
        set(raw) == _REFERENCE_LENGTH_BINDING_FIELDS,
        "reference_length_inventory_binding fields are invalid",
    )
    _require(
        raw.get("kind") == REFERENCE_LENGTH_KIND,
        "reference-length binding kind mismatch",
    )
    _require(
        raw.get("schema_version") == REFERENCE_LENGTH_SCHEMA_VERSION,
        "reference-length binding schema version mismatch",
    )
    inventory_digest = _require_sha256(
        raw.get(REFERENCE_LENGTH_DIGEST_FIELD),
        f"reference_length_inventory_binding.{REFERENCE_LENGTH_DIGEST_FIELD}",
    )
    for binding_field, schedule_field in (
        ("artifact_mode", "artifact_mode"),
        ("scientific_use", "scientific_use"),
        ("split_sha256", "split_sha256"),
        ("split_selection_sha256", "split_selection_sha256"),
        ("partition", "partition"),
        ("selected_motion_keys", "selected_motion_keys"),
        ("selection_complete_for_d_atlas", "selection_complete_for_d_atlas"),
    ):
        _require(
            raw.get(binding_field) == manifest.get(schedule_field),
            f"reference-length binding {binding_field} does not match the schedule",
        )
    _require_sha256(
        raw.get("source_file_set_sha256"),
        "reference_length_inventory_binding.source_file_set_sha256",
    )
    target_fps = raw.get("target_fps")
    _require(
        isinstance(target_fps, int) and not isinstance(target_fps, bool) and target_fps > 0,
        "reference_length_inventory_binding.target_fps must be a positive integer",
    )
    runtime_contract = raw.get("runtime_contract")
    _require(
        isinstance(runtime_contract, Mapping),
        "reference_length_inventory_binding.runtime_contract must be a mapping",
    )
    _require(
        runtime_contract.get("target_fps") == target_fps
        and runtime_contract.get("sim_fps") == target_fps
        and runtime_contract.get("motion_fps_scale") == {"numerator": 1, "denominator": 1}
        and runtime_contract.get("max_len") == -1
        and runtime_contract.get("reference_num_steps_equals_target_num_frames") is True,
        "reference-length binding does not preserve the frozen runtime configuration",
    )
    _require(
        isinstance(raw.get("dataset_provenance"), Mapping),
        "reference_length_inventory_binding.dataset_provenance must be a mapping",
    )
    if manifest.get("scientific_use") is True:
        _require(
            target_fps == 50
            and runtime_contract.get("sim_fps") == 50
            and runtime_contract.get("motion_fps_scale") == {"numerator": 1, "denominator": 1}
            and runtime_contract.get("max_len") == -1,
            "scientific schedule must bind the release 50 Hz/scale-1/no-crop configuration",
        )
    if inventory is not None:
        expected = _reference_length_binding(
            inventory,
            split_manifest=split_manifest,
        )
        _require(
            dict(raw) == expected,
            "schedule reference-length binding does not match the supplied inventory",
        )
        _require(
            inventory_digest == inventory[REFERENCE_LENGTH_DIGEST_FIELD],
            "schedule reference-length digest does not match the supplied inventory",
        )
        schedule_steps = {
            str(record["motion_key"]): int(record["reference_num_steps"])
            for record in manifest["reference_num_steps"]
        }
        _require(
            schedule_steps == reference_num_steps_by_motion(inventory),
            "schedule reference_num_steps were not derived from the supplied inventory",
        )


def _rollout_identity(
    *,
    split_sha256: str,
    split_selection_sha256: str,
    motion_key: str,
    policy_id: str,
    checkpoint_sha256: str,
    seed: int,
    runtime_rng_seed: int,
    phase_id: str,
    target_fraction: float,
    reference_num_steps: int,
    start_step: int,
    realized_fraction: float,
    repeat_index: int,
) -> dict[str, Any]:
    return {
        "identity_version": 1,
        "split_sha256": split_sha256,
        "split_selection_sha256": split_selection_sha256,
        "motion_key": motion_key,
        "probe_policy_id": policy_id,
        "checkpoint_sha256": checkpoint_sha256,
        "domain_randomization_seed": seed,
        "runtime_rng_seed": runtime_rng_seed,
        "phase_id": phase_id,
        "target_fraction": target_fraction,
        "reference_num_steps": reference_num_steps,
        "start_step": start_step,
        "realized_fraction": realized_fraction,
        "repeat_index": repeat_index,
    }


def _expected_rollout_id(prefix: str, identity: Mapping[str, Any]) -> str:
    return f"{prefix}:{canonical_sha256(identity)}"


def build_rollout_schedule(
    split_manifest: Mapping[str, Any],
    *,
    reference_length_inventory: Mapping[str, Any],
    probe_policies: Sequence[Mapping[str, Any]],
    domain_randomization_seeds: Sequence[int],
    phase_targets: Sequence[Mapping[str, Any]],
    repeats: int,
    rollout_id_prefix: str = "lace-rollout",
) -> dict[str, Any]:
    """Build a schema-v2 schedule whose lengths come only from an inventory."""

    validate_split_manifest(split_manifest)
    inventory_binding = _reference_length_binding(
        reference_length_inventory,
        split_manifest=split_manifest,
    )
    artifact_mode = str(reference_length_inventory["artifact_mode"])
    _require(artifact_mode in SCHEDULE_MODES, f"artifact_mode must be one of {SCHEDULE_MODES}")
    _require(
        isinstance(rollout_id_prefix, str) and rollout_id_prefix,
        "rollout_id_prefix must be a non-empty string",
    )
    _require(":" not in rollout_id_prefix, "rollout_id_prefix must not contain ':'")

    selected = list(reference_length_inventory["selected_motion_keys"])

    policies = _validate_policies(probe_policies)
    seeds = _validate_seeds(domain_randomization_seeds)
    phases = _validate_phase_targets(phase_targets)
    repeat_count = _validate_repeat_count(repeats)
    repeat_indices = list(range(repeat_count))
    runtime_seed_schedule = _build_runtime_seed_schedule(seeds, phases, repeat_count)
    runtime_seed_by_condition = {
        (
            int(record["domain_randomization_seed"]),
            str(record["phase_id"]),
            int(record["repeat_index"]),
        ): int(record["runtime_rng_seed"])
        for record in runtime_seed_schedule
    }
    steps_by_motion = _validate_reference_steps(
        reference_num_steps_by_motion(reference_length_inventory),
        selected,
    )

    split_sha256 = str(split_manifest["split_sha256"])
    split_selection_sha256 = str(split_manifest["selection_sha256"])
    rollouts: list[dict[str, Any]] = []
    for motion_key in selected:
        num_steps = steps_by_motion[motion_key]
        for policy in policies:
            for seed in seeds:
                for phase in phases:
                    target_fraction = float(phase["target_fraction"])
                    start_step = quantize_start_step(target_fraction, num_steps)
                    realized_fraction = start_step / (num_steps - 1)
                    for repeat_index in repeat_indices:
                        runtime_rng_seed = runtime_seed_by_condition[
                            (seed, str(phase["phase_id"]), repeat_index)
                        ]
                        identity = _rollout_identity(
                            split_sha256=split_sha256,
                            split_selection_sha256=split_selection_sha256,
                            motion_key=motion_key,
                            policy_id=policy["id"],
                            checkpoint_sha256=policy["checkpoint_sha256"],
                            seed=seed,
                            runtime_rng_seed=runtime_rng_seed,
                            phase_id=str(phase["phase_id"]),
                            target_fraction=target_fraction,
                            reference_num_steps=num_steps,
                            start_step=start_step,
                            realized_fraction=realized_fraction,
                            repeat_index=repeat_index,
                        )
                        rollouts.append(
                            {
                                "rollout_id": _expected_rollout_id(rollout_id_prefix, identity),
                                "partition": "D_atlas",
                                **{
                                    key: value
                                    for key, value in identity.items()
                                    if key != "identity_version"
                                },
                            }
                        )

    manifest: dict[str, Any] = {
        "kind": SCHEDULE_KIND,
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "artifact_mode": artifact_mode,
        "scientific_use": artifact_mode == "scientific",
        "pilot_status": (
            None
            if artifact_mode == "scientific"
            else "non_scientific_subset_for_contract_or_runtime_validation_only"
        ),
        "split_sha256": split_sha256,
        "split_selection_sha256": split_selection_sha256,
        "partition": "D_atlas",
        "selected_motion_keys": selected,
        "selection_complete_for_d_atlas": artifact_mode == "scientific",
        "motion_count": len(selected),
        "reference_length_inventory_binding": inventory_binding,
        "reference_num_steps": [
            {"motion_key": motion_key, "reference_num_steps": steps_by_motion[motion_key]}
            for motion_key in selected
        ],
        "probe_policies": policies,
        "policy_order": [policy["id"] for policy in policies],
        "domain_randomization_seeds": seeds,
        "domain_randomization_seed_semantics": DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "phase_targets": phases,
        "repeat_count": repeat_count,
        "repeat_indices": repeat_indices,
        "runtime_rng_seed_derivation": dict(RUNTIME_RNG_SEED_DERIVATION_SPEC),
        "runtime_rng_seed_schedule": runtime_seed_schedule,
        "quantization": dict(QUANTIZATION_SPEC),
        "rollout_id_prefix": rollout_id_prefix,
        "cartesian_order": list(_CARTESIAN_ORDER),
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    manifest[SCHEDULE_DIGEST_FIELD] = canonical_sha256(
        manifest,
        digest_field=SCHEDULE_DIGEST_FIELD,
    )
    validate_rollout_schedule(
        manifest,
        split_manifest=split_manifest,
        reference_length_inventory=reference_length_inventory,
    )
    return manifest


def validate_rollout_schedule(
    manifest: Mapping[str, Any],
    *,
    split_manifest: Mapping[str, Any] | None = None,
    reference_length_inventory: Mapping[str, Any] | None = None,
    verify_digest: bool = True,
) -> None:
    """Validate bindings, Cartesian coverage, quantization, and pairing.

    Schema v1 remains readable for frozen historical artifacts. Schema v2 is
    the only buildable format and can be externally revalidated against the
    exact split and reference-length inventory that produced it.
    """

    _require(manifest.get("kind") == SCHEDULE_KIND, f"kind must be {SCHEDULE_KIND!r}")
    schema_version = manifest.get("schema_version")
    _require(
        schema_version in SUPPORTED_SCHEDULE_SCHEMA_VERSIONS,
        "unsupported rollout schedule schema version",
    )
    if schema_version == LEGACY_SCHEDULE_SCHEMA_VERSION:
        _require(
            reference_length_inventory is None,
            "schema-v1 schedules cannot claim a reference-length inventory binding",
        )
    else:
        _require(
            (split_manifest is None) == (reference_length_inventory is None),
            "schema-v2 external validation requires split_manifest and "
            "reference_length_inventory together",
        )
    artifact_mode = manifest.get("artifact_mode")
    _require(artifact_mode in SCHEDULE_MODES, f"artifact_mode must be one of {SCHEDULE_MODES}")
    _require(
        manifest.get("scientific_use") is (artifact_mode == "scientific"),
        "scientific_use must agree with artifact_mode",
    )
    expected_pilot_status = (
        None
        if artifact_mode == "scientific"
        else "non_scientific_subset_for_contract_or_runtime_validation_only"
    )
    _require(
        manifest.get("pilot_status") == expected_pilot_status,
        "pilot_status must explicitly mark pilot schedules non-scientific",
    )
    split_sha256 = _require_sha256(manifest.get("split_sha256"), "split_sha256")
    split_selection_sha256 = _require_sha256(
        manifest.get("split_selection_sha256"),
        "split_selection_sha256",
    )
    _require(manifest.get("partition") == "D_atlas", "partition must be D_atlas")

    selected_raw = _require_sequence(manifest.get("selected_motion_keys"), "selected_motion_keys")
    selected = list(selected_raw)
    _require(bool(selected), "selected_motion_keys must be non-empty")
    _require(
        all(isinstance(key, str) and key for key in selected),
        "selected_motion_keys must contain non-empty strings",
    )
    _require(
        selected == sorted(set(selected)),
        "selected_motion_keys must be unique and canonically sorted",
    )
    _require(
        manifest.get("selection_complete_for_d_atlas") is (artifact_mode == "scientific"),
        "selection_complete_for_d_atlas must agree with artifact_mode",
    )
    _require(manifest.get("motion_count") == len(selected), "motion_count mismatch")

    reference_records = _require_sequence(
        manifest.get("reference_num_steps"),
        "reference_num_steps",
    )
    _require(
        len(reference_records) == len(selected),
        "reference_num_steps must contain one record per selected motion",
    )
    steps_by_motion: dict[str, int] = {}
    for index, item in enumerate(reference_records):
        _require(isinstance(item, Mapping), f"reference_num_steps[{index}] must be a mapping")
        motion_key = item.get("motion_key")
        num_steps = item.get("reference_num_steps")
        _require(
            motion_key == selected[index],
            "reference_num_steps records must follow selected_motion_keys order",
        )
        _require(
            isinstance(num_steps, int) and not isinstance(num_steps, bool) and num_steps >= 2,
            f"reference_num_steps[{index}] must be an integer >= 2",
        )
        steps_by_motion[str(motion_key)] = int(num_steps)

    if schema_version == SCHEDULE_SCHEMA_VERSION:
        _validate_reference_length_binding(
            manifest.get("reference_length_inventory_binding"),
            manifest=manifest,
            inventory=reference_length_inventory,
            split_manifest=split_manifest,
        )
    else:
        _require(
            "reference_length_inventory_binding" not in manifest,
            "schema-v1 schedule must not contain a schema-v2 inventory binding",
        )

    policies = _validate_policies(manifest.get("probe_policies"))
    policy_ids = [policy["id"] for policy in policies]
    _require(manifest.get("policy_order") == policy_ids, "policy_order mismatch")
    checkpoint_by_policy = {policy["id"]: policy["checkpoint_sha256"] for policy in policies}
    seeds = _validate_seeds(manifest.get("domain_randomization_seeds"))
    _require(
        manifest.get("domain_randomization_seed_semantics") == DOMAIN_RANDOMIZATION_SEED_SEMANTICS,
        "domain_randomization_seed_semantics mismatch",
    )
    phases = _validate_phase_targets(manifest.get("phase_targets"))
    repeat_count = _validate_repeat_count(manifest.get("repeat_count"))
    repeat_indices = manifest.get("repeat_indices")
    _require(
        repeat_indices == list(range(repeat_count)),
        "repeat_indices must be contiguous from zero to repeat_count - 1",
    )
    _require(
        manifest.get("runtime_rng_seed_derivation") == RUNTIME_RNG_SEED_DERIVATION_SPEC,
        "runtime_rng_seed_derivation spec mismatch",
    )
    expected_runtime_seed_schedule = _build_runtime_seed_schedule(seeds, phases, repeat_count)
    runtime_seed_records_raw = _require_sequence(
        manifest.get("runtime_rng_seed_schedule"),
        "runtime_rng_seed_schedule",
    )
    runtime_seed_records = list(runtime_seed_records_raw)
    _require(
        all(
            isinstance(record, Mapping) and set(record) == _RUNTIME_SEED_FIELDS
            for record in runtime_seed_records
        ),
        "runtime_rng_seed_schedule records must match the exact schema",
    )
    declared_runtime_seeds = [record["runtime_rng_seed"] for record in runtime_seed_records]
    _require(
        all(
            isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= 0xFFFFFFFF
            for seed in declared_runtime_seeds
        ),
        "runtime_rng_seed values must be uint32 integers",
    )
    _require(
        len(declared_runtime_seeds) == len(set(declared_runtime_seeds)),
        "runtime RNG seed collision in runtime_rng_seed_schedule",
    )
    _require(
        runtime_seed_records == expected_runtime_seed_schedule,
        "runtime_rng_seed_schedule does not match the frozen derivation",
    )
    runtime_seed_by_condition = {
        (
            int(record["domain_randomization_seed"]),
            str(record["phase_id"]),
            int(record["repeat_index"]),
        ): int(record["runtime_rng_seed"])
        for record in expected_runtime_seed_schedule
    }
    _require(manifest.get("quantization") == QUANTIZATION_SPEC, "quantization spec mismatch")
    prefix = manifest.get("rollout_id_prefix")
    _require(isinstance(prefix, str) and prefix, "rollout_id_prefix must be non-empty")
    _require(":" not in prefix, "rollout_id_prefix must not contain ':'")
    _require(manifest.get("cartesian_order") == _CARTESIAN_ORDER, "cartesian_order mismatch")

    rollouts_raw = _require_sequence(manifest.get("rollouts"), "rollouts")
    rollouts = list(rollouts_raw)
    expected_count = len(selected) * len(policies) * len(seeds) * len(phases) * repeat_count
    _require(manifest.get("rollout_count") == expected_count, "rollout_count mismatch")
    _require(len(rollouts) == expected_count, "rollouts do not provide full Cartesian coverage")

    phase_by_id = {str(phase["phase_id"]): float(phase["target_fraction"]) for phase in phases}
    expected_keys = [
        (motion_key, policy_id, seed, phase["phase_id"], repeat_index)
        for motion_key in selected
        for policy_id in policy_ids
        for seed in seeds
        for phase in phases
        for repeat_index in range(repeat_count)
    ]
    observed_keys: list[tuple[Any, ...]] = []
    rollout_ids: set[str] = set()
    common_starts: dict[tuple[str, int, str, int], set[tuple[int, float]]] = {}
    common_runtime_seeds: dict[tuple[int, str, int], set[int]] = {}
    for index, item in enumerate(rollouts):
        _require(isinstance(item, Mapping), f"rollouts[{index}] must be a mapping")
        _require(
            set(item) == _ROLLOUT_FIELDS,
            f"rollouts[{index}] fields do not match the exact tuple schema",
        )
        _require(item.get("partition") == "D_atlas", f"rollouts[{index}].partition must be D_atlas")
        _require(
            item.get("split_sha256") == split_sha256,
            f"rollouts[{index}].split_sha256 mismatch",
        )
        _require(
            item.get("split_selection_sha256") == split_selection_sha256,
            f"rollouts[{index}].split_selection_sha256 mismatch",
        )
        motion_key = item.get("motion_key")
        policy_id = item.get("probe_policy_id")
        seed = item.get("domain_randomization_seed")
        runtime_rng_seed = item.get("runtime_rng_seed")
        phase_id = item.get("phase_id")
        repeat_index = item.get("repeat_index")
        key = (motion_key, policy_id, seed, phase_id, repeat_index)
        observed_keys.append(key)
        _require(motion_key in steps_by_motion, f"rollouts[{index}] has undeclared motion")
        _require(policy_id in checkpoint_by_policy, f"rollouts[{index}] has undeclared policy")
        _require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"rollouts[{index}].domain_randomization_seed must be an integer",
        )
        _require(seed in seeds, f"rollouts[{index}] has undeclared DR seed")
        _require(phase_id in phase_by_id, f"rollouts[{index}] has undeclared phase_id")
        _require(
            isinstance(repeat_index, int)
            and not isinstance(repeat_index, bool)
            and repeat_index in range(repeat_count),
            f"rollouts[{index}] has invalid repeat_index",
        )
        _require(
            isinstance(runtime_rng_seed, int)
            and not isinstance(runtime_rng_seed, bool)
            and 0 <= runtime_rng_seed <= 0xFFFFFFFF,
            f"rollouts[{index}].runtime_rng_seed must be a uint32 integer",
        )
        runtime_condition = (int(seed), str(phase_id), int(repeat_index))
        _require(
            runtime_rng_seed == runtime_seed_by_condition[runtime_condition],
            f"rollouts[{index}].runtime_rng_seed violates the frozen derivation",
        )
        _require(
            item.get("checkpoint_sha256") == checkpoint_by_policy[policy_id],
            f"rollouts[{index}] checkpoint does not match its declared policy",
        )
        num_steps = steps_by_motion[str(motion_key)]
        _require(
            item.get("reference_num_steps") == num_steps,
            f"rollouts[{index}] reference_num_steps mismatch",
        )
        target = phase_by_id[str(phase_id)]
        row_target = item.get("target_fraction")
        _require(
            isinstance(row_target, (int, float))
            and not isinstance(row_target, bool)
            and math.isfinite(float(row_target))
            and row_target == target,
            f"rollouts[{index}] target_fraction mismatch",
        )
        start_step = item.get("start_step")
        realized = item.get("realized_fraction")
        _require(
            isinstance(start_step, int) and not isinstance(start_step, bool),
            f"rollouts[{index}].start_step must be an integer",
        )
        _require(
            isinstance(realized, (int, float))
            and not isinstance(realized, bool)
            and math.isfinite(float(realized)),
            f"rollouts[{index}].realized_fraction must be finite",
        )
        pairing_key = (str(motion_key), int(seed), str(phase_id), int(repeat_index))
        common_starts.setdefault(pairing_key, set()).add((start_step, float(realized)))
        common_runtime_seeds.setdefault(runtime_condition, set()).add(runtime_rng_seed)

    _require(
        observed_keys == expected_keys,
        "rollouts do not exactly follow the declared full Cartesian coverage and order",
    )
    _require(
        all(len(values) == 1 for values in common_starts.values()),
        "common integer starts across policies are violated",
    )
    _require(
        all(len(values) == 1 for values in common_runtime_seeds.values()),
        "runtime RNG seeds are not shared across motions and policies",
    )

    for index, item in enumerate(rollouts):
        motion_key = str(item["motion_key"])
        policy_id = str(item["probe_policy_id"])
        seed = int(item["domain_randomization_seed"])
        runtime_rng_seed = int(item["runtime_rng_seed"])
        phase_id = str(item["phase_id"])
        repeat_index = int(item["repeat_index"])
        num_steps = steps_by_motion[motion_key]
        target = phase_by_id[phase_id]
        expected_start = quantize_start_step(target, num_steps)
        _require(
            item["start_step"] == expected_start,
            f"rollouts[{index}].start_step violates {QUANTIZATION_RULE_ID}",
        )
        expected_realized = expected_start / (num_steps - 1)
        _require(
            math.isclose(
                float(item["realized_fraction"]),
                expected_realized,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"rollouts[{index}].realized_fraction does not match start_step",
        )
        identity = _rollout_identity(
            split_sha256=split_sha256,
            split_selection_sha256=split_selection_sha256,
            motion_key=motion_key,
            policy_id=policy_id,
            checkpoint_sha256=checkpoint_by_policy[policy_id],
            seed=seed,
            runtime_rng_seed=runtime_rng_seed,
            phase_id=phase_id,
            target_fraction=target,
            reference_num_steps=num_steps,
            start_step=expected_start,
            realized_fraction=expected_realized,
            repeat_index=repeat_index,
        )
        rollout_id = item.get("rollout_id")
        _require(
            rollout_id == _expected_rollout_id(str(prefix), identity),
            f"rollouts[{index}].rollout_id does not bind its exact tuple",
        )
        _require(rollout_id not in rollout_ids, f"duplicate rollout_id: {rollout_id}")
        rollout_ids.add(str(rollout_id))

    if split_manifest is not None:
        validate_split_manifest(split_manifest)
        _require(
            split_sha256 == split_manifest["split_sha256"],
            "schedule split_sha256 does not match split manifest",
        )
        _require(
            split_selection_sha256 == split_manifest["selection_sha256"],
            "schedule split_selection_sha256 does not match split manifest",
        )
        atlas_keys = {
            record["motion_key"]
            for record in split_manifest["motions"]
            if record["partition"] == "D_atlas"
        }
        _require(
            set(selected) <= atlas_keys,
            "selected_motion_keys include a motion outside D_atlas",
        )
        if artifact_mode == "scientific":
            _require(
                set(selected) == atlas_keys,
                "scientific schedule must exactly cover the full D_atlas partition",
            )
        else:
            _require(
                set(selected) < atlas_keys,
                "pilot selected_motion_keys must be a strict D_atlas subset",
            )

    if verify_digest:
        expected_digest = _require_sha256(
            manifest.get(SCHEDULE_DIGEST_FIELD),
            SCHEDULE_DIGEST_FIELD,
        )
        actual_digest = canonical_sha256(manifest, digest_field=SCHEDULE_DIGEST_FIELD)
        _require(
            expected_digest == actual_digest,
            f"{SCHEDULE_DIGEST_FIELD} mismatch: expected {expected_digest}, computed {actual_digest}",
        )
