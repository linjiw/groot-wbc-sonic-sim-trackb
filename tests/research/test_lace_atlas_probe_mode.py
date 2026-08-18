from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from gear_sonic.research.lace.atlas_probe_mode import (
    require_runtime_rng_seed_readback,
    resolve_atlas_probe_batch,
)
from gear_sonic.research.lace.schedule import derive_runtime_rng_seed, quantize_start_step
from gear_sonic.research.lace.schema import canonical_sha256
from gear_sonic.research.lace.split import build_source_disjoint_split


def _schedule() -> tuple[dict[str, object], dict[str, int]]:
    motions = [
        {
            "motion_key": f"motion_{index:02d}__A{index:03d}",
            "release_filter_key": f"motion_{index:02d}__A{index:03d}.pkl",
            "source_group_id": f"actor_{index:03d}",
            "duration_source_frames": 100 + index,
            "stratum": f"duration_{index % 3}",
        }
        for index in range(25)
    ]
    split = build_source_disjoint_split(motions, seed=83)
    atlas_keys = sorted(
        row["motion_key"] for row in split["motions"] if row["partition"] == "D_atlas"
    )
    reference_steps = {key: 20 + index for index, key in enumerate(atlas_keys)}
    rows = []
    for motion_key in atlas_keys:
        for policy_id, checkpoint_sha256 in (("weak", "a" * 64), ("strong", "b" * 64)):
            num_steps = reference_steps[motion_key]
            start_step = quantize_start_step(0.5, num_steps)
            identity = {
                "identity_version": 1,
                "split_sha256": split["split_sha256"],
                "split_selection_sha256": split["selection_sha256"],
                "motion_key": motion_key,
                "probe_policy_id": policy_id,
                "checkpoint_sha256": checkpoint_sha256,
                "domain_randomization_seed": 101,
                "runtime_rng_seed": derive_runtime_rng_seed(101, "middle", 0),
                "phase_id": "middle",
                "target_fraction": 0.5,
                "reference_num_steps": num_steps,
                "start_step": start_step,
                "realized_fraction": start_step / (num_steps - 1),
                "repeat_index": 0,
            }
            rows.append(
                {
                    "rollout_id": f"lace-rollout:{canonical_sha256(identity)}",
                    "partition": "D_atlas",
                    **{key: value for key, value in identity.items() if key != "identity_version"},
                }
            )
    schedule = {"schedule_sha256": "c" * 64, "rollouts": rows}
    return schedule, reference_steps


def _policy_rows(schedule: dict[str, object], policy_id: str) -> list[dict[str, object]]:
    return [dict(row) for row in schedule["rollouts"] if row["probe_policy_id"] == policy_id]


def _loaded_library(
    reference_steps: dict[str, int],
) -> tuple[list[str], list[int]]:
    loaded_keys = ["unused_motion", *reversed(sorted(reference_steps))]
    loaded_lengths = [12, *(reference_steps[key] for key in loaded_keys[1:])]
    return loaded_keys, loaded_lengths


def test_resolver_binds_exact_motion_ids_starts_and_common_rng_cell() -> None:
    schedule, reference_steps = _schedule()
    rows = _policy_rows(schedule, "weak")
    loaded_keys, loaded_lengths = _loaded_library(reference_steps)

    batch = resolve_atlas_probe_batch(
        rows,
        schedule_sha256=schedule["schedule_sha256"],
        loaded_motion_keys=loaded_keys,
        loaded_reference_num_steps=loaded_lengths,
        num_envs=len(rows),
    )

    assert batch.num_envs == len(rows)
    assert batch.motion_keys == tuple(row["motion_key"] for row in rows)
    assert batch.motion_ids == tuple(loaded_keys.index(key) for key in batch.motion_keys)
    assert batch.start_steps == tuple(row["start_step"] for row in rows)
    assert batch.reference_num_steps == tuple(row["reference_num_steps"] for row in rows)
    assert batch.probe_policy_id == "weak"
    assert batch.domain_randomization_seed == 101
    assert batch.runtime_rng_seed == rows[0]["runtime_rng_seed"]
    assert batch.phase_id == "middle"
    assert batch.repeat_index == 0
    assert batch.schedule_sha256 == schedule["schedule_sha256"]
    assert (
        require_runtime_rng_seed_readback(batch, batch.runtime_rng_seed) == batch.runtime_rng_seed
    )

    with pytest.raises(ValueError, match="does not match atlas runtime_rng_seed"):
        require_runtime_rng_seed_readback(batch, batch.runtime_rng_seed + 1)


def test_resolver_fails_closed_on_env_count_or_loaded_length_mismatch() -> None:
    schedule, reference_steps = _schedule()
    rows = _policy_rows(schedule, "weak")
    loaded_keys, loaded_lengths = _loaded_library(reference_steps)

    with pytest.raises(ValueError, match="does not match num_envs"):
        resolve_atlas_probe_batch(
            rows,
            schedule_sha256=schedule["schedule_sha256"],
            loaded_motion_keys=loaded_keys,
            loaded_reference_num_steps=loaded_lengths,
            num_envs=len(rows) + 1,
        )

    wrong_lengths = list(loaded_lengths)
    wrong_lengths[loaded_keys.index(rows[0]["motion_key"])] += 1
    with pytest.raises(ValueError, match="does not match loaded length"):
        resolve_atlas_probe_batch(
            rows,
            schedule_sha256=schedule["schedule_sha256"],
            loaded_motion_keys=loaded_keys,
            loaded_reference_num_steps=wrong_lengths,
            num_envs=len(rows),
        )


def test_resolver_rejects_unloaded_motion_and_tampered_integer_start() -> None:
    schedule, reference_steps = _schedule()
    rows = _policy_rows(schedule, "weak")
    loaded_keys, loaded_lengths = _loaded_library(reference_steps)
    absent = loaded_keys.index(rows[0]["motion_key"])
    del loaded_keys[absent]
    del loaded_lengths[absent]

    with pytest.raises(ValueError, match="absent from the loaded motion library"):
        resolve_atlas_probe_batch(
            rows,
            schedule_sha256=schedule["schedule_sha256"],
            loaded_motion_keys=loaded_keys,
            loaded_reference_num_steps=loaded_lengths,
            num_envs=len(rows),
        )

    loaded_keys, loaded_lengths = _loaded_library(reference_steps)
    tampered = deepcopy(rows)
    tampered[0]["start_step"] += 1
    with pytest.raises(ValueError, match="frozen quantization"):
        resolve_atlas_probe_batch(
            tampered,
            schedule_sha256=schedule["schedule_sha256"],
            loaded_motion_keys=loaded_keys,
            loaded_reference_num_steps=loaded_lengths,
            num_envs=len(tampered),
        )


def test_resolver_requires_one_policy_seed_phase_repeat_cell() -> None:
    schedule, reference_steps = _schedule()
    weak_rows = _policy_rows(schedule, "weak")
    strong_rows = _policy_rows(schedule, "strong")
    mixed = [*weak_rows[:-1], strong_rows[-1]]
    loaded_keys, loaded_lengths = _loaded_library(reference_steps)

    with pytest.raises(ValueError, match="must share policy"):
        resolve_atlas_probe_batch(
            mixed,
            schedule_sha256=schedule["schedule_sha256"],
            loaded_motion_keys=loaded_keys,
            loaded_reference_num_steps=loaded_lengths,
            num_envs=len(mixed),
        )


def test_tracking_command_wiring_is_opt_in_and_precedes_evaluation_sampling() -> None:
    source = Path("gear_sonic/envs/manager_env/mdp/commands.py").read_text(encoding="utf-8")
    helper_source = Path("gear_sonic/research/lace/atlas_probe_mode.py").read_text(encoding="utf-8")
    config_source = Path("gear_sonic/config/manager_env/commands/terms/motion.yaml").read_text(
        encoding="utf-8"
    )

    assert "atlas_probe_mode: bool = False" in source
    assert "atlas_probe_assignments: list[dict[str, Any]] | None = None" in source
    assert "atlas_probe_schedule_sha256: str | None = None" in source
    assert source.index(
        "if self._atlas_probe_enabled:\n                self._apply_atlas_probe"
    ) < (source.index("elif self.is_evaluating:"))
    assert "self.time_left[env_ids] = torch.inf" in source
    assert "motion_time_out termination" in source
    assert "atlas_probe_mode: false" in config_source
    assert "atlas_probe_assignments: null" in config_source
    assert "atlas_probe_schedule_sha256: null" in config_source
    assert "import torch" not in helper_source
    assert "import isaaclab" not in helper_source


def test_eval_wrapper_keeps_only_atlas_motion_command_non_evaluating() -> None:
    source = Path("gear_sonic/envs/wrapper/manager_env_wrapper.py").read_text(encoding="utf-8")
    method = source[
        source.index("    def set_is_evaluating(") : source.index(
            "    def begin_seq_motion_samples("
        )
    ]

    assert 'getattr(self.motion_command, "atlas_probe_batch", None)' in method
    assert "False if atlas_probe_batch is not None else is_evaluating" in method
    assert "if is_evaluating and atlas_probe_batch is None:" in method
    assert "self.begin_seq_motion_samples(global_rank)" in method
