from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from gear_sonic.research.lace.atlas_probe_mode import AtlasProbeBatch
import gear_sonic.research.lace.isaac_recorder as recorder_module
from gear_sonic.research.lace.isaac_recorder import (
    EpisodeAssembler,
    IsaacBindings,
    RolloutRun,
    assemble_atlas_probe_metadata,
    assemble_probe_frame_batch,
    assemble_rollout_metadata,
    bind_runtime_realizations,
    build_runtime_realizations,
    capture_resolved_event_configuration,
    consume_termination_trace,
    extract_isaac_post_reset_realizations,
    extract_isaac_post_step,
    extract_isaac_rollout_metadata,
    install_termination_trace,
    termination_compute_source_sha256,
    validate_termination_trace_unions,
)
from gear_sonic.research.lace.probes import MECHANISM_NAMES, ProbeThresholds
from gear_sonic.research.lace.schedule import derive_runtime_rng_seed


class _FakeTensor:
    def __init__(self, values: object) -> None:
        self._values = np.asarray(values)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeEntity:
    def __init__(
        self,
        data: object,
        body_names: tuple[str, ...],
        *,
        joint_names: tuple[str, ...] = (),
        root_physx_view: object | None = None,
    ) -> None:
        self.data = data
        self.body_names = body_names
        self.joint_names = joint_names
        self.root_physx_view = root_physx_view

    def find_bodies(
        self,
        names: list[str],
        preserve_order: bool = False,
    ) -> tuple[list[int], list[str]]:
        assert preserve_order
        resolved = [name for name in names if name in self.body_names]
        return [self.body_names.index(name) for name in resolved], resolved


class _FakeTerminationManager:
    def __init__(self, terms: dict[str, object]) -> None:
        self._terms = terms
        self.active_terms = list(terms)

    def get_term(self, name: str) -> object:
        return self._terms[name]


def _startup_event(env: object, env_ids: object, **kwargs: object) -> None:
    del env, env_ids, kwargs


class _FakeEventCfg:
    def __init__(self, mode: str, params: dict[str, object] | None = None) -> None:
        self.mode = mode
        self.func = _startup_event
        self.params = params or {}

    def to_dict(self) -> dict[str, object]:
        return {"mode": self.mode, "func": self.func, "params": self.params}


class _FakeEventManager:
    def __init__(self, modes: dict[str, list[str]]) -> None:
        self.active_terms = modes
        self._configs = {
            term_name: _FakeEventCfg(mode, {"resolved_ids": slice(None)})
            for mode, term_names in modes.items()
            for term_name in term_names
        }

    def get_term_cfg(self, name: str) -> _FakeEventCfg:
        return self._configs[name]


class _FakePhysxView:
    def __init__(self, num_envs: int, num_bodies: int) -> None:
        self.materials = np.full((num_envs, num_bodies, 3), [0.8, 0.7, 0.1])
        self.masses = (
            np.arange(num_envs * num_bodies, dtype=float).reshape(num_envs, num_bodies) + 1.0
        )
        self.inertias = np.ones((num_envs, num_bodies, 9))
        self.coms = np.zeros((num_envs, num_bodies, 7))
        self.coms[:, :, 3] = 1.0

    def get_material_properties(self) -> _FakeTensor:
        return _FakeTensor(self.materials)

    def get_masses(self) -> _FakeTensor:
        return _FakeTensor(self.masses)

    def get_inertias(self) -> _FakeTensor:
        return _FakeTensor(self.inertias)

    def get_coms(self) -> _FakeTensor:
        return _FakeTensor(self.coms)


class _TraceTerminationManager:
    def __init__(
        self,
        term_values: dict[str, torch.Tensor],
        timeout_terms: tuple[str, ...],
    ) -> None:
        self._env = SimpleNamespace(common_step_counter=1)
        self.num_envs = int(next(iter(term_values.values())).shape[0])
        self._term_names = list(term_values)
        self._calls = {name: 0 for name in self._term_names}

        def make_term(name: str):
            def term(env: object) -> torch.Tensor:
                assert env is self._env
                self._calls[name] += 1
                return term_values[name]

            return term

        self._term_cfgs = [
            SimpleNamespace(
                func=make_term(name),
                params={},
                time_out=name in timeout_terms,
            )
            for name in self._term_names
        ]
        self._term_name_to_term_idx = {name: index for index, name in enumerate(self._term_names)}
        self._term_dones = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.bool)
        self._truncated_buf = torch.zeros(self.num_envs, dtype=torch.bool)
        self._terminated_buf = torch.zeros(self.num_envs, dtype=torch.bool)

    @property
    def active_terms(self) -> list[str]:
        return self._term_names

    @property
    def terminated(self) -> torch.Tensor:
        return self._terminated_buf

    @property
    def time_outs(self) -> torch.Tensor:
        return self._truncated_buf

    def compute(self) -> torch.Tensor:
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            rows = value.nonzero(as_tuple=True)[0]
            if rows.numel() > 0:
                self._term_dones[rows] = False
                self._term_dones[rows, i] = True
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> torch.Tensor:
        return self._term_dones[:, self._term_name_to_term_idx[name]]


def _trace_manager(num_envs: int = 3) -> _TraceTerminationManager:
    if num_envs == 1:
        values = {
            "anchor_pos": torch.tensor([True]),
            "anchor_ori_full": torch.tensor([True]),
            "time_out": torch.tensor([True]),
        }
    else:
        values = {
            "anchor_pos": torch.tensor([True, False, False]),
            "anchor_ori_full": torch.tensor([True, True, False]),
            "time_out": torch.tensor([False, True, False]),
        }
    return _TraceTerminationManager(values, ("time_out",))


def _enabled_runtime_cfg(output_path: Path, source_sha256: str) -> SimpleNamespace:
    return SimpleNamespace(
        enabled=True,
        partition="D_atlas",
        output_path=str(output_path),
        allow_append_existing=False,
        command_name="motion",
        robot_name="robot",
        joint_action_name="joint_pos",
        contact_sensor_name="contact_forces",
        foot_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        contact_force_threshold=10.0,
        ground_normal_axis=2,
        fall_termination_terms=[],
        timeout_termination_terms=["time_out"],
        expected_termination_compute_source_sha256=source_sha256,
        probe_thresholds={},
        max_episode_steps=100,
    )


def _atlas_batch(num_envs: int = 2) -> AtlasProbeBatch:
    if num_envs not in (1, 2):
        raise ValueError("test fixture supports one or two environments")
    runtime_seed = derive_runtime_rng_seed(101, "middle", 0)
    return AtlasProbeBatch(
        rollout_ids=("lace-rollout:" + "1" * 64, "lace-rollout:" + "2" * 64)[:num_envs],
        motion_keys=("walk", "idle")[:num_envs],
        motion_ids=(1, 0)[:num_envs],
        start_steps=(50, 40)[:num_envs],
        reference_num_steps=(101, 81)[:num_envs],
        probe_policy_id="release",
        checkpoint_sha256="a" * 64,
        domain_randomization_seed=101,
        runtime_rng_seed=runtime_seed,
        phase_id="middle",
        target_fraction=0.5,
        repeat_index=0,
        schedule_sha256="b" * 64,
        split_sha256="c" * 64,
        split_selection_sha256="d" * 64,
    )


def _atlas_metadata(num_envs: int = 2) -> list[recorder_module.RolloutMetadata]:
    batch = _atlas_batch(num_envs)
    return assemble_atlas_probe_metadata(
        batch=batch,
        motion_ids=batch.motion_ids,
        motion_start_steps=batch.start_steps,
        motion_num_steps=batch.reference_num_steps,
        motion_keys=["idle", "walk"],
        process_seed_readback=batch.runtime_rng_seed,
    )


def _runtime_realization_inputs(
    metadata: list[recorder_module.RolloutMetadata],
) -> dict[str, object]:
    num_envs = len(metadata)
    num_bodies = 2
    num_joints = 3
    events = capture_resolved_event_configuration(
        _FakeEventManager({"startup": ["physics_material", "joint_defaults"], "reset": []})
    )
    root_quaternion = np.zeros((num_envs, 4))
    root_quaternion[:, 0] = 1.0
    body_quaternion = np.zeros((num_envs, num_bodies, 4))
    body_quaternion[:, :, 0] = 1.0
    centers_of_mass = np.zeros((num_envs, num_bodies, 7))
    centers_of_mass[:, :, 3] = 1.0
    motion_ids = np.asarray([1, 0][:num_envs], dtype=np.int64)
    return {
        "metadata": metadata,
        "event_configuration": events,
        "body_names": ["pelvis", "torso"],
        "joint_names": ["hip", "knee", "ankle"],
        "action_joint_names": ["hip", "knee", "ankle"],
        "reference_body_names": ["pelvis", "torso"],
        "robot_contract_state": {
            "joint_pos_limits": np.tile(
                np.array([-1.0, 1.0]),
                (num_envs, num_joints, 1),
            ),
            "soft_joint_pos_limits": np.tile(
                np.array([-0.9, 0.9]),
                (num_envs, num_joints, 1),
            ),
            "joint_vel_limits": np.full((num_envs, num_joints), 20.0),
            "soft_joint_vel_limits": np.full((num_envs, num_joints), 18.0),
        },
        "robot_randomization": {
            "material_properties": np.full(
                (num_envs, num_bodies, 3),
                [0.8, 0.7, 0.1],
            ),
            "masses": np.arange(num_envs * num_bodies, dtype=float).reshape(num_envs, num_bodies)
            + 1.0,
            "inertias": np.ones((num_envs, num_bodies, 9)),
            "centers_of_mass": centers_of_mass,
            "default_joint_position": np.zeros((num_envs, num_joints)),
            "joint_action_offset": np.zeros((num_envs, num_joints)),
        },
        "post_reset_state": {
            "root_position_w": np.zeros((num_envs, 3)),
            "root_quaternion_wxyz": root_quaternion,
            "root_linear_velocity_w": np.zeros((num_envs, 3)),
            "root_angular_velocity_w": np.zeros((num_envs, 3)),
            "joint_position": np.zeros((num_envs, num_joints)),
            "joint_velocity": np.zeros((num_envs, num_joints)),
        },
        "scheduled_reference_state": {
            "command_time_step": np.zeros(num_envs, dtype=np.int64),
            "motion_id": motion_ids,
            "anchor_position_w": np.zeros((num_envs, 3)),
            "anchor_quaternion_wxyz": root_quaternion,
            "body_position_w": np.zeros((num_envs, num_bodies, 3)),
            "body_quaternion_wxyz": body_quaternion,
            "body_linear_velocity_w": np.zeros((num_envs, num_bodies, 3)),
            "body_angular_velocity_w": np.zeros((num_envs, num_bodies, 3)),
            "joint_position": np.zeros((num_envs, num_joints)),
            "joint_velocity": np.zeros((num_envs, num_joints)),
            "left_foot_contact": np.zeros(num_envs, dtype=bool),
            "right_foot_contact": np.ones(num_envs, dtype=bool),
        },
    }


def _raw_frame_inputs(num_envs: int = 2) -> dict[str, object]:
    reference_left = np.array([True, False][:num_envs])
    reference_right = np.array([False, True][:num_envs])
    contact_force = np.zeros((num_envs, 2, 3))
    contact_force[0, 0, 2] = 12.0
    velocity = np.zeros((num_envs, 2, 3))
    velocity[0, 0] = [3.0, 4.0, 99.0]
    reference_position = np.zeros((num_envs, 3))
    actual_position = np.zeros((num_envs, 3))
    actual_position[:, 0] = 0.2
    identity = np.tile([1.0, 0.0, 0.0, 0.0], (num_envs, 1))
    actual_quaternion = identity.copy()
    actual_quaternion[0] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
    requested_torque = np.zeros((num_envs, 3))
    torque = np.zeros((num_envs, 3))
    effort = np.full((num_envs, 3), 100.0)
    reference_joint_position = np.zeros((num_envs, 3))
    joint_position = np.zeros((num_envs, 3))
    soft_limits = np.tile(np.array([-1.0, 1.0]), (num_envs, 3, 1))
    reference_body = np.zeros((num_envs, 4, 3))
    actual_body = reference_body.copy()
    actual_body[:, :, 0] = 0.1
    end = np.zeros(num_envs, dtype=bool)
    failure = np.zeros(num_envs, dtype=bool)
    fall = np.zeros(num_envs, dtype=bool)
    return {
        "reference_left_contact": reference_left,
        "reference_right_contact": reference_right,
        "foot_contact_force_w": contact_force,
        "foot_linear_velocity_w": velocity,
        "reference_anchor_position_w": reference_position,
        "actual_anchor_position_w": actual_position,
        "reference_anchor_quaternion_wxyz": identity,
        "actual_anchor_quaternion_wxyz": actual_quaternion,
        "requested_torque": requested_torque,
        "applied_torque": torque,
        "effort_limits": effort,
        "reference_joint_position": reference_joint_position,
        "joint_position": joint_position,
        "joint_soft_limits": soft_limits,
        "aligned_reference_body_position_w": reference_body,
        "actual_body_position_w": actual_body,
        "episode_end_mask": end,
        "failure_mask": failure,
        "fall_mask": fall,
        "termination_terms": {"anchor_pos": failure, "time_out": end},
        "fall_termination_terms": ("anchor_pos",),
        "timeout_termination_terms": ("time_out",),
        "contact_force_threshold": 10.0,
    }


def test_cpu_import_does_not_load_torch_or_isaaclab() -> None:
    source = Path("gear_sonic/research/lace/isaac_recorder.py").read_text()

    import_lines = [
        line.strip() for line in source.splitlines() if line.startswith(("import ", "from "))
    ]
    assert not any(line.startswith(("import torch", "from torch")) for line in import_lines)
    assert not any(line.startswith(("import isaaclab", "from isaaclab")) for line in import_lines)


def test_lazy_runtime_exposes_opt_in_manager_config_without_eager_imports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class FakeRecorderTerm:
        def __init__(self, cfg: object, env: object) -> None:
            del cfg, env

    class FakeRecorderTermCfg:
        pass

    class FakeRecorderManagerBaseCfg:
        pass

    class FakeDatasetExportMode:
        EXPORT_NONE = "export_none"

    isaaclab = ModuleType("isaaclab")
    managers = ModuleType("isaaclab.managers")
    managers.manager_term_cfg = SimpleNamespace(RecorderTermCfg=FakeRecorderTermCfg)
    managers.recorder_manager = SimpleNamespace(
        DatasetExportMode=FakeDatasetExportMode,
        RecorderManagerBaseCfg=FakeRecorderManagerBaseCfg,
        RecorderTerm=FakeRecorderTerm,
    )
    utils = ModuleType("isaaclab.utils")
    utils.configclass = lambda value: value
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setattr(recorder_module, "_RUNTIME_TYPES", None)

    term_type, term_config_type, manager_config_type = recorder_module.load_isaac_recorder_types()

    assert issubclass(manager_config_type, FakeRecorderManagerBaseCfg)
    assert manager_config_type.dataset_export_mode == "export_none"
    assert manager_config_type.export_in_record_pre_reset is False
    assert manager_config_type.failure_atlas is None
    assert term_config_type.enabled is False
    assert term_config_type.partition == "D_atlas"
    assert term_config_type.joint_action_name == "joint_pos"
    assert term_config_type.ground_normal_axis == 2
    assert term_config_type.expected_termination_compute_source_sha256 == (
        recorder_module.EXPECTED_ISAACLAB_TERMINATION_COMPUTE_SHA256
    )
    assert term_config_type.probe_thresholds == {}
    assert term_config_type.max_episode_steps == 10_000

    disabled_manager = _trace_manager()
    original_compute = disabled_manager.compute.__func__
    term_type(SimpleNamespace(enabled=False), SimpleNamespace(termination_manager=disabled_manager))
    assert disabled_manager.compute.__func__ is original_compute
    assert not hasattr(disabled_manager, "_lace_trace_installed")

    # Real Isaac order: RecorderManager constructs this term before
    # ManagerBasedRLEnv assigns env.termination_manager.
    reset_manager = _trace_manager(num_envs=1)
    reset_source_sha256 = termination_compute_source_sha256(reset_manager)
    reset_env = SimpleNamespace(num_envs=1, step_dt=0.02)
    reset_term = term_type(
        _enabled_runtime_cfg(tmp_path / "reset.jsonl", reset_source_sha256),
        reset_env,
    )
    assert reset_term._termination_trace_contract is None
    assert not hasattr(reset_manager, "_lace_trace_installed")
    with pytest.raises(RuntimeError, match="not installed before termination computation"):
        reset_term.record_post_step()

    reset_env.termination_manager = reset_manager
    reset_term.record_pre_reset([0])
    installed_compute = reset_manager.compute
    assert reset_manager._lace_trace_installed is True
    assert reset_term._termination_trace_contract is not None
    reset_term.record_pre_reset([0])
    assert reset_manager.compute == installed_compute

    # No-reset callers are still intercepted by record_pre_step, which Isaac
    # invokes before physics and before TerminationManager.compute.
    step_manager = _trace_manager(num_envs=1)
    step_source_sha256 = termination_compute_source_sha256(step_manager)
    step_env = SimpleNamespace(
        num_envs=1,
        step_dt=0.02,
        termination_manager=step_manager,
    )
    step_term = term_type(
        _enabled_runtime_cfg(tmp_path / "step.jsonl", step_source_sha256),
        step_env,
    )
    assert not hasattr(step_manager, "_lace_trace_installed")
    step_term.record_pre_step()
    assert step_manager._lace_trace_installed is True


def test_single_evaluation_trace_retains_simultaneous_terms_and_legacy_last_winner() -> None:
    manager = _trace_manager()
    baseline = _trace_manager()
    # Prove the pinned implementation's stale-row behavior remains unchanged.
    manager._term_dones[2, 0] = True
    baseline._term_dones[2, 0] = True
    baseline_dones = baseline.compute()
    source_sha256 = termination_compute_source_sha256(manager)

    contract, contract_sha256 = install_termination_trace(
        manager,
        expected_compute_source_sha256=source_sha256,
    )
    with pytest.raises(RuntimeError, match="already installed"):
        install_termination_trace(manager, expected_compute_source_sha256=source_sha256)

    dones = manager.compute()

    assert manager._calls == {"anchor_pos": 1, "anchor_ori_full": 1, "time_out": 1}
    assert baseline._calls == manager._calls
    assert torch.equal(dones, baseline_dones)
    assert torch.equal(manager.terminated, baseline.terminated)
    assert torch.equal(manager.time_outs, baseline.time_outs)
    assert torch.equal(manager._term_dones, baseline._term_dones)
    assert dones.tolist() == [True, True, False]
    assert manager.terminated.tolist() == [True, True, False]
    assert manager.time_outs.tolist() == [False, True, False]
    # Rows 0 and 1 preserve Isaac's last-trigger-wins behavior; row 2 remains stale.
    assert manager.get_term("anchor_pos").tolist() == [False, False, True]
    assert manager.get_term("anchor_ori_full").tolist() == [True, False, False]
    assert manager.get_term("time_out").tolist() == [False, True, False]

    with pytest.raises(RuntimeError, match="stale for env.common_step_counter"):
        consume_termination_trace(manager, expected_common_step_counter=2)
    snapshot = consume_termination_trace(manager, expected_common_step_counter=1)

    assert snapshot.values.tolist() == [
        [True, True, False],
        [False, True, True],
        [False, False, False],
    ]
    assert snapshot.time_out_flags == (False, False, True)
    assert snapshot.contract_sha256 == contract_sha256
    assert contract["manager_compute_source_sha256"] == source_sha256
    assert len(contract["instrument_compute_source_sha256"]) == 64
    assert contract["raw_trace_semantics"] == (
        "ordered_independent_values_from_single_manager_evaluation"
    )
    with pytest.raises(RuntimeError, match="stale or has already been consumed"):
        consume_termination_trace(manager, expected_common_step_counter=1)


def test_trace_rejects_unconsumed_compute_source_and_shape_drift() -> None:
    manager = _trace_manager()
    source_sha256 = termination_compute_source_sha256(manager)
    with pytest.raises(RuntimeError, match="compute source drifted"):
        install_termination_trace(
            manager,
            expected_compute_source_sha256="0" * 64,
        )
    assert not hasattr(manager, "_lace_trace_installed")

    install_termination_trace(manager, expected_compute_source_sha256=source_sha256)
    manager.compute()
    with pytest.raises(RuntimeError, match="was not consumed"):
        manager.compute()
    assert manager._calls == {"anchor_pos": 1, "anchor_ori_full": 1, "time_out": 1}

    malformed = _trace_manager()
    malformed._term_dones = torch.zeros((3, 2), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="_term_dones shape drifted"):
        install_termination_trace(
            malformed,
            expected_compute_source_sha256=termination_compute_source_sha256(malformed),
        )
    assert not hasattr(malformed, "_lace_trace_installed")


def test_trace_union_validation_separates_timeouts_and_fails_on_mismatch() -> None:
    manager = _trace_manager()
    install_termination_trace(
        manager,
        expected_compute_source_sha256=termination_compute_source_sha256(manager),
    )
    dones = manager.compute()
    snapshot = consume_termination_trace(manager, expected_common_step_counter=1)

    terms = validate_termination_trace_unions(
        snapshot,
        reset_terminated=manager.terminated,
        reset_time_outs=manager.time_outs,
        reset_buf=dones,
    )

    assert list(terms) == ["anchor_pos", "anchor_ori_full", "time_out"]
    assert terms["anchor_ori_full"].tolist() == [True, True, False]
    assert terms["time_out"].tolist() == [False, True, False]
    with pytest.raises(RuntimeError, match="non-timeout termination union"):
        validate_termination_trace_unions(
            snapshot,
            reset_terminated=[False, True, False],
            reset_time_outs=manager.time_outs,
            reset_buf=dones,
        )
    with pytest.raises(RuntimeError, match="timeout termination union"):
        validate_termination_trace_unions(
            snapshot,
            reset_terminated=manager.terminated,
            reset_time_outs=[False, False, False],
            reset_buf=dones,
        )
    with pytest.raises(RuntimeError, match="raw termination union"):
        validate_termination_trace_unions(
            snapshot,
            reset_terminated=manager.terminated,
            reset_time_outs=manager.time_outs,
            reset_buf=[True, False, False],
        )


def test_episode_exports_canonical_independent_multi_hot_trace() -> None:
    manager = _trace_manager(num_envs=1)
    install_termination_trace(
        manager,
        expected_compute_source_sha256=termination_compute_source_sha256(manager),
    )
    dones = manager.compute()
    snapshot = consume_termination_trace(manager, expected_common_step_counter=1)
    terms = validate_termination_trace_unions(
        snapshot,
        reset_terminated=manager.terminated,
        reset_time_outs=manager.time_outs,
        reset_buf=dones,
    )
    frame_inputs = _raw_frame_inputs(num_envs=1)
    frame_inputs.update(
        {
            "episode_end_mask": dones.numpy(),
            "failure_mask": manager.terminated.numpy(),
            "fall_mask": np.array([False]),
            "termination_terms": terms,
            "fall_termination_terms": (),
            "timeout_termination_terms": ("time_out",),
            "termination_semantics": ("instrumented_single_evaluation_ordered_raw_boolean_matrix"),
            "termination_multi_hot_available": True,
            "termination_trace_contract": snapshot.contract,
            "termination_trace_contract_sha256": snapshot.contract_sha256,
            "termination_trace_generation": snapshot.generation,
            "termination_trace_step_counter": snapshot.common_step_counter,
        }
    )
    frame = assemble_probe_frame_batch(**frame_inputs)
    metadata = _atlas_metadata(num_envs=1)
    realizations = build_runtime_realizations(**_runtime_realization_inputs(metadata))
    metadata = bind_runtime_realizations(metadata, realizations)

    record = EpisodeAssembler(num_envs=1, timestep_seconds=0.02).append(frame, metadata)[0]

    assert record["termination_multi_hot_available"] is True
    assert record["termination_multi_hot"]["term_names"] == [
        "anchor_pos",
        "anchor_ori_full",
        "time_out",
    ]
    assert record["termination_multi_hot"]["time_out_flags"] == [False, False, True]
    assert record["termination_multi_hot"]["values"] == [[True, True, True]]
    assert record["termination_multi_hot"]["trace_generations"] == [1]
    assert record["termination_multi_hot"]["common_step_counters"] == [1]
    assert record["termination_multi_hot"]["contract_sha256"] == snapshot.contract_sha256
    assert all(
        record["termination_terms"][name]["occurred"]
        for name in ("anchor_pos", "anchor_ori_full", "time_out")
    )
    assert record["scientific_runtime_ready"] is True
    assert record["scientific_runtime_blockers"] == []


def test_pure_frame_mapping_computes_contacts_slip_and_residuals() -> None:
    frame = assemble_probe_frame_batch(**_raw_frame_inputs())

    assert frame.reference_contacts.tolist() == [[True, False], [False, True]]
    assert frame.actual_contacts.tolist() == [[True, False], [False, False]]
    assert frame.foot_tangential_speed[0, 0] == pytest.approx(5.0)
    assert frame.base_translation_error[:, 0] == pytest.approx([-0.2, -0.2])
    assert frame.base_orientation_error[0] == pytest.approx(np.pi / 2)
    assert frame.base_tilt[0] == pytest.approx(np.pi / 2)
    assert frame.requested_torque.shape == (2, 3)
    assert frame.reference_joint_position.shape == (2, 3)
    assert frame.local_pose_error.shape == (2, 12)
    assert frame.local_pose_error[:, ::3] == pytest.approx(-0.1)


def test_slip_proxy_uses_configured_plane_not_resultant_force_direction() -> None:
    inputs = _raw_frame_inputs(num_envs=1)
    contact_force = np.zeros((1, 2, 3))
    contact_force[0, 0] = [20.0, 0.0, 0.0]
    inputs["foot_contact_force_w"] = contact_force
    velocity = np.zeros((1, 2, 3))
    velocity[0, 0] = [3.0, 4.0, 0.0]
    inputs["foot_linear_velocity_w"] = velocity

    frame = assemble_probe_frame_batch(**inputs)

    assert frame.foot_tangential_speed[0, 0] == pytest.approx(5.0)
    assert frame.foot_tangential_speed[0, 1] == 0.0


def test_tilt_is_angle_between_body_frame_gravity_not_difference_of_magnitudes() -> None:
    inputs = _raw_frame_inputs(num_envs=1)
    half_angle = np.deg2rad(15.0)
    inputs["reference_anchor_quaternion_wxyz"] = np.array(
        [[np.cos(half_angle), np.sin(half_angle), 0.0, 0.0]]
    )
    inputs["actual_anchor_quaternion_wxyz"] = np.array(
        [[np.cos(half_angle), -np.sin(half_angle), 0.0, 0.0]]
    )

    frame = assemble_probe_frame_batch(**inputs)

    assert frame.base_tilt[0] == pytest.approx(np.deg2rad(60.0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("foot_contact_force_w", np.zeros((2, 1, 3)), "shape"),
        ("effort_limits", np.zeros((2, 3)), "strictly positive"),
        ("applied_torque", np.full((2, 3), np.nan), "finite"),
        ("actual_anchor_quaternion_wxyz", np.zeros((2, 4)), "zero-norm"),
    ),
)
def test_frame_mapping_fails_closed_on_invalid_live_values(
    field: str,
    value: object,
    message: str,
) -> None:
    inputs = _raw_frame_inputs()
    inputs[field] = value

    with pytest.raises(ValueError, match=message):
        assemble_probe_frame_batch(**inputs)


def test_rollout_metadata_is_deterministic_and_rejects_duplicate_schedules() -> None:
    run = RolloutRun(
        policy_id="lite_early",
        domain_randomization_seed=101,
        repeat_indices_by_env=(0, 1),
    )
    first = assemble_rollout_metadata(
        motion_ids=[1, 1],
        motion_start_steps=[25, 25],
        motion_num_steps=[101, 101],
        motion_keys=["idle", "walk"],
        run=run,
    )
    second = assemble_rollout_metadata(
        motion_ids=[1, 1],
        motion_start_steps=[25, 25],
        motion_num_steps=[101, 101],
        motion_keys=["idle", "walk"],
        run=run,
    )

    assert first == second
    assert first[0].initial_phase == pytest.approx(0.25)
    assert first[0].reference_start_step == 25
    assert first[0].reference_num_steps == 101
    assert first[0].partition == "D_atlas"
    assert first[0].rollout_id != first[1].rollout_id

    with pytest.raises(ValueError, match="duplicate rollout schedules"):
        assemble_rollout_metadata(
            motion_ids=[1, 1],
            motion_start_steps=[25, 25],
            motion_num_steps=[101, 101],
            motion_keys=["idle", "walk"],
            run=RolloutRun(policy_id="lite_early", domain_randomization_seed=101),
        )


def test_atlas_metadata_uses_exact_batch_rows_and_requires_process_seed_readback() -> None:
    batch = _atlas_batch()
    metadata = assemble_atlas_probe_metadata(
        batch=batch,
        motion_ids=[1, 0],
        motion_start_steps=[50, 40],
        motion_num_steps=[101, 81],
        motion_keys=["idle", "walk"],
        process_seed_readback=batch.runtime_rng_seed,
    )

    assert tuple(item.rollout_id for item in metadata) == batch.rollout_ids
    assert tuple(item.motion_key for item in metadata) == batch.motion_keys
    assert all(item.checkpoint_sha256 == batch.checkpoint_sha256 for item in metadata)
    assert all(item.runtime_rng_seed == batch.runtime_rng_seed for item in metadata)
    assert all(item.runtime_rng_seed_readback == batch.runtime_rng_seed for item in metadata)
    assert all(item.phase_id == "middle" for item in metadata)
    assert all(item.target_fraction == 0.5 for item in metadata)
    assert all(item.schedule_sha256 == batch.schedule_sha256 for item in metadata)
    assert tuple(item.schedule_entry_id for item in metadata) == batch.rollout_ids

    with pytest.raises(ValueError, match="does not match atlas runtime_rng_seed"):
        assemble_atlas_probe_metadata(
            batch=batch,
            motion_ids=[1, 0],
            motion_start_steps=[50, 40],
            motion_num_steps=[101, 81],
            motion_keys=["idle", "walk"],
            process_seed_readback=batch.runtime_rng_seed + 1,
        )

    with pytest.raises(ValueError, match="start steps drifted"):
        assemble_atlas_probe_metadata(
            batch=batch,
            motion_ids=[1, 0],
            motion_start_steps=[49, 40],
            motion_num_steps=[101, 81],
            motion_keys=["idle", "walk"],
            process_seed_readback=batch.runtime_rng_seed,
        )


def test_resolved_event_capture_preserves_order_and_rejects_interval_events() -> None:
    record = capture_resolved_event_configuration(
        _FakeEventManager(
            {
                "startup": ["physics_material", "joint_defaults"],
                "reset": ["root_state"],
            }
        )
    )

    assert record["mode_order"] == ["startup", "reset"]
    assert [term["term_name"] for term in record["modes"][0]["terms"]] == [
        "physics_material",
        "joint_defaults",
    ]
    assert record["modes"][0]["terms"][0]["callable"].endswith(":_startup_event")
    assert record["interval_events_instrumented"] is False

    with pytest.raises(RuntimeError, match="rejects active interval events"):
        capture_resolved_event_configuration(_FakeEventManager({"interval": ["push_robot"]}))


def test_runtime_realization_is_deterministic_policy_neutral_and_state_sensitive() -> None:
    metadata = _atlas_metadata()
    inputs = _runtime_realization_inputs(metadata)

    first = build_runtime_realizations(**inputs)
    second = build_runtime_realizations(**inputs)

    assert first == second
    assert len({item.sha256 for item in first}) == 2
    assert first[0].record["ordered_names"]["joint_names"] == ["hip", "knee", "ankle"]
    assert first[0].record["resolved_event_configuration"]["mode_order"] == [
        "startup",
        "reset",
    ]
    assert (
        first[0].record["realized_parameters"]["masses"]["values"]
        == inputs["robot_randomization"]["masses"][0].tolist()
    )
    assert first[0].record["scheduled_reference"]["identity"]["motion_key"] == "walk"
    assert first[0].robot_contract_readback["ordered_joint_names"] == [
        "hip",
        "knee",
        "ankle",
    ]
    assert first[0].robot_contract_readback["limits"]["joint_vel_limits"]["values"] == [
        20.0,
        20.0,
        20.0,
    ]
    assert "probe_policy_id" not in first[0].record["scheduled_reference"]["identity"]
    assert "checkpoint_sha256" not in first[0].record["scheduled_reference"]["identity"]
    assert "rollout_id" not in first[0].record["scheduled_reference"]["identity"]

    other_policy = [
        replace(
            item,
            rollout_id=f"other-policy:{item.env_index}",
            policy_id="strong",
            checkpoint_sha256="e" * 64,
            schedule_entry_id=f"other-policy:{item.env_index}",
        )
        for item in metadata
    ]
    other_inputs = {**inputs, "metadata": other_policy}
    other = build_runtime_realizations(**other_inputs)
    assert [item.sha256 for item in other] == [item.sha256 for item in first]
    assert [item.robot_contract_readback_sha256 for item in other] == [
        item.robot_contract_readback_sha256 for item in first
    ]

    changed_randomization = dict(inputs["robot_randomization"])
    changed_masses = np.asarray(changed_randomization["masses"]).copy()
    changed_masses[0, 0] += 0.25
    changed_randomization["masses"] = changed_masses
    changed = build_runtime_realizations(**{**inputs, "robot_randomization": changed_randomization})
    assert changed[0].sha256 != first[0].sha256
    assert changed[1].sha256 == first[1].sha256

    varying_contract = dict(inputs["robot_contract_state"])
    varying_velocity_limits = np.asarray(varying_contract["joint_vel_limits"]).copy()
    varying_velocity_limits[1, 0] += 1.0
    varying_contract["joint_vel_limits"] = varying_velocity_limits
    with pytest.raises(ValueError, match="varies by environment"):
        build_runtime_realizations(**{**inputs, "robot_contract_state": varying_contract})


def test_runtime_realization_fails_closed_on_bad_reset_state_or_missing_binding() -> None:
    metadata = _atlas_metadata(num_envs=1)
    inputs = _runtime_realization_inputs(metadata)
    bad_reference = dict(inputs["scheduled_reference_state"])
    bad_reference["command_time_step"] = np.array([1])

    with pytest.raises(ValueError, match="command_time_step=0"):
        build_runtime_realizations(**{**inputs, "scheduled_reference_state": bad_reference})
    with pytest.raises(RuntimeError, match="no record_post_reset runtime realization"):
        bind_runtime_realizations(metadata, [None])

    two_metadata = _atlas_metadata(num_envs=2)
    second_only = build_runtime_realizations(
        **_runtime_realization_inputs(two_metadata),
        env_indices=[1],
    )[0]
    selectively_bound = bind_runtime_realizations(
        two_metadata,
        [None, second_only],
        env_indices=[1],
    )
    assert selectively_bound[0].domain_randomization_realization is None
    assert selectively_bound[1].domain_randomization_realization_sha256 == second_only.sha256


def test_live_post_reset_extractor_reads_exact_physics_action_and_reference_state() -> None:
    metadata = _atlas_metadata(num_envs=2)
    inputs = _runtime_realization_inputs(metadata)
    num_envs = 2
    body_names = tuple(inputs["body_names"])
    joint_names = tuple(inputs["joint_names"])
    randomization = inputs["robot_randomization"]
    state = inputs["post_reset_state"]
    reference = inputs["scheduled_reference_state"]
    physx_view = _FakePhysxView(num_envs, len(body_names))
    # Match the pure fixture exactly so this also checks the live field mapping.
    physx_view.materials = np.asarray(randomization["material_properties"])
    physx_view.masses = np.asarray(randomization["masses"])
    physx_view.inertias = np.asarray(randomization["inertias"])
    physx_view.coms = np.asarray(randomization["centers_of_mass"])
    robot_data = SimpleNamespace(
        default_joint_pos=_FakeTensor(randomization["default_joint_position"]),
        joint_pos_limits=_FakeTensor(inputs["robot_contract_state"]["joint_pos_limits"]),
        soft_joint_pos_limits=_FakeTensor(inputs["robot_contract_state"]["soft_joint_pos_limits"]),
        joint_vel_limits=_FakeTensor(inputs["robot_contract_state"]["joint_vel_limits"]),
        soft_joint_vel_limits=_FakeTensor(inputs["robot_contract_state"]["soft_joint_vel_limits"]),
        root_pos_w=_FakeTensor(state["root_position_w"]),
        root_quat_w=_FakeTensor(state["root_quaternion_wxyz"]),
        root_lin_vel_w=_FakeTensor(state["root_linear_velocity_w"]),
        root_ang_vel_w=_FakeTensor(state["root_angular_velocity_w"]),
        joint_pos=_FakeTensor(state["joint_position"]),
        joint_vel=_FakeTensor(state["joint_velocity"]),
    )
    robot = _FakeEntity(
        robot_data,
        body_names,
        joint_names=joint_names,
        root_physx_view=physx_view,
    )
    action_term = SimpleNamespace(
        _joint_names=list(inputs["action_joint_names"]),
        _offset=_FakeTensor(randomization["joint_action_offset"]),
    )
    command = SimpleNamespace(
        cmd_body_names=list(inputs["reference_body_names"]),
        time_steps=_FakeTensor(reference["command_time_step"]),
        motion_ids=_FakeTensor(reference["motion_id"]),
        anchor_pos_w=_FakeTensor(reference["anchor_position_w"]),
        anchor_quat_w=_FakeTensor(reference["anchor_quaternion_wxyz"]),
        body_pos_w=_FakeTensor(reference["body_position_w"]),
        body_quat_w=_FakeTensor(reference["body_quaternion_wxyz"]),
        body_lin_vel_w=_FakeTensor(reference["body_linear_velocity_w"]),
        body_ang_vel_w=_FakeTensor(reference["body_angular_velocity_w"]),
        joint_pos=_FakeTensor(reference["joint_position"]),
        joint_vel=_FakeTensor(reference["joint_velocity"]),
        feet_l=_FakeTensor(np.asarray(reference["left_foot_contact"])[:, None]),
        feet_r=_FakeTensor(np.asarray(reference["right_foot_contact"])[:, None]),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        event_manager=_FakeEventManager({"startup": ["physics_material"], "reset": []}),
        scene={"robot": robot},
        action_manager=SimpleNamespace(get_term=lambda name: action_term),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )

    realizations = extract_isaac_post_reset_realizations(
        env,
        IsaacBindings(),
        metadata,
        _FakeTensor([1]),
    )

    assert [item.env_index for item in realizations] == [1]
    record = realizations[0].record
    assert record["realized_parameters"]["masses"]["values"] == [3.0, 4.0]
    assert record["post_reset_state"]["root_position_w"]["shape"] == [3]
    assert record["scheduled_reference"]["state"]["motion_id"]["values"] == 0
    assert realizations[0].robot_contract_readback["ordered_joint_names"] == list(joint_names)


def test_live_metadata_reads_exact_command_batch_and_env_config_seed() -> None:
    batch = _atlas_batch()
    command = SimpleNamespace(
        atlas_probe_batch=batch,
        motion_lib=SimpleNamespace(curr_motion_keys=["idle", "walk"]),
        motion_ids=_FakeTensor([1, 0]),
        motion_start_time_steps=_FakeTensor([50, 40]),
        motion_num_steps=_FakeTensor([101, 81]),
    )
    env = SimpleNamespace(
        cfg=SimpleNamespace(seed=batch.runtime_rng_seed),
        command_manager=SimpleNamespace(get_term=lambda name: command),
    )

    metadata = extract_isaac_rollout_metadata(env)

    assert tuple(item.rollout_id for item in metadata) == batch.rollout_ids
    env.cfg.seed += 1
    with pytest.raises(ValueError, match="env.cfg.seed"):
        extract_isaac_rollout_metadata(env)


def test_episode_assembler_finalizes_pre_reset_failure_as_atlas_ready_record() -> None:
    first_inputs = _raw_frame_inputs(num_envs=1)
    first = assemble_probe_frame_batch(**first_inputs)
    last_inputs = _raw_frame_inputs(num_envs=1)
    last_inputs["episode_end_mask"] = np.array([True])
    last_inputs["failure_mask"] = np.array([True])
    last_inputs["fall_mask"] = np.array([True])
    last_inputs["termination_terms"] = {
        "anchor_pos": np.array([True]),
        "time_out": np.array([False]),
    }
    requested_torque = np.asarray(last_inputs["requested_torque"]).copy()
    requested_torque[0, 0] = 120.0
    last_inputs["requested_torque"] = requested_torque
    applied_torque = np.asarray(last_inputs["applied_torque"]).copy()
    applied_torque[0, 0] = 100.0
    last_inputs["applied_torque"] = applied_torque
    last = assemble_probe_frame_batch(**last_inputs)
    metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[20],
        motion_num_steps=[101],
        motion_keys=["walk__A001"],
        run=RolloutRun(policy_id="lite_early", domain_randomization_seed=101),
    )
    thresholds = ProbeThresholds(slip_speed_threshold=0.20)
    assembler = EpisodeAssembler(
        num_envs=1,
        timestep_seconds=0.02,
        thresholds=thresholds,
    )

    assert assembler.append(first, metadata) == []
    records = assembler.append(last, metadata)

    assert len(records) == 1
    record = records[0]
    assert record["rollout_id"] == metadata[0].rollout_id
    assert record["probe_policy_id"] == record["policy_id"] == "lite_early"
    assert record["domain_randomization_seed"] == 101
    assert record["domain_randomization_seed_semantics"] == (
        "declared_run_seed_not_runtime_verified"
    )
    assert record["partition"] == "D_atlas"
    assert record["reference_start_step"] == 20
    assert record["reference_num_steps"] == 101
    assert record["initial_phase"] == pytest.approx(0.2)
    assert record["repeat_index"] == 0
    assert record["failed"] is True
    assert tuple(record["mechanism_scores"]) == MECHANISM_NAMES
    assert record["probe"]["scores"] == record["mechanism_scores"]
    assert set(record["probe"]) >= {"scores", "onsets", "diagnostics"}
    episode_diagnostics = record["probe"]["episode_diagnostics"]
    assert episode_diagnostics["reference_failure_step"] == 21
    assert episode_diagnostics["reference_progress_to_failure"] == pytest.approx(0.21)
    assert episode_diagnostics["failure_progress_censored"] is False
    assert record["termination_terms"]["anchor_pos"]["occurred"] is True
    assert record["termination_multi_hot_available"] is False
    assert record["fall_signal_available"] is True
    assert record["fall_termination_terms"] == ["anchor_pos"]
    assert record["termination_terms"]["anchor_pos"]["onset_index"] == 1
    assert record["probe"]["scores"]["actuation_saturation"] > 0.0
    actuation_diagnostics = record["probe"]["diagnostics"]["actuation_saturation"]
    assert actuation_diagnostics["max_requested_torque_ratio"] == pytest.approx(1.2)
    assert actuation_diagnostics["max_applied_torque_ratio"] == pytest.approx(1.0)
    assert actuation_diagnostics["max_clip_gap_ratio"] == pytest.approx(0.2)
    assert record["buffering_semantics"] == "bounded_full_episode_cpu_smoke"
    assert record["foot_slip_velocity_proxy"] == "link_origin_velocity_tangent_to_configured_plane"
    assert record["ground_normal_axis"] == 2
    assert record["buffered_step_count"] == 2
    assert record["probe_thresholds"]["slip_speed_threshold"] == pytest.approx(0.20)
    threshold_payload = json.dumps(
        record["probe_thresholds"],
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert record["probe_thresholds_sha256"] == hashlib.sha256(threshold_payload).hexdigest()
    json.dumps(record, allow_nan=False)
    assert assembler.active_env_indices() == ()


def test_atlas_record_exports_frozen_schedule_identity_and_realized_dr() -> None:
    batch = _atlas_batch(num_envs=1)
    metadata = assemble_atlas_probe_metadata(
        batch=batch,
        motion_ids=[1],
        motion_start_steps=[50],
        motion_num_steps=[101],
        motion_keys=["idle", "walk"],
        process_seed_readback=batch.runtime_rng_seed,
    )
    runtime_inputs = _runtime_realization_inputs(metadata)
    realizations = build_runtime_realizations(**runtime_inputs)
    metadata = bind_runtime_realizations(metadata, realizations)
    frame_inputs = _raw_frame_inputs(num_envs=1)
    frame_inputs["episode_end_mask"] = np.array([True])
    frame = assemble_probe_frame_batch(**frame_inputs)

    record = EpisodeAssembler(num_envs=1, timestep_seconds=0.02).append(frame, metadata)[0]

    assert record["rollout_id"] == batch.rollout_ids[0]
    assert record["schedule_entry_id"] == batch.rollout_ids[0]
    assert record["checkpoint_sha256"] == batch.checkpoint_sha256
    assert record["runtime_rng_seed"] == batch.runtime_rng_seed
    assert record["runtime_rng_seed_readback"] == batch.runtime_rng_seed
    assert record["phase_id"] == batch.phase_id
    assert record["target_fraction"] == batch.target_fraction
    assert record["realized_fraction"] == pytest.approx(0.5)
    assert record["schedule_sha256"] == batch.schedule_sha256
    assert record["domain_randomization_realization_sha256"] == realizations[0].sha256
    assert record["domain_randomization_realization"] == realizations[0].record
    assert record["robot_contract_readback_sha256"] == (
        realizations[0].robot_contract_readback_sha256
    )
    assert record["runtime_rng_seed_semantics"] == (
        "runtime_rng_seed_applied_and_realization_hash_verified"
    )
    assert record["scientific_runtime_ready"] is False
    assert record["scientific_runtime_blockers"] == [
        "independent_termination_multi_hot_unavailable"
    ]


def test_timeout_ends_rollout_without_fabricating_a_failure() -> None:
    inputs = _raw_frame_inputs(num_envs=1)
    inputs["episode_end_mask"] = np.array([True])
    inputs["termination_terms"] = {
        "anchor_pos": np.array([False]),
        "time_out": np.array([True]),
    }
    frame = assemble_probe_frame_batch(**inputs)
    metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7),
    )

    record = EpisodeAssembler(num_envs=1, timestep_seconds=0.02).append(frame, metadata)[0]

    assert record["failed"] is False
    assert record["termination_terms"]["time_out"]["occurred"] is True
    assert record["probe"]["episode_diagnostics"]["reference_progress_to_failure"] is None
    assert record["probe"]["episode_diagnostics"]["failure_progress_censored"] is True


def test_live_duck_adapter_maps_exact_verified_paths_and_terms() -> None:
    body_names = ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link", "head")
    robot_data = SimpleNamespace(
        body_link_lin_vel_w=_FakeTensor(np.zeros((1, 4, 3))),
        computed_torque=_FakeTensor(np.zeros((1, 3))),
        applied_torque=_FakeTensor(np.zeros((1, 3))),
        joint_effort_limits=_FakeTensor(np.full((1, 3), 100.0)),
        joint_pos=_FakeTensor(np.zeros((1, 3))),
        soft_joint_pos_limits=_FakeTensor(np.tile([-1.0, 1.0], (1, 3, 1))),
    )
    sensor_forces = np.zeros((1, 4, 3))
    sensor_forces[0, 1, 2] = 20.0
    sensor_data = SimpleNamespace(net_forces_w=_FakeTensor(sensor_forces))
    command = SimpleNamespace(
        # SONIC retains a singleton configured-foot body axis.
        feet_l=_FakeTensor([[True]]),
        feet_r=_FakeTensor([[False]]),
        anchor_pos_w=_FakeTensor(np.zeros((1, 3))),
        robot_anchor_pos_w=_FakeTensor(np.zeros((1, 3))),
        anchor_quat_w=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
        robot_anchor_quat_w=_FakeTensor([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=_FakeTensor(np.zeros((1, 3))),
        body_pos_relative_w=_FakeTensor(np.zeros((1, 4, 3))),
        robot_body_pos_w=_FakeTensor(np.zeros((1, 4, 3))),
    )
    scene = {
        "robot": _FakeEntity(robot_data, body_names),
        "contact_forces": _FakeEntity(sensor_data, body_names),
    }
    env = SimpleNamespace(
        scene=scene,
        command_manager=SimpleNamespace(get_term=lambda name: command),
        termination_manager=_FakeTerminationManager(
            {
                "tracking_failure": _FakeTensor([True]),
                "fallen": _FakeTensor([False]),
                "time_out": _FakeTensor([False]),
            }
        ),
        reset_buf=_FakeTensor([True]),
        reset_terminated=_FakeTensor([True]),
    )

    frame = extract_isaac_post_step(
        env,
        IsaacBindings(fall_termination_terms=("fallen",)),
    )

    assert frame.reference_contacts.tolist() == [[True, False]]
    assert frame.actual_contacts.tolist() == [[True, False]]
    assert frame.failure_mask.tolist() == [True]
    assert frame.fall_mask.tolist() == [False]
    assert set(frame.termination_terms) == {"tracking_failure", "fallen", "time_out"}


def test_live_contact_boundary_rejects_unaggregated_multi_body_labels() -> None:
    with pytest.raises(ValueError, match="shape"):
        recorder_module._sonic_reference_foot_contact(  # noqa: SLF001
            _FakeTensor([[True, False]]),
            "motion_command.feet_l",
        )


def test_live_duck_adapter_rejects_missing_or_inactive_bindings() -> None:
    bindings = IsaacBindings(fall_termination_terms=("not_active",))
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_term=lambda name: SimpleNamespace()),
        scene={},
    )

    with pytest.raises(RuntimeError, match="scene entity"):
        extract_isaac_post_step(env, bindings)


def test_empty_fall_binding_means_explicitly_unavailable_not_generic_failure() -> None:
    inputs = _raw_frame_inputs(num_envs=1)
    inputs["episode_end_mask"] = np.array([True])
    inputs["failure_mask"] = np.array([True])
    inputs["termination_terms"] = {"tracking_failure": np.array([True])}
    inputs["fall_termination_terms"] = ()
    inputs["timeout_termination_terms"] = ()
    frame = assemble_probe_frame_batch(**inputs)
    metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7),
    )

    record = EpisodeAssembler(num_envs=1, timestep_seconds=0.02).append(frame, metadata)[0]

    assert record["failed"] is True
    assert record["fall_signal_available"] is False
    assert record["fall_termination_terms"] == []
    assert record["probe"]["diagnostics"]["balance_orientation"]["fall_incidence"] == 0.0


def test_stale_last_trigger_term_is_masked_off_timeout_only_episode() -> None:
    first_inputs = _raw_frame_inputs(num_envs=1)
    first_inputs["episode_end_mask"] = np.array([True])
    first_inputs["failure_mask"] = np.array([True])
    first_inputs["termination_terms"] = {
        "tracking_failure": np.array([True]),
        "time_out": np.array([False]),
    }
    first_inputs["fall_termination_terms"] = ()
    first = assemble_probe_frame_batch(**first_inputs)
    second_inputs = _raw_frame_inputs(num_envs=1)
    second_inputs["episode_end_mask"] = np.array([True])
    second_inputs["failure_mask"] = np.array([False])
    second_inputs["termination_terms"] = {
        # Stale cached value from the preceding failed episode.
        "tracking_failure": np.array([True]),
        "time_out": np.array([True]),
    }
    second_inputs["fall_termination_terms"] = ()
    second = assemble_probe_frame_batch(**second_inputs)
    first_metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7, repeat_index=0),
    )
    second_metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7, repeat_index=1),
    )
    assembler = EpisodeAssembler(num_envs=1, timestep_seconds=0.02)

    failed_record = assembler.append(first, first_metadata)[0]
    timeout_record = assembler.append(second, second_metadata)[0]

    assert failed_record["termination_terms"]["tracking_failure"]["occurred"] is True
    assert timeout_record["failed"] is False
    assert timeout_record["termination_terms"]["tracking_failure"]["occurred"] is False
    assert timeout_record["termination_terms"]["time_out"]["occurred"] is True
    assert timeout_record["termination_semantics"] == (
        "manager_last_trigger_wins_masked_to_current_end"
    )


def test_post_reset_guard_prevents_mixing_two_episodes() -> None:
    frame = assemble_probe_frame_batch(**_raw_frame_inputs(num_envs=1))
    metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7),
    )
    assembler = EpisodeAssembler(num_envs=1, timestep_seconds=0.02)
    assembler.append(frame, metadata)

    with pytest.raises(RuntimeError, match="reset without a post-step"):
        assembler.assert_empty([0])


def test_episode_assembler_rejects_unlabelled_motion_reassignment() -> None:
    frame = assemble_probe_frame_batch(**_raw_frame_inputs(num_envs=1))
    first_metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7, repeat_index=0),
    )
    reassigned_metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7, repeat_index=1),
    )
    assembler = EpisodeAssembler(
        num_envs=1,
        timestep_seconds=0.02,
        quiesce_after_first_completion=True,
    )
    assembler.append(frame, first_metadata)

    with pytest.raises(RuntimeError, match="motion assignment changed"):
        assembler.append(frame, reassigned_metadata)


def test_atlas_episode_assembler_quiesces_early_done_environment() -> None:
    metadata = assemble_rollout_metadata(
        motion_ids=[0, 1],
        motion_start_steps=[0, 0],
        motion_num_steps=[10, 30],
        motion_keys=["short", "long"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7),
    )
    assembler = EpisodeAssembler(
        num_envs=2,
        timestep_seconds=0.02,
        quiesce_after_first_completion=True,
    )

    first_inputs = _raw_frame_inputs(num_envs=2)
    first_inputs["episode_end_mask"] = np.array([True, False])
    first_inputs["failure_mask"] = np.array([True, False])
    first_inputs["termination_terms"] = {
        "anchor_pos": np.array([True, False]),
        "time_out": np.array([False, False]),
    }
    first = assembler.append(assemble_probe_frame_batch(**first_inputs), metadata)
    assert assembler.active_env_indices() == (1,)
    assert assembler.quiescent_env_indices() == (0,)
    assert assembler.unfinished_env_indices() == (1,)

    middle = assembler.append(
        assemble_probe_frame_batch(**_raw_frame_inputs(num_envs=2)),
        metadata,
    )

    last_inputs = _raw_frame_inputs(num_envs=2)
    last_inputs["episode_end_mask"] = np.array([True, True])
    last_inputs["failure_mask"] = np.array([True, True])
    last_inputs["termination_terms"] = {
        "anchor_pos": np.array([True, True]),
        "time_out": np.array([False, False]),
    }
    last = assembler.append(assemble_probe_frame_batch(**last_inputs), metadata)
    after_all_done = assembler.append(
        assemble_probe_frame_batch(**last_inputs),
        metadata,
    )

    records = first + middle + last + after_all_done
    assert [record["rollout_id"] for record in records] == [
        metadata[0].rollout_id,
        metadata[1].rollout_id,
    ]
    assert assembler.active_env_indices() == ()
    assert assembler.quiescent_env_indices() == (0, 1)
    assert assembler.unfinished_env_indices() == ()


def test_episode_assembler_enforces_full_buffer_smoke_bound() -> None:
    frame = assemble_probe_frame_batch(**_raw_frame_inputs(num_envs=1))
    metadata = assemble_rollout_metadata(
        motion_ids=[0],
        motion_start_steps=[0],
        motion_num_steps=[10],
        motion_keys=["stand"],
        run=RolloutRun(policy_id="release", domain_randomization_seed=7),
    )
    assembler = EpisodeAssembler(
        num_envs=1,
        timestep_seconds=0.02,
        max_episode_steps=1,
    )
    assembler.append(frame, metadata)

    with pytest.raises(RuntimeError, match="exceeded max_episode_steps=1"):
        assembler.append(frame, metadata)
