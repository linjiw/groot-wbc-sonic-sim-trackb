"""Evaluation-only alternate bank, with a first-step source-route audit."""

import json
from pathlib import Path
import random

from isaaclab.utils import configclass
import joblib
import numpy as np
import torch

from .motion2scene_overhang_execution import (
    OverhangCommand,
    OverhangExecutionEnvCfg,
    OverhangRecorder,
)
from .motion2scene_reactive_execution import ReactiveRecorderCfg
from .motion2scene_reference_bank import audit_root_route


class EvaluationBankCommand(OverhangCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Keep downstream randomization stream unchanged by this deterministic reload.
        py_state, np_state = random.getstate(), np.random.get_state()
        cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        try:
            self._interface_libraries[1].load_motions_for_evaluation(start_idx=0)
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(gpu_state)
        self._bank_verified = False

    def _verify_bank(self):
        # The wrapper reloads the primary clip for evaluation before the first step.
        roots, dofs, errors = [], [], []
        for lib in self._interface_libraries:
            n = int(lib.get_time_step_total(self.motion_ids)[0])
            ids = torch.zeros(n, dtype=torch.long, device=self.device)
            steps = torch.arange(n, device=self.device)
            root = lib.get_root_pos_w(ids, steps).detach().cpu().numpy().copy()
            dof = lib.get_dof_pos(ids, steps).detach().cpu().numpy().copy()
            entry = next(iter(joblib.load(lib.m_cfg.motion_file).values()))
            error = audit_root_route(root, entry["root_trans_offset"], float(entry["fps"]))
            if not np.isfinite(dof).all():
                raise RuntimeError("nonfinite loaded joint references")
            roots.append(root)
            dofs.append(dof)
            errors.append(error)
        self.bank_arrays = {
            "root_xyz": np.array(roots),
            "joint_pos": np.array(dofs),
            "fps": np.array(50.0),
        }
        self.bank_audit = {
            "source_route_max_error_m": errors,
            "alternate_loader": "load_motions_for_evaluation",
            "same_root_xy_max_error_m": float(np.max(np.abs(roots[0][:, :2] - roots[1][:, :2]))),
            "random_stream_restored_after_reload": True,
        }
        self._bank_verified = True

    def _update_command(self):
        if not self._bank_verified:
            self._verify_bank()
        super()._update_command()


class EvaluationBankEnvCfg(OverhangExecutionEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = EvaluationBankCommand


class EvaluationBankRecorder(OverhangRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_bank_saved", False):
            return
        command = self.env.command_manager.get_term("motion")
        if not command._bank_verified:
            raise RuntimeError("reference bank was not verified")
        np.savez_compressed(Path(self.save_dir, "loaded_reference_bank.npz"), **command.bank_arrays)
        Path(self.save_dir, "loaded_reference_bank.json").write_text(
            json.dumps(command.bank_audit, indent=2)
        )
        self._bank_saved = True


@configclass
class EvaluationBankRecorderCfg(ReactiveRecorderCfg):
    class_type = EvaluationBankRecorder
