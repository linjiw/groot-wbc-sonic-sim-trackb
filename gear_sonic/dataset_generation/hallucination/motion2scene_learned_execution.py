"""Learned encounter request from direct pre-command observations and state.

Derived from the frozen comparison executor. No scripted scene decision overrides the model.
"""

import json
from pathlib import Path

from isaaclab.utils import configclass
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand

from .motion2scene_action_contract import command_permission, requested_skill
from .motion2scene_learned_readout import load_readout, readout
from .motion2scene_observation_delay import ObservationDelay
from .motion2scene_outcome_learner import decision_features
from .motion2scene_overhang_eval_execution import (
    EvaluationBankCommand,
    EvaluationBankEnvCfg,
    EvaluationBankRecorder,
    EvaluationBankRecorderCfg,
)
from .motion2scene_overhang_observer import observe_overhang


class LearnedCommand(EvaluationBankCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.observation_delay = ObservationDelay(cfg.observation_delay_s)
        self.decision_issued = False
        self.decision_capture = None
        self.learned_readout = load_readout(cfg.learned_policy_path, cfg.learned_policy_sha256)

    def _update_command(self):
        from omni.physx import get_physx_scene_query_interface

        if not self._bank_verified:
            self._verify_bank()
        TrackingCommand._update_command(self)
        if not self._env.sim.cfg.enable_scene_query_support:
            raise RuntimeError("scene queries disabled")
        occupied, origin, rays = observe_overhang(
            self.robot.data.root_pos_w[0].detach().cpu().numpy(),
            self.robot.data.root_quat_w[0].detach().cpu().numpy(),
            get_physx_scene_query_interface(),
        )
        time_s = float((self.time_steps + self.motion_start_time_steps)[0]) / 50
        delivered = self.observation_delay.push(
            {"time_s": time_s, "occupied": occupied, "origin": origin, "rays": rays}
        )
        delivered_occupied = bool(delivered and delivered["occupied"])
        current_lib = self.motion_lib
        before_joint_reference = self.joint_pos.clone()
        before_root_reference = self.anchor_pos_w.clone()
        state = self.robot.data.root_state_w.clone()
        joints = self.robot.data.joint_pos.clone()
        clock = self.time_steps.clone()
        # Read the alternate reference at the same phase, then restore it before
        # deciding. This changes no robot state and issues no simulator writes.
        try:
            self.motion_lib = self._interface_libraries[1 - self.interface_active]
            joint_jump = float((self.joint_pos - before_joint_reference).abs().max())
            root_jump = float(torch.linalg.vector_norm(self.anchor_pos_w - before_root_reference))
        finally:
            self.motion_lib = current_lib
        if not self.decision_issued and abs(time_s - self.cfg.decision_time_s) < 1e-8:
            if delivered is None or self.decision_capture is not None:
                raise RuntimeError("missing or duplicate first decision packet")
            data = self.robot.data
            state_values = {
                name: getattr(data, source)[0].detach().cpu().tolist()
                for name, source in (
                    ("projected_gravity_b", "projected_gravity_b"),
                    ("root_lin_vel_w", "root_lin_vel_w"),
                    ("root_ang_vel_w", "root_ang_vel_w"),
                    ("dof_pos", "joint_pos"),
                    ("dof_vel", "joint_vel"),
                )
            }
            frame = len(self.observation_delay.packets) - 1
            age = (frame - delivered["capture_frame"]) / 50
            self.decision_capture = {
                "phase_s": time_s,
                "capture_frame": frame,
                "capture_elapsed_s": frame / 50,
                "requested_action": self.cfg.encounter_action,
                "active_before": self.interface_active,
                "packet": delivered,
                "observation_age_s": age,
                "state": state_values,
                "root_pos_w": data.root_pos_w[0].detach().cpu().tolist(),
                "root_quat_w": data.root_quat_w[0].detach().cpu().tolist(),
                "joint_names": list(self.robot.joint_names),
                "features": decision_features(
                    delivered, state_values, time_s, self.interface_active, age
                ).tolist(),
            }
            decision = readout(self.learned_readout, self.decision_capture["features"])
            self.cfg.encounter_action = decision["requested_action"]
            self.decision_capture["requested_action"] = decision["requested_action"]
            self.decision_capture["learned_decision"] = decision
            self.decision_capture["policy_sha256"] = self.cfg.learned_policy_sha256
        requested, self.decision_issued = requested_skill(
            self.cfg.encounter_action,
            self.cfg.decision_time_s,
            time_s,
            self.interface_active,
            self.decision_issued,
        )
        allowed, reasons = command_permission(
            requested,
            self.interface_active,
            time_s,
            joint_jump,
            root_jump,
        )
        attempted = requested != self.interface_active
        if attempted and allowed:
            old = self.interface_active
            self.motion_lib = self._interface_libraries[requested]
            self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
            self.interface_active = requested
            self.interface_switches.append(
                {
                    "time_s": time_s,
                    "from": old,
                    "to": requested,
                    "joint_reference_jump_rad": joint_jump,
                    "root_reference_jump_m": root_jump,
                    "root_state_unchanged": bool(torch.equal(state, self.robot.data.root_state_w)),
                    "joint_state_unchanged": bool(torch.equal(joints, self.robot.data.joint_pos)),
                    "clock_unchanged": bool(torch.equal(clock, self.time_steps)),
                }
            )
        self.interface_rows.append(
            {
                "time_s": time_s,
                "encounter_action": self.cfg.encounter_action,
                "decision_time_s": self.cfg.decision_time_s,
                "decision_issued": self.decision_issued,
                "observation_age_s": (
                    None
                    if delivered is None
                    else (len(self.observation_delay.packets) - 1 - delivered["capture_frame"]) / 50
                ),
                "occupied": occupied,
                "delivered": delivered,
                "delivered_occupied": delivered_occupied,
                "capture_frame": len(self.observation_delay.packets) - 1,
                "active": self.interface_active,
                "origin": origin,
                "rays": rays,
                "upper_occupied": any(r["upper_candidate"] for r in rays),
                "transition": {
                    "requested": requested,
                    "attempted": attempted,
                    "allowed": allowed,
                    "reasons": reasons,
                    "joint_jump_rad": joint_jump,
                    "root_jump_m": root_jump,
                    "root_state_unchanged": bool(torch.equal(state, self.robot.data.root_state_w)),
                    "joint_state_unchanged": bool(torch.equal(joints, self.robot.data.joint_pos)),
                    "clock_unchanged": bool(torch.equal(clock, self.time_steps)),
                },
            }
        )


class LearnedEnvCfg(EvaluationBankEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = LearnedCommand
        self.commands.motion.observation_delay_s = float(config["observation_delay_s"])
        self.commands.motion.encounter_action = int(config["encounter_action"])
        self.commands.motion.decision_time_s = float(config["decision_time_s"])
        self.commands.motion.learned_policy_path = str(config["learned_policy_path"])
        self.commands.motion.learned_policy_sha256 = str(config["learned_policy_sha256"])


class LearnedRecorder(EvaluationBankRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_action_contract_saved", False):
            return
        path = Path(self.save_dir, "reactive_interface.json")
        data = json.loads(path.read_text())
        data["rule_inputs"] = (
            "learned outcome predictor from directly captured causal features; shared phase/jump guard"
        )
        data["limitations"] = (
            "one decision, no reconsideration; refusal is not stopping; ideal rays are observations only"
        )
        path.write_text(json.dumps(data, indent=2))
        command = self.env.command_manager.get_term("motion")
        if command.decision_capture is None:
            raise RuntimeError("the registered decision callback was not captured")
        Path(self.save_dir, "decision_capture.json").write_text(
            json.dumps(command.decision_capture, indent=2)
        )
        self._action_contract_saved = True


@configclass
class LearnedRecorderCfg(EvaluationBankRecorderCfg):
    class_type = LearnedRecorder
