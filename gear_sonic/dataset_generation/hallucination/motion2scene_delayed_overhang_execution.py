"""Opt-in delayed packet delivery; retained evaluation-bank and switch guards."""

from isaaclab.utils import configclass
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand

from .motion2scene_observation_delay import ObservationDelay
from .motion2scene_overhang_eval_execution import (
    EvaluationBankCommand,
    EvaluationBankEnvCfg,
    EvaluationBankRecorderCfg,
)
from .motion2scene_overhang_observer import observe_overhang, transition_decision


class DelayedOverhangCommand(EvaluationBankCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.observation_delay = ObservationDelay(cfg.observation_delay_s)

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
        requested, allowed, reasons = transition_decision(
            self.cfg.reactive_mode,
            time_s,
            delivered_occupied,
            self.interface_active,
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


class DelayedOverhangEnvCfg(EvaluationBankEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = DelayedOverhangCommand
        self.commands.motion.observation_delay_s = float(config["observation_delay_s"])


@configclass
class DelayedOverhangRecorderCfg(EvaluationBankRecorderCfg):
    pass
