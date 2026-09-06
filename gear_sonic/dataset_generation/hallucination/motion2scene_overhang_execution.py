"""Opt-in overhang observation and guarded online reference transitions."""

import json
from pathlib import Path

from isaaclab.utils import configclass
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand

from .motion2scene_overhang_observer import observe_overhang, transition_decision
from .motion2scene_reactive_execution import ReactiveCommand, ReactiveRecorder, ReactiveRecorderCfg
from .motion2scene_reactive_query_execution import QueryEnabledExecutionEnvCfg


class OverhangCommand(ReactiveCommand):
    def _update_command(self):
        from omni.physx import get_physx_scene_query_interface

        TrackingCommand._update_command(self)
        if not self._env.sim.cfg.enable_scene_query_support:
            raise RuntimeError("scene queries disabled")
        occupied, origin, rays = observe_overhang(
            self.robot.data.root_pos_w[0].detach().cpu().numpy(),
            self.robot.data.root_quat_w[0].detach().cpu().numpy(),
            get_physx_scene_query_interface(),
        )
        time_s = float((self.time_steps + self.motion_start_time_steps)[0]) / 50
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
            self.cfg.reactive_mode, time_s, occupied, self.interface_active, joint_jump, root_jump
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


class OverhangExecutionEnvCfg(QueryEnabledExecutionEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = OverhangCommand


class OverhangRecorder(ReactiveRecorder):
    def close_writers(self):
        super().close_writers()
        if not self._initialized or getattr(self, "_overhang_metadata_saved", False):
            return
        path = Path(self.save_dir, "reactive_interface.json")
        data = json.loads(path.read_text())
        data["sensor"] = "PhysX rays: 12 upper plus up to 36 conditional lower rays per 50 Hz frame"
        data["rule_inputs"] = (
            "upper/lower geometry, clock, reference jumps and active skill; no hit identity"
        )
        data["limitations"] = (
            "flat known floor; sparse ideal free-space rays; fixed legal motion phases"
        )
        path.write_text(json.dumps(data, indent=2))
        self._overhang_metadata_saved = True


@configclass
class OverhangRecorderCfg(ReactiveRecorderCfg):
    class_type = OverhangRecorder
