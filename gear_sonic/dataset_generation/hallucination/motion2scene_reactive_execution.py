"""Opt-in PhysX ray sensing, phase-aligned reference switching and 200-Hz contacts.

This is a scripted interface diagnostic, not a trained traversal policy. Scene
queries use collision geometry; hit identities are logged but not used by the rule.
"""

import copy
import json
import math
from pathlib import Path

from isaaclab.utils import configclass
import numpy as np
import torch

from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.utils.motion_lib.motion_lib_robot import MotionLibRobot

from .motion2scene_beam_execution import (
    BeamExecutionEnvCfg,
    BeamTrajectoryRecorder,
    BeamTrajectoryRecorderCfg,
)
from .motion2scene_reactive_rule import select_skill


class ReactiveCommand(TrackingCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        alternate_cfg = copy.deepcopy(self.motion_lib.m_cfg)
        alternate_cfg.motion_file = cfg.reactive_alternate_path
        alternate_cfg.filter_motion_keys = None
        alternate = MotionLibRobot(alternate_cfg, self.num_envs, self.device)
        alternate.load_motions_for_training(max_num_seqs=1)
        self._interface_libraries = [self.motion_lib, alternate]
        if self.num_envs != 1 or any(lib._num_motions != 1 for lib in self._interface_libraries):
            raise ValueError("interface pilot requires one environment and one clip per library")
        if not torch.equal(
            self.motion_lib.get_time_step_total(self.motion_ids),
            alternate.get_time_step_total(self.motion_ids),
        ):
            raise ValueError("phase-aligned references require equal durations")
        self.interface_active = 0
        self.interface_rows = []
        self.interface_switches = []

    def _observe_overhead(self):
        from omni.physx import get_physx_scene_query_interface

        root = self.robot.data.root_pos_w[0].detach().cpu().numpy()
        w, x, y, z = self.robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        origin = root + np.array([0.2 * math.cos(yaw), 0.2 * math.sin(yaw), 0.4])
        rays = []
        query = get_physx_scene_query_interface()
        for elevation in (-5, 0, 5, 10):
            for azimuth in (-10, 0, 10):
                a, e = yaw + math.radians(azimuth), math.radians(elevation)
                direction = (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))
                hits = []

                def collect(hit):
                    path = str(hit.get("collision", hit.get("rigidBody", "")))
                    if not path.startswith("/World/envs/env_0/Robot"):
                        hits.append(
                            {
                                "distance": float(hit["distance"]),
                                "position": [float(v) for v in hit["position"]],
                                "path": path,
                            }
                        )
                    return True

                query.raycast_all(tuple(float(v) for v in origin), direction, 3.0, collect)
                nearest = min(hits, key=lambda hit: hit["distance"]) if hits else None
                rays.append({"direction": direction, "hit": nearest})
        occupied = any(r["hit"] and 1.15 <= r["hit"]["position"][2] <= 1.5 for r in rays)
        return bool(occupied), origin.tolist(), rays

    def _update_command(self):
        super()._update_command()
        occupied, origin, rays = self._observe_overhead()
        time_s = float((self.time_steps + self.motion_start_time_steps)[0]) / 50
        new = select_skill(self.cfg.reactive_mode, time_s, occupied, self.interface_active)
        if new != self.interface_active:
            before_state = self.robot.data.root_state_w.clone()
            before_joint = self.robot.data.joint_pos.clone()
            before_clock = self.time_steps.clone()
            old_reference = self.joint_pos.clone()
            old = self.interface_active
            self.motion_lib = self._interface_libraries[new]
            self.motion_num_steps = self.motion_lib.get_motion_num_steps(self.motion_ids)
            self.interface_active = new
            self.interface_switches.append(
                {
                    "time_s": time_s,
                    "from": old,
                    "to": new,
                    "joint_reference_jump_rad": float((self.joint_pos - old_reference).abs().max()),
                    "root_state_unchanged": bool(
                        torch.equal(before_state, self.robot.data.root_state_w)
                    ),
                    "joint_state_unchanged": bool(
                        torch.equal(before_joint, self.robot.data.joint_pos)
                    ),
                    "clock_unchanged": bool(torch.equal(before_clock, self.time_steps)),
                }
            )
        self.interface_rows.append(
            {"time_s": time_s, "occupied": occupied, "active": new, "origin": origin, "rays": rays}
        )


class ReactiveExecutionEnvCfg(BeamExecutionEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = ReactiveCommand
        self.commands.motion.reactive_alternate_path = config["reactive_alternate_path"]
        self.commands.motion.reactive_mode = config["reactive_mode"]


class ReactiveRecorder(BeamTrajectoryRecorder):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._physics_forces = []
        self._physics_steps = []
        self._control_steps = []
        self._reactive_saved = False

    def record_post_physics_decimation_step(self):
        # The hook precedes scene.update: bypass the timestamp-cached Sensor.data.
        sensor = self.env.scene.sensors["counterfactual_beam_contact"]
        values = sensor.contact_physx_view.get_contact_force_matrix(dt=self.env.physics_dt)
        values = values.detach().cpu().numpy().reshape(1, 1, -1, 3)[0, 0].copy()
        if values.shape != (30, 3) or not np.isfinite(values).all():
            raise ValueError("invalid physics-step force matrix")
        self._physics_forces.append(values)
        self._physics_steps.append(self.env._sim_step_counter)
        return None, None

    def record_post_step(self):
        result = super().record_post_step()
        self._control_steps.append(self.env._sim_step_counter)
        return result

    def close_writers(self):
        super().close_writers()
        if self._reactive_saved or not self._initialized:
            return
        np.savez_compressed(
            Path(self.save_dir, "physics_beam_contacts.npz"),
            force_w=np.asarray(self._physics_forces),
            physics_steps=np.asarray(self._physics_steps),
            control_steps=np.asarray(self._control_steps),
            physics_dt=self.env.physics_dt,
        )
        command = self.env.command_manager.get_term("motion")
        Path(self.save_dir, "reactive_interface.json").write_text(
            json.dumps(
                {
                    "mode": command.cfg.reactive_mode,
                    "switches": command.interface_switches,
                    "observations": command.interface_rows,
                    "sensor": "PhysX collision raycast_all; 12 rays at 50 Hz; robot excluded",
                    "rule_inputs": "nearest-hit positions, motion clock and latched skill only",
                },
                indent=2,
            )
        )
        self._reactive_saved = True


@configclass
class ReactiveRecorderCfg(BeamTrajectoryRecorderCfg):
    class_type = ReactiveRecorder
