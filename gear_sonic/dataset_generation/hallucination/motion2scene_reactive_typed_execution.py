"""Typed-ray repair; reuse frozen switching and physics recording unchanged."""

from .motion2scene_ray_observer import observe_overhead
from .motion2scene_reactive_execution import ReactiveCommand
from .motion2scene_reactive_query_execution import QueryEnabledExecutionEnvCfg


class TypedRayCommand(ReactiveCommand):
    def _observe_overhead(self):
        from omni.physx import get_physx_scene_query_interface

        if not self._env.sim.cfg.enable_scene_query_support:
            raise RuntimeError("scene queries must be enabled")
        return observe_overhead(
            self.robot.data.root_pos_w[0].detach().cpu().numpy(),
            self.robot.data.root_quat_w[0].detach().cpu().numpy(),
            get_physx_scene_query_interface(),
        )


class TypedRayExecutionEnvCfg(QueryEnabledExecutionEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.commands.motion.class_type = TypedRayCommand
