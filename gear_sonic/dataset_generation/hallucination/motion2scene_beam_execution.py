"""Opt-in Isaac instrumentation for the registered beam intervention.

Imported only inside Isaac Sim. The release controller and default recorder stay intact.
Forces are sampled normal forces, not penetration depths or continuous-time certificates.
"""

import json
from pathlib import Path

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import numpy as np

from gear_sonic.envs.manager_env.mdp.recorders import (
    TrajectoryRecorderCfg,
    TrajectoryRecorderTerm,
)
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import ModularTrackingEnvCfg

from .motion2scene_collision_inventory import collision_inventory

BEAM_PATH = "/World/ground/terrain/CounterfactualBeam"


class BeamExecutionEnvCfg(ModularTrackingEnvCfg):
    def __init__(self, config, **kwargs):
        super().__init__(config=config, **kwargs)
        if self.scene.num_envs != 1:
            raise ValueError("beam intervention requires exactly one environment")
        names = list(config["beam_contact_body_names"])
        if not names or len(set(names)) != len(names):
            raise ValueError("beam force filters require unique robot body names")
        self.scene.counterfactual_beam_contact = ContactSensorCfg(
            prim_path=BEAM_PATH,
            filter_prim_paths_expr=[f"/World/envs/env_0/Robot/{name}" for name in names],
            update_period=0.0,
            history_length=0,
            track_air_time=False,
            force_threshold=0.0,
        )


class BeamTrajectoryRecorder(TrajectoryRecorderTerm):
    def _initialize(self):
        super()._initialize()
        import omni.usd

        self._beam_sensor = self.env.scene.sensors["counterfactual_beam_contact"]
        self._beam_frames = []
        self._beam_saved = False
        stage = omni.usd.get_context().get_stage()
        inventory = collision_inventory(stage)
        if not any(r["path"] == BEAM_PATH and r["collision"] for r in inventory):
            raise RuntimeError("beam missing from imported native collision inventory")
        Path(self.save_dir, "native_collision_inventory.json").write_text(
            json.dumps(
                {
                    "beam_path": BEAM_PATH,
                    "filter_paths": list(self._beam_sensor.cfg.filter_prim_paths_expr),
                    "physics_dt": self.env.physics_dt,
                    "control_dt": self.env.step_dt,
                    "capture": "composed USD at first recorded state; not a per-frame collider pose trace",
                    "shapes": inventory,
                },
                indent=2,
            )
        )

    def record_post_step(self):
        result = super().record_post_step()
        # Registered trajectory profile captures every 50-Hz control step.
        if self.render_frame_skip != 1:
            raise RuntimeError("beam capture requires every control frame")
        values = self._beam_sensor.data.force_matrix_w
        if values is None or values.shape[0:2] != (1, 1):
            raise RuntimeError("missing single-beam filtered contact matrix")
        forces = values.detach().cpu().numpy()[0, 0].copy()
        if not np.isfinite(forces).all():
            raise FloatingPointError("nonfinite beam contact forces")
        self._beam_frames.append(forces)
        return result

    def close_writers(self):
        super().close_writers()
        if not getattr(self, "_initialized", False) or getattr(self, "_beam_saved", True):
            return
        if len(self._beam_frames) != len(self._frame_data[0]["dof_pos"]):
            raise RuntimeError("beam force and trajectory lengths disagree")
        np.savez_compressed(
            Path(self.save_dir, "beam_contacts.npz"),
            force_w=np.asarray(self._beam_frames),
            filter_paths=np.array(self._beam_sensor.cfg.filter_prim_paths_expr),
            fps=1 / self.env.step_dt,
        )
        self._beam_saved = True


@configclass
class BeamTrajectoryRecorderCfg(TrajectoryRecorderCfg):
    class_type = BeamTrajectoryRecorder
