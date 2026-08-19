"""Custom recorder terms for the manager environment MDP."""

from __future__ import annotations

import json
import os
import pickle
from typing import TYPE_CHECKING

import cv2
import imageio
from isaaclab.managers import manager_term_cfg, recorder_manager
from isaaclab.utils import configclass
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

if TYPE_CHECKING:
    from isaaclab import envs


@configclass
class RecordersCfg(recorder_manager.RecorderManagerBaseCfg):
    """Recorders terms for the MDP."""

    render_envs = None
    running_ref_root_height = None
    trajectory = None


def _as_viewable(buffer):
    """Turn a camera annotator buffer into frames a video encoder accepts.

    Depth arrives as float metres with an infinite sky and segmentation as integer ids. Both are
    written as 8-bit images so a reviewer can watch them; the authoritative values stay in the
    trajectory capture, because a video is lossy and a training signal must not be read off one.
    """
    array = buffer.detach().cpu().numpy() if hasattr(buffer, "detach") else np.asarray(buffer)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.dtype.kind == "f":
        finite = np.isfinite(array)
        array = np.where(finite, array, 0.0)
        top = float(array[finite].max()) if finite.any() else 1.0
        return (np.clip(array / top if top > 0 else array, 0.0, 1.0) * 255).astype(np.uint8)
    if array.dtype.kind in "iu":
        # Stable pseudo-colour: an id keeps its colour across frames and across episodes.
        ids = array.astype(np.int64)
        return np.stack(
            [((ids * 67) % 255), ((ids * 149) % 255), ((ids * 223) % 255)], axis=-1
        ).astype(np.uint8)
    return array.astype(np.uint8)


class RenderEnvsRecorderTerm(recorder_manager.RecorderTerm):
    """Recorder term for rendering environments with advanced features like text overlay and frame skipping."""

    cfg: RenderEnvsRecorderCfg

    def __init__(self, cfg: RenderEnvsRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        # Determine save directory (backward compatibility)
        self.save_dir = self.cfg.video_save_path
        logger.info(f"=== Start recording video to {self.save_dir} ===")

        # Create directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
        self.video_writers = []
        # One writer per extra modality the camera was asked for. A navigation corpus needs depth
        # and segmentation as much as colour, and the camera already accepts them through
        # cameras.camera_data_types -- they were captured and dropped, because this recorder only
        # ever read output["rgb"].
        self.modality_writers: dict[str, list] = {}
        self._writers_closed = False
        self.frame_id = 0
        self.first_render = True
        self._fixed_eye = None
        self._fixed_target = None

    def _initialize_writers(self):
        """Initialize video writers for each environment."""
        logger.info(f"Saving rendering to {self.save_dir}")
        # Get configuration parameters with defaults
        self.group_camera = self.env.wrapper.config.get("group_camera", False)
        self.max_render_envs = self.env.wrapper.config.get("max_render_envs", self.env.num_envs)
        if self.group_camera:
            self.max_render_envs = 1  # single video from overview_camera
        self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        self.start_idx = self.env.wrapper.start_idx
        self.camera_name = self.cfg.camera_name
        self.track_root = self.cfg.track_root

        for i in range(self.max_render_envs):
            file_name = f"{self.save_dir}/{self.start_idx+i:06d}.mp4"
            fps = 1 / (self.env.step_dt * self.render_frame_skip)
            writer = imageio.get_writer(
                file_name,
                fps=fps,
                codec="libx264",
                quality=self.cfg.video_quality,
                pixelformat="yuv420p",
            )
            self.video_writers.append(writer)
            for modality in self._extra_modalities():
                self.modality_writers.setdefault(modality, []).append(
                    imageio.get_writer(
                        f"{self.save_dir}/{self.start_idx+i:06d}__{modality}.mp4",
                        fps=self.fps,
                        macro_block_size=1,
                    )
                )

    def _extra_modalities(self) -> tuple[str, ...]:
        """Camera outputs to encode beside colour, as the camera was configured."""
        config = getattr(getattr(self.env, "wrapper", None), "config", {}) or {}
        cameras = config.get("cameras", {}) or {}
        return tuple(n for n in tuple(cameras.get("camera_data_types", ("rgb",))) if n != "rgb")

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record video frames after each step with frame skipping and text overlay support."""
        if len(self.video_writers) == 0:
            self._initialize_writers()

        # Check if we should render this frame
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

        cam = self.env.scene[self.camera_name]
        if cam is None:
            raise RuntimeError(
                f"Camera {self.camera_name!r} was requested by the recorder but is not configured"
            )

        if self.track_root:
            # Move a free camera with the robot. Attached cameras such as the
            # dataset ego camera retain their authored link-relative transform.
            root_pos = self.env.command_manager.get_term("motion").robot_body_pos_w[:, 0]
            camera_offset = self.env.wrapper.config.get("eval_camera_offset", [2, 2, 1])
            fix_camera = self.env.wrapper.config.get("fix_camera_after_first_frame", False)

            if fix_camera and self._fixed_eye is not None:
                eye, target = self._fixed_eye, self._fixed_target
            elif self.group_camera:
                center = root_pos.mean(dim=0, keepdim=True).expand_as(root_pos)
                eye = center + torch.tensor(camera_offset, device=self.env.device)
                target = center
                if fix_camera:
                    self._fixed_eye = eye.clone()
                    self._fixed_target = center.clone()
            else:
                eye = root_pos + torch.tensor(camera_offset, device=self.env.device)
                target = root_pos
                if fix_camera:
                    self._fixed_eye = eye.clone()
                    self._fixed_target = root_pos.clone()

            # Write world poses to Fabric and sync to USD so both renderer paths see it.
            cam._view._sync_usd_on_fabric_write = True  # noqa: SLF001
            cam.set_world_poses_from_view(eye, target)

        # Two render calls: 1st flushes pose to render pipeline, 2nd captures at new pose
        if hasattr(self.env, "sim"):
            self.env.sim.render()
            self.env.sim.render()

        # Mark sensor as outdated so update actually re-reads the annotator buffers
        cam._is_outdated[:] = True  # noqa: SLF001
        cam.update(dt=0.0, force_recompute=True)

        # Get RGB data
        rgb_viewer = cam.data.output["rgb"].clone()
        extra_frames = {
            modality: _as_viewable(cam.data.output[modality])
            for modality in self._extra_modalities()
            if modality in cam.data.output
        }

        # Get render info if available
        cur_render_info = None
        if self.env.wrapper.config.get("render_info", None) is not None:
            end_idx = self.start_idx + self.max_render_envs
            cur_render_info = self.env.wrapper.config.render_info[self.start_idx : end_idx]

        # Record the first post-step frame as well. The trajectory recorder also
        # starts at frame zero; dropping it here creates a one-frame dataset skew.
        loop = (
            tqdm(range(self.max_render_envs)) if self.first_render else range(self.max_render_envs)
        )
        for i in loop:
            frame = rgb_viewer[i].cpu().numpy()

            # Add text overlay if render info is provided
            if cur_render_info is not None and i < len(cur_render_info):
                for j, text in enumerate(cur_render_info[i]):
                    frame = cv2.putText(
                        frame,
                        str(text),
                        (10, 30 + j * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                    )

            self.video_writers[i].append_data(frame)
            for modality, writers in self.modality_writers.items():
                if i < len(writers) and modality in extra_frames:
                    writers[i].append_data(extra_frames[modality][i])
        self.first_render = False

        self.frame_id += 1
        return "record_post_step", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Explicitly close all video writers."""
        if not self._writers_closed:
            for i, writer in enumerate(self.video_writers):
                try:
                    writer.close()
                    logger.info(f"Closed video writer {i}")
                except Exception as e:  # noqa: BLE001
                    logger.info(f"Error closing video writer {i}: {e}")
            for writers in self.modality_writers.values():
                for writer in writers:
                    try:
                        writer.close()
                    except Exception as error:  # noqa: BLE001
                        logger.info(f"Error closing modality writer: {error}")
            self.modality_writers.clear()
            self.video_writers.clear()
            self._writers_closed = True
            self.frame_id = 0
            self.first_render = True
            self._fixed_eye = None
            self._fixed_target = None
            logger.info("=== All video writers closed ===")

    def __del__(self):
        """Ensure writers are closed when object is destroyed."""
        self.close_writers()


@configclass
class RenderEnvsRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for environment rendering recorder with advanced features."""

    class_type = RenderEnvsRecorderTerm
    video_save_path: str = None
    video_quality: int = 5
    camera_name: str = "eval_camera"
    track_root: bool = True


class TrajectoryRecorderTerm(recorder_manager.RecorderTerm):
    """Record physics state plus SONIC/reference actions for each environment.

    The original recorder only stored enough state for kinematic replay. The
    additional fields make the same artifact useful as the authoritative raw
    trajectory for synthetic-dataset generation. Existing replay consumers keep
    working because the original keys and file naming are unchanged.
    """

    cfg: TrajectoryRecorderCfg
    _ALLOWED_FOOT_CONTACT_BODIES = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    def __init__(self, cfg: TrajectoryRecorderCfg, env: envs.ManagerBasedEnv):
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env

        self.save_dir = self.cfg.save_path
        os.makedirs(self.save_dir, exist_ok=True)
        logger.info(f"=== TrajectoryRecorder: saving to {self.save_dir} ===")

        self._initialized = False
        self._closed = False
        self._frame_data: dict[int, dict] = {}  # env_idx -> {field: [frames]}
        self.frame_id = 0

    def _initialize(self):
        """Initialize per-env data buffers after environment is ready."""
        self.num_record_envs = self.env.num_envs
        self.start_idx = (
            getattr(self.env.wrapper, "start_idx", 0) if hasattr(self.env, "wrapper") else 0
        )

        # Match video recorder's frame skip to keep trajectory in sync with video
        if hasattr(self.env, "wrapper"):
            self.render_frame_skip = self.env.wrapper.config.get("render_frame_skip", 2)
        else:
            self.render_frame_skip = 2

        # Detect available scene entities
        self._has_object = "object" in self.env.scene.rigid_objects
        self._has_table = "table" in self.env.scene.rigid_objects
        self._dof_joint_names = tuple(self.env.scene["robot"].joint_names)
        self._contact_sensor = self.env.scene.sensors.get("contact_forces")
        self._left_foot_support_sensor = self.env.scene.sensors.get("left_foot_support_contact")
        self._right_foot_support_sensor = self.env.scene.sensors.get("right_foot_support_contact")
        self._contact_body_names = (
            tuple(self._contact_sensor.body_names) if self._contact_sensor is not None else ()
        )
        # Per-body world positions come from the articulation itself, not from the
        # motion command: TrackingCommand.robot_body_pos_w covers only the bodies the
        # command tracks, whereas the articulation order matches the contact sensor so
        # geometry and contact can be indexed together.
        self._body_names = tuple(self.env.scene["robot"].body_names)

        # Get motion command for root pose
        try:
            self._motion_cmd = self.env.command_manager.get_term("motion")
        except Exception:  # noqa: BLE001
            self._motion_cmd = None

        for i in range(self.num_record_envs):
            self._frame_data[i] = self._create_empty_data()

        self._initialized = True

    def _create_empty_data(self) -> dict:
        data = {
            "dof_pos": [],
            "dof_vel": [],
            "root_pos_w": [],
            "root_quat_w": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "root_lin_vel_w": [],
            "root_ang_vel_w": [],
            "projected_gravity_b": [],
            "applied_joint_action": [],
            "action_motion_token": [],
            "policy_meta_action": [],
            "reference_g1_qpos": [],
            "motion_id": [],
            "motion_time_step": [],
            "motion_time_s": [],
            "tracking_metrics": {},
            "robot_contact_force_norm_w": [],
            "robot_contact_force_w": [],
            "left_foot_ground_contact_force_w": [],
            "right_foot_ground_contact_force_w": [],
            "max_nonfoot_contact_force_n": [],
            "left_foot_contact_force_n": [],
            "right_foot_contact_force_n": [],
        }
        if self._has_object:
            data["object_pos_w"] = []
            data["object_quat_w"] = []
        if self._has_table:
            data["table_pos_w"] = []
            data["table_quat_w"] = []
        return data

    def record_post_step(self) -> tuple[str | None, torch.Tensor | dict | None]:
        """Record trajectory state after each step, synced with video frame skip."""
        if not self._initialized:
            self._initialize()

        # Skip frames to match video recorder cadence
        if self.frame_id % self.render_frame_skip != 0:
            self.frame_id += 1
            return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

        robot = self.env.scene["robot"]
        env_origins = self.env.scene.env_origins.detach().cpu().numpy()

        # Move each batched value to the CPU once instead of once per environment.
        joint_pos = robot.data.joint_pos.detach().cpu().numpy()
        joint_vel = robot.data.joint_vel.detach().cpu().numpy()
        root_quat = robot.data.root_quat_w.detach().cpu().numpy()
        root_lin_vel = robot.data.root_lin_vel_w.detach().cpu().numpy()
        root_ang_vel = robot.data.root_ang_vel_w.detach().cpu().numpy()
        projected_gravity = robot.data.projected_gravity_b.detach().cpu().numpy()
        body_pos = robot.data.body_pos_w.detach().cpu().numpy()
        # Orientation is needed to place body-local collision capsules in world
        # frame; positions alone only support conservative bounding spheres.
        body_quat = robot.data.body_quat_w.detach().cpu().numpy()

        if self._motion_cmd is not None:
            root_pos = self._motion_cmd.robot_body_pos_w[:, 0].detach().cpu().numpy()
        else:
            root_pos = robot.data.root_pos_w.detach().cpu().numpy()

        action_manager = getattr(self.env, "action_manager", None)
        applied_action_tensor = getattr(action_manager, "action", None)
        applied_action = (
            applied_action_tensor.detach().cpu().numpy()
            if isinstance(applied_action_tensor, torch.Tensor)
            else None
        )

        full_latent_tensor = getattr(self.env, "_full_latent", None)
        full_latent = (
            full_latent_tensor.detach().cpu().numpy()
            if isinstance(full_latent_tensor, torch.Tensor)
            else None
        )
        meta_action_tensor = getattr(self.env, "_last_meta_action", None)
        meta_action = (
            meta_action_tensor.detach().cpu().numpy()
            if isinstance(meta_action_tensor, torch.Tensor)
            else None
        )

        reference_qpos = None
        motion_ids = None
        motion_time_steps = None
        motion_times_s = None
        tracking_metrics: dict[str, np.ndarray] = {}
        if self._motion_cmd is not None:
            motion_ids_tensor = self._motion_cmd.motion_ids
            motion_time_steps_tensor = (
                self._motion_cmd.motion_start_time_steps + self._motion_cmd.time_steps
            )
            reference_root_pos = self._motion_cmd.motion_lib.get_root_pos_w(
                motion_ids_tensor, motion_time_steps_tensor
            )
            reference_root_quat = self._motion_cmd.motion_lib.get_root_quat_w(
                motion_ids_tensor, motion_time_steps_tensor
            )
            reference_qpos = (
                torch.cat(
                    [reference_root_pos, reference_root_quat, self._motion_cmd.joint_pos],
                    dim=-1,
                )
                .detach()
                .cpu()
                .numpy()
            )
            motion_ids = motion_ids_tensor.detach().cpu().numpy()
            motion_time_steps = motion_time_steps_tensor.detach().cpu().numpy()
            motion_times_s = motion_time_steps * float(self.env.step_dt)
            tracking_metrics = {
                name: value.detach().cpu().numpy()
                for name, value in self._motion_cmd.metrics.items()
                if isinstance(value, torch.Tensor)
            }

        contact_force_norm = None
        contact_force_w = None
        left_foot_ground_contact_force_w = None
        right_foot_ground_contact_force_w = None
        max_nonfoot_contact_force = None
        left_foot_contact_force = None
        right_foot_contact_force = None
        if self._contact_sensor is not None:
            contact_force_w = self._contact_sensor.data.net_forces_w.detach().cpu().numpy()
            contact_force_norm = (
                torch.linalg.vector_norm(self._contact_sensor.data.net_forces_w, dim=-1)
                .detach()
                .cpu()
                .numpy()
            )
            nonfoot_indices = [
                index
                for index, body_name in enumerate(self._contact_body_names)
                if body_name not in self._ALLOWED_FOOT_CONTACT_BODIES
            ]
            if nonfoot_indices:
                max_nonfoot_contact_force = contact_force_norm[:, nonfoot_indices].max(axis=1)
            else:
                max_nonfoot_contact_force = np.zeros(self.num_record_envs, dtype=np.float32)
            for body_name, destination in (
                ("left_ankle_roll_link", "left"),
                ("right_ankle_roll_link", "right"),
            ):
                if body_name not in self._contact_body_names:
                    continue
                values = contact_force_norm[:, self._contact_body_names.index(body_name)]
                if destination == "left":
                    left_foot_contact_force = values
                else:
                    right_foot_contact_force = values
        for sensor, destination in (
            (self._left_foot_support_sensor, "left"),
            (self._right_foot_support_sensor, "right"),
        ):
            if sensor is None or sensor.data.force_matrix_w is None:
                continue
            support_force = sensor.data.force_matrix_w[:, 0, 0, :].detach().cpu().numpy()
            if destination == "left":
                left_foot_ground_contact_force_w = support_force
            else:
                right_foot_ground_contact_force_w = support_force

        for i in range(self.num_record_envs):
            data = self._frame_data[i]
            data["dof_pos"].append(joint_pos[i].copy())
            data["dof_vel"].append(joint_vel[i].copy())
            data["root_pos_w"].append((root_pos[i] - env_origins[i]).copy())
            data["root_quat_w"].append(root_quat[i].copy())
            data["root_lin_vel_w"].append(root_lin_vel[i].copy())
            data["root_ang_vel_w"].append(root_ang_vel[i].copy())
            data["projected_gravity_b"].append(projected_gravity[i].copy())
            # Same frame convention as root_pos_w: scene-local, env origin removed.
            data["body_pos_w"].append((body_pos[i] - env_origins[i]).copy())
            data["body_quat_w"].append(body_quat[i].copy())

            if applied_action is not None:
                data["applied_joint_action"].append(applied_action[i].copy())
            if full_latent is not None:
                data["action_motion_token"].append(full_latent[i].copy())
            if meta_action is not None:
                data["policy_meta_action"].append(meta_action[i].copy())
            if reference_qpos is not None:
                data["reference_g1_qpos"].append(reference_qpos[i].copy())
                data["motion_id"].append(int(motion_ids[i]))
                data["motion_time_step"].append(int(motion_time_steps[i]))
                data["motion_time_s"].append(float(motion_times_s[i]))
            for name, values in tracking_metrics.items():
                data["tracking_metrics"].setdefault(name, []).append(np.asarray(values[i]).copy())
            if contact_force_norm is not None:
                data["robot_contact_force_w"].append(contact_force_w[i].copy())
                data["robot_contact_force_norm_w"].append(contact_force_norm[i].copy())
                data["max_nonfoot_contact_force_n"].append(float(max_nonfoot_contact_force[i]))
                data["left_foot_contact_force_n"].append(
                    float(left_foot_contact_force[i])
                    if left_foot_contact_force is not None
                    else 0.0
                )
                # Both feet are recorded here, under the contact-force guard. The right one
                # used to sit inside the `right_foot_ground_contact_force_w` block below,
                # so whenever the ground-contact sensor was absent -- which is the case for
                # every bare-plane rollout -- the right foot's scalar was silently dropped
                # while the left's was kept. Every such episode then failed evaluation with
                # "missing required field: right_foot_contact_force_n" and was classified
                # unevaluable, which reads as a recording failure rather than the asymmetry
                # it was.
                data["right_foot_contact_force_n"].append(
                    float(right_foot_contact_force[i])
                    if right_foot_contact_force is not None
                    else 0.0
                )
            if left_foot_ground_contact_force_w is not None:
                data["left_foot_ground_contact_force_w"].append(
                    left_foot_ground_contact_force_w[i].copy()
                )
            if right_foot_ground_contact_force_w is not None:
                data["right_foot_ground_contact_force_w"].append(
                    right_foot_ground_contact_force_w[i].copy()
                )

            # Object state
            if self._has_object:
                obj = self.env.scene["object"]
                obj_pos = obj.data.root_pos_w[i].cpu().numpy().copy()
                obj_pos_rel = obj_pos - env_origins[i]
                obj_quat = obj.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["object_pos_w"].append(obj_pos_rel)
                self._frame_data[i]["object_quat_w"].append(obj_quat)

            # Table state
            if self._has_table:
                table = self.env.scene["table"]
                table_pos = table.data.root_pos_w[i].cpu().numpy().copy()
                table_pos_rel = table_pos - env_origins[i]
                table_quat = table.data.root_quat_w[i].cpu().numpy().copy()
                self._frame_data[i]["table_pos_w"].append(table_pos_rel)
                self._frame_data[i]["table_quat_w"].append(table_quat)

        self.frame_id += 1
        return "trajectory_record", torch.ones(self.env.num_envs, 1, device=self.env.device)

    def close_writers(self):
        """Save all trajectory data to pkl files."""
        if self._closed or not self._initialized:
            return
        self._closed = True

        # FPS matches the video (after frame skip)
        effective_fps = 1.0 / (self.env.step_dt * self.render_frame_skip)

        scene_metadata = {}

        for i in range(self.num_record_envs):
            env_idx = self.start_idx + i
            data = self._frame_data[i]

            if not data["dof_pos"]:
                continue

            # Stack frame arrays
            trajectory = {
                "schema_version": 2,
                "kind": "sonic_physics_trajectory",
                "dof_pos": np.array(data["dof_pos"]),
                "dof_vel": np.array(data["dof_vel"]),
                "root_pos_w": np.array(data["root_pos_w"]),
                "root_quat_w": np.array(data["root_quat_w"]),
                "root_lin_vel_w": np.array(data["root_lin_vel_w"]),
                "root_ang_vel_w": np.array(data["root_ang_vel_w"]),
                "projected_gravity_b": np.array(data["projected_gravity_b"]),
                "quat_format": "wxyz",
                "fps": effective_fps,
                "num_joints": data["dof_pos"][0].shape[0],
                "total_frames": len(data["dof_pos"]),
                # RecorderManager calls this term after physics and observation
                # updates, while the policy token was selected before the same
                # step. Dataset export uses these declarations to perform the
                # required one-frame causal shift.
                "recording_phase": "post_physics_step",
                "policy_action_phase": "pre_physics_step_policy_output",
                # Isaac Articulation tensors and TrackingCommand.joint_pos both
                # use the articulation's Isaac Lab order. The exporter must
                # reorder these values before labeling them as MuJoCo/G1 data.
                "dof_order": "isaaclab",
                "dof_joint_names": self._dof_joint_names,
                "reference_g1_qpos_dof_order": "isaaclab",
            }

            for key in (
                "body_pos_w",
                "body_quat_w",
                "applied_joint_action",
                "action_motion_token",
                "policy_meta_action",
                "reference_g1_qpos",
                "motion_id",
                "motion_time_step",
                "motion_time_s",
                "robot_contact_force_norm_w",
                "robot_contact_force_w",
                "left_foot_ground_contact_force_w",
                "right_foot_ground_contact_force_w",
                "max_nonfoot_contact_force_n",
                "left_foot_contact_force_n",
                "right_foot_contact_force_n",
            ):
                if data[key]:
                    trajectory[key] = np.asarray(data[key])
            if "body_pos_w" in trajectory:
                trajectory["body_names"] = self._body_names
            if "robot_contact_force_norm_w" in trajectory:
                trajectory["contact_body_names"] = self._contact_body_names
                trajectory["allowed_foot_contact_body_names"] = self._ALLOWED_FOOT_CONTACT_BODIES
            if "left_foot_ground_contact_force_w" in trajectory:
                trajectory["support_floor_prim_path"] = "/World/ground/terrain/Structure/Floor"
            if data["tracking_metrics"]:
                trajectory["tracking_metrics"] = {}
                trajectory["partial_tracking_metrics"] = {}
                for name, values in data["tracking_metrics"].items():
                    destination = (
                        trajectory["tracking_metrics"]
                        if len(values) == trajectory["total_frames"]
                        else trajectory["partial_tracking_metrics"]
                    )
                    destination[name] = np.asarray(values)
                if not trajectory["partial_tracking_metrics"]:
                    trajectory.pop("partial_tracking_metrics")

            if data.get("object_pos_w"):
                trajectory["object_pos_w"] = np.array(data["object_pos_w"])
                trajectory["object_quat_w"] = np.array(data["object_quat_w"])
            else:
                trajectory["object_pos_w"] = None
                trajectory["object_quat_w"] = None

            if data.get("table_pos_w"):
                trajectory["table_pos_w"] = np.array(data["table_pos_w"])
                trajectory["table_quat_w"] = np.array(data["table_quat_w"])
            else:
                trajectory["table_pos_w"] = None
                trajectory["table_quat_w"] = None

            # Save pkl
            pkl_path = os.path.join(self.save_dir, f"{env_idx:06d}.trajectory.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved trajectory: {pkl_path} ({trajectory['total_frames']} frames)")

            # Build metadata entry
            meta = {
                "schema_version": trajectory["schema_version"],
                "trajectory_file": f"{env_idx:06d}.trajectory.pkl",
                "video_file": f"{env_idx:06d}.mp4",
                "num_frames": trajectory["total_frames"],
                "num_joints": trajectory["num_joints"],
                "fps": effective_fps,
                "has_motion_token": "action_motion_token" in trajectory,
                "motion_token_dim": (
                    int(trajectory["action_motion_token"].shape[-1])
                    if "action_motion_token" in trajectory
                    else None
                ),
                "has_reference_g1_qpos": "reference_g1_qpos" in trajectory,
                "has_robot_contact_forces": "robot_contact_force_norm_w" in trajectory,
                "has_pair_resolved_ground_contact": (
                    "left_foot_ground_contact_force_w" in trajectory
                    and "right_foot_ground_contact_force_w" in trajectory
                ),
                "has_object": trajectory["object_pos_w"] is not None,
                "has_table": trajectory["table_pos_w"] is not None,
            }

            # Add object USD path if available from config
            if hasattr(self.env, "wrapper"):
                obj_usd = self.env.wrapper.config.get("object_usd_path", None)
                if obj_usd:
                    meta["object_usd_path"] = obj_usd

            scene_metadata[str(env_idx)] = meta

        # Save scene metadata JSON
        meta_path = os.path.join(self.save_dir, "scene_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(scene_metadata, f, indent=2)
        logger.info(f"Saved scene metadata: {meta_path}")
        logger.info("=== TrajectoryRecorder: all data saved ===")

    def __del__(self):
        self.close_writers()


@configclass
class TrajectoryRecorderCfg(manager_term_cfg.RecorderTermCfg):
    """Configuration for trajectory recording alongside video."""

    class_type = TrajectoryRecorderTerm
    save_path: str = None
