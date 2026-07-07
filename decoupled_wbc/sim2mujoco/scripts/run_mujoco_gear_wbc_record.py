"""Headless MuJoCo rollout of GR00T decoupled WBC policies, recording MP4.

Adapted from run_mujoco_gear_wbc.py — drops pynput/torch/cuda deps and the
interactive viewer; uses mujoco.Renderer for offscreen frames and imageio
for the MP4 writer. Good for unattended verification in sandboxed/CI shells.

Usage:
    python run_mujoco_gear_wbc_record.py \
        --config g1_gear_wbc_headless.yaml \
        --out rollouts/wbc_demo.mp4 \
        --cmd walk          # or "balance"
"""
import argparse
import collections
import os
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
import yaml


def quat_rotate_inverse(q, v):
    w, x, y, z = q
    qc = np.array([w, -x, -y, -z])
    return np.array([
        v[0] * (qc[0]**2 + qc[1]**2 - qc[2]**2 - qc[3]**2)
        + v[1] * 2 * (qc[1]*qc[2] - qc[0]*qc[3])
        + v[2] * 2 * (qc[1]*qc[3] + qc[0]*qc[2]),
        v[0] * 2 * (qc[1]*qc[2] + qc[0]*qc[3])
        + v[1] * (qc[0]**2 - qc[1]**2 + qc[2]**2 - qc[3]**2)
        + v[2] * 2 * (qc[2]*qc[3] - qc[0]*qc[1]),
        v[0] * 2 * (qc[1]*qc[3] - qc[0]*qc[2])
        + v[1] * 2 * (qc[2]*qc[3] + qc[0]*qc[1])
        + v[2] * (qc[0]**2 - qc[1]**2 - qc[2]**2 + qc[3]**2),
    ])


def get_gravity_orientation(quat):
    return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def compute_observation(d, config, action, control_dict, n_joints, single_obs_dim=86):
    command = np.zeros(7, dtype=np.float32)
    command[:3] = control_dict["loco_cmd"][:3] * config["cmd_scale"]
    command[3] = control_dict["height_cmd"]
    command[4:7] = control_dict["rpy_cmd"]

    qj = d.qpos[7:7 + n_joints].copy()
    dqj = d.qvel[6:6 + n_joints].copy()
    quat = d.qpos[3:7].copy()
    omega = d.qvel[3:6].copy()

    padded_defaults = np.zeros(n_joints, dtype=np.float32)
    L = min(len(config["default_angles"]), n_joints)
    padded_defaults[:L] = config["default_angles"][:L]

    qj_scaled = (qj - padded_defaults) * config["dof_pos_scale"]
    dqj_scaled = dqj * config["dof_vel_scale"]
    gravity = get_gravity_orientation(quat)
    omega_scaled = omega * config["ang_vel_scale"]

    single = np.zeros(single_obs_dim, dtype=np.float32)
    single[0:7] = command[:7]
    single[7:10] = omega_scaled
    single[10:13] = gravity
    single[13:13 + n_joints] = qj_scaled
    single[13 + n_joints:13 + 2 * n_joints] = dqj_scaled
    single[13 + 2 * n_joints:13 + 2 * n_joints + 15] = action
    return single


def load_onnx(path):
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    def run(x):
        return sess.run(None, {in_name: x.astype(np.float32)})[0]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="g1_gear_wbc_headless.yaml")
    ap.add_argument("--out", default="rollouts/wbc_demo.mp4")
    ap.add_argument("--cmd", choices=["balance", "walk"], default="walk")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    cfg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_root = os.path.join(cfg_root, "resources", "robots", "g1")
    with open(os.path.join(cfg_root, args.config), "r") as f:
        config = yaml.safe_load(f)
    for k in ("policy_path", "walk_policy_path", "xml_path"):
        config[k] = os.path.join(cfg_root, config[k])
    for k in ("kps", "kds", "default_angles", "cmd_scale", "cmd_init"):
        config[k] = np.array(config[k], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(config["xml_path"])
    data = mujoco.MjData(model)
    model.opt.timestep = config["simulation_dt"]
    n_joints = data.qpos.shape[0] - 7

    balance_policy = load_onnx(config["policy_path"])
    walk_policy = load_onnx(config["walk_policy_path"])

    if args.cmd == "walk":
        control_dict = {
            "loco_cmd": np.array([0.5, 0.0, 0.0], dtype=np.float32),
            "height_cmd": config["height_cmd"],
            "rpy_cmd": np.array(config.get("rpy_cmd", [0.0, 0.0, 0.0]), dtype=np.float32),
            "freq_cmd": config.get("freq_cmd", 1.5),
        }
    else:
        control_dict = {
            "loco_cmd": config["cmd_init"].astype(np.float32),
            "height_cmd": config["height_cmd"],
            "rpy_cmd": np.array(config.get("rpy_cmd", [0.0, 0.0, 0.0]), dtype=np.float32),
            "freq_cmd": config.get("freq_cmd", 1.5),
        }

    action = np.zeros(config["num_actions"], dtype=np.float32)
    target_dof_pos = config["default_angles"].copy()
    single_obs_dim = 86
    obs_hist = collections.deque(
        [np.zeros(single_obs_dim, dtype=np.float32)] * config["obs_history_len"],
        maxlen=config["obs_history_len"],
    )
    obs = np.zeros(config["num_obs"], dtype=np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.5
    cam.azimuth = 135.0
    cam.elevation = -15.0

    sim_duration = float(config["simulation_duration"])
    frames_every = int(round(1.0 / (args.fps * model.opt.timestep)))
    counter = 0
    wall_start = time.time()
    t_sim = 0.0
    frames_written = 0
    print(f"[record] mode={args.cmd} duration={sim_duration}s fps={args.fps} -> {args.out}")

    try:
        while t_sim < sim_duration:
            leg_tau = pd_control(
                target_dof_pos,
                data.qpos[7:7 + config["num_actions"]],
                config["kps"],
                np.zeros_like(config["kps"]),
                data.qvel[6:6 + config["num_actions"]],
                config["kds"],
            )
            data.ctrl[:config["num_actions"]] = leg_tau
            if n_joints > config["num_actions"]:
                extra = n_joints - config["num_actions"]
                arm_tau = pd_control(
                    np.zeros(extra, dtype=np.float32),
                    data.qpos[7 + config["num_actions"]:7 + n_joints],
                    np.full(extra, 100.0),
                    np.zeros(extra),
                    data.qvel[6 + config["num_actions"]:6 + n_joints],
                    np.full(extra, 0.5),
                )
                data.ctrl[config["num_actions"]:] = arm_tau

            mujoco.mj_step(model, data)
            counter += 1
            t_sim += model.opt.timestep

            if counter % config["control_decimation"] == 0:
                so = compute_observation(data, config, action, control_dict, n_joints, single_obs_dim)
                obs_hist.append(so)
                for i, h in enumerate(obs_hist):
                    obs[i * single_obs_dim:(i + 1) * single_obs_dim] = h
                x = obs[None, :]
                pol = balance_policy if np.linalg.norm(control_dict["loco_cmd"]) <= 0.05 else walk_policy
                action = pol(x).squeeze()
                target_dof_pos = action * config["action_scale"] + config["default_angles"]

            if counter % frames_every == 0:
                cam.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=cam)
                writer.append_data(renderer.render())
                frames_written += 1
    finally:
        writer.close()
        renderer.close()

    wall = time.time() - wall_start
    print(f"[record] done  sim_time={t_sim:.2f}s  wall={wall:.2f}s  frames={frames_written}  file={args.out}")
    print(f"[record] final base height z={data.qpos[2]:.3f}m  root_xy=({data.qpos[0]:.3f}, {data.qpos[1]:.3f})")


if __name__ == "__main__":
    main()
