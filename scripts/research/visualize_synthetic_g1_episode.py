#!/usr/bin/env python3
"""Render a human-inspectable summary figure for one synthetic_g1 episode.

Schema validation says the numbers are well-formed. It cannot say the robot walked
where it was told, that the camera was actually looking out of the robot's head, or
that the behaviour is worth training on. Those need eyes.

The figure has four panels:

1. **Ego contact sheet** -- frames sampled across the episode, so you can see what
   the policy saw and whether the view moves with the body.
2. **Top-down path** -- executed root vs commanded reference, drawn over the scene's
   real obstacle footprints, so route error and clearance are visible in context.
3. **Root height and tilt** -- executed against reference, which is what the
   acceptance gates compare (see design doc section 20).
4. **Action and contact** -- motion-token activity plus the self/lateral/support
   contact split.

Usage::

    python scripts/research/visualize_synthetic_g1_episode.py \\
        --rollout /path/rollouts/<config_id> --out figure.png
    python scripts/research/visualize_synthetic_g1_episode.py \\
        --rollout ... --scene factory_aisle --frames 8 --out figure.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from gear_sonic.dataset_generation.contact_decomposition import (  # noqa: E402
    decompose_payload_contacts,
)
from gear_sonic.dataset_generation.scene_asset_preflight import (  # noqa: E402
    load_scene_obstacle_map,
)


def _tilt(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternions, dtype=np.float64).T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    return np.maximum(np.abs(roll), np.abs(pitch))


def sample_video_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            index += 1
    finally:
        capture.release()
    return [frames[i] for i in indices if i in frames]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=6, help="ego frames to sample")
    parser.add_argument("--scene", default=None, help="scene id; read from manifest if omitted")
    args = parser.parse_args()

    traj_paths = sorted(args.rollout.glob("trajectories/*.trajectory.pkl"))
    video_paths = sorted(args.rollout.glob("renders/*.mp4"))
    if not traj_paths or not video_paths:
        print(f"ERROR: {args.rollout} has no trajectory/video", file=sys.stderr)
        return 2
    with traj_paths[0].open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local recorder artifact

    scene_id = args.scene
    manifest_path = args.rollout / "success_manifest.json"
    task = ""
    if manifest_path.is_file():
        capture = json.loads(manifest_path.read_text(encoding="utf-8"))["capture_context"]
        scene_id = scene_id or capture.get("scene_id")
        task = capture.get("task", "")

    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)
    frame_count = len(root)
    fps = float(payload["fps"])
    time_s = np.arange(frame_count) / fps
    decomposition = decompose_payload_contacts(payload)

    indices = np.linspace(0, frame_count - 1, args.frames).astype(int).tolist()
    images = sample_video_frames(video_paths[0], indices)

    figure = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, max(len(images), 4))

    # Panel 1: ego contact sheet.
    for column, (image, index) in enumerate(zip(images, indices)):
        axis = figure.add_subplot(grid[0, column])
        axis.imshow(image)
        axis.set_title(f"t={index / fps:.2f}s", fontsize=9)
        axis.axis("off")

    # Panel 2: top-down path over real obstacle footprints.
    path_axis = figure.add_subplot(grid[1, :2])
    if scene_id:
        try:
            obstacle_map = load_scene_obstacle_map(scene_id)
            for _, (min_x, min_y, max_x, max_y) in obstacle_map.obstacles:
                path_axis.add_patch(
                    mpatches.Rectangle(
                        (min_x, min_y),
                        max_x - min_x,
                        max_y - min_y,
                        facecolor="0.75",
                        edgecolor="0.4",
                        linewidth=0.5,
                    )
                )
            path_axis.set_xlim(obstacle_map.walkable_min_xy[0], obstacle_map.walkable_max_xy[0])
            path_axis.set_ylim(obstacle_map.walkable_min_xy[1], obstacle_map.walkable_max_xy[1])
        except (ValueError, OSError) as exc:  # scene not in the package
            path_axis.set_title(f"(no scene geometry: {exc})", fontsize=8)
    path_axis.plot(reference[:, 0], reference[:, 1], "--", color="tab:orange", label="reference")
    path_axis.plot(root[:, 0], root[:, 1], "-", color="tab:blue", label="executed")
    path_axis.plot(root[0, 0], root[0, 1], "o", color="green", markersize=8, label="start")
    path_axis.plot(root[-1, 0], root[-1, 1], "s", color="red", markersize=8, label="end")
    path_axis.set_aspect("equal")
    path_axis.set_xlabel("x [m]")
    path_axis.set_ylabel("y [m]")
    path_axis.legend(fontsize=8, loc="best")
    path_axis.set_title("top-down path over scene obstacles", fontsize=10)

    # Panel 3: what the fall gates actually compare.
    height_axis = figure.add_subplot(grid[1, 2:])
    height_axis.plot(time_s, reference[:, 2], "--", color="tab:orange", label="reference height")
    height_axis.plot(time_s, root[:, 2], "-", color="tab:blue", label="executed height")
    height_axis.set_ylabel("root height [m]")
    height_axis.set_xlabel("time [s]")
    tilt_axis = height_axis.twinx()
    tilt_axis.plot(
        time_s, _tilt(payload["root_quat_w"]), color="tab:red", alpha=0.6, label="executed tilt"
    )
    tilt_axis.plot(
        time_s,
        _tilt(reference[:, 3:7]),
        ":",
        color="tab:red",
        alpha=0.6,
        label="reference tilt",
    )
    tilt_axis.set_ylabel("tilt [rad]", color="tab:red")
    height_axis.legend(fontsize=7, loc="lower left")
    tilt_axis.legend(fontsize=7, loc="lower right")
    height_axis.set_title("height and tilt vs command", fontsize=10)

    # Panel 4: action activity and the contact split.
    token = np.asarray(payload["action_motion_token"], dtype=np.float64)
    action_axis = figure.add_subplot(grid[2, :2])
    action_axis.imshow(token.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
    action_axis.set_title(
        f"action.motion_token (64 dims x {frame_count} frames), "
        f"{np.unique(token).size} distinct values",
        fontsize=10,
    )
    action_axis.set_xlabel("frame")
    action_axis.set_ylabel("latent dim")

    contact_axis = figure.add_subplot(grid[2, 2:])
    contact_axis.plot(time_s, decomposition.self_contact_by_frame, label="self-contact")
    contact_axis.plot(time_s, decomposition.external_lateral_by_frame, label="lateral (scene)")
    contact_axis.plot(time_s, decomposition.external_support_by_frame, label="support (non-foot)")
    contact_axis.set_xlabel("time [s]")
    contact_axis.set_ylabel("force [N]")
    contact_axis.legend(fontsize=8)
    contact_axis.set_title("contact decomposition", fontsize=10)

    figure.suptitle(
        f"{args.rollout.name}   |   scene={scene_id}   |   task={task!r}   |   "
        f"{frame_count} frames @ {fps:g} Hz",
        fontsize=12,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    plt.close(figure)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
