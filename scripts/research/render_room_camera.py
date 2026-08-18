#!/usr/bin/env python
"""A static observer camera in the room, rendered from recorded body poses.

The ego camera rides the head and the chase camera follows the robot; against a moving background
you cannot see gait, only the limbs. A camera pinned to the room shows the robot walk *past* you,
which is what makes the behaviour legible -- and, in a counterfactual pair, makes the difference
between the two behaviours legible side by side.

This projects the executed collision capsules and the room's own geometry through a fixed pinhole
camera, so it needs no GPU and no second Isaac pass. Bodies are depth-sorted so nearer parts occlude
further ones.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import build_counterfactual_family as cf  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
FLOOR = "#cfd4da"
WALL = "#e4e7ea"
SHELF = "#b8a68f"
SHELF_EDGE = "#7a6a54"
#: Parent-child body links, so the figure reads as a body rather than as scattered parts. G1's
#: collision model is 14 capsules with real gaps between them, which is correct for physics and
#: illegible as a silhouette; these segments are drawn for the eye only and carry no collision
#: meaning.
SKELETON = (
    ("pelvis", "waist_yaw_link"),
    ("waist_yaw_link", "torso_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("right_shoulder_pitch_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("left_hip_pitch_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("right_hip_pitch_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
)
ROBOT_OK = "#2f6b4f"
ROBOT_HIT = "#a33a2c"


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World-to-camera rotation for a pinhole at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=np.float64))
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, forward]), eye


def project(points, rotation, eye, focal):
    """World points to image plane. Returns ``(uv, depth)``; depth <= 0 is behind the camera."""
    local = (np.asarray(points, dtype=np.float64) - eye) @ rotation.T
    depth = local[..., 2]
    safe = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    uv = np.stack([focal * local[..., 0] / safe, focal * local[..., 1] / safe], axis=-1)
    return uv, depth


def box_faces(box):
    x0, y0, z0, x1, y1, z1 = box
    corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [corners[list(face)] for face in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--focal", type=float, default=760.0)
    ap.add_argument("--stride", type=int, default=2)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    with open(sorted(args.cell.glob("trajectories/*.trajectory.pkl"))[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    outcome = classify_episode(args.cell.name, payload)
    colour = ROBOT_HIT if outcome.outcome == "rejected" else ROBOT_OK

    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
    body_index = {name: i for i, name in enumerate(payload["body_names"])}
    links = [
        (body_index[a], body_index[b]) for a, b in SKELETON if a in body_index and b in body_index
    ]

    log = args.cell.parent / "logs" / f"{args.cell.name}.runner.log"
    scene_id = None
    if log.exists():
        match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
        scene_id = match.group(1) if match else None
    shelf = None
    room = (10.0, 5.0)
    if scene_id and (SCENES / f"{scene_id}.usda").exists():
        text = (SCENES / f"{scene_id}.usda").read_text()
        floor = text[text.index('Plane "Floor"') :]
        room = (
            float(re.search(r"double width = ([\d.]+)", floor).group(1)),
            float(re.search(r"double length = ([\d.]+)", floor).group(1)),
        )
        try:
            shelf = cf.rendered_shelf_box(SCENES / f"{scene_id}.usda")
        except Exception:  # noqa: BLE001
            shelf = None

    # Pinned beside and behind the obstacle, at standing eye height, so the robot walks across
    # the frame rather than toward the lens.
    focus_x = float((shelf[0] + shelf[3]) / 2) if shelf is not None else float(root[:, 0].mean())
    eye = np.array([focus_x - 0.9, -2.7, 1.35])
    rotation, eye = look_at(eye, (focus_x + 0.25, 0.0, 0.80))

    half_w, half_l = room[0] / 2, room[1] / 2
    # The floor is drawn as tiles rather than one quad. A single ground plane has corners behind
    # the camera, and rejecting a quad when any corner is behind it drops the whole floor -- which
    # is why the first version rendered the robot against blank space.
    tiles = []
    step = 0.5
    for x in np.arange(-half_w, half_w, step):
        for y in np.arange(-half_l, half_l, step):
            tiles.append(
                np.array(
                    [[x, y, 0.0], [x + step, y, 0.0], [x + step, y + step, 0.0], [x, y + step, 0.0]]
                )
            )
    back_wall = np.array(
        [
            [-half_w, half_l, 0.0],
            [half_w, half_l, 0.0],
            [half_w, half_l, 2.8],
            [-half_w, half_l, 2.8],
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.05), dpi=130)
    frames = []
    for frame in range(0, len(root), args.stride):
        ax.clear()
        uv, depth = project(back_wall, rotation, eye, args.focal)
        if (depth > 0.05).all():
            ax.add_patch(Polygon(uv, closed=True, facecolor=WALL, edgecolor="none", zorder=0))
        for tile in tiles:
            uv, depth = project(tile, rotation, eye, args.focal)
            if (depth > 0.05).all():
                # A faint checker keeps the floor readable as a receding surface, which is what
                # makes the walking speed legible from a still frame.
                shade = FLOOR if int(tile[0, 0] * 2 + tile[0, 1] * 2) % 2 else "#c4c9d0"
                ax.add_patch(Polygon(uv, closed=True, facecolor=shade, edgecolor="none", zorder=0))

        drawable = []
        for index in range(starts.shape[1]):
            a, b, r = starts[frame, index], ends[frame, index], radii[index]
            uv, depth = project(np.stack([a, b]), rotation, eye, args.focal)
            if (depth <= 0.05).any():
                continue
            width = args.focal * r / depth.mean()
            direction = uv[1] - uv[0]
            norm = float(np.hypot(*direction))
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            normal = np.array([-direction[1], direction[0]]) / norm * width
            drawable.append(
                (
                    float(depth.mean()),
                    np.array([uv[0] + normal, uv[1] + normal, uv[1] - normal, uv[0] - normal]),
                )
            )
        for _, quad in sorted(drawable, key=lambda item: -item[0]):
            ax.add_patch(
                Polygon(quad, closed=True, facecolor=colour, edgecolor="none", alpha=0.85, zorder=2)
            )
        for first, second in links:
            uv, depth = project(
                np.stack([body_pos[frame, first], body_pos[frame, second]]),
                rotation,
                eye,
                args.focal,
            )
            if (depth > 0.05).all():
                ax.plot(
                    uv[:, 0],
                    uv[:, 1],
                    color=colour,
                    linewidth=2.4,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=1,
                )

        if shelf is not None:
            for face in box_faces(shelf):
                uv, depth = project(face, rotation, eye, args.focal)
                if (depth > 0.05).all():
                    ax.add_patch(
                        Polygon(
                            uv,
                            closed=True,
                            facecolor=SHELF,
                            edgecolor=SHELF_EDGE,
                            linewidth=0.8,
                            alpha=0.97,
                            zorder=3,
                        )
                    )

        ax.set_xlim(-430, 430)
        ax.set_ylim(-300, 250)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{args.cell.name}   {outcome.outcome}   frame {frame}", fontsize=10, loc="left"
        )
        fig.tight_layout(pad=0.4)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    plt.close(fig)

    imageio.mimsave(args.out, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"{args.cell.name}: {outcome.outcome}, {len(frames)} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
