#!/usr/bin/env python
"""Render a counterfactual family as third-person video and stills, from recorded trajectories.

The rollouts already carry a first-person ego camera. A third-person view would need another GPU
pass per cell through Isaac, which is the wrong cost when the executed body poses and the scene's
obstacle box are both already on disk. Drawing the collision capsules directly also shows the thing
that actually decides the outcome -- the swept body volume against the shelf underside -- which a
photoreal chase camera would hide behind surfaces.

Produces, per cell: an mp4 of the side view, and a still at the frame the gate flagged.
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

import build_counterfactual_family as cf  # noqa: E402

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
#: Drawn in a warm grey so the shelf reads as furniture rather than as part of the robot.
SHELF_FACE = "#b8a68f"
ROBOT_OK = "#2f6b4f"
ROBOT_HIT = "#a33a2c"


def load(cell: Path):
    paths = sorted(cell.glob("trajectories/*.trajectory.pkl"))
    with open(paths[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))
    return payload


def scene_of(cell: Path) -> str | None:
    log = cell.parent / "logs" / f"{cell.name}.runner.log"
    if not log.exists():
        return None
    match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
    return match.group(1) if match else None


def capsule_patches(a, b, r, colour):
    """A capsule in the x-z plane, as real geometry rather than a thick line.

    Line width is measured in points, so scaling it by the radius makes the body's size depend on
    figure dpi and zoom -- the robot changes shape when the framing does. Drawn as a rectangle
    between the endpoints plus a disc at each end, the silhouette is correct in metres, which
    matters here because the silhouette is exactly what the shelf decides against.
    """
    p0 = np.array([a[0], a[2]])
    p1 = np.array([b[0], b[2]])
    direction = p1 - p0
    length = float(np.hypot(*direction))
    patches = [
        Circle(p0, r, facecolor=colour, edgecolor="none", alpha=0.45, zorder=2),
        Circle(p1, r, facecolor=colour, edgecolor="none", alpha=0.45, zorder=2),
    ]
    if length > 1e-9:
        normal = np.array([-direction[1], direction[0]]) / length * r
        patches.append(
            Polygon(
                [p0 + normal, p1 + normal, p1 - normal, p0 - normal],
                closed=True,
                facecolor=colour,
                edgecolor="none",
                alpha=0.45,
                zorder=2,
            )
        )
    return patches


def draw(ax, starts, ends, radii, frame, shelf, colour, xlim, title):
    """Side elevation: x forward, z up, capsules drawn to scale in metres."""
    ax.clear()
    ax.axhspan(-0.08, 0.0, facecolor="#c9ced6", zorder=0)
    if shelf is not None:
        ax.add_patch(
            Rectangle(
                (shelf[0], shelf[2]),
                shelf[3] - shelf[0],
                shelf[5] - shelf[2],
                facecolor=SHELF_FACE,
                edgecolor="#6b5d4a",
                linewidth=1.2,
                zorder=3,
            )
        )
        ax.annotate(
            f"shelf {shelf[2]:.3f} m",
            xy=((shelf[0] + shelf[3]) / 2, shelf[2]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#5a4d3d",
        )
    for index in range(starts.shape[1]):
        for patch in capsule_patches(
            starts[frame, index], ends[frame, index], radii[index], colour
        ):
            ax.add_patch(patch)
    top = float((np.maximum(starts[frame, :, 2], ends[frame, :, 2]) + radii).max())
    ax.axhline(top, color=colour, linewidth=0.9, linestyle=":", zorder=4)
    ax.annotate(
        f"peak {top:.3f} m",
        xy=(xlim[0] + 0.04, top),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=7.5,
        color=colour,
        weight="bold",
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.08, 1.72)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("height (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.12, linewidth=0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", type=Path, required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio

    for name in args.cells:
        cell = args.family / name
        payload = load(cell)
        outcome = classify_episode(name, payload)
        diagnostics = outcome.diagnostics or {}
        flagged = diagnostics.get("max_nonfoot_contact_frame")

        starts, ends, radii, _ = body_capsules_world(
            np.asarray(payload["body_pos_w"], dtype=np.float64),
            np.asarray(payload["body_quat_w"], dtype=np.float64),
            list(payload["body_names"]),
            capsules=G1_COLLISION_CAPSULES,
        )
        scene_id = scene_of(cell)
        shelf = None
        if scene_id and (SCENES / f"{scene_id}.usda").exists():
            try:
                shelf = cf.rendered_shelf_box(SCENES / f"{scene_id}.usda")
            except Exception:  # noqa: BLE001 - a scene without a shelf is fine to draw bare
                shelf = None

        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        colour = ROBOT_HIT if outcome.outcome == "rejected" else ROBOT_OK
        label = f"{name}   {outcome.outcome}"

        fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=120)
        frames = []
        for frame in range(0, len(root), 2):
            centre = float(root[frame, 0])
            draw(
                ax,
                starts,
                ends,
                radii,
                frame,
                shelf,
                colour,
                (centre - 1.35, centre + 1.35),
                f"{label}   frame {frame}",
            )
            fig.tight_layout()
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(image.copy())
        imageio.mimsave(
            args.out / f"{name}_third_person.mp4",
            frames,
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

        still = int(flagged) if flagged is not None else int(len(root) * 0.55)
        still = min(still, len(root) - 1)
        centre = float(root[still, 0])
        draw(
            ax,
            starts,
            ends,
            radii,
            still,
            shelf,
            colour,
            (centre - 1.35, centre + 1.35),
            f"{label}   frame {still}" + ("  (contact)" if flagged is not None else ""),
        )
        fig.tight_layout()
        fig.savefig(args.out / f"{name}_still.png", dpi=150)
        plt.close(fig)
        print(
            f"{name:>18s}  {outcome.outcome:>9s}  {len(frames)} frames"
            f"  still at {still}{'  (gate-flagged contact)' if flagged is not None else ''}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
