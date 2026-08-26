#!/usr/bin/env python3
"""Render the reproducible LFH GitHub Pages progress surface and visual evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.stage_geometry import (  # noqa: E402
    read_stage_geometry,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

COLORS = {
    "nominal": "#d95f4f",
    "adapted": "#2f8f6b",
    "binding": "#4b78c2",
    "context": "#9ba7b5",
    "station": "#d4962d",
}

SKELETON_CHAINS = (
    ("pelvis", "torso_link"),
    ("pelvis", "left_hip_pitch_link"),
    ("left_hip_pitch_link", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_hip_yaw_link"),
    ("left_hip_yaw_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_pitch_link"),
    ("left_ankle_pitch_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_pitch_link"),
    ("right_hip_pitch_link", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_hip_yaw_link"),
    ("right_hip_yaw_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_pitch_link"),
    ("right_ankle_pitch_link", "right_ankle_roll_link"),
    ("torso_link", "left_shoulder_pitch_link"),
    ("left_shoulder_pitch_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_shoulder_yaw_link"),
    ("left_shoulder_yaw_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_roll_link"),
    ("torso_link", "right_shoulder_pitch_link"),
    ("right_shoulder_pitch_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_shoulder_yaw_link"),
    ("right_shoulder_yaw_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_roll_link"),
)


def load(path: Path) -> dict:
    with path.open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    if payload is None:
        raise ValueError(f"{path}: no evaluable trajectory")
    return payload


def cube_faces(box: tuple[float, ...]) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0, x1, y1, z1 = box
    points = {
        "000": (x0, y0, z0),
        "001": (x0, y0, z1),
        "010": (x0, y1, z0),
        "011": (x0, y1, z1),
        "100": (x1, y0, z0),
        "101": (x1, y0, z1),
        "110": (x1, y1, z0),
        "111": (x1, y1, z1),
    }
    return [
        [points[key] for key in ("000", "100", "110", "010")],
        [points[key] for key in ("001", "101", "111", "011")],
        [points[key] for key in ("000", "100", "101", "001")],
        [points[key] for key in ("010", "110", "111", "011")],
        [points[key] for key in ("000", "010", "011", "001")],
        [points[key] for key in ("100", "110", "111", "101")],
    ]


def torso_path(payload: dict) -> np.ndarray:
    names = list(payload["body_names"])
    index = names.index("torso_link")
    return np.asarray(payload["body_pos_w"], dtype=np.float64)[:, index]


def render_route_map(
    nominal: dict,
    adapted: dict,
    stage,
    nominal_contact: dict,
    out: Path,
) -> None:
    nominal_root = np.asarray(nominal["root_pos_w"], dtype=np.float64)
    adapted_root = np.asarray(adapted["root_pos_w"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.2, 4.8), constrained_layout=True)
    for cube in stage.cubes:
        if cube.role not in {"binding_constraint", "constraint_context"}:
            continue
        x0, y0, _, x1, y1, _ = cube.box
        color = COLORS["binding"] if cube.role == "binding_constraint" else COLORS["context"]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=color,
                edgecolor=color,
                alpha=0.34,
                linewidth=1.5,
            )
        )
    ax.plot(
        nominal_root[:, 0],
        nominal_root[:, 1],
        color=COLORS["nominal"],
        linewidth=2.2,
        label="nominal — strikes",
    )
    ax.plot(
        adapted_root[:, 0],
        adapted_root[:, 1],
        color=COLORS["adapted"],
        linewidth=2.2,
        label="adapted — clears",
    )
    contact_frame = nominal_contact["first_frame"]
    drift_frame = nominal_contact["drift_onset_frame"]
    if contact_frame is not None:
        ax.scatter(
            nominal_root[contact_frame, 0],
            nominal_root[contact_frame, 1],
            marker="X",
            s=90,
            color=COLORS["nominal"],
            zorder=5,
            label=f"binding contact f{contact_frame}",
        )
    if drift_frame is not None:
        ax.scatter(
            nominal_root[drift_frame, 0],
            nominal_root[drift_frame, 1],
            marker="o",
            s=70,
            facecolor="none",
            edgecolor=COLORS["station"],
            linewidth=2,
            zorder=5,
            label=f"reference drift f{drift_frame}",
        )
    ax.set_title("Top-down executed routes and five-obstacle footprints")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", ncols=2, fontsize=9)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def render_scene_3d(nominal: dict, adapted: dict, stage, out: Path) -> None:
    fig = plt.figure(figsize=(9.2, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for cube in stage.cubes:
        if cube.role not in {"binding_constraint", "constraint_context"}:
            continue
        color = COLORS["binding"] if cube.role == "binding_constraint" else COLORS["context"]
        collection = Poly3DCollection(
            cube_faces(cube.box), facecolors=color, edgecolors=color, linewidths=0.7, alpha=0.28
        )
        ax.add_collection3d(collection)
    for payload, label, color in (
        (nominal, "nominal torso path", COLORS["nominal"]),
        (adapted, "adapted torso path", COLORS["adapted"]),
    ):
        path = torso_path(payload)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=2.2, label=label)
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color=color, s=25)
    all_paths = np.concatenate((torso_path(nominal), torso_path(adapted)), axis=0)
    plotted = [
        cube.box
        for cube in stage.cubes
        if cube.role in {"binding_constraint", "constraint_context"}
    ]
    x_limits = [all_paths[:, 0].min(), all_paths[:, 0].max()]
    y_limits = [all_paths[:, 1].min(), all_paths[:, 1].max()]
    z_max = all_paths[:, 2].max()
    for x0, y0, _, x1, y1, z1 in plotted:
        x_limits.extend((x0, x1))
        y_limits.extend((y0, y1))
        z_max = max(z_max, z1)
    ax.set_xlim(min(x_limits) - 0.3, max(x_limits) + 0.3)
    ax.set_ylim(min(y_limits) - 0.25, max(y_limits) + 0.25)
    ax.set_zlim(0.0, max(2.3, z_max + 0.2))
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_zlabel("height (m)")
    ax.set_title("Isometric five-obstacle scene with executed torso trajectories")
    ax.view_init(elev=23, azim=-63)
    ax.legend(loc="upper left")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def render_distribution(q_lfh: dict, e7: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), constrained_layout=True)
    sources = [atom["source_pair_id"] for atom in q_lfh["atoms"]]
    palette = plt.get_cmap("tab10")
    offsets = ((6, 7), (6, -15), (6, 8), (6, -15))
    for index, atom in enumerate(q_lfh["atoms"]):
        axes[0].scatter(
            atom["route_progress"],
            atom["face_along_route_m"],
            s=35 + 5 * atom["engineering_width_mm"],
            color=palette(index),
            alpha=0.78,
            edgecolor="white",
            linewidth=1,
        )
        axes[0].annotate(
            atom["source_pair_id"].replace("lfh_", "").replace("_crouch", ""),
            (atom["route_progress"], atom["face_along_route_m"]),
            xytext=offsets[index],
            textcoords="offset points",
            fontsize=9,
        )
    axes[0].set_title("Trajectory-conditioned geometry atoms")
    axes[0].set_xlabel("route progress $u^*$")
    axes[0].set_ylabel("finite exposure (m)")
    axes[0].grid(alpha=0.2)

    columns = ["shelf_plank", "door_lintel", "ibeam", "hanging_panel"]
    verified = {
        atom["source_pair_id"]: {row["archetype_id"] for row in atom["verified_archetypes"]}
        for atom in q_lfh["atoms"]
    }
    refused = {
        (row["source_pair_id"], row["archetype_id"])
        for row in e7["variants"]
        if not row["verified"]
    }
    matrix = np.zeros((len(sources), len(columns)), dtype=int)
    for row, source in enumerate(sources):
        for column, archetype in enumerate(columns):
            if archetype in verified[source]:
                matrix[row, column] = 2
            elif (source, archetype) in refused:
                matrix[row, column] = 1
    cmap = ListedColormap(["#edf0f4", "#d95f4f", "#2f8f6b"])
    axes[1].imshow(matrix, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    axes[1].set_xticks(range(len(columns)), [value.replace("_", "\n") for value in columns])
    axes[1].set_yticks(range(len(sources)), [value.replace("lfh_", "") for value in sources])
    axes[1].set_title("Physics-verified source × archetype support")
    for row in range(len(sources)):
        for column in range(len(columns)):
            label = {0: "—", 1: "refused", 2: "verified"}[matrix[row, column]]
            axes[1].text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if matrix[row, column] else "#55606e",
                fontsize=8,
            )
    axes[1].tick_params(length=0)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def render_direction_support(out: Path) -> None:
    """Show represented 3D face directions separately from physics-verified support."""

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    plan, vertical = axes
    plan.set_title("Route-frame plan view")
    plan.arrow(0, 0, 1.15, 0, width=0.018, head_width=0.13, color="#66717f")
    plan.text(1.2, 0, "tangent +t\nrepresentation", va="center", color="#55606e")
    for y, label in ((0.95, "left +l"), (-0.95, "right −l")):
        plan.arrow(0, 0, 0, y, width=0.018, head_width=0.13, color=COLORS["station"])
        plan.text(0.08, y, f"{label}\ncontext-only", va="center", color="#8b641b")
    plan.scatter([0], [0], s=95, color="#263238", label="executed path station")
    plan.set(xlim=(-0.25, 1.75), ylim=(-1.35, 1.35), xlabel="along route", ylabel="lateral")
    plan.set_aspect("equal")
    plan.grid(alpha=0.16)

    vertical.set_title("Vertical / oblique support")
    vertical.arrow(0, 0, 0, -1.0, width=0.018, head_width=0.13, color=COLORS["adapted"])
    vertical.text(-0.40, -1.03, "overhead −z\nVERIFIED CRITICAL", color=COLORS["adapted"])
    vertical.arrow(0, 0, 0, 0.95, width=0.018, head_width=0.13, color=COLORS["context"])
    vertical.text(0.08, 0.82, "floor +z\nunsupported", color="#66717f")
    vertical.plot([0, 0.9], [0, -0.75], linestyle="--", color=COLORS["context"], linewidth=2)
    vertical.text(0.78, -0.56, "oblique\nunsupported", color="#66717f")
    vertical.scatter([0], [0], s=95, color="#263238")
    vertical.set(
        xlim=(-0.45, 1.45), ylim=(-1.35, 1.35), xlabel="lateral component", ylabel="vertical"
    )
    vertical.set_aspect("equal")
    vertical.grid(alpha=0.16)
    fig.suptitle("LFH can represent all directions; evidence currently certifies only overhead")
    fig.savefig(out, dpi=170)
    plt.close(fig)


def skeleton_points(payload: dict, frame: int) -> tuple[np.ndarray, np.ndarray]:
    names = list(payload["body_names"])
    positions = np.asarray(payload["body_pos_w"], dtype=np.float64)[frame]
    segments = []
    for start, end in SKELETON_CHAINS:
        if start not in names or end not in names:
            continue
        a = positions[names.index(start)]
        b = positions[names.index(end)]
        segments.extend(((a[0], a[2]), (b[0], b[2]), (np.nan, np.nan)))
    points = np.asarray(segments, dtype=np.float64)
    return points[:, 0], points[:, 1]


def render_behavior_replay(cells: list[dict], out: Path, columns: int) -> None:
    """Render an evidence replay from measured body poses when sim cameras are occluded."""
    rows = (len(cells) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 3.3 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    artists = []
    for axis, cell in zip(axes.flat, cells, strict=False):
        payload = cell["payload"]
        positions = np.asarray(payload["body_pos_w"], dtype=np.float64)
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        binding_z_max = positions[:, :, 2].max()
        for cube in cell["stage"].cubes:
            if cube.role != "binding_constraint":
                continue
            x0, _, z0, x1, _, z1 = cube.box
            binding_z_max = max(binding_z_max, z1)
            axis.add_patch(
                plt.Rectangle(
                    (x0, z0),
                    x1 - x0,
                    z1 - z0,
                    facecolor=COLORS["binding"],
                    edgecolor=COLORS["binding"],
                    alpha=0.35,
                )
            )
        color = cell["color"]
        (line,) = axis.plot([], [], color=color, linewidth=2.2)
        nodes = axis.scatter([], [], s=13, color=color, zorder=3)
        (trail,) = axis.plot([], [], color=color, linewidth=1.1, alpha=0.32)
        marker = axis.scatter([], [], marker="X", s=85, color="#f6c344", zorder=5)
        marker.set_visible(False)
        status = axis.text(
            0.02,
            0.04,
            "",
            transform=axis.transAxes,
            fontsize=9,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "#263238", "alpha": 0.78},
        )
        axis.set_title(cell["title"], fontsize=10)
        axis.set_xlim(root[:, 0].min() - 0.35, root[:, 0].max() + 0.35)
        axis.set_ylim(max(0.0, positions[:, :, 2].min() - 0.15), binding_z_max + 0.1)
        axis.set_xlabel("route x (m)")
        axis.set_ylabel("height (m)")
        axis.grid(alpha=0.17)
        artists.append((line, nodes, trail, marker, status, cell))
    for axis in axes.flat[len(cells) :]:
        axis.set_visible(False)

    frame_count = min(int(cell["payload"]["total_frames"]) for cell in cells)
    stride = max(1, round(float(cells[0]["payload"].get("fps", 50)) / 25))
    frames = list(range(0, frame_count, stride))

    def update(frame: int):
        updated = []
        for line, nodes, trail, marker, status, cell in artists:
            payload = cell["payload"]
            x, z = skeleton_points(payload, frame)
            line.set_data(x, z)
            body = np.asarray(payload["body_pos_w"], dtype=np.float64)[frame]
            nodes.set_offsets(np.column_stack((body[:, 0], body[:, 2])))
            torso = torso_path(payload)
            start = max(0, frame - 35)
            trail.set_data(torso[start : frame + 1, 0], torso[start : frame + 1, 2])
            contact_frame = cell.get("contact_frame")
            if contact_frame is not None and abs(frame - contact_frame) <= stride * 2:
                torso_index = list(payload["body_names"]).index("torso_link")
                point = body[torso_index]
                marker.set_offsets([[point[0], point[2]]])
                marker.set_visible(True)
                status.set_text(f"BINDING CONTACT · frame {contact_frame}")
            else:
                marker.set_visible(False)
                status.set_text(cell["status"])
            updated.extend((line, nodes, trail, marker, status))
        return updated

    animation = FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    animation.save(out, writer=FFMpegWriter(fps=25, codec="libx264", bitrate=1800), dpi=110)
    plt.close(fig)


def video_poster(video: Path, out: Path, time_s: float = 2.4) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            str(time_s),
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )


def montage(inputs: list[tuple[Path, str]], out: Path, columns: int) -> None:
    width, height = 640, 360
    filters = []
    for index, (_, label) in enumerate(inputs):
        escaped = label.replace("'", "\\'").replace(":", "\\:")
        filters.append(
            f"[{index}:v]scale={width}:{height},"
            f"drawtext=text='{escaped}':x=10:y=10:fontsize=17:fontcolor=white:"
            "box=1:boxcolor=black@0.58[v{index}]".format(index=index)
        )
    rows = (len(inputs) + columns - 1) // columns
    layout = "|".join(
        f"{(index % columns) * width}_{(index // columns) * height}" for index in range(len(inputs))
    )
    stack_inputs = "".join(f"[v{index}]" for index in range(len(inputs)))
    filters.append(f"{stack_inputs}xstack=inputs={len(inputs)}:layout={layout}:fill=black[vout]")
    command = ["ffmpeg", "-y", "-loglevel", "error"]
    for path, _ in inputs:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[vout]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-shortest",
            str(out),
        ]
    )
    subprocess.run(command, check=True)
    if rows * height <= 0 or not out.exists():
        raise RuntimeError("video montage was not created")


def page(q_lfh: dict, e7: dict, e7c: dict, e9a: dict, e10: dict) -> str:
    total_gpu = e7["actual_contended_gpu_hours"] + e7c["actual_contended_gpu_hours"]
    fresh = {name.rsplit("__", 1)[-1]: cell["scientific"] for name, cell in e9a["cells"].items()}
    hard_contact = e10["contacts"]["nominal_hard"]
    atom_rows = []
    for atom in q_lfh["atoms"]:
        archetypes = ", ".join(row["archetype_id"] for row in atom["verified_archetypes"])
        atom_rows.append(
            f"| `{atom['source_pair_id']}` | {atom['route_progress']:.3f} | "
            f"{atom['face_along_route_m']:.2f} | {atom['engineering_width_mm']:.2f} | "
            f"{atom['source_weight']:.2f} | {archetypes} |"
        )
    return f"""# LFH Trajectory-Conditioned Scene Research

<div class="lfh-stats">
  <div><strong>5</strong><span>obstacles in latest scene</span></div>
  <div><strong>4/4</strong><span>E10b physics pattern</span></div>
  <div><strong>1</strong><span>verified critical direction</span></div>
  <div><strong>0/1</strong><span>fresh motion pairs ready</span></div>
</div>

LFH does not estimate natural obstacle frequency and never predicts a physics verdict. It builds a
source-balanced proposal distribution over obstacles that are feasible for exact executed nominal
and adapted trajectories. Deterministic geometry proposes; the unchanged physics scorer labels.

## Motion → LFH → physics loop

The implemented loop now begins with a freshly sampled Kimodo motion. A pair is allowed to reach
scene design only after both the original and deterministic adaptation track in empty-scene Isaac
physics. Failed motion pairs stop; LFH does not invent an obstacle around an invalid controller
response.

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/fresh-motion-wide-poster.png">
  <source src="../_static/lfh_progress/fresh-motion-wide.mp4" type="video/mp4">
</video>

This is actual Isaac Lab rendering of the new curved walking sample. Nominal accepted. Its 80 mm
crouch twin rejected with 0 N external contact: endpoint error rose from
{fresh["nominal"]["diagnostics"]["endpoint_error_m"]:.3f} m to
{fresh["adapted"]["diagnostics"]["endpoint_error_m"]:.3f} m and p95 path error from
{fresh["nominal"]["diagnostics"]["schedule_error_p95_m"]:.3f} m to
{fresh["adapted"]["diagnostics"]["schedule_error_p95_m"]:.3f} m. No scene was generated for this
pair.

## Completed multi-obstacle scene

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e10b-isaac-wide-poster.png">
  <source src="../_static/lfh_progress/e10b-isaac-wide.mp4" type="video/mp4">
</video>

This seed-matched Isaac replay shows the completed five-obstacle E10b hard pair. LFH places one
blue hanging panel as the causal binding obstacle and four route-relative context obstacles on the
left, right, floor level, and overhead. Nominal strikes the binding panel at frame
{hard_contact["first_frame"]}; adapted crouches and clears. The three intended-clear physics cells
have zero external contact, and no context object becomes causal. The wide videos are deterministic
presentation reruns; verdicts come from the hash-pinned experiment trajectories.

## Scene and trajectory maps

![Top-down route and multi-obstacle map](../_static/lfh_progress/route-map-2d.png)

The measured executed paths, binding footprint, and four context footprints are all shown in world
coordinates. Context placement is rejected against both swept bodies before physics.

![Isometric multi-obstacle scene and torso paths](../_static/lfh_progress/scene-map-3d.png)

The isometric view exposes the full 3D relationship rather than reducing the scene to one height.
The current v1 scene has one causal obstacle so the training label remains identifiable; context
objects increase scene complexity without receiving the causal label.

![Route-frame direction support](../_static/lfh_progress/direction-support.png)

LFH's representation can address route tangent, lateral, vertical, and oblique face normals. The
evidence is narrower: only overhead `−z` has a verified intervention window. Lateral faces are
context-only here, and floor/oblique critical placement stays fail-closed.

## Physics and causal metrics

| experiment | nominal easy | adapted easy | nominal hard | adapted hard | decision |
|---|---|---|---|---|---|
| E7c, one obstacle | accepted | accepted | rejected | accepted | verified |
| E10, 5 obstacles, seed 33101 | accepted | rejected | rejected | rejected | seed-confounded refusal |
| E10b, 5 obstacles, seed 32301 | accepted | accepted | rejected | accepted | verified |

E10b's CPU keep-out minimum is {e10["minimum_cpu_context_clearance_mm"]:.2f} mm versus the 50 mm
requirement. Nominal-hard torso contact is uniquely attributed to
`{hard_contact["attributed_prim_path"]}` before reference drift at frame
{hard_contact["drift_onset_frame"]}. The E10/E10b contrast also warns that a single successful seed
does not establish population-level context invariance.

## Existing crossed support

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e7c-four-cell-poster.png">
  <source src="../_static/lfh_progress/e7c-four-cell-replay.mp4" type="video/mp4">
</video>

The pose replay above remains useful for exact contact timing across all four source-086 cells.

<video controls muted loop playsinline preload="metadata" width="100%"
       poster="../_static/lfh_progress/e7-cross-source-hard-poster.png">
  <source src="../_static/lfh_progress/e7-cross-source-hard-replay.mp4" type="video/mp4">
</video>

The cross-source view holds the I-beam constraint class fixed. The top row contains nominal hard
strikes; the bottom row contains adapted clears for sources 086, 089, and 090.

## Designed proposal distribution

![Trajectory-conditioned proposal support](../_static/lfh_progress/proposal-distribution.png)

`q_LFH_v1` assigns 0.25 mass to each independent source, then balances verified archetypes within
that source and easy/hard geometry equally. Window size or scene multiplication cannot give one
source more training weight.

| source | route progress | exposure (m) | engineering window (mm) | source mass | verified archetypes |
|---|---:|---:|---:|---:|---|
{chr(10).join(atom_rows)}

The count-only gate is green: every source has three verified archetypes. The stricter crossed
learning gate remains **closed** because only `shelf_plank` and `ibeam` are common to all four
sources. The next experiment should transfer `hanging_panel` to `cf_005_056`, 089, and 090; only
then can E5 compare critical, uniform-feasible, DCS-ranked, and visual-only samplers without
source–archetype confounding.

**Archetype conditioning is refuted (audit, 2026-08-26).** `q_LFH_conditional_v1` scores 1.51373
nats leave-one-source-out against 1.60944 for a uniform prior, but **1.46416** for the same sampler
with the trajectory kernel replaced by a constant. Feature-blind counting wins; the kernel costs
0.0496 nats, and its top-3 recall and 400/400 in-support sampling are true by construction. The
executed pair determines *where* the face must be — closed form, unlearned — and not what it looks
like. Use source-balanced counting for archetypes; the retained learned component is `D_phi`
(`REPORT_DELIVERY_MODEL.md`), which predicts executed reach and beats its identity baseline by
35.4% RMSE under leave-one-motion-out over 28 clips.

## LFH contract and next research step

1. Generate a motion, then execute both the original and proposed adaptation in empty-scene physics.
2. Measure trajectory-relative feasible support `(u*, normal, finite extent, keypoint, xi)` only
   from accepted executed pairs.
3. Sample one binding obstacle plus keep-out-certified context from the designed distribution
   `q_LFH`; use DCS only as a novelty/reporting term.
4. Execute all cells in physics. Complete a family only on
   accepted/accepted/rejected/accepted with unique binding attribution.
5. Learn proposal density, never natural obstacle frequency or a physics-verdict surrogate — and
   only for quantities the executed pair actually constrains. Archetype identity is not one of
   them; executed reach is.

Next, do not add more overhead clutter. First make fresh motion→adaptation calibration reliable;
then obtain executed separation for a lateral or oblique operator. In parallel, finish the common
`hanging_panel` cross needed for the equal-budget learner comparison.

Machine-readable evidence: `docs/hallucination/q_lfh_v1.json`, `e7_transfer.json`, and
`e10_context_rich.json`. Total E7/E7c spend is {total_gpu:.3f} contended GPU-h; E10/E10b and all
generated families remain excluded from claim 5.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--q-lfh", type=Path, default=REPO_ROOT / "docs/hallucination/q_lfh_v1.json"
    )
    parser.add_argument(
        "--e7", type=Path, default=REPO_ROOT / "docs/hallucination/e7_transfer.json"
    )
    parser.add_argument(
        "--e7c", type=Path, default=REPO_ROOT / "docs/hallucination/e7c_replacement.json"
    )
    parser.add_argument(
        "--e7c-cpu",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e7c_replacement_cpu.json",
    )
    parser.add_argument(
        "--e7-cpu",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e7_transfer_cpu.json",
    )
    parser.add_argument(
        "--e9a-run",
        type=Path,
        default=Path(
            "/data/robotixx/groot-wbc-kimodo-m0/hallucination/run_records/"
            "E9A_MOTION_SCENE_CALIBRATION_2026-08-21.json"
        ),
    )
    parser.add_argument(
        "--e10",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e10_context_rich.json",
    )
    parser.add_argument(
        "--e10-cpu",
        type=Path,
        default=REPO_ROOT / "docs/hallucination/e10b_context_rich_cpu.json",
    )
    parser.add_argument(
        "--e10-run",
        type=Path,
        default=Path(
            "/data/robotixx/groot-wbc-kimodo-m0/hallucination/run_records/"
            "E10B_CONTEXT_RICH_SEED_CONTROL_2026-08-21.json"
        ),
    )
    parser.add_argument(
        "--wide-root",
        type=Path,
        default=Path("/data/robotixx/groot-wbc-kimodo-m0"),
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=REPO_ROOT / "docs/source/_static/lfh_progress",
    )
    parser.add_argument(
        "--page",
        type=Path,
        default=REPO_ROOT / "docs/source/research/lfh_progress.md",
    )
    args = parser.parse_args()

    q_lfh = json.loads(args.q_lfh.read_text())
    e7 = json.loads(args.e7.read_text())
    e7c = json.loads(args.e7c.read_text())
    e7c_cpu = json.loads(args.e7c_cpu.read_text())
    e7_cpu = json.loads(args.e7_cpu.read_text())
    e9a = json.loads(args.e9a_run.read_text())
    e10 = json.loads(args.e10.read_text())
    e10_cpu = json.loads(args.e10_cpu.read_text())
    e10_run = json.loads(args.e10_run.read_text())
    replacement = e7c["replacement"]
    hard_scene = REPO_ROOT / e10_cpu["scenes"]["hard"]
    stage = read_stage_geometry(hard_scene)
    by_role = {
        cell["cell_role"]: e10_run["cells"][cell["cell_id"]]["scientific"]
        for cell in json.loads(
            (
                REPO_ROOT
                / "docs/hallucination/manifests/E10B_CONTEXT_RICH_SEED_CONTROL_APPROVED.json"
            ).read_text()
        )["cells"]
    }
    nominal = load(Path(by_role["nominal_hard"]["artifacts"]["trajectory"]))
    adapted = load(Path(by_role["adapted_hard"]["artifacts"]["trajectory"]))

    args.assets.mkdir(parents=True, exist_ok=True)
    render_route_map(
        nominal,
        adapted,
        stage,
        e10["contacts"]["nominal_hard"],
        args.assets / "route-map-2d.png",
    )
    render_scene_3d(nominal, adapted, stage, args.assets / "scene-map-3d.png")
    render_direction_support(args.assets / "direction-support.png")
    render_distribution(q_lfh, e7, args.assets / "proposal-distribution.png")

    fresh_wide = [
        (
            args.wide_root / "lfh_3d_loop/wide_replay/nominal/wide/000000.mp4",
            "Fresh nominal · accepted",
        ),
        (
            args.wide_root / "lfh_3d_loop/wide_replay/adapted/wide/000000.mp4",
            "Crouch twin · tracking rejected",
        ),
    ]
    fresh_video = args.assets / "fresh-motion-wide.mp4"
    montage(fresh_wide, fresh_video, columns=2)
    video_poster(fresh_video, args.assets / "fresh-motion-wide-poster.png", time_s=1.6)
    e10_wide = [
        (
            args.wide_root / "hallucination/wide_replay/e10b/nominal_hard/wide/000000.mp4",
            "Nominal · binding strike",
        ),
        (
            args.wide_root / "hallucination/wide_replay/e10b/adapted_hard/wide/000000.mp4",
            "Adapted · clears all 5 obstacles",
        ),
    ]
    e10_video = args.assets / "e10b-isaac-wide.mp4"
    montage(e10_wide, e10_video, columns=2)
    video_poster(e10_video, args.assets / "e10b-isaac-wide-poster.png", time_s=2.4)

    replay_cells = []
    for role, label, status in (
        ("nominal_easy", "Nominal · easy", "CLEAR · accepted"),
        ("adapted_easy", "Adapted · easy", "CLEAR · accepted"),
        ("nominal_hard", "Nominal · hard", "STRIKE · rejected"),
        ("adapted_hard", "Adapted · hard", "CLEAR · accepted"),
    ):
        phase = role.split("_")[1]
        trajectory = (
            Path(replacement["videos"][role]).parents[1] / "trajectories/000000.trajectory.pkl"
        )
        replay_cells.append(
            {
                "payload": load(trajectory),
                "stage": read_stage_geometry(
                    REPO_ROOT / e7c_cpu["variants_detail"][0]["scenes"][phase]
                ),
                "title": label,
                "status": status,
                "contact_frame": replacement["contacts"][role]["first_frame"],
                "color": COLORS[role.split("_")[0]],
            }
        )
    e7c_replay = args.assets / "e7c-four-cell-replay.mp4"
    render_behavior_replay(replay_cells, e7c_replay, columns=2)
    video_poster(e7c_replay, args.assets / "e7c-four-cell-poster.png")
    chosen = [
        ("lfh_086_crouch", "ibeam"),
        ("lfh_089_crouch", "ibeam"),
        ("lfh_090_crouch", "ibeam"),
    ]
    variants = {(row["source_pair_id"], row["archetype_id"]): row for row in e7["variants"]}
    cpu_variants = {
        (row["source_pair_id"], row["archetype_id"]): row for row in e7_cpu["variants_detail"]
    }
    cross_cells = []
    for role, outcome in (("nominal_hard", "strikes"), ("adapted_hard", "clears")):
        for source, archetype in chosen:
            label = f"{source.removeprefix('lfh_').removesuffix('_crouch')} {role.split('_')[0]} · {outcome}"
            report_variant = variants[(source, archetype)]
            trajectory = (
                Path(report_variant["videos"][role]).parents[1]
                / "trajectories/000000.trajectory.pkl"
            )
            cross_cells.append(
                {
                    "payload": load(trajectory),
                    "stage": read_stage_geometry(
                        REPO_ROOT / cpu_variants[(source, archetype)]["scenes"]["hard"]
                    ),
                    "title": label,
                    "status": outcome.upper(),
                    "contact_frame": report_variant["contacts"][role]["first_frame"],
                    "color": COLORS[role.split("_")[0]],
                }
            )
    cross_replay = args.assets / "e7-cross-source-hard-replay.mp4"
    render_behavior_replay(cross_cells, cross_replay, columns=3)
    video_poster(cross_replay, args.assets / "e7-cross-source-hard-poster.png")

    args.page.parent.mkdir(parents=True, exist_ok=True)
    args.page.write_text(page(q_lfh, e7, e7c, e9a, e10))
    print(f"PASS: rendered LFH progress page and {len(list(args.assets.iterdir()))} assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
