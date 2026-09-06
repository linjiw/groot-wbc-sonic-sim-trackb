#!/usr/bin/env python3
"""Make a labelled body-origin replay of actual Isaac closed-loop recordings."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import numpy as np


def render(result_path, output):
    result = json.loads(result_path.read_text())
    ref = result["manifest"]
    manifest = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    beam = manifest["beam"]
    normal = np.array([np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"])])
    chains = [
        ["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link"],
        *[
            ["pelvis"]
            + [
                f"{side}_{part}_link"
                for part in (
                    "hip_pitch",
                    "hip_roll",
                    "hip_yaw",
                    "knee",
                    "ankle_pitch",
                    "ankle_roll",
                )
            ]
            for side in ("left", "right")
        ],
        *[
            ["torso_link"]
            + [
                f"{side}_{part}_link"
                for part in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
            for side in ("left", "right")
        ],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    traces = []
    source_refs = []
    for ax, mode in zip(axes, ("blind", "reactive")):
        row = next(
            r
            for r in result["rows"]
            if r["condition"] == "present" and r["mode"] == mode and r["seed"] == 8021
        )
        for key in ("trajectory", "sensor", "physics_contacts"):
            ref = row[key]
            checked(Path(ref["path"]), ref["sha256"])
            source_refs.append(ref)
        with Path(row["trajectory"]["path"]).open("rb") as stream:
            payload = pickle.load(stream)
        sensor = json.loads(Path(row["sensor"]["path"]).read_text())
        # The command update follows the recorded physics frame. Match the
        # preceding packet by frame order, never by a global clock comparison:
        # the final command packet may wrap to zero after the capture ends.
        observed_times = np.array([o["time_s"] for o in sensor["observations"]])
        recorded_times = np.asarray(payload["motion_time_s"]).reshape(-1)
        if len(observed_times) != len(recorded_times) or not np.allclose(
            observed_times[:-1], recorded_times[1:], atol=1e-6
        ):
            raise ValueError("command/trajectory clock alignment changed")
        with np.load(row["physics_contacts"]["path"]) as raw:
            force = np.linalg.norm(raw["force_w"], axis=-1).max(1).reshape(-1, 4).max(1)
        names = list(payload["body_names"])
        body = np.asarray(payload["body_pos_w"])
        horizontal = (body[:, :, :2] - beam["center_xy_m"]) @ normal
        heights = body[:, :, 2]
        indices = [[names.index(n) for n in chain] for chain in chains]
        artists = [ax.plot([], [], "o-", markersize=3, lw=2)[0] for _ in chains]
        annotation = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top", fontsize=10)
        ax.add_patch(Rectangle((-0.05, manifest["beam_underside_m"]), 0.1, 0.1, color="#a7753d"))
        ax.axhline(0, c="#777777", lw=1)
        ax.set(
            xlim=(-3, 2),
            ylim=(-0.08, 1.65),
            xlabel="Distance from beam along route (m)",
            ylabel="World height (m)",
        )
        ax.set_title(
            f"{'Blind walking' if mode == 'blind' else 'Sensed reference switching'} · seed 8021"
        )
        traces.append((horizontal, heights, indices, artists, annotation, payload, sensor, force))
    fig.suptitle(
        "Actual Isaac Lab execution replay · body origins/kinematic chains, not collision meshes"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(output), dpi=100):
        for i in np.linspace(0, 198, 100).round().astype(int):
            for horizontal, heights, indices, artists, annotation, payload, sensor, force in traces:
                time_s = float(np.asarray(payload["motion_time_s"])[i])
                state = sensor["observations"][i - 1] if i > 0 else {"active": 0, "occupied": False}
                for ids, artist in zip(indices, artists):
                    artist.set_data(horizontal[i, ids], heights[i, ids])
                    artist.set_color("#bd664b" if force[i] > 1 else "#2a8c87")
                annotation.set_text(
                    f"t = {time_s:.2f} s | reference: {'d040' if state['active'] else 'walk'}\n"
                    f"sensed occupancy: {bool(state['occupied'])} | substep peak: {force[i]:.1f} N"
                )
            writer.grab_frame()
    plt.close(fig)
    receipt = {
        "result": artifact(result_path),
        "sources": source_refs,
        "video": artifact(output),
        "scope": "one fixed seed; side projection of measured body origins; no extra physics run",
    }
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/motion2scene/assets/reactive-interface-replay.mp4",
    )
    args = parser.parse_args()
    render(args.result, args.output)
