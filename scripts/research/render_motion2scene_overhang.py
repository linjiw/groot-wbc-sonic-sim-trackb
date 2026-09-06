#!/usr/bin/env python3
"""Verify and publish the complete overhang/guard experiment and actual query rays."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_reactive_interface import physics_windows
from motion2scene_timing_diagnostic import ROOT, checked
import numpy as np
from render_motion2scene_icra_progress import checked_trace, export, save

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage


def render(path, out, name="overhang-interface-v1"):
    result, receipt = export(path, name, out)
    supplement, extra_receipt = export(
        Path(result["prior_orphan_reservation_supplement"]["path"]), "reactive-orphan-cleanup", out
    )
    receipt["resource_supplement"] = extra_receipt
    manifest = json.loads(Path(result["manifest"]["path"]).read_text())
    traces = {}
    for row in result["rows"]:
        for key in ("trajectory", "contacts", "inventory", "sensor", "physics_contacts"):
            ref = row[key]
            checked(Path(ref["path"]), ref["sha256"])
        with Path(row["trajectory"]["path"]).open("rb") as f:
            payload = pickle.load(f)
        beam = {
            **manifest["beam"],
            "underside_m": 2.0 if row["condition"] == "raised" else manifest["beam_underside_m"],
        }
        with np.load(row["contacts"]["path"]) as a, np.load(row["physics_contacts"]["path"]) as b:
            sampled = a["force_w"]
            checked_trace(
                {
                    **row,
                    **score_passage(payload, sampled, beam),
                    "condition": "present" if row["condition"] == "raised" else row["condition"],
                },
                beam,
            )
            blocks, aggregated, error = physics_windows(b, sampled)
            if blocks.shape != (199, 4, 30, 3) or error != row["contact_sync_max_error_n"]:
                raise ValueError("physics record mismatch")
            recomputed = score_passage(payload, aggregated, beam)
            if any(row[k] != v for k, v in recomputed.items()):
                raise ValueError("passage result changed")
            force = np.linalg.norm(blocks.reshape(-1, 30, 3), axis=-1).max(1)
        sensor = json.loads(Path(row["sensor"]["path"]).read_text())
        if sum(o["occupied"] for o in sensor["observations"]) != row["raw_overhang_frames"]:
            raise ValueError("sensor count changed")
        traces[row["cell_id"]] = (force, sensor)
    rows = result["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    reactive = [r for r in rows if r["mode"] == "reactive"]
    x = np.arange(len(reactive))
    axes[0, 0].bar(
        x - 0.18,
        [r["raw_upper_frames"] for r in reactive],
        0.36,
        label="Head-height hit",
        color="#d19742",
    )
    axes[0, 0].bar(
        x + 0.18,
        [r["raw_overhang_frames"] for r in reactive],
        0.36,
        label="Lower space clear",
        color="#2a8c87",
    )
    axes[0, 0].set_xticks(x, [f"{r['condition']}\n{r['seed']}" for r in reactive])
    axes[0, 0].set_ylabel("Positive frames / 199")
    axes[0, 0].set_title("Raw sensing over the full capture; independent of guard")
    axes[0, 0].legend(fontsize=8)
    colors = {
        "reactive": "#2a8c87",
        "oracle": "#8b6bb1",
        "blind": "#b85a45",
        "late_oracle": "#b18c27",
    }
    for r in rows:
        force, sensor = traces[r["cell_id"]]
        style = "-" if r["seed"] == 8031 else "--"
        if r["condition"] == "present":
            axes[0, 1].plot(
                (np.arange(len(force)) + 1) * 0.005,
                force,
                style,
                color=colors[r["mode"]],
                alpha=0.8,
                label=f"{r['mode']} / {r['seed']}",
            )
        obs = sensor["observations"]
        y = rows.index(r)
        allowed = [o for o in obs if o["transition"]["attempted"] and o["transition"]["allowed"]]
        denied = [o for o in obs if o["transition"]["attempted"] and not o["transition"]["allowed"]]
        axes[1, 0].scatter(
            [o["time_s"] for o in allowed],
            [y] * len(allowed),
            color="#2a8c87",
            marker="o",
            s=20,
        )
        axes[1, 0].scatter(
            [o["time_s"] for o in denied], [y] * len(denied), color="#b85a45", marker="x", s=12
        )
    axes[0, 1].axhline(1, color="black", ls="--", lw=0.8)
    axes[0, 1].set(
        yscale="symlog",
        ylabel="Peak per-body beam normal force (N)",
        xlabel="Physics time (s)",
        title="Critical beam · all four modes and both seeds",
    )
    axes[0, 1].set_ylim(bottom=0)
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].set_yticks(
        range(len(rows)), [f"{r['condition']}/{r['mode']} {r['seed']}" for r in rows], fontsize=7
    )
    axes[1, 0].set(
        xlim=(0, 4),
        xlabel="Command time (s)",
        title="Accepted switches (green) and denied requests (red)",
    )
    axes[1, 0].invert_yaxis()
    axes[1, 1].bar(
        np.arange(len(rows)),
        [r["maximum_beam_normal_force_n_through_passage"] for r in rows],
        color=[colors[r["mode"]] for r in rows],
    )
    axes[1, 1].axhline(1, color="black", ls="--", lw=0.8)
    axes[1, 1].set_yscale("symlog", linthresh=1)
    axes[1, 1].set_xticks(
        range(len(rows)),
        [f"{r['condition']}/{r['mode']}\n{r['seed']}" for r in rows],
        rotation=65,
        ha="right",
        fontsize=7,
    )
    axes[1, 1].set(
        ylabel="Peak through passage/stabilization (N)",
        title="All 14 requests; late refusal is not avoidance success",
    )
    fig.suptitle(
        "Isaac Lab · upper/lower sparse rays + legal reference switches · one carrier, two new seeds"
    )
    receipt["figures"] = save(fig, out, name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, condition in zip(axes, ("present", "absent")):
        row = next(
            r
            for r in rows
            if r["mode"] == "reactive" and r["condition"] == condition and r["seed"] == 8031
        )
        _, sensor = traces[row["cell_id"]]
        observation, ray = next(
            (o, r)
            for o in sensor["observations"]
            for r in o["rays"]
            if r["upper_candidate"] and r["overhang"] == (condition == "present")
        )
        origin = np.asarray(observation["origin"])
        position = np.asarray(ray["hit"]["position"])
        distance = np.linalg.norm(position[:2] - origin[:2])
        ax.plot([0, distance], [origin[2], position[2]], color="#d19742", label="Upper ray")
        ax.scatter([distance], [position[2]], color="#d19742", s=40)
        for i, lower in enumerate(ray["lower_rays"]):
            hit = lower["hit"]
            d = hit["distance"] if hit else lower["range_m"]
            z = lower["origin"][2]
            ax.plot(
                [0, d],
                [z, z],
                color="#b85a45" if hit else "#2a8c87",
                ls="-" if hit else "--",
                label=("Lower hit" if hit else "Lower clear") if i == 0 else None,
            )
            ax.scatter([d], [z], marker="x" if hit else "|", color="#b85a45" if hit else "#2a8c87")
        ax.set(
            xlim=(0, 3.1),
            ylim=(0.2, 1.6),
            xlabel="Horizontal query range (m)",
            ylabel="World height (m)",
            title=(
                f"{'Beam accepted' if condition=='present' else 'Wall rejected'} · "
                f"first candidate at {observation['time_s']:.2f} s"
            ),
        )
        ax.legend(fontsize=8)
    fig.suptitle("Actual recorded PhysX queries; sparse evidence, not a swept-volume certificate")
    receipt["figures"] += save(fig, out, name + "-rays")
    receipt["verification"] = (
        "14 raw captures hashed; imported flags/geometry and 200 Hz passage scores independently checked"
    )
    (out / "evidence" / (name + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "predictions": result["predictions"],
                "new_gpu_hours": result["new_contended_gpu_hours"],
                "prior_orphan_extra_hours": supplement[
                    "additional_reserved_gpu_hours_conservative"
                ],
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    p.add_argument("--name", default="overhang-interface-v1")
    a = p.parse_args()
    render(a.result, a.out, a.name)
