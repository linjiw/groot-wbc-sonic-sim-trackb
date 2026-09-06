#!/usr/bin/env python3
"""Publish both interface attempts and plots checked against their raw captures."""

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
from render_motion2scene_icra_progress import export, save

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage


def render(prior_path, current_path, out):
    out.joinpath("assets").mkdir(parents=True, exist_ok=True)
    out.joinpath("evidence").mkdir(parents=True, exist_ok=True)
    old, old_receipt = export(prior_path, "reactive-interface-v1", out)
    new, receipt = export(current_path, "reactive-interface-typed-v3", out)
    failure, failure_receipt = export(
        current_path.parent / "prior_failure.json", "reactive-interface-v2-failure", out
    )
    for ref in failure["logs"]:
        checked(Path(ref["path"]), ref["sha256"])
    manifest = json.loads(Path(new["manifest"]["path"]).read_text())
    traces = {}
    for result in (old, new):
        for row in result["rows"]:
            for key in ("trajectory", "contacts", "inventory", "sensor", "physics_contacts"):
                ref = row[key]
                checked(Path(ref["path"]), ref["sha256"])
            with Path(row["trajectory"]["path"]).open("rb") as stream:
                payload = pickle.load(stream)
            inventory = json.loads(Path(row["inventory"]["path"]).read_text())
            native = next(s for s in inventory["shapes"] if s["path"] == inventory["beam_path"])
            if native["attributes"]["physics:collisionEnabled"] != str(
                row["condition"] != "absent"
            ):
                raise ValueError("imported collision flag mismatch")
            if native["attributes"]["physics:kinematicEnabled"] != "True":
                raise ValueError("beam is not kinematic")
            matrix = np.asarray(native["local_to_world_at_capture_start"])
            yaw = manifest["beam"]["yaw_rad"]
            c, s = np.cos(yaw), np.sin(yaw)
            expected = np.diag([0.1, 1.2, 0.1]) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            height = 2.0 if row["condition"] == "raised" else manifest["beam_underside_m"]
            if not np.allclose(matrix[:3, :3], expected, atol=1e-6) or not np.allclose(
                matrix[3, :3], [*manifest["beam"]["center_xy_m"], height + 0.05], atol=1e-6
            ):
                raise ValueError("imported beam pose or dimensions mismatch")
            with np.load(row["contacts"]["path"]) as a, np.load(
                row["physics_contacts"]["path"]
            ) as b:
                if b["force_w"].shape != (796, 30, 3) or a["force_w"].shape != (199, 30, 3):
                    raise ValueError("incomplete capture")
                blocks, aggregated, error = physics_windows(b, a["force_w"])
                if error != row["contact_sync_max_error_n"]:
                    raise ValueError("contact synchronization result changed")
                replay = score_passage(payload, aggregated, manifest["beam"])
                if any(replay[k] != row[k] for k in replay):
                    raise ValueError("passage result changed")
                forces = np.linalg.norm(blocks.reshape(-1, 30, 3), axis=-1).max(1)
            sensor = json.loads(Path(row["sensor"]["path"]).read_text())
            traces[row["cell_id"]] = (payload, forces, sensor)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    rows = new["rows"]
    labels = [f"{r['condition']} / {r['mode']}\n{r['seed']}" for r in rows]
    axes[0, 0].bar(
        np.arange(len(rows)),
        [r["maximum_beam_normal_force_n_through_passage"] for r in rows],
        color=["#2a8c87" if r["pass"] else "#bd664b" for r in rows],
    )
    axes[0, 0].axhline(1, c="black", ls="--", lw=1)
    axes[0, 0].set_yscale("symlog", linthresh=1)
    axes[0, 0].set_xticks(np.arange(len(rows)), labels, rotation=65, ha="right", fontsize=7)
    axes[0, 0].set_ylabel("200 Hz peak body–beam normal force (N)")
    axes[0, 0].set_title("All 12 requests · force through first passage/stabilization")
    colors = {"reactive": "#2a8c87", "oracle": "#9466b4", "blind": "#bd664b"}
    for row in rows:
        payload, forces, sensor = traces[row["cell_id"]]
        style = "-" if row["seed"] == 8021 else "--"
        if row["condition"] == "present":
            label = f"{row['mode']} / {row['seed']}"
            axes[0, 1].plot(
                (np.arange(len(forces)) + 1) * 0.005,
                forces,
                style,
                color=colors[row["mode"]],
                label=label,
                alpha=0.8,
            )
            axes[1, 1].plot(
                np.asarray(payload["motion_time_s"]).reshape(-1),
                np.asarray(payload["root_pos_w"])[:, 2],
                style,
                color=colors[row["mode"]],
                label=label,
                alpha=0.8,
            )
        if row["mode"] == "reactive":
            obs = sensor["observations"]
            offset = 0 if row["condition"] == "absent" else 1 if row["condition"] == "raised" else 2
            offset += 0.35 * (row["seed"] == 8022)
            t = [o["time_s"] for o in obs]
            axes[1, 0].plot(
                t,
                [offset + 0.22 * o["active"] for o in obs],
                style,
                label=f"{row['condition']} / {row['seed']}",
            )
            occupied = [o for o in obs if o["occupied"]]
            axes[1, 0].scatter(
                [o["time_s"] for o in occupied], [offset + 0.27] * len(occupied), s=2, color="black"
            )
    axes[0, 1].set_yscale("symlog", linthresh=1)
    axes[0, 1].axhline(1, c="black", ls="--", lw=1)
    axes[0, 1].set_title("Critical beam · actual 200 Hz forces, full capture")
    axes[0, 1].set_ylabel("Maximum per-body normal force (N)")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].set_title("Reactive command (+0.22 = d040); black dots = sensed occupancy")
    axes[1, 0].set_yticks([0, 1, 2], ["absent", "raised", "critical"])
    axes[1, 0].legend(fontsize=7, ncol=2, loc="center left")
    axes[1, 1].set_title("Critical beam · achieved pelvis height")
    axes[1, 1].set_ylabel("World height (m)")
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Time (s)")
    fig.suptitle(
        "Isaac Lab + frozen SONIC · scene queries enabled · one development carrier, two seeds"
    )
    receipt["figures"] = save(fig, out, "reactive-interface-typed-v3")
    receipt["prior_attempt"] = old_receipt
    receipt["interrupted_query_attempt"] = failure_receipt
    receipt["verification"] = (
        "All 24 raw trajectories, sensor and contact artifacts hashed; passage rescored."
    )
    target = out / "evidence/reactive-interface-receipt.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return new, old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    render(args.prior, args.current, args.out)


if __name__ == "__main__":
    main()
