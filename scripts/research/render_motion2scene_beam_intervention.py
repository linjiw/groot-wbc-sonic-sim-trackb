#!/usr/bin/env python3
"""Render completed beam results without changing scoring or selecting rollouts."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    results = {}
    records = {}
    inventories = []
    contact_witnesses = []
    traces = {}
    for batch in ("controls", "paired"):
        result = json.loads((out / f"{batch}_result.json").read_text())
        results[batch] = result
        for key in ("manifest", "run_record"):
            ref = result[key]
            checked(Path(ref["path"]), ref["sha256"])
        records[batch] = json.loads(Path(result["run_record"]["path"]).read_text())
        for row in result["rows"]:
            refs = records[batch]["cells"][row["cell_id"]]["scientific"]["artifacts"]
            with checked(Path(refs["trajectory"]), refs["trajectory_sha256"]).open("rb") as f:
                payload = pickle.load(f)
            ref = row["contacts"]
            with np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False) as f:
                force = f["force_w"].copy()
                filters = list(f["filter_paths"])
            ref = row["inventory"]
            inventory = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
            assert filters == inventory["filter_paths"]
            assert len(force) == row["capture_frames"]
            beam = next(r for r in inventory["shapes"] if r["path"] == inventory["beam_path"])
            assert beam["attributes"]["physics:collisionEnabled"] == str(
                row["condition"] != "absent"
            )
            assert beam["attributes"]["physics:kinematicEnabled"] == "True"
            assert len(filters) == force.shape[1] == 30
            assert set(filters) <= {r["path"] for r in inventory["shapes"] if r["rigid_body"]}
            assert np.isclose(inventory["physics_dt"], 0.005)
            assert np.isclose(inventory["control_dt"], 0.02)
            expected_height = {"clear": 2.0, "intersecting": 1.1}.get(
                row["condition"], 1.2652671813964844
            )
            matrix = np.asarray(beam["local_to_world_at_capture_start"])
            assert np.allclose(
                matrix[3, :3], [2.696616322673983, 0.22645592215640215, expected_height + 0.05]
            )
            assert np.allclose(np.linalg.norm(matrix[:3, :3], axis=1), [0.1, 1.2, 0.1])
            robot_shapes = [
                r for r in inventory["shapes"] if r["path"].startswith("/World/envs/env_0/Robot/")
            ]
            inventories.append(
                {
                    "cell_id": row["cell_id"],
                    "beam": beam,
                    "filter_count": len(filters),
                    "physics_dt": inventory["physics_dt"],
                    "control_dt": inventory["control_dt"],
                    "robot_collision_declarations": sum(r["collision"] for r in robot_shapes),
                    "robot_geometry_types": {
                        t: sum(r["type"] == t for r in robot_shapes)
                        for t in ("Capsule", "Sphere", "Mesh")
                    },
                    "inventory": ref,
                    "scope": "running composed USD inventory; not a cooked-PhysX shape dump",
                }
            )
            end = row["first_episode_frames"]
            horizon = row["passage_finish_frame_exclusive"] or end
            norms = np.linalg.norm(force[:horizon], axis=-1)
            frame, body = np.unravel_index(np.argmax(norms), norms.shape)
            contact_witnesses.append(
                {
                    "cell_id": row["cell_id"],
                    "peak_force_n_through_passage": float(norms[frame, body]),
                    "peak_frame": int(frame),
                    "peak_filter": filters[body] if norms[frame, body] > 0 else None,
                    "passage_pass_with_full_sequence_tracker_rejection": row["pass"]
                    and row["tracker_outcome"] != "accepted",
                }
            )
            beam_spec = json.loads(Path(result["manifest"]["path"]).read_text())["beam"]
            normal = np.array([np.cos(beam_spec["yaw_rad"]), np.sin(beam_spec["yaw_rad"])])
            downstream = (
                (np.asarray(payload["body_pos_w"])[:end, :, :2] - beam_spec["center_xy_m"]) @ normal
            ).min(1)
            traces[row["cell_id"]] = (
                np.arange(end) / float(payload["fps"]),
                np.linalg.norm(force[:end], axis=-1).max(1),
                downstream - beam_spec["length_m"] / 2 - 0.1,
            )
    summary = {
        "results": {b: artifact(out / f"{b}_result.json") for b in results},
        "cells": sum(len(r["rows"]) for r in results.values()),
        "contended_gpu_hours": sum(
            r["budget"]["actual_contended_gpu_hours"] for r in records.values()
        ),
        "elapsed_seconds_by_cell": {
            cell_id: cell["elapsed_seconds"]
            for r in records.values()
            for cell_id, cell in r["cells"].items()
        },
        "runtime_inventory_checks": inventories,
        "contact_witnesses": contact_witnesses,
        "limitations": [
            "One selected source and one frozen development beam",
            "50 Hz force samples and body-origin crossing",
            "No learned-generator execution-yield claim",
        ],
    }
    summary_path = out / "execution_summary.json"
    if summary_path.exists():
        if json.loads(summary_path.read_text()) != summary:
            raise ValueError("existing execution summary differs; preserve it and investigate")
    else:
        write_new(summary_path, summary)
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 6), sharex=True, sharey="row", constrained_layout=True
    )
    for col, (label, condition) in enumerate(
        (("neutral", "absent"), ("d055", "absent"), ("neutral", "present"), ("d055", "present"))
    ):
        rows = [
            r
            for r in results["paired"]["rows"]
            if r["label"] == label and r["condition"] == condition
        ]
        for row in rows:
            t, force, downstream = traces[row["cell_id"]]
            axes[0, col].plot(t, downstream, label=str(row["seed"]))
            axes[1, col].plot(t, force)
        axes[0, col].axhline(0, color="black", ls="--", lw=0.8)
        axes[1, col].axhline(1, color="black", ls="--", lw=0.8)
        axes[1, col].set_yscale("symlog", linthresh=1)
        axes[0, col].set_title(
            f"{'Upright' if label == 'neutral' else 'Crouch'} / beam {condition}\n"
            f"Pass {sum(r['pass'] for r in rows)}/3"
        )
        axes[1, col].set_xlabel("First-episode elapsed time (s)")
        axes[0, col].grid(alpha=0.2)
        axes[1, col].grid(alpha=0.2)
    axes[0, 0].set_ylabel("Crossing threshold margin (m)")
    axes[1, 0].set_ylabel("Sampled beam normal force (N)")
    axes[0, 0].legend(title="Physics seed")
    fig.suptitle("41002 development intervention · fixed controller and beam · all first episodes")
    for suffix in ("svg", "png"):
        fig.savefig(out / f"beam-intervention.{suffix}", dpi=160)
    plt.close(fig)
    print(
        json.dumps({k: v for k, v in summary.items() if k != "runtime_inventory_checks"}, indent=2)
    )


if __name__ == "__main__":
    main()
