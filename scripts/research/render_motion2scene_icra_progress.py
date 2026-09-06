#!/usr/bin/env python3
"""Render hash-checked temporal/source/intermediate evidence for the ICRA roadmap."""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import numpy as np

from gear_sonic.dataset_generation.hallucination.motion2scene_passage import score_passage


def export(result_path, name, out):
    result = json.loads(result_path.read_text())
    for key in (
        "registration",
        "manifest",
        "run_record",
        "qualification",
        "proposals",
        "prior_result",
    ):
        if key in result:
            ref = result[key]
            checked(Path(ref["path"]), ref["sha256"])
    content = json.dumps(result, indent=2)
    content = content.replace(str(ROOT.parent / "research-data/groot-wbc"), "research-data")
    content = content.replace(str(ROOT), "repository")
    target = out / "evidence" / f"{name}.json"
    target.write_text(content + "\n")
    return result, {"original": artifact(result_path), "public": artifact(target)}


def save(fig, out, name):
    refs = []
    for suffix in ("png", "svg"):
        target = out / "assets" / f"{name}.{suffix}"
        fig.savefig(target, dpi=150)
        refs.append(artifact(target))
    plt.close(fig)
    return refs


def temporal(path, out):
    result, receipt = export(path, "temporal-native-audit", out)
    for row in result["rows"]:
        checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    rows = [
        r
        for r in result["rows"]
        if r["geometry"] == "native_outer" and r["hz"] == 200 and r["label"] != "neutral"
    ]
    x = np.arange(len(rows))
    axes[0].bar(
        x - 0.18,
        [1000 * r["sampled_minimum_m"] for r in rows],
        0.36,
        label="200 Hz sampled minimum",
        color="#2a8c87",
    )
    axes[0].bar(
        x + 0.18,
        [1000 * r["interval_minimum_lower_bound_m"] for r in rows],
        0.36,
        label="Interval lower bound",
        color="#d19035",
    )
    axes[0].axhline(10, color="black", ls="--", lw=1, label="10 mm target requirement")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_xticks(x, [r["label"] + "\n" + r["cell_id"].split("_")[3] for r in rows])
    axes[0].set_ylabel("Worst over all 81 beam offsets (mm)")
    axes[0].set_title("Native outer geometry · empty-scene recordings")
    axes[0].legend(fontsize=8)
    for i, hz in enumerate((50, 200)):
        values = [
            sum(
                r["query_seconds"]
                for r in result["rows"]
                if r["hz"] == hz and r["geometry"] == kind
            )
            for kind in ("proxy", "native_inner", "native_outer")
        ]
        axes[1].bar(np.arange(3) + (i - 0.5) * 0.36, values, 0.36, label=f"{hz} Hz")
    axes[1].set_xticks(
        np.arange(3), ["Proxy\n29 shapes", "Native subset\n27 shapes", "Native outer\n45 shapes"]
    )
    axes[1].set_ylabel("Query + interval-bound time, 9 × 81 queries (s)")
    axes[1].set_title("Same-machine panels · no GPU queries")
    axes[1].legend()
    fig.suptitle("Declared rigid-body interpolation; no certificate of unrecorded physics")
    receipt["figures"] = save(fig, out, "temporal-native-audit")
    return receipt


def checked_trace(row, beam):
    with checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"]).open("rb") as f:
        payload = pickle.load(f)
    with np.load(
        checked(Path(row["contacts"]["path"]), row["contacts"]["sha256"]), allow_pickle=False
    ) as data:
        forces = data["force_w"].copy()
        filters = list(data["filter_paths"])
        if float(data["fps"]) != float(payload["fps"]):
            raise ValueError("cadence mismatch")
    inventory = json.loads(
        checked(Path(row["inventory"]["path"]), row["inventory"]["sha256"]).read_text()
    )
    expected_filters = [f"/World/envs/env_0/Robot/{name}" for name in payload["body_names"]]
    if filters != inventory["filter_paths"] or filters != expected_filters or len(filters) != 30:
        raise ValueError("body/force filter mismatch")
    if not np.isclose(inventory["physics_dt"], 0.005) or not np.isclose(
        inventory["control_dt"], 0.02
    ):
        raise ValueError("runtime cadence changed")
    native = next(r for r in inventory["shapes"] if r["path"] == inventory["beam_path"])
    attributes = native["attributes"]
    if (
        attributes["physics:collisionEnabled"] != str(row["condition"] == "present")
        or attributes["physics:kinematicEnabled"] != "True"
    ):
        raise ValueError("unexpected collision/kinematic flag")
    matrix = np.asarray(native["local_to_world_at_capture_start"])
    c, s = np.cos(beam["yaw_rad"]), np.sin(beam["yaw_rad"])
    expected_rotation = np.diag([0.1, 1.2, 0.1]) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    if not np.allclose(matrix[:3, :3], expected_rotation, atol=1e-6):
        raise ValueError("beam rotation or dimensions changed")
    if not np.allclose(matrix[3, :3], [*beam["center_xy_m"], beam["underside_m"] + 0.05]):
        raise ValueError("beam translation changed")
    scores = score_passage(payload, forces, beam)
    for key, value in scores.items():
        if row[key] != value:
            raise ValueError(f"registered score differs: {row['cell_id']} {key}")
    end = row["first_episode_frames"]
    t = np.arange(end) / float(payload["fps"])
    normal = np.array([c, s])
    progress = ((payload["body_pos_w"][:end, :, :2] - beam["center_xy_m"]) @ normal).min(1) - 0.15
    return t, np.linalg.norm(forces[:end], axis=-1).max(1), progress


def source(path, out):
    result, receipt = export(path, "source-execution", out)
    qualification = json.loads(Path(result["qualification"]["path"]).read_text())
    proposals = json.loads(Path(result["proposals"]["path"]).read_text())
    receipt["qualification_export"] = export(
        Path(result["qualification"]["path"]), "source-execution-qualification", out
    )[1]
    receipt["proposal_export"] = export(
        Path(result["proposals"]["path"]), "source-execution-proposals", out
    )[1]
    proposals = {r["source"]: r for r in proposals["rows"]}
    # Absent runtime beam is inherited from the 41002 sensor-control scene.
    absent_beam = {
        "center_xy_m": [2.696616322673983, 0.22645592215640215],
        "yaw_rad": 0.07179232999993775,
        "underside_m": 1.2652671813964844,
        "length_m": 0.1,
    }
    for row in qualification["rows"]:
        # The registered absent score uses each source's virtual qualification plane.
        virtual = score_for_plane(row, proposals[row["source"]]["canonical_beam"])
        if any(row[key] != value for key, value in virtual.items()):
            raise ValueError("registered source-qualification passage score changed")
        physical = {**row, **score_for_plane(row, absent_beam)}
        checked_trace(physical, absent_beam)
    sources = sorted(proposals)
    fig, axes = plt.subplots(
        2, 8, figsize=(18, 5), sharex=True, sharey="row", constrained_layout=True
    )
    for col, source_id in enumerate(sources):
        subset = [r for r in result["rows"] if r["source"] == source_id]
        axes[0, col].set_title(
            f"{source_id}\n{result['separated_slots_per_source'][str(source_id)]}/3 slots",
            fontsize=10,
        )
        if not subset:
            axes[0, col].text(
                0.5, 0.5, "Source\nrefused", ha="center", transform=axes[0, col].transAxes
            )
        for row in subset:
            beam = proposals[source_id]["slots"][row["slot"]]["beam"]
            t, force, progress = checked_trace(row, beam)
            color = "#bb4c3e" if row["label"] == "neutral" else "#167f86"
            axes[0, col].plot(t, progress, color=color, alpha=0.8, lw=1)
            axes[1, col].plot(t, force, color=color, alpha=0.8, lw=1)
        axes[0, col].axhline(0, color="black", ls="--", lw=0.7)
        axes[1, col].axhline(1, color="black", ls="--", lw=0.7)
        axes[1, col].set_yscale("symlog", linthresh=1)
        axes[1, col].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Crossing margin (m)")
    axes[1, 0].set_ylabel("Sampled beam force (N)")
    fig.suptitle(
        "Frozen learned beams · all requested source groups · upright red / d055 teal · all first episodes"
    )
    receipt["figures"] = save(fig, out, "source-execution")
    receipt["runtime_inventories_checked"] = len(qualification["rows"]) + len(result["rows"])
    return receipt


def score_for_plane(row, beam):
    with checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"]).open("rb") as f:
        payload = pickle.load(f)
    with np.load(
        checked(Path(row["contacts"]["path"]), row["contacts"]["sha256"]), allow_pickle=False
    ) as data:
        return score_passage(payload, data["force_w"], beam)


def middle(path, out):
    result, receipt = export(path, "d040-beam-execution", out)
    manifest = json.loads(Path(result["manifest"]["path"]).read_text())
    beam = {**manifest["beam"], "underside_m": manifest["beam_underside_m"]}
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 6), sharex=True, sharey="row", constrained_layout=True
    )
    for col, label in enumerate(("neutral", "d040", "d055")):
        for row in result["rows"]:
            if row["label"] != label:
                continue
            t, force, progress = checked_trace(row, beam)
            color = f"C{row['seed'] - 7911}"
            style = "-" if row["condition"] == "present" else ":"
            axes[0, col].plot(
                t, progress, style, color=color, label=f"{row['seed']} {row['condition']}"
            )
            axes[1, col].plot(t, force, style, color=color)
        count = result["pass_counts_out_of_three"][f"{label}_present"]
        axes[0, col].set_title(f"{label} · beam-present contact-free {count}/3")
        axes[0, col].axhline(0, color="black", ls="--", lw=0.7)
        axes[1, col].axhline(1, color="black", ls="--", lw=0.7)
        axes[1, col].set_yscale("symlog", linthresh=1)
        axes[1, col].set_xlabel("First-episode time (s)")
    axes[0, 0].set_ylabel("Crossing margin (m)")
    axes[1, 0].set_ylabel("Sampled beam force (N)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("41002 three-motion development panel · all 18 cells · six new d040 executions")
    receipt["figures"] = save(fig, out, "d040-beam-execution")
    return receipt


def local_pose(path, out):
    result, receipt = export(path, "fresh-local-pose-certificate", out)
    audit, audit_receipt = export(path.with_name("cap_audit.json"), "local-pose-cap-audit", out)
    ref = audit["original_result"]
    checked(Path(ref["path"]), ref["sha256"])
    for row in result["rows"]:
        checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
    receipt["cap_audit"] = audit_receipt
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temporal", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--middle", type=Path)
    parser.add_argument("--local-pose", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    for name in ("evidence", "assets"):
        (args.out / name).mkdir(parents=True, exist_ok=True)
    for name, function in (
        ("temporal", temporal),
        ("source", source),
        ("middle", middle),
        ("local_pose", local_pose),
    ):
        path = getattr(args, name)
        if path:
            receipt = function(path, args.out)
            receipt["renderer"] = artifact(Path(__file__))
            (args.out / "assets" / f"{name}-progress-receipt.json").write_text(
                json.dumps(receipt, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
