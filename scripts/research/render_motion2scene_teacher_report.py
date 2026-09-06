#!/usr/bin/env python3
"""Publish portable repeatability and analytic beam-teacher diagnostic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from render_motion2scene_timing_report import portable, sha

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def verified_json(path):
    value = json.loads(path.read_text())
    for key in (
        "manifest",
        "run_record",
        "driver",
        "geometry_implementation",
        "capsule_model_implementation",
    ):
        if key in value:
            item = value[key]
            if sha(Path(item["path"])) != item["sha256"]:
                raise ValueError(f"{key} hash mismatch")
    for item in value.get("sources", {}).values():
        if sha(Path(item["path"])) != item["sha256"]:
            raise ValueError("source hash mismatch")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeatability", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--middle-audit", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    repeat, teacher = verified_json(args.repeatability), verified_json(args.teacher)
    sources = {"repeatability": args.repeatability, "beam-teacher": args.teacher}
    if args.extension:
        verified_json(args.extension)
        sources["ladder-extension"] = args.extension
    if args.middle_audit:
        verified_json(args.middle_audit)
        sources["intermediate-beam-audit"] = args.middle_audit
    assets, evidence = args.out / "assets", args.out / "evidence"
    assets.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, path in list(sources.items()):
        value = json.loads(path.read_text())
        for key in ("manifest", "run_record"):
            if key in value:
                sources[f"{name}-{key}"] = Path(value[key]["path"])
    for name, path in sources.items():
        target = evidence / f"{name}.json"
        target.write_text(json.dumps(portable(json.loads(path.read_text())), indent=2) + "\n")
        outputs.append(target)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-teacher-v1",
            "font.size": 11,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    seeds = [7901, 7902, 7903]
    rows = {(r["runtime_seed"], r["label"]): r for r in repeat["rows"]}
    for label, color in (("neutral", "#567ba3"), ("d055", "#187f72")):
        axes[0].plot(
            seeds,
            [100 * rows[seed, label]["diagnostics"]["endpoint_error_m"] for seed in seeds],
            marker="o",
            color=color,
            label=label,
        )
    axes[0].axhline(35, color="#bd752e", linestyle="--", label="Frozen tracker limit")
    axes[0].set(
        title="Tracking repeats across three physics seeds",
        ylabel="Tracker endpoint error (cm)",
        ylim=(0, 40),
    )
    axes[0].legend()
    retained = [
        100
        * rows[seed, "d055"]["achieved_semantic"]["peak_reduction"]
        / rows[seed, "d055"]["reference_semantic"]["peak_reduction"]
        for seed in seeds
    ]
    axes[1].bar(seeds, retained, color="#187f72", width=0.55)
    axes[1].axhline(60, color="#bd752e", linestyle="--", label="Frozen magnitude-retention limit")
    axes[1].set(
        title="Paired crouch magnitude retained",
        ylabel="Achieved / reference peak reduction (%)",
        ylim=(0, 100),
    )
    axes[1].legend(loc="upper right")
    for axis in axes:
        axis.set_xticks(seeds)
        axis.set_xlabel("Physics seed (repeated measurements of one carrier)")
        axis.grid(axis="y", alpha=0.2)
    figure.supxlabel(
        "All six tracker and route checks passed; all three crouches passed "
        "the localized event checks. No Q4 admission.",
        fontsize=10,
    )
    path = assets / "repeatability.svg"
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for length, color in ((0.1, "#187f72"), (0.2, "#567ba3"), (0.3, "#bd752e")):
        candidates = [
            c
            for c in teacher["candidates"]
            if c["length_m"] == length and "width_m" in c["interval"]
        ]
        axes[0].plot(
            [c["route_progress"] for c in candidates],
            [1000 * c["interval"]["width_m"] for c in candidates],
            marker="o",
            color=color,
            label=f"{int(100 * length)} cm beam depth",
        )
    axes[0].axhline(0, color="#53625f", linewidth=1)
    axes[0].set(
        title="Common roof intervals at all 27 placements",
        xlabel="Reference route progress at beam center",
        ylabel="Usable beam-height interval (mm)",
    )
    axes[0].legend(loc="lower center", fontsize=9)
    quantiles = [p["quantile"] for p in teacher["proposals"]]
    if quantiles:
        target = [
            1000 * min(r["target_min_clearance_m"] for r in p["jitter_audit"]["rows"])
            for p in teacher["proposals"]
        ]
        strike = [
            -1000 * max(r["weaker_worst_clearance_m"] for r in p["jitter_audit"]["rows"])
            for p in teacher["proposals"]
        ]
        axes[1].plot(quantiles, target, "o-", color="#187f72", label="Worst crouch clearance")
        axes[1].plot(
            quantiles, strike, "o-", color="#567ba3", label="Weakest walk overlap criterion"
        )
        axes[1].axhline(10, color="#bd752e", linestyle="--", label="Required margin")
        axes[1].set_xticks(
            quantiles, [f'{p["beam_underside_m"]:.4f}' for p in teacher["proposals"]]
        )
    axes[1].set(
        title="Finite beam: all eight sources × 81 jitter points",
        xlabel="Beam underside height (m)",
        ylabel="Geometric margin criterion (mm)",
    )
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.supxlabel(
        "Capsule geometry at recorded frames; discrete jitter checks. "
        "Overlap criterion is not penetration depth or a physics verdict.",
        fontsize=10,
    )
    path = assets / "beam-teacher.svg"
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    outputs.append(path)

    # Capsule side view of the actual nominal/target poses closest to the chosen beam.
    import pickle

    from motion2scene_beam_teacher import capsules
    from motion2scene_timing_diagnostic import checked

    from gear_sonic.dataset_generation.capsule_box_exact import capsule_box_clearance
    from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

    candidate = teacher["selected_candidate"]
    if candidate:
        proposal = next(p for p in teacher["proposals"] if p["quantile"] == 0.5)
        height = proposal["beam_underside_m"]
        figure, axes = plt.subplots(1, 2, figsize=(11, 5.4), layout="constrained")
        for axis, label in zip(axes, ("neutral", "d055")):
            row = rows[7902, label]
            path = checked(Path(row["trajectory"]["path"]), row["trajectory"]["sha256"])
            with path.open("rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
            state = capsules(payload)
            yaw = candidate["yaw_rad"]
            rotation = np.array(
                [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
            )
            origin = np.array([*candidate["center_xy_m"], 0.0])
            starts, ends = (state["starts"] - origin) @ rotation, (
                state["ends"] - origin
            ) @ rotation
            length, width = candidate["length_m"], candidate["width_m"]
            distance = capsule_box_clearance(
                starts,
                ends,
                state["radii"],
                [-length / 2, -width / 2, height],
                [length / 2, width / 2, height + 0.1],
            )
            frame = int(np.argmin(distance.min(axis=1)))
            for start, end, radius in zip(starts[frame], ends[frame], state["radii"]):
                for t in np.linspace(0, 1, 9):
                    center = start + t * (end - start)
                    axis.add_patch(
                        plt.Circle(
                            (center[0], center[2]),
                            radius,
                            color="#567ba3" if label == "neutral" else "#187f72",
                            alpha=0.22,
                            linewidth=0,
                        )
                    )
            axis.add_patch(
                plt.Rectangle((-length / 2, height), length, 0.1, facecolor="#bd752e", alpha=0.8)
            )
            axis.set(
                title=f"{label} · closest sampled pose at {frame / float(payload['fps']):.2f} s",
                xlabel="Along-beam local route axis (m)",
                ylabel="World height (m)",
                xlim=(-0.6, 0.6),
                ylim=(-0.05, 1.5),
                aspect="equal",
            )
            axis.grid(alpha=0.2)
        figure.supxlabel(
            "Recorded empty-scene poses with a proposed beam overlay · seed 7902 · "
            "panels show different elapsed times\n"
            "2D capsule projection for illustration; clearance queries use all 3D capsules. "
            "No obstacle-present simulation.",
            fontsize=10,
        )
        path = assets / "beam-teacher-poses.svg"
        figure.savefig(path, metadata={"Date": None})
        plt.close(figure)
        outputs.append(path)
    if args.extension:
        extension = json.loads(args.extension.read_text())
        rows = sorted(extension["rows"], key=lambda row: row["runtime_seed"])
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
        peaks = [1000 * row["achieved_semantic"]["peak_reduction"] for row in rows]
        axes[0].bar(
            seeds,
            peaks,
            width=0.55,
            color=["#187f72" if row["paired_behavior_retained"] else "#bd752e" for row in rows],
        )
        axes[0].axhline(50, color="#53625f", linestyle="--", label="Frozen minimum effect")
        axes[0].set(
            title="Intermediate crouch: 1/3 retains the required effect",
            ylabel="Peak whole-body lowering (mm)",
            ylim=(0, 70),
        )
        for index, label in enumerate(("Neutral to d040", "d040 to d055")):
            axes[1].plot(
                seeds,
                [1000 * row["minimum_adjacent_gap_m"][index] for row in rows],
                marker="o",
                label=label,
            )
        axes[1].axhline(0, color="#53625f", linewidth=1)
        axes[1].set(
            title="Central height ordering passes every seed",
            ylabel="Minimum adjacent central height gap (mm)",
            ylim=(0, 25),
        )
        for axis in axes:
            axis.set_xticks(seeds)
            axis.set_xlabel("Physics seed")
            axis.grid(axis="y", alpha=0.2)
            axis.legend(fontsize=9)
        figure.supxlabel(
            "All three intermediate tracker and route checks pass. "
            "Ordered profiles alone do not establish behavior retention or fixed-world clearance.",
            fontsize=10,
        )
        path = assets / "ladder-extension.svg"
        figure.savefig(path, metadata={"Date": None})
        plt.close(figure)
        outputs.append(path)
    scene_manifest = args.teacher.parent / "scene_manifest.json"
    if scene_manifest.exists():
        scene = json.loads(scene_manifest.read_text())
        path = Path(scene["scene"]["path"])
        if (
            sha(path) != scene["scene"]["sha256"]
            or sha(args.teacher) != scene["teacher_result"]["sha256"]
        ):
            raise ValueError("diagnostic scene identity mismatch")
        target = assets / "diagnostic-beam.usda"
        target.write_bytes(path.read_bytes())
        outputs.append(target)
        sources["visual_scene"] = path
        sources["visual_scene_manifest"] = scene_manifest
    # Normalize SVG whitespace before hashing, matching the existing report convention.
    for path in outputs:
        if path.suffix == ".svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    manifest = {
        "schema_version": "motion2scene_teacher_public_export_v1",
        "sources": {
            name: {"path": portable(str(path)), "sha256": sha(path)}
            for name, path in sources.items()
        },
        "outputs": [
            {"path": str(path.relative_to(args.out)), "sha256": sha(path)} for path in outputs
        ],
        "renderer_sha256": sha(Path(__file__)),
        "training_eligible": False,
    }
    (assets / "teacher-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {len(outputs)} artifacts")


if __name__ == "__main__":
    main()
