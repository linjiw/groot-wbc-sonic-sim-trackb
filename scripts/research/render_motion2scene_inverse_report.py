#!/usr/bin/env python3
"""Export a portable evidence bundle and figures for the inverse-learning diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_timing_diagnostic import artifact, checked, write_new
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ORDER = ["full", "no_kl", "clearance_only", "per_motion", "random_prior", "grid_oracle"]
LABELS = [
    "Encoder\npreference + KL",
    "Encoder\npreference, no KL",
    "Encoder\nclearance only",
    "Direct mixture\noptimization",
    "Random\nprior",
    "Grid sampler\n+2,100 queries",
]
COLORS = ["#187f72", "#567ba3", "#c77f45", "#766994", "#999184", "#46515c"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    ref = result["manifest"]
    manifest_path = checked(Path(ref["path"]), ref["sha256"])
    manifest = json.loads(manifest_path.read_text())
    for ref in [
        manifest["bank"],
        manifest["protocol"],
        manifest["geometry_inventory"],
        *manifest["implementations"],
        *manifest["sources"],
    ]:
        checked(Path(ref["path"]), ref["sha256"])
    for row in result["rows"]:
        for key in ("artifact", "checkpoint", "samples_artifact"):
            if key in row:
                checked(Path(row[key]["path"]), row[key]["sha256"])
    evidence, assets = args.out / "evidence", args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    sources = {
        "inverse-learning": args.result,
        "inverse-learning-manifest": manifest_path,
        "inverse-geometry-inventory": Path(manifest["geometry_inventory"]["path"]),
    }
    outputs = []
    for name, path in sources.items():
        out = evidence / f"{name}.json"
        write_new(out, portable(json.loads(path.read_text())))
        outputs.append(out)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-inverse-v1",
            "font.size": 10,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7.5), layout="constrained")
    for index, arm in enumerate(ORDER):
        rows = [r for r in result["rows"] if r["arm"] == arm]
        valid = [
            100 * r["valid_count"] / r["sample_count"] if r["sample_count"] else 0 for r in rows
        ]
        coverage = [
            100 * r["valid_grid_center_bins_covered"] / result["grid_valid_centers"] for r in rows
        ]
        for ax, values in zip(axes, (valid, coverage)):
            ax.bar(index, np.mean(values), color=COLORS[index], alpha=0.7)
            ax.scatter(
                index + np.linspace(-0.12, 0.12, len(values)),
                values,
                color="#20292e",
                s=25,
                zorder=3,
            )
            ax.text(
                index,
                max(values) + 3,
                " / ".join(f"{v:.0f}" for v in values),
                ha="center",
                fontsize=9,
            )
    axes[0].set(
        ylabel="Joint-valid proposals (%)",
        title="Raw geometric yield: target clears and every upright record overlaps by the criterion",
        ylim=(0, 115),
    )
    axes[1].set(
        ylabel="Valid grid-center bins reached (%)",
        title=f"Coverage proxy: {result['grid_valid_centers']} valid centers in a 30 × 70 grid",
        ylim=(0, 115),
    )
    for ax in axes:
        ax.set_xticks(range(len(ORDER)), LABELS)
        ax.grid(axis="y", alpha=0.15)
    figure.suptitle(
        "First motion-only inverse experiment · one development carrier\n"
        "256 proposals per optimizer seed; dots show three seeds, not three carriers",
        fontsize=14,
    )
    path = assets / "inverse-learning.svg"
    if path.exists():
        raise FileExistsError(path)
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    outputs.append(path)

    figure, axes = plt.subplots(
        1, 3, figsize=(12, 4.8), sharex=True, sharey=True, layout="constrained"
    )
    ns, nh = manifest["grid"]["station_bins"], manifest["grid"]["height_bins"]
    valid_grid = np.asarray(result["grid"]["valid"]).reshape(ns, nh)
    grid_scenes = np.asarray(result["grid"]["scenes"])
    for ax, arm, label in zip(
        axes,
        ("full", "no_kl", "random_prior"),
        ("Encoder + preference + KL", "Encoder + preference, no KL", "Random prior"),
    ):
        row = next(r for r in result["rows"] if r["arm"] == arm and r["seed"] == 8121)
        scenes, valid = np.asarray(row["scenes"]), np.asarray(row["valid"])
        ax.scatter(
            grid_scenes[valid_grid.ravel(), 0],
            grid_scenes[valid_grid.ravel(), 1],
            color="#9ab7a1",
            marker="s",
            s=12,
            label="Valid grid centers",
            zorder=1,
        )
        ax.scatter(
            scenes[~valid, 0],
            scenes[~valid, 1],
            color="#c77f45",
            s=9,
            alpha=0.5,
            label="Rejected proposal",
        )
        ax.scatter(
            scenes[valid, 0],
            scenes[valid, 1],
            color="#187f72",
            s=9,
            alpha=0.6,
            label="Valid proposal",
        )
        ax.set(title=label, xlabel="Route station", xlim=(0.35, 0.65), ylim=(1.10, 1.45))
    axes[0].set_ylabel("Beam underside (m)")
    axes[-1].legend(loc="lower right", fontsize=8)
    figure.suptitle(
        "Learned obstacle samples · optimizer seed 8121\n"
        "All eight recordings checked independently; no physics or continuous-time verdict",
        fontsize=13,
    )
    path = assets / "inverse-distributions.svg"
    if path.exists():
        raise FileExistsError(path)
    figure.savefig(path, metadata={"Date": None})
    plt.close(figure)
    outputs.append(path)
    write_new(
        assets / "inverse-manifest.json",
        portable(
            {
                "role": "portable_inverse_learning_diagnostic",
                "sources": {k: artifact(v) for k, v in sources.items()},
                "renderer": artifact(Path(__file__)),
                "outputs": [artifact(path) for path in outputs],
                "training_eligible": False,
            }
        ),
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
