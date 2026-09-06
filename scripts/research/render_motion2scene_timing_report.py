#!/usr/bin/env python3
"""Export the completed timing diagnostic as portable evidence and a paired-results figure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sha(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    if isinstance(value, str) and value.startswith("/home/"):
        for marker in ("research-data/groot-wbc/", "groot-wbc-sonic-sim-trackb/", "motion2scene/"):
            if marker in value:
                return value.split(marker, 1)[1]
        return Path(value).name
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--route-audit", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result["analysis_complete"] or len(result["rows"]) != 12:
        raise ValueError("requires the complete twelve-cell diagnostic")
    sources = {"result": args.result}
    for key in ("manifest", "run_record", "driver"):
        path = Path(result[key]["path"])
        if sha(path) != result[key]["sha256"]:
            raise ValueError(f"{key} hash mismatch")
        sources[key] = path
    manifest = json.loads(sources["manifest"].read_text())
    predictions = Path(manifest["registered_predictions"]["path"])
    if sha(predictions) != manifest["registered_predictions"]["sha256"]:
        raise ValueError("prediction hash mismatch")
    sources["predictions"] = predictions
    evidence = args.out / "evidence"
    assets = args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    outputs = []
    for key in ("result", "manifest", "run_record"):
        path = evidence / f"timing-diagnostic-{key}.json"
        path.write_text(json.dumps(portable(json.loads(sources[key].read_text())), indent=2) + "\n")
        outputs.append(path)

    plt.rcParams.update({"svg.hashsalt": "motion2scene-timing-v1", "font.size": 11})
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    figure.set_facecolor("#f6f4ed")
    colors = {41001: "#187f72", 41002: "#bd752e", 41003: "#567ba3"}
    rows = {(r["generation_seed"], r["condition"], r["label"]): r for r in result["rows"]}
    for axis, level in zip(axes[:2], ("neutral", "d055")):
        axis.set_title("Walk tracking" if level == "neutral" else "Crouch tracking")
        for seed, color in colors.items():
            values = [rows[seed, condition, level] for condition in ("original", "shared_clock")]
            if all(r["status"] == "completed" for r in values):
                axis.plot(
                    [0, 1],
                    [r["diagnostics"]["endpoint_error_m"] for r in values],
                    color=color,
                    alpha=0.8,
                    label=str(seed),
                )
            for position, row in enumerate(values):
                if row["status"] == "completed":
                    axis.scatter(
                        position,
                        row["diagnostics"]["endpoint_error_m"],
                        color=color,
                        marker="o" if row["tracker_accepted"] else "x",
                        s=70,
                        zorder=3,
                    )
        axis.set_ylabel("Tracker endpoint error (m); lower is better")
        axis.set_xticks([0, 1], ["Original", "Shared clock"])
        axis.set_xlim(-0.25, 1.25)
        axis.set_ylim(bottom=0)
    axis = axes[2]
    axis.set_title("Executed crouch effect")
    for seed, color in colors.items():
        for position, condition in enumerate(("original", "shared_clock")):
            row = rows[seed, condition, "d055"]
            if "achieved_semantic" in row:
                axis.scatter(
                    position + (seed - 41002) * 0.08,
                    100 * row["achieved_semantic"]["peak_reduction"],
                    color=color,
                    marker="o" if row["paired_behavior_retained"] else "x",
                    s=70,
                )
    axis.axhline(5, color="#53625f", linestyle="--", label="Frozen 5 cm effect threshold")
    axis.set_ylabel("Peak whole-body lowering vs matched walk (cm)")
    axis.set_xticks([0, 1], ["Original", "Shared clock"])
    axis.set_xlim(-0.25, 1.25)
    axis.set_ylim(bottom=0)
    axes[0].legend(title="Carrier", loc="best")
    for axis in axes:
        axis.set_facecolor("#f6f4ed")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Paired timing diagnostic · 3 development carriers · physics seed 7900", fontsize=15
    )
    figure.supxlabel(
        "Circles: tracker accepted (left/middle), paired behavior retained (right). Crosses: failed.\n"
        "Whole-body lowering alone does not establish a localized, route-retained crouch.",
        fontsize=10,
    )
    plot = assets / "timing-diagnostic.svg"
    figure.savefig(plot, metadata={"Date": None})
    plt.close(figure)
    outputs.append(plot)
    if args.route_audit:
        audit = json.loads(args.route_audit.read_text())
        sources["route_audit"] = args.route_audit
        for source in audit["sources"].values():
            if sha(Path(source["path"])) != source["sha256"]:
                raise ValueError("route audit source hash mismatch")
        path = evidence / "route-sensitivity.json"
        path.write_text(json.dumps(portable(audit), indent=2) + "\n")
        outputs.append(path)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        figure.set_facecolor("#f6f4ed")
        for index, row in enumerate(audit["rows"]):
            grid = row["grid"]
            frozen = next(
                c for c in grid if c["smoothing_seconds"] == 0.5 and c["spacing_metres"] == 0.5
            )
            axes[0].plot(
                row["heading_error_range_rad"], [index, index], color="#187f72", linewidth=3
            )
            axes[0].scatter(
                frozen["metrics"]["signed_heading_error_rad"], index, color="#bd752e", zorder=3
            )
            axes[1].scatter(100 * frozen["metrics"]["cross_track_rmse_m"], index, color="#187f72")
        for axis in axes:
            axis.set_yticks(
                range(len(audit["rows"])), [str(r["generation_seed"]) for r in audit["rows"]]
            )
            axis.set_facecolor("#f6f4ed")
            axis.spines[["top", "right"]].set_visible(False)
            axis.invert_yaxis()
            axis.grid(alpha=0.2)
        axes[0].axvline(0.3490658503988659, color="#bd752e", linestyle="--")
        axes[0].axvline(-0.3490658503988659, color="#bd752e", linestyle="--")
        axes[0].set_title("Heading estimate across all 12 measurement scales")
        axes[0].set_xlabel("Signed heading error (rad); dot = frozen policy")
        axes[1].axvline(10, color="#bd752e", linestyle="--")
        axes[1].set_title("Cross-track error is unchanged across the grid")
        axes[1].set_xlabel("Cross-track RMSE (cm)")
        figure.supxlabel(
            "Post-outcome diagnostic; no scale selected and no historical label changed.",
            fontsize=10,
        )
        plot = assets / "route-sensitivity.svg"
        figure.savefig(plot, metadata={"Date": None})
        plt.close(figure)
        outputs.append(plot)
    export = {
        "schema_version": "motion2scene_timing_public_export_v1",
        "sources": {
            key: {"path": portable(str(path)), "sha256": sha(path)} for key, path in sources.items()
        },
        "outputs": [
            {"path": str(path.relative_to(args.out)), "sha256": sha(path)} for path in outputs
        ],
        "renderer_sha256": sha(Path(__file__)),
        "note": "Public paths are shortened; source and public snapshot hashes differ.",
    }
    (assets / "timing-manifest.json").write_text(json.dumps(export, indent=2) + "\n")
    print(f"Exported {len(outputs)} artifacts and timing-manifest.json")


if __name__ == "__main__":
    main()
