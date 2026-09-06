#!/usr/bin/env python3
"""Verify and export the explicit-margin experiment, including every ablation seed."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_margin_learning import load, summarize
from motion2scene_timing_diagnostic import artifact, checked, write_new
import numpy as np
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--certificate", type=Path)
    parser.add_argument("--certificate-audit", type=Path)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result["analysis_complete"]:
        raise ValueError("requires completed analysis")
    ref = result["manifest"]
    manifest_path = checked(Path(ref["path"]), ref["sha256"])
    manifest, source, _, _ = load(manifest_path)
    ref = result["training_completion"]
    completion = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    for ref in completion["cells"]:
        saved = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ("checkpoint", "samples"):
            checked(Path(saved[key]["path"]), saved[key]["sha256"])
    for row in result["rows"]:
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["offsets"], manifest["evaluation_offsets"])
        for key, value in summarize(raw["clearance_m"], raw["scenes"], source).items():
            if value != row[key]:
                raise ValueError(f"raw evidence differs from summary: {row['arm']} {key}")
    evidence, assets = args.out / "evidence", args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, value in (("margin-learning", result), ("margin-learning-manifest", manifest)):
        path = evidence / f"{name}.json"
        write_new(path, portable(value))
        outputs.append(path)
    certificate_source = None
    if args.certificate is not None:
        certificate_source = artifact(args.certificate)
        certificate = json.loads(args.certificate.read_text())
        ref = certificate["manifest"]
        cm = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for ref in [
            cm["source_result"],
            cm["source_manifest"],
            cm["protocol"],
            *cm["implementations"],
        ]:
            checked(Path(ref["path"]), ref["sha256"])
        if cm["source_result"] != artifact(args.result):
            raise ValueError("certificate must refer to this margin result")
        summaries = []
        for row in certificate["rows"]:
            ref = row["artifact"]
            detail = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
            summaries.append(
                {
                    **row,
                    "certificate_summary": {
                        k: v
                        for k, v in detail["certificate"].items()
                        if k not in ("trace", "remaining_domains")
                    },
                }
            )
        for name, value in (
            ("placement-certificate", {**certificate, "rows": summaries}),
            ("placement-certificate-manifest", cm),
        ):
            path = evidence / f"{name}.json"
            write_new(path, portable(value))
            outputs.append(path)
        if args.certificate_audit is not None:
            audit = json.loads(args.certificate_audit.read_text())
            if audit["source"] != artifact(args.certificate):
                raise ValueError("certificate audit source differs")
            ref = audit["auditor"]
            checked(Path(ref["path"]), ref["sha256"])
            path = evidence / "placement-certificate-audit.json"
            write_new(path, portable(audit))
            outputs.append(path)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-margin-v1",
            "font.size": 10,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    methods = ("robust", "margin", "constraints")
    colors = ("#567ba3", "#187f72", "#c77f45")
    labels = ("Previous preference", "Preference + both margins", "Both margins only")
    for column, family in enumerate(("full", "no_kl")):
        for method, color, label, shift in zip(methods, colors, labels, (-0.26, 0, 0.26)):
            rows = [
                next(
                    r
                    for r in result["rows"]
                    if r["arm"] == f"{method}_{family}" and r["seed"] == seed
                )
                for seed in manifest["seeds"]
            ]
            for ax, key in zip(axes[:, column], ("combined_113_valid", "occupied_accepted_bins")):
                counts = [r[key] for r in rows]
                positions = np.arange(3) + shift
                ax.bar(positions, counts, width=0.23, color=color, label=label)
                for x, count in zip(positions, counts):
                    ax.text(x, count + 0.7, str(count), ha="center", fontsize=9, color=color)
                ax.set(xticks=np.arange(3), xticklabels=manifest["seeds"])
                ax.grid(axis="y", alpha=0.15)
        axes[0, column].set(title="KL = 0.02" if family == "full" else "No KL", ylim=(0, 78))
        axes[1, column].set(
            xlabel="Optimizer seed",
            ylim=(0, max(r["occupied_accepted_bins"] for r in result["rows"]) + 4),
        )
    axes[0, 0].set_ylabel("Valid at all 113 placements / 64 draws")
    axes[1, 0].set_ylabel("Distinct bins occupied by accepted draws")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle(
        "Explicit interference margins: matched geometry-query budgets\n"
        "One carrier · three optimizer seeds · dispersion counts are not feasible-support coverage",
        fontsize=13,
    )
    path = assets / "margin-learning.svg"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    outputs.append(path)
    write_new(
        assets / "margin-manifest.json",
        portable(
            {
                "source": artifact(args.result),
                "certificate_source": certificate_source,
                "renderer": artifact(Path(__file__)),
                "outputs": [artifact(p) for p in outputs],
                "training_eligible": False,
                "execution_eligible": False,
            }
        ),
    )
    print(json.dumps({"outputs": [str(p) for p in outputs]}))


if __name__ == "__main__":
    main()
