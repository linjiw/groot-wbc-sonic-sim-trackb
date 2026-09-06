#!/usr/bin/env python3
"""Verify raw uncertainty evidence and export a compact, portable visual report."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import load, verdict
import numpy as np
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result["analysis_complete"]:
        raise ValueError("requires completed analysis")
    ref = result["manifest"]
    manifest_path = checked(Path(ref["path"]), ref["sha256"])
    manifest, source, _, _ = load(manifest_path)
    ref = result["training_completion"]
    checked(Path(ref["path"]), ref["sha256"])
    target_mask = np.array([r["label"] == "d055" for r in source["records"]])
    records = []
    for row in result["rows"]:
        for ref in row["sources"]:
            checked(Path(ref["path"]), ref["sha256"])
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        values, scenes = raw["clearance_m"], raw["scenes"]
        np.testing.assert_array_equal(raw["offsets"], manifest["evaluation_offsets"])
        valid = verdict(values, target_mask)
        if valid.all(-1).tolist() != row["per_proposal_113_valid"]:
            raise ValueError("raw-array verdict differs from saved result")
        for name, count in (
            ("nominal_valid", valid[:, 0].sum()),
            ("grid_81_valid", valid[:, :81].all(-1).sum()),
            ("interior_32_valid", valid[:, 81:].all(-1).sum()),
            ("combined_113_valid", valid.all(-1).sum()),
        ):
            if int(count) != row[name]:
                raise ValueError(f"raw-array count differs: {name}")
        target = values[..., target_mask].min(axis=(1, 2))
        weaker = values[..., ~target_mask].max(axis=(1, 2))
        records.append(
            {
                **row,
                "scenes": scenes.tolist(),
                "worst_target_clearance_m": target.tolist(),
                "worst_upright_clearance_m": weaker.tolist(),
                "failure_counts": {
                    "target_only": int(((target < 0.01) & (weaker <= -0.01)).sum()),
                    "upright_only": int(((target >= 0.01) & (weaker > -0.01)).sum()),
                    "both": int(((target < 0.01) & (weaker > -0.01)).sum()),
                },
            }
        )
    evidence, assets = args.out / "evidence", args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    exports = {
        "uncertainty-learning": {**result, "rows": records},
        "uncertainty-learning-manifest": manifest,
        "uncertainty-carrier-audit": json.loads(
            Path(manifest["carrier_audit"]["path"]).read_text()
        ),
    }
    outputs = []
    for name, data in exports.items():
        path = evidence / f"{name}.json"
        write_new(path, portable(data))
        outputs.append(path)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-uncertainty-v1",
            "font.size": 10,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained", sharey=True)
    for ax, family, title in zip(axes, ("full", "no_kl"), ("Preference + KL", "Preference, no KL")):
        for kind, color, shift, label in zip(
            ("nominal", "robust"),
            ("#567ba3", "#187f72"),
            (-0.18, 0.18),
            ("Nominal training", "Perturbation training"),
        ):
            counts = [
                next(r for r in records if r["arm"] == f"{kind}_{family}" and r["seed"] == seed)[
                    "combined_113_valid"
                ]
                for seed in manifest["seeds"]
            ]
            positions = np.arange(3) + shift
            ax.bar(positions, np.array(counts) / 64 * 100, width=0.32, color=color, label=label)
            for x, count in zip(positions, counts):
                ax.text(
                    x, count / 64 * 100 + 2, f"{count}/64", ha="center", color=color, fontsize=9
                )
        ax.set(
            xticks=np.arange(3),
            xticklabels=manifest["seeds"],
            xlabel="Optimizer seed",
            title=title,
            ylim=(0, 115),
        )
        ax.grid(axis="y", alpha=0.15)
    axes[0].set_ylabel("Proposals valid at all 113 placements (%)")
    axes[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(
        "Does placement uncertainty improve raw scene yield?\n"
        "One carrier · 64 unfiltered proposals per cell · eight recordings · capsule geometry",
        fontsize=13,
    )
    path = assets / "uncertainty-learning.svg"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    outputs.append(path)
    write_new(
        assets / "uncertainty-manifest.json",
        portable(
            {
                "source": artifact(args.result),
                "renderer": artifact(Path(__file__)),
                "outputs": [artifact(path) for path in outputs],
                "role": "finite_placement_development_diagnostic",
                "training_eligible": False,
                "execution_eligible": False,
            }
        ),
    )
    print(json.dumps({"outputs": [str(path) for path in outputs]}))


if __name__ == "__main__":
    main()
