#!/usr/bin/env python3
"""Export verified grouped motion-conditioning results and a carrier-level figure."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_carrier_learning import load
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import verdict
import numpy as np
from render_motion2scene_timing_report import portable
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_inverse import BeamMixture

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["conditioned", "constant", "shuffled", "swapped_input", "direct", "random"]
LABELS = [
    "Correct\nmotion input",
    "Constant\ninput",
    "Shuffled\ntraining",
    "Swapped\ntest event",
    "Per-motion\nsearch",
    "Random\nprior",
]


def input_sensitivity(cells, cases):
    """Post-hoc diagnosis; does not select or modify any checkpoint/proposal."""
    rows = []
    torch.set_num_threads(2)
    for cell in cells:
        if cell["mode"] != "conditioned":
            continue
        ref = cell["checkpoint"]
        path = checked(Path(ref["path"]), ref["sha256"])
        model = BeamMixture(29).double()
        model.load_state_dict(torch.load(path, weights_only=True)["model"])
        model.eval()
        for carrier in range(41005, 41009):
            features = [cases[f"{carrier}_event{event}"]["feature"] for event in range(3)]
            with torch.no_grad():
                parameters = [torch.cat([v.flatten() for v in model(f)]) for f in features]
                pooled = [
                    model.temporal(model.body(f).flatten(1).T.unsqueeze(0)).flatten()
                    for f in features
                ]
            predictions = json.loads(Path(cell["samples"]["path"]).read_text())["predictions"]
            scenes = [
                np.array(
                    next(
                        p["scenes"]
                        for p in predictions
                        if p["case_id"] == f"{carrier}_event{event}" and "mode" not in p
                    )
                )
                for event in range(3)
            ]
            rows.append(
                {
                    "seed": cell["seed"],
                    "carrier_seed": carrier,
                    "checkpoint": ref,
                    "feature_max_change_from_early": max(
                        float((f - features[0]).abs().max()) for f in features
                    ),
                    "parameter_max_change_from_early": max(
                        float((p - parameters[0]).abs().max()) for p in parameters
                    ),
                    "pooled_feature_max_change_from_early": max(
                        float((p - pooled[0]).abs().max()) for p in pooled
                    ),
                    "pooled_abs_above_0_999_fraction": float(
                        (torch.stack(pooled).abs() > 0.999).double().mean()
                    ),
                    "event_mean_station": [float(s[:, 0].mean()) for s in scenes],
                    "paired_draw_max_station_change_from_early": max(
                        float(np.abs(s[:, 0] - scenes[0][:, 0]).max()) for s in scenes
                    ),
                    "paired_draw_max_height_change_m_from_early": max(
                        float(np.abs(s[:, 1] - scenes[0][:, 1]).max()) for s in scenes
                    ),
                }
            )
    return {
        "role": "post_hoc_checkpoint_diagnosis_not_registered_hypothesis",
        "comparison": "middle_and_late_against_early_input_same_carrier_same_draw_seed",
        "rows": rows,
        "causal_architecture_claim": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    if not result["analysis_complete"]:
        raise ValueError("requires completed analysis")
    ref = result["manifest"]
    path = checked(Path(ref["path"]), ref["sha256"])
    manifest = json.loads(path.read_text())
    reg, cases = load(path)
    ref = result["training_completion"]
    completion = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    cells = []
    for ref in completion["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ("checkpoint", "samples"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        cells.append(cell)
    for row in result["rows"]:
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["offsets"], reg["evaluation_offsets"])
        np.testing.assert_array_equal(raw["scenes"], row["scenes"])
        valid = verdict(raw["clearances"], np.array([False, True]))
        assert valid.all(1).tolist() == row["per_proposal_valid"]
        assert int(valid.all(1).sum()) == row["all_113_valid"]
        assert int(valid[:, 0].sum()) == row["nominal_valid"]
    for aggregate in result["summary"]:
        rows = [
            r
            for r in result["rows"]
            if r["mode"] == aggregate["mode"] and r["carrier_seed"] == aggregate["carrier_seed"]
        ]
        assert len(rows) == 9 and sum(r["all_113_valid"] for r in rows) == aggregate["valid"]
        assert sum(r["sample_count"] for r in rows) == aggregate["proposals"]
    evidence, assets = args.out / "evidence", args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    timing = {
        mode: {
            "cells": sum(c["mode"] == mode for c in cells),
            "training_seconds": sum(c["training_seconds"] for c in cells if c["mode"] == mode),
        }
        for mode in ("conditioned", "constant", "shuffled", "direct")
    }
    outputs = []
    exports = {
        "carrier-learning": {**result, "training_cost": timing},
        "carrier-learning-manifest": manifest,
        "carrier-learning-registration": reg,
        "carrier-registry": json.loads(Path(manifest["registry"]["path"]).read_text()),
        "carrier-input-sensitivity": input_sensitivity(cells, cases),
    }
    for name, value in exports.items():
        out = evidence / f"{name}.json"
        write_new(out, portable(value))
        outputs.append(out)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-carrier-v1",
            "font.size": 10,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
        }
    )
    counts = np.array(
        [
            [
                next(
                    r["valid"]
                    for r in result["summary"]
                    if r["mode"] == mode and r["carrier_seed"] == seed
                )
                for mode in ORDER
            ]
            for seed in range(41005, 41009)
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained")
    visual = ax.imshow(counts / 288 * 100, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    for i in range(4):
        for j in range(6):
            count = counts[i, j]
            ax.text(
                j,
                i,
                f"{100 * count / 288:.1f}%\n{count}/288",
                ha="center",
                va="center",
                color="white" if count / 288 > 0.55 else "#172b36",
                fontsize=11,
            )
    ax.set_xticks(range(6), LABELS)
    ax.set_yticks(
        range(4), ["41005 · validation", "41006 · validation", "41007 · test", "41008 · test"]
    )
    ax.tick_params(length=0)
    ax.axhline(1.5, color="#a76d43", linewidth=2)
    ax.set_title(
        "Does learned motion input help beyond the route-based scene builder?\n"
        "Excluded source carriers · 3 events × 3 optimizer seeds × 32 draws per cell",
        pad=18,
    )
    fig.colorbar(visual, ax=ax, label="Valid at all 113 placements (%)", shrink=0.85)
    path = assets / "carrier-learning.svg"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    outputs.append(path)
    write_new(
        assets / "carrier-manifest.json",
        portable(
            {
                "source": artifact(args.result),
                "renderer": artifact(Path(__file__)),
                "outputs": [artifact(p) for p in outputs],
                "reference_only": True,
                "training_eligible": False,
                "execution_eligible": False,
            }
        ),
    )
    print(json.dumps({"outputs": [str(p) for p in outputs]}), flush=True)


if __name__ == "__main__":
    main()
