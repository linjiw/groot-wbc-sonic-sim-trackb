#!/usr/bin/env python3
"""Export verified event/scale evidence, learning curves and all test-case samples."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_event_scaling import load
from motion2scene_timing_diagnostic import artifact, checked, write_new
from motion2scene_uncertainty_learning import verdict
import numpy as np
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
from matplotlib.lines import Line2D  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["local", "pooled", "constant", "shuffled", "wide", "half_data"]
LABELS = [
    "Local features",
    "Pooled features",
    "Constant input",
    "Shuffled input",
    "Wider local model",
    "Local, 2 parents",
]
COLORS = ["#087f73", "#617393", "#a66b42", "#ae628d", "#316ec4", "#978a34"]


def support_diagnostic(result):
    """Accepted-bin occupancy is descriptive; it is not feasible-support coverage."""
    groups = {}
    for row in result["rows"]:
        key = (row["arm"], row["budget"], row["case_id"])
        groups.setdefault(key, []).append(row)
    rows = []
    for (arm, budget, case), group in groups.items():
        assert len(group) == 3
        scenes = np.concatenate([np.array(r["scenes"])[r["per_proposal_valid"]] for r in group])
        bins = np.floor((scenes - [0.1, 1.1]) / [0.02, 0.01]).astype(int)
        bins = np.minimum(bins, [39, 34])
        rows.append(
            {
                "arm": arm,
                "budget": budget,
                "case_id": case,
                "proposals": 48,
                "accepted": len(scenes),
                "occupied_accepted_bins": len(np.unique(bins, axis=0)),
                "accepted_station_std": float(scenes[:, 0].std()) if len(scenes) else None,
                "accepted_height_std_m": float(scenes[:, 1].std()) if len(scenes) else None,
            }
        )
    return {
        "role": "post_hoc_descriptive_accepted_bin_occupancy_not_registered_prediction",
        "bin_width_station_height_m": [0.02, 0.01],
        "bins": [40, 35],
        "feasible_support_coverage_estimated": False,
        "rows": rows,
    }


def sampler_exports(root, reg_ref):
    path, plan_path = root / "fresh-sampler/result.json", root / "sampler-plan.json"
    if not path.exists():
        return {}
    plan, sample = json.loads(plan_path.read_text()), json.loads(path.read_text())
    assert plan["registration"] == sample["registration"] == reg_ref
    assert plan["sampler"] == sample["sampler"]
    for ref in [sample["sampler"], sample["cell"], sample["raw"]]:
        checked(Path(ref["path"]), ref["sha256"])
    cell = json.loads(Path(sample["cell"]["path"]).read_text())
    assert (cell["arm"], cell["seed"], cell["budget"]) == (
        plan["arm"],
        plan["model_seed"],
        plan["budget"],
    )
    assert (sample["case"], sample["seed"], sample["requested"]) == (
        plan["case"],
        plan["seed"],
        plan["count"],
    )
    raw = np.load(sample["raw"]["path"], allow_pickle=False)
    valid = verdict(raw["clearances"], np.array([False, True])).all(1)
    assert np.flatnonzero(valid).tolist() == sample["accepted_indices"]
    assert int(valid.sum()) == sample["accepted"]
    return {"event-fresh-sampler": sample, "event-sampler-plan": plan}


def verify(result):
    ref = result["registration"]
    reg, _, _ = load(checked(Path(ref["path"]), ref["sha256"]))
    ref = result["training_completion"]
    completion = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
    cells = []
    for ref in completion["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ["checkpoint", "samples"]:
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        cells.append(cell)
    for row in result["rows"]:
        ref = row["raw"]
        raw = np.load(checked(Path(ref["path"]), ref["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["scenes"][:16], row["scenes"])
        np.testing.assert_array_equal(raw["offsets"], reg["evaluation_offsets"])
        valid = verdict(raw["clearances"][:16], np.array([False, True]))
        assert valid.all(1).tolist() == row["per_proposal_valid"]
        assert int(valid[:, 0].sum()) == row["nominal_valid"]
        assert int(valid.all(1).sum()) == row["all_113_valid"]
    for row in result["summary"]:
        selected = [
            r
            for r in result["rows"]
            if r["arm"] == row["arm"]
            and r["budget"] == row["budget"]
            and r["carrier_seed"] == row["carrier_seed"]
        ]
        assert len(selected) == 9
        assert sum(r["all_113_valid"] for r in selected) == row["valid"]
        assert sum(r["sample_count"] for r in selected) == row["proposals"]
    return reg, completion, cells


def curves(result, out):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained", sharex=True, sharey=True)
    for ax, carrier in zip(axes.flat, range(41005, 41009)):
        for arm, label, color in zip(ARMS, LABELS, COLORS):
            rates, low, high = [], [], []
            for budget in [600, 2400]:
                rows = [
                    r
                    for r in result["rows"]
                    if r["arm"] == arm and r["budget"] == budget and r["carrier_seed"] == carrier
                ]
                values = [
                    sum(r["all_113_valid"] for r in rows if r["seed"] == seed) / 48 * 100
                    for seed in [8321, 8322, 8323]
                ]
                rates.append(np.mean(values))
                low.append(max(0.0, rates[-1] - min(values)))
                high.append(max(0.0, max(values) - rates[-1]))
            ax.errorbar(
                [600, 2400],
                rates,
                yerr=[low, high],
                label=label,
                color=color,
                marker="o",
                markersize=4,
                capsize=3,
                linewidth=1.6,
                elinewidth=0.7,
            )
        for arm, style in [("previous_direct", ":"), ("previous_random", "--")]:
            row = next(
                r for r in result["summary"] if r["arm"] == arm and r["carrier_seed"] == carrier
            )
            ax.axhline(row["valid"] / 144 * 100, color="#7b7b7b", linestyle=style, linewidth=1)
        ax.set_title(f"{carrier} · {'validation' if carrier < 41007 else 'test'}")
        ax.set_xticks([600, 2400], ["600", "2,400"])
        ax.set_xlim(350, 2650)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", alpha=0.18)
        ax.set_xlabel("Optimizer updates · same eight proposals/update")
        ax.set_ylabel("Valid at all 113 placements (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [
        Line2D([], [], color="#7b7b7b", linestyle=":"),
        Line2D([], [], color="#7b7b7b", linestyle="--"),
    ]
    labels += ["Earlier direct search", "Earlier random prior"]
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        "Event representation and scale: all source-level outcomes\n"
        "Means of 3 seeds; whiskers show seed min–max, not confidence intervals",
        fontsize=13,
    )
    path = out / "event-scaling.svg"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def samples(result, out):
    fig, axes = plt.subplots(
        2, 3, figsize=(12, 7.2), layout="constrained", sharex=True, sharey=True
    )
    colors = ["#087f73", "#316ec4", "#a66b42"]
    for i, carrier in enumerate([41007, 41008]):
        for event in range(3):
            ax = axes[i, event]
            rows = [
                r
                for r in result["rows"]
                if r["arm"] == "local"
                and r["budget"] == 2400
                and r["case_id"] == f"{carrier}_event{event}"
            ]
            assert len(rows) == 3
            for row, color in zip(sorted(rows, key=lambda r: r["seed"]), colors):
                scenes, valid = np.array(row["scenes"]), np.array(row["per_proposal_valid"])
                ax.scatter(scenes[valid, 0], scenes[valid, 1], color=color, s=25, marker="o")
                ax.scatter(
                    scenes[~valid, 0], scenes[~valid, 1], color=color, s=22, marker="x", alpha=0.75
                )
            ax.axvline(rows[0]["event_station"], color="#888888", linewidth=0.9, linestyle=":")
            count = sum(r["all_113_valid"] for r in rows)
            ax.set_title(
                f"{carrier} · event {['early', 'middle', 'late'][event]} · {count}/48 valid"
            )
            ax.set_xlim(0.1, 0.9)
            ax.set_ylim(1.1, 1.45)
            ax.set_xlabel("Route station")
            ax.set_ylabel("Beam underside height (m)")
            ax.grid(alpha=0.13)
    fig.suptitle(
        "All local-model test proposals at 2,400 updates\n"
        "Circle: all-113 valid · cross: rejected · dotted line: constructed event location",
        fontsize=13,
    )
    fig.legend(
        [Line2D([], [], marker="o", color=c, linestyle="") for c in colors],
        ["Seed 8321", "Seed 8322", "Seed 8323"],
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    path = out / "event-proposals.svg"
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/motion2scene"))
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    assert result["analysis_complete"]
    reg, completion, cells = verify(result)
    cost = [
        {
            "arm": arm,
            "parameters": reg["arms"][arm]["parameters"],
            "budget": budget,
            "fitting_seconds": [
                c["cumulative_fitting_seconds"]
                for c in cells
                if c["arm"] == arm and c["budget"] == budget
            ],
            "sampling_seconds": sum(
                c["sampling_seconds"] for c in cells if c["arm"] == arm and c["budget"] == budget
            ),
        }
        for arm in ARMS
        for budget in reg["budgets"]
    ]
    evidence, assets = args.out / "evidence", args.out / "assets"
    evidence.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    outputs = []
    exports = {
        "event-scaling": {
            **result,
            "cost": cost,
            "training_wall_seconds": completion["elapsed_seconds"],
        },
        "event-scaling-registration": reg,
        "event-support-diagnostic": support_diagnostic(result),
        **sampler_exports(args.result.parent, result["registration"]),
    }
    for name, value in exports.items():
        path = evidence / f"{name}.json"
        write_new(path, portable(value))
        outputs.append(path)
    plt.rcParams.update(
        {
            "svg.hashsalt": "motion2scene-event-scaling-v1",
            "font.size": 9,
            "figure.facecolor": "#f6f4ed",
            "axes.facecolor": "#f6f4ed",
        }
    )
    outputs.extend([curves(result, assets), samples(result, assets)])
    write_new(
        assets / "event-manifest.json",
        portable(
            {
                "source": artifact(args.result),
                "renderer": artifact(Path(__file__)),
                "outputs": [artifact(p) for p in outputs],
                "reference_only": True,
                "execution_eligible": False,
                "training_eligible": False,
            }
        ),
    )
    print(json.dumps({"outputs": [str(p) for p in outputs]}), flush=True)


if __name__ == "__main__":
    main()
