#!/usr/bin/env python3
"""Independently reconcile distillation traces, source counts and public artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_distill_study import (
    ARMS,
    FITS,
    checkpoint,
    load,
    original_cells,
    predicates,
    retained_indices,
)
from motion2scene_event_scaling import normalized, raw_features
from motion2scene_refinement_study import measure
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import tensor, verdict
import numpy as np
from render_motion2scene_refinement import slack
from render_motion2scene_timing_report import portable
import torch

from gear_sonic.dataset_generation.hallucination.motion2scene_events import sample_scenes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = [
    "Original raw",
    "Geometry 600",
    "Geometry 2666",
    "Set only 600",
    "Hybrid 600",
    "Original + 17",
    "Hybrid + 17",
    "Hybrid + 5",
    "Uniform + 5",
]


def selected_output(trace):
    selected = np.argmax(slack(trace["clearances"]), axis=0)
    np.testing.assert_array_equal(selected, trace["selected"])
    return trace["scenes"][selected, np.arange(8)]


def verify(path):
    result = json.loads(path.read_text())
    for k in ["registration", "teacher", "fitting"]:
        checked(Path(result[k]["path"]), result[k]["sha256"])
    reg, parent, train, dev = load(Path(result["registration"]["path"]))
    teacher = json.loads(Path(result["teacher"]["path"]).read_text())
    fitted = json.loads(Path(result["fitting"]["path"]).read_text())
    assert [r["case_id"] for r in teacher["rows"]] == reg["training_ids"]
    for r in teacher["rows"]:
        trace = np.load(checked(Path(r["trace"]["path"]), r["trace"]["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(selected_output(trace), trace["output"])
        assert trace["clearances"].shape == (17, 8, 17, 2)
        valid = verdict(trace["audit_clearances"], train[r["case_id"]]["mask"].numpy()).all(1)
        np.testing.assert_array_equal(valid, r["accepted"])
        retained = retained_indices(trace["output"], valid)
        assert retained == r["retained_indices"]
        np.testing.assert_array_equal(trace["retained"], retained)
        np.testing.assert_array_equal(trace["output"][retained], r["targets"])
        assert abs(trace["scenes"] - trace["initial"]).max(axis=(0, 1))[0] <= 0.05 + 1e-12
        assert abs(trace["scenes"] - trace["initial"]).max(axis=(0, 1))[1] <= 0.03 + 1e-12
        assert np.isfinite(trace["audit_clearances"]).all() and r["query_error_m"] <= 1e-8
    assert teacher["all_cases_nonempty"] == all(r["targets"] for r in teacher["rows"])
    assert teacher["total_queries"] == 165216
    assert len(fitted["cells"]) == 12 and {(c["arm"], c["seed"]) for c in fitted["cells"]} == {
        (a, s) for a in FITS for s in reg["seeds"]
    }
    expected = {(a, s, c) for a in ARMS for s in reg["seeds"] for c in dev}
    assert len(result["rows"]) == len(expected) == 432
    assert {(r["arm"], r["seed"], r["case_id"]) for r in result["rows"]} == expected
    draws, refs = {}, {}
    raw = raw_features(dev)
    for original in original_cells(parent):
        seed = original["seed"]
        refs[seed] = {
            "original": original["checkpoint"],
            **{c["arm"]: c["checkpoint"] for c in fitted["cells"] if c["seed"] == seed},
        }
        original_saved, _ = checkpoint(original["checkpoint"])
        features = normalized(raw, original_saved["config"])
        for name, ref in refs[seed].items():
            saved, model = checkpoint(ref)
            assert saved["config"] == original_saved["config"]
            with torch.no_grad():
                for case in dev:
                    draws[(seed, name, case)] = sample_scenes(
                        model(features[case]), 256, torch.Generator().manual_seed(seed + 250000)
                    )[:8].numpy()
        for case in dev:
            draws[(seed, "uniform", case)] = (
                tensor([0.1, 1.1])
                + torch.rand(
                    (8, 2),
                    dtype=torch.float64,
                    generator=torch.Generator().manual_seed(seed + 260000),
                )
                * tensor([0.8, 0.35])
            ).numpy()
    for row in result["rows"]:
        trace = np.load(
            checked(Path(row["trace"]["path"]), row["trace"]["sha256"]), allow_pickle=False
        )
        base = row["arm"].split("_")[0]
        if base == "hybrid":
            base = "hybrid600"
        assert row["checkpoint"] == refs[row["seed"]].get(base)
        assert row["draw_seed"] == row["seed"] + (260000 if base == "uniform" else 250000)
        np.testing.assert_array_equal(trace["initial"], draws[(row["seed"], base, row["case_id"])])
        np.testing.assert_array_equal(trace["output"], row["scenes"])
        np.testing.assert_array_equal(trace["offsets"], parent["audit_offsets"])
        assert (
            trace["audit_clearances"].shape == (8, 113, 2)
            and np.isfinite(trace["audit_clearances"]).all()
        )
        assert (
            measure(
                row,
                dev[row["case_id"]],
                trace["audit_clearances"],
                row["query_error_m"],
                row["raw"],
            )
            == row
        )
        assert np.isfinite(trace["scenes"]).all()
        assert (trace["scenes"] >= [0.1 - 1e-12, 1.1 - 1e-12]).all() and (
            trace["scenes"] <= [0.9 + 1e-12, 1.45 + 1e-12]
        ).all()
        if row["arm"].endswith("raw"):
            assert row["queries"] == 0
            np.testing.assert_array_equal(trace["output"], trace["initial"])
        else:
            evaluations = 17 if row["arm"].endswith("17") else 5
            assert trace["scenes"].shape == (evaluations, 8, 2)
            assert trace["clearances"].size == row["queries"] == evaluations * 8 * 17 * 2
            assert (abs(trace["scenes"] - trace["initial"]) <= [0.05 + 1e-12, 0.03 + 1e-12]).all()
            np.testing.assert_array_equal(trace["output"], selected_output(trace))
    for c in fitted["cells"]:
        assert c["steps"] == FITS[c["arm"]]
        assert c["training_queries"] == (0 if c["arm"] == "set600" else FITS[c["arm"]] * 80)
        assert c["teacher_queries_charged_per_student"] == (
            165216 if c["arm"] in ["set600", "hybrid600"] else 0
        )
    summary, outcomes = predicates(result["rows"], teacher)
    assert summary == result["summary"] and outcomes == result["predicates"]
    assert result["search_queries"] == sum(r["queries"] for r in result["rows"]) == 574464
    assert result["audit_queries"] == 781056 and result["crosscheck_queries"] == 195264
    assert result["max_query_error_m"] == max(r["query_error_m"] for r in result["rows"]) <= 1e-8
    return result, reg, teacher, fitted


def table(result):
    sources = sorted({r["carrier_seed"] for r in result["rows"]})
    lines = [
        "| Method | " + " | ".join(f"{s} /48" for s in sources) + " | Total /384 |",
        "|---|" + "---:|" * 9,
    ]
    for arm, label in zip(ARMS, LABELS):
        counts = [
            sum(r["valid"] for r in result["rows"] if r["arm"] == arm and r["carrier_seed"] == s)
            for s in sources
        ]
        lines.append("| " + label + " | " + " | ".join(map(str, counts + [sum(counts)])) + " |")
    return "\n".join(lines)


def plot(result, fitted, path):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
    counts = [sum(r["valid"] for r in result["rows"] if r["arm"] == a) for a in ARMS]
    bars = axes[0].bar(
        np.arange(9),
        np.array(counts) * 100 / 384,
        color=["#8793a2"] * 3 + ["#b08b43", "#267b6b"] + ["#2772b8"] * 4,
    )
    axes[0].bar_label(bars, labels=[f"{n}/384" for n in counts], fontsize=9)
    axes[0].set_xticks(np.arange(9), LABELS, rotation=18)
    axes[0].set_ylim(0, 115)
    axes[0].set_ylabel("Accepted development outputs (%)")
    axes[0].set_title(
        "Can training-source search improve the learned proposal?\n"
        "Observed 410xx/420xx development sources; the 430xx audit pool is excluded"
    )
    times = [sum(c["fitting_seconds"] for c in fitted["cells"] if c["arm"] == a) for a in FITS]
    bars = axes[1].bar(np.arange(4), times, color="#418678")
    axes[1].bar_label(bars, fmt="%.1f s")
    axes[1].set_xticks(np.arange(4), list(FITS))
    axes[1].set_ylim(0, max(times) * 1.2)
    axes[1].set_ylabel("Additional fitting time, three seeds (s)")
    axes[1].set_title("Teacher acquisition and inference verification are charged separately")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    result, reg, teacher, fitted = verify(args.result)
    outputs = []
    for name, value in [
        ("distillation", result),
        ("distillation-registration", reg),
        ("distillation-teacher", teacher),
        ("distillation-fits", fitted),
    ]:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    path = args.out / "assets/distillation.svg"
    if path.exists():
        raise FileExistsError(path)
    plot(result, fitted, path)
    outputs.append(artifact(path))
    write_new(
        args.out / "assets/distillation-manifest.json",
        {"source": artifact(args.result), "renderer": artifact(Path(__file__)), "outputs": outputs},
    )
    print(table(result))
    print(json.dumps(result["predicates"], indent=2))


if __name__ == "__main__":
    main()
