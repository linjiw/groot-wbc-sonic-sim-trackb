#!/usr/bin/env python3
"""Verify saved search trajectories and export the independent refinement audit."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_refinement_study import load
from motion2scene_source_phase import summarize
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import verdict
import numpy as np
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARMS = ["raw", "learned_rank", "uniform_rank", "learned_refine", "uniform_refine"]
LABELS = ["Raw learned", "Rank learned", "Rank uniform", "Refine learned", "Refine uniform"]


def slack(values):
    return np.minimum(values[..., 1].min(-1) - 0.01, -values[..., 0].max(-1) - 0.01)


def verify(path):
    result = json.loads(path.read_text())
    for key in ["registration", "search"]:
        checked(Path(result[key]["path"]), result[key]["sha256"])
    reg, cases, previous = load(Path(result["registration"]["path"]))
    if len(result["rows"]) != 240:
        raise ValueError("incomplete audit")
    keys = set()
    for row in result["rows"]:
        key = (row["arm"], row["seed"], row["case_id"])
        if key in keys:
            raise ValueError("duplicate audit")
        keys.add(key)
        raw = np.load(checked(Path(row["raw"]["path"]), row["raw"]["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["scenes"], row["scenes"])
        np.testing.assert_array_equal(raw["offsets"], reg["audit_offsets"])
        if raw["clearances"].shape != (8, 113, 2) or not np.isfinite(raw["clearances"]).all():
            raise ValueError("invalid clearances")
        case = cases[row["case_id"]]
        valid = verdict(raw["clearances"], case["mask"].numpy()).all(1)
        assert row["valid"] == int(valid.sum()) and row["proposals"] == 8
        assert row["per_proposal_valid"] == valid.tolist()
        assert row["target_failures"] == int((raw["clearances"][:, :, 1].min(1) < 0.01).sum())
        assert row["neutral_failures"] == int((raw["clearances"][:, :, 0].max(1) > -0.01).sum())
        for field in ["carrier_seed", "split", "event_station"]:
            assert row[field] == case["metadata"][field]
        assert row["phase"] == "unseen" and row["budget"] == 0
        bins = np.floor((raw["scenes"][valid] - [0.1, 1.1]) / [0.02, 0.01]).astype(int)
        assert row["occupied_accepted_bins"] == len(np.unique(bins, axis=0))
        if row["arm"] == "raw":
            assert row["raw"] == previous[(row["seed"], row["case_id"])]["raw"]
            continue
        trace = np.load(
            checked(Path(row["trace"]["path"]), row["trace"]["sha256"]), allow_pickle=False
        )
        assert row["queries"] == trace["clearances"].size == 4624
        if row["arm"].endswith("rank"):
            indices = np.argsort(-slack(trace["clearances"]), kind="stable")[:8]
            scenes = trace["scenes"][indices]
        else:
            indices = np.argmax(slack(trace["clearances"]), axis=0)
            scenes = trace["scenes"][indices, np.arange(8)]
            assert np.all(
                abs(trace["scenes"] - trace["scenes"][0]) <= [0.050000000001, 0.030000000001]
            )
        np.testing.assert_array_equal(trace["selected"], indices)
        np.testing.assert_array_equal(raw["scenes"], scenes)
        assert np.all(trace["scenes"] >= np.array([0.1, 1.1]) - 1e-14)
        assert np.all(trace["scenes"] <= np.array([0.9, 1.45]) + 1e-14)
        if row["arm"] == "learned_refine":
            old = previous[(row["seed"], row["case_id"])]
            original = np.load(old["raw"]["path"], allow_pickle=False)
            np.testing.assert_allclose(trace["scenes"][0], original["scenes"], rtol=0, atol=1e-14)
            was_valid = verdict(original["clearances"], case["mask"].numpy()).all(1)
            assert row["rescued"] == int((valid & ~was_valid).sum())
            assert row["newly_failed"] == int((~valid & was_valid).sum())
    expected = {(a, s, c) for a in ARMS for s in reg["seeds"] for c in cases}
    assert keys == expected
    assert summarize(result["rows"]) == result["summary"]
    for comparator in ["raw", "uniform_refine", "learned_rank"]:
        delta = {}
        for parent in [41007, 41008, 42007, 42008]:
            scores = [
                next(
                    r["valid"]
                    for r in result["summary"]
                    if r["arm"] == arm and r["carrier_seed"] == parent
                )
                for arm in ["learned_refine", comparator]
            ]
            delta[str(parent)] = scores[0] - scores[1]
        assert result["predicates"]["beats_" + comparator] == {
            "per_test_parent_delta": delta,
            "improves_each_parent": all(d > 0 for d in delta.values()),
        }
    hard = sum(
        r["valid"]
        for r in result["rows"]
        if (r["arm"], r["carrier_seed"], r["event_station"]) == ("learned_refine", 42007, 0.35)
    )
    assert result["predicates"]["hard_case"] == {
        "valid": hard,
        "proposals": 24,
        "gains_any": hard > 0,
    }
    return result


def figure(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout="constrained")
    colors = ["#617393", "#369286", "#a88a42", "#1d69ac", "#a2618d"]
    for ax, parent in zip(axes.flat, [41007, 41008, 42007, 42008]):
        selected = [r for r in result["summary"] if r["carrier_seed"] == parent]
        counts = [next(r["valid"] for r in selected if r["arm"] == a) for a in ARMS]
        ax.bar(range(5), np.array(counts) / 48 * 100, color=colors)
        for i, n in enumerate(counts):
            ax.text(i, n / 48 * 100 + 2, f"{n}/48", ha="center", fontsize=9)
        ax.set_xticks(
            range(5),
            ["Raw", "Rank\nlearned", "Rank\nuniform", "Refine\nlearned", "Refine\nuniform"],
        )
        ax.set_title(f"Test source {parent}")
        ax.set_ylim(0, 113)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Independent audit acceptance (%)")
    fig.suptitle(
        "Can geometric feedback rescue learned proposals?\n"
        "Four assisted methods: 4,624 search queries each; raw has zero search queries"
    )
    fig.savefig(path, format="svg")
    plt.close(fig)


def table(result, split):
    parents = [41007, 41008, 42007, 42008] if split == "test" else [41005, 41006, 42005, 42006]
    lines = [
        "| Method | " + " | ".join(str(p) + " /48" for p in parents) + " | Total /192 |",
        "|---|" + "---:|" * 5,
    ]
    for arm, label in zip(ARMS, LABELS):
        values = [
            next(
                r["valid"] for r in result["summary"] if r["arm"] == arm and r["carrier_seed"] == p
            )
            for p in parents
        ]
        lines.append("| " + label + " | " + " | ".join(map(str, values + [sum(values)])) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    result = verify(args.result)
    outputs = []
    for name, value in [
        ("refinement", result),
        ("refinement-registration", json.loads(Path(result["registration"]["path"]).read_text())),
    ]:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    (args.out / "assets").mkdir(parents=True, exist_ok=True)
    path = args.out / "assets/refinement.svg"
    if path.exists():
        raise FileExistsError(path)
    figure(result, path)
    outputs.append(artifact(path))
    write_new(
        args.out / "assets/refinement-manifest.json",
        {"source": artifact(args.result), "renderer": artifact(Path(__file__)), "outputs": outputs},
    )
    print(table(result, "test") + "\n\nVALIDATION\n" + table(result, "validation"))
    print(json.dumps(result["predicates"], indent=2))


if __name__ == "__main__":
    main()
