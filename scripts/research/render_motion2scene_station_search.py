#!/usr/bin/env python3
"""Verify the complete fixed-budget station-search study and export its evidence."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_refinement_study import measure
from motion2scene_station_study import ARMS, load, predicates
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np
from render_motion2scene_refinement import slack
from render_motion2scene_timing_report import portable

from gear_sonic.dataset_generation.hallucination.motion2scene_station_search import METHODS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABELS = [
    "Raw",
    "Rank learned",
    "Original gradient",
    "Probe then refine",
    "Multiple starts",
    "Pattern search",
]


def verify(path):
    result = json.loads(path.read_text())
    for key in ["registration", "search"]:
        checked(Path(result[key]["path"]), result[key]["sha256"])
    reg, parent, cases, replay = load(Path(result["registration"]["path"]))
    searched = json.loads(Path(result["search"]["path"]).read_text())
    lookup = {(r["panel"], r["arm"], r["seed"], r["case_id"]): r for r in searched["rows"]}
    expected = {("main", a, s, c) for a in ARMS for s in reg["seeds"] for c in cases}
    expected |= {("replay", a, replay["checkpoint_seed"], replay["case_id"]) for a in ARMS}
    assert len(result["rows"]) == len(searched["rows"]) == 294 and set(lookup) == expected
    identities = set()
    for row in result["rows"]:
        key = tuple(row[k] for k in ["panel", "arm", "seed", "case_id"])
        assert key not in identities
        identities.add(key)
        for k, v in lookup[key].items():
            assert row[k] == v
        raw = np.load(checked(Path(row["raw"]["path"]), row["raw"]["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["scenes"], row["scenes"])
        np.testing.assert_array_equal(raw["offsets"], parent["audit_offsets"])
        assert raw["clearances"].shape == (8, 113, 2) and np.isfinite(raw["clearances"]).all()
        assert (
            measure(row, cases[row["case_id"]], raw["clearances"], row["query_error_m"], row["raw"])
            == row
        )
        trace = np.load(
            checked(Path(row["trace"]["path"]), row["trace"]["sha256"]), allow_pickle=False
        )
        initial = lookup[(row["panel"], "raw", row["seed"], row["case_id"])]["scenes"]
        np.testing.assert_array_equal(trace["initial"], initial)
        assert np.isfinite(trace["scenes"]).all()
        assert np.all(trace["scenes"] >= np.array([0.1, 1.1]) - 1e-12)
        assert np.all(trace["scenes"] <= np.array([0.9, 1.45]) + 1e-12)
        if row["arm"] == "raw":
            np.testing.assert_array_equal(trace["scenes"], raw["scenes"])
            continue
        assert trace["clearances"].size == row["queries"] == 4624
        if row["arm"] == "rank":
            selected = np.argsort(-slack(trace["clearances"]), kind="stable")[:8]
            output = trace["scenes"][selected]
        else:
            assert trace["scenes"].shape == (17, 8, 2)
            assert np.all(abs(trace["scenes"] - initial) <= np.array([0.05, 0.03]) + 1e-12)
            selected = np.argmax(slack(trace["clearances"]), axis=0)
            output = trace["scenes"][selected, np.arange(8)]
        np.testing.assert_array_equal(trace["selected"], selected)
        np.testing.assert_array_equal(output, raw["scenes"])
    assert identities == expected
    grads = {
        (r["panel"], r["seed"], r["case_id"]): np.array(r["per_proposal_valid"])
        for r in result["rows"]
        if r["arm"] == "gradient"
    }
    for r in result["rows"]:
        if r["arm"] in METHODS:
            before = grads[(r["panel"], r["seed"], r["case_id"])]
            after = np.array(r["per_proposal_valid"])
            assert r["rescued_gradient"] == int((after & ~before).sum())
            assert r["lost_gradient"] == int((~after & before).sum())
    summary, outcomes = predicates(result["rows"])
    assert summary == result["summary"] and outcomes == result["predicates"]
    gradient = next(r for r in result["rows"] if r["panel"] == "replay" and r["arm"] == "gradient")
    assert gradient["valid"] == replay["accepted_count"] == 5
    np.testing.assert_allclose(gradient["scenes"], replay["refined_scenes"], rtol=0, atol=1e-12)
    return result


def table(result, split):
    parents = [41007, 41008, 42007, 42008] if split == "test" else [41005, 41006, 42005, 42006]
    lines = [
        "| Method | " + " | ".join(str(p) + " /48" for p in parents) + " | Total /192 |",
        "|---|" + "---:|" * 5,
    ]
    for arm, label in zip(ARMS, LABELS):
        scores = [
            next(
                r["valid"] for r in result["summary"] if r["arm"] == arm and r["carrier_seed"] == p
            )
            for p in parents
        ]
        lines.append("| " + label + " | " + " | ".join(map(str, scores + [sum(scores)])) + " |")
    return "\n".join(lines)


def plot(result, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True, layout="constrained")
    for ax, parent in zip(axes.flat, [41007, 41008, 42007, 42008]):
        counts = [
            next(
                r["valid"]
                for r in result["summary"]
                if r["arm"] == a and r["carrier_seed"] == parent
            )
            for a in ARMS
        ]
        ax.bar(
            range(6),
            np.array(counts) * 100 / 48,
            color=["#8793a2", "#418678", "#2772b8", "#af823c", "#987296", "#27604d"],
        )
        for i, n in enumerate(counts):
            ax.text(i, n * 100 / 48 + 2, f"{n}/48", ha="center", fontsize=9)
        ax.set_xticks(
            range(6), ["Raw", "Rank", "Gradient", "Probe +\nrefine", "Multiple\nstarts", "Pattern"]
        )
        ax.set_title(f"Test source {parent}")
        ax.set_ylim(0, 113)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Valid fresh-draw outputs (%)")
    fig.suptitle(
        "Exploring station positions at a fixed query budget\n"
        "Fresh draws on observed sources; known-failure replay excluded"
    )
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    result = verify(args.result)
    outputs = []
    for name, value in [
        ("station-search", result),
        (
            "station-search-registration",
            json.loads(Path(result["registration"]["path"]).read_text()),
        ),
    ]:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    (args.out / "assets").mkdir(parents=True, exist_ok=True)
    path = args.out / "assets/station-search.svg"
    if path.exists():
        raise FileExistsError(path)
    plot(result, path)
    outputs.append(artifact(path))
    write_new(
        args.out / "assets/station-search-manifest.json",
        {"source": artifact(args.result), "renderer": artifact(Path(__file__)), "outputs": outputs},
    )
    print(table(result, "test") + "\nVALIDATION\n" + table(result, "validation"))
    print(json.dumps(result["predicates"], indent=2))


if __name__ == "__main__":
    main()
