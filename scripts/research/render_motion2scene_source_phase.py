#!/usr/bin/env python3
"""Verify and export the registered source/phase experiment."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_source_phase import SEEN, load, summarize
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
from motion2scene_uncertainty_learning import verdict
import numpy as np
from render_motion2scene_timing_report import portable

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def verify(path):
    result = json.loads(path.read_text())
    for key in ("registration", "manifest", "runs"):
        ref = result[key]
        checked(Path(ref["path"]), ref["sha256"])
    reg, cases = load(Path(result["manifest"]["path"]))
    complete = json.loads(Path(result["runs"]["path"]).read_text())
    cells = {}
    for ref in complete["cells"]:
        cell = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        for key in ("checkpoint", "samples"):
            checked(Path(cell[key]["path"]), cell[key]["sha256"])
        cells[(cell["arm"], cell["seed"], cell["budget"])] = {
            p["case_id"]: p["scenes"]
            for p in json.loads(Path(cell["samples"]["path"]).read_text())["predictions"]
        }
    if len(cells) != 24 or len(result["rows"]) != 960:
        raise ValueError("incomplete result")
    identities = set()
    for row in result["rows"]:
        key = (row["arm"], row["seed"], row["budget"])
        identity = (*key, row["case_id"])
        if identity in identities:
            raise ValueError("duplicate audit row")
        identities.add(identity)
        raw = np.load(checked(Path(row["raw"]["path"]), row["raw"]["sha256"]), allow_pickle=False)
        np.testing.assert_array_equal(raw["scenes"], cells[key][row["case_id"]])
        np.testing.assert_array_equal(raw["scenes"], row["scenes"])
        np.testing.assert_array_equal(raw["offsets"], reg["evaluation_offsets"])
        values = raw["clearances"]
        if values.shape != (8, 113, 2) or not np.isfinite(values).all():
            raise ValueError("invalid raw query array")
        meta = cases[row["case_id"]]["metadata"]
        assert row["carrier_seed"] == meta["carrier_seed"]
        assert row["split"] == meta["split"] != "train"
        assert row["event_station"] == meta["event_station"]
        assert row["phase"] == ("seen" if meta["event_station"] in SEEN else "unseen")
        mask = cases[row["case_id"]]["mask"].numpy()
        assert row["valid"] == int(verdict(values, mask).all(axis=1).sum())
        assert row["target_failures"] == int((values[:, :, 1].min(axis=1) < 0.01).sum())
        assert row["neutral_failures"] == int((values[:, :, 0].max(axis=1) > -0.01).sum())
        assert row["proposals"] == 8
    assert summarize(result["rows"]) == result["summary"]
    for label, new_budget, old_budget in [
        ("equal_queries", 600, 600),
        ("equal_visits", 1200, 600),
        ("extra_update_control", 1200, 1200),
    ]:
        differences = {}
        for parent in [41007, 41008, 42007, 42008]:
            counts = []
            for arm, budget in [("all8", new_budget), ("base4", old_budget)]:
                counts.append(
                    next(
                        r["valid"]
                        for r in result["summary"]
                        if (r["arm"], r["budget"], r["carrier_seed"], r["phase"])
                        == (arm, budget, parent, "unseen")
                    )
                )
            differences[str(parent)] = counts[0] - counts[1]
        assert result["predicates"][label] == {
            "unseen_test_per_parent_delta": differences,
            "improves_each_parent": all(d > 0 for d in differences.values()),
        }
    return result


def figure(result, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True, layout="constrained")
    parents = [41007, 41008, 42007, 42008]
    for ax, title, budgets in zip(
        axes,
        ["Equal queries: 600 updates", "Equal visits: 50 per case"],
        [[600, 600, 600, 600], [600, 900, 900, 1200]],
    ):
        for parent, color in zip(parents, ["#087f73", "#316ec4", "#b1762c", "#a35086"]):
            values = []
            for arm, budget in zip(["base4", "add_a6", "add_b6", "all8"], budgets):
                row = next(
                    r
                    for r in result["summary"]
                    if (r["arm"], r["budget"], r["carrier_seed"], r["phase"])
                    == (arm, budget, parent, "unseen")
                )
                values.append(100 * row["valid"] / row["proposals"])
            ax.plot(range(4), values, marker="o", lw=2, color=color, label=str(parent))
        ax.set_xticks(range(4), ["4 parents", "6 (A)", "6 (B)", "8 parents"])
        ax.set_title(title)
        ax.set_ylim(-3, 103)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Valid proposals on unseen phases (%)")
    axes[1].legend(title="Excluded test parent", fontsize=8)
    fig.suptitle(
        "Source diversity × new event locations\n"
        "48 fixed proposals per parent and cell; development references only"
    )
    fig.savefig(out, format="svg")
    plt.close(fig)


def table(result, split):
    parents = (
        [41005, 41006, 42005, 42006] if split == "validation" else [41007, 41008, 42007, 42008]
    )
    lines = [
        "| Arm / updates | "
        + " | ".join(str(p) + " unseen /48" for p in parents)
        + " | Seen /288 | Unseen /192 |",
        "|---|" + "---:|" * 6,
    ]
    for arm, budget in [
        ("base4", 600),
        ("add_a6", 600),
        ("add_b6", 600),
        ("all8", 600),
        ("base4", 1200),
        ("add_a6", 900),
        ("add_b6", 900),
        ("all8", 1200),
    ]:
        selected = [
            r
            for r in result["summary"]
            if r["arm"] == arm and r["budget"] == budget and r["split"] == split
        ]
        scores = [
            next(r["valid"] for r in selected if r["carrier_seed"] == p and r["phase"] == "unseen")
            for p in parents
        ]
        scores += [
            sum(r["valid"] for r in selected if r["phase"] == phase) for phase in ("seen", "unseen")
        ]
        lines.append(f"| {arm} / {budget} | " + " | ".join(map(str, scores)) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--out", default=ROOT / "docs/motion2scene", type=Path)
    args = parser.parse_args()
    result = verify(args.result)
    (args.out / "assets").mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, value in [
        ("source-phase", result),
        ("source-phase-registration", json.loads(Path(result["registration"]["path"]).read_text())),
        (
            "source-phase-registry",
            json.loads((args.result.parent / "carrier_registry.json").read_text()),
        ),
    ]:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    plot = args.out / "assets/source-phase.svg"
    if plot.exists():
        raise FileExistsError(plot)
    figure(result, plot)
    outputs.append(artifact(plot))
    write_new(
        args.out / "assets/source-phase-manifest.json",
        {"source": artifact(args.result), "renderer": artifact(Path(__file__)), "outputs": outputs},
    )
    print("TEST\n" + table(result, "test") + "\nVALIDATION\n" + table(result, "validation"))
    print(json.dumps(result["predicates"], indent=2))


if __name__ == "__main__":
    main()
