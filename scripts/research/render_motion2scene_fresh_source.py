#!/usr/bin/env python3
"""Verify and export the complete fresh-source acquisition and optional inference audit."""

import argparse
import json
from pathlib import Path

import matplotlib
from motion2scene_fresh_audit import ARMS, load, predicates, uniform_initial
from motion2scene_fresh_sources import SOURCE_IDS, registration, validate_generation
from motion2scene_refinement_study import measure
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
    "Learned pattern",
    "Uniform pattern",
]


def verify(path):
    result = json.loads(path.read_text())
    for key in ["registration", "search"]:
        checked(Path(result[key]["path"]), result[key]["sha256"])
    reg, parent, cases = load(Path(result["registration"]["path"]))
    searched = json.loads(Path(result["search"]["path"]).read_text())
    lookup = {(r["panel"], r["arm"], r["seed"], r["case_id"]): r for r in searched["rows"]}
    expected = {("main", a, s, c) for a in ARMS for s in reg["seeds"] for c in cases}
    assert len(result["rows"]) == len(searched["rows"]) == 336 and set(lookup) == expected
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
        if row["arm"] == "uniform_pattern":
            initial = uniform_initial(row["seed"] + reg["uniform_seed_offset"]).numpy()
        assert row["draw_seed"] == row["seed"] + reg["main_draw_seed_offset"]
        assert row["uniform_seed"] == row["seed"] + reg["uniform_seed_offset"]
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
    assert result["search_queries"] == sum(r["queries"] for r in result["rows"]) == 1331712
    assert result["audit_queries"] == 336 * 8 * 113 * 2
    assert result["crosscheck_queries"] == 336 * 2 * 113 * 2
    assert result["max_query_error_m"] == max(r["query_error_m"] for r in result["rows"])
    assert np.isfinite(result["max_query_error_m"]) and result["max_query_error_m"] <= 1e-8
    assert result["search_seconds"] == searched["elapsed_seconds"]
    for key in ["preprocessing_seconds", "checkpoint_setup_seconds", "inherited_training_seconds"]:
        assert result[key] == searched[key]
    assert result["sampling_seconds"] == sum(
        r["sampling_seconds"] for r in result["rows"] if r["arm"] == "raw"
    )
    assert result["uniform_sampling_seconds"] == sum(
        r["uniform_sampling_seconds"] for r in result["rows"] if r["arm"] == "uniform_pattern"
    )
    return result


def table(result):
    lines = [
        "| Method | " + " | ".join(str(p) + " /48" for p in SOURCE_IDS) + " | Total /384 |",
        "|---|" + "---:|" * 9,
    ]
    for arm, label in zip(ARMS, LABELS):
        scores = [
            next(
                r["valid"] for r in result["summary"] if r["arm"] == arm and r["carrier_seed"] == p
            )
            for p in SOURCE_IDS
        ]
        lines.append("| " + label + " | " + " | ".join(map(str, scores + [sum(scores)])) + " |")
    return "\n".join(lines)


def acquisition(root):
    reg, parent = registration(root / "registration.json")
    generated = json.loads((root / "generation_result.json").read_text())
    for ref in [generated["launch"], generated["log"], *generated["retained_outputs"]]:
        checked(Path(ref["path"]), ref["sha256"])
    launch = json.loads(Path(generated["launch"]["path"]).read_text())
    checked(root / "registration.json", launch["registration"]["sha256"])
    result = {
        "registration": artifact(root / "registration.json"),
        "generation_record": artifact(root / "generation_result.json"),
        "generation": generated,
        "source_candidates": 8,
        "target_candidates": 16,
        "training_eligible": False,
        "execution_eligible": False,
    }
    if not generated["complete"]:
        result.update(status="generation_failed", qualified_targets=0, predictions="unassessed")
        return result
    assert validate_generation(root / "generated", reg) == generated["rows"]
    old_manifest = json.loads(Path(parent["bank"]["path"]).read_text())
    old_sources = json.loads(Path(old_manifest["registration"]["path"]).read_text())
    old_hashes = {r["csv_sha256"] for r in old_sources["base_carriers"]}
    new_hashes = {r["csv"]["sha256"] for r in generated["rows"]}
    result["source_content_check"] = {
        "prior_bank": parent["bank"],
        "unique_new_csv_hashes": len(new_hashes),
        "matching_prior_csv_hashes": len(new_hashes & old_hashes),
        "role": "descriptive exact-file check, not an additional acquisition gate",
    }
    manifest = json.loads((root / "manifest.json").read_text())
    for ref in manifest.values():
        if isinstance(ref, dict) and "sha256" in ref:
            checked(Path(ref["path"]), ref["sha256"])
    registry = json.loads(Path(manifest["registry"]["path"]).read_text())
    assert [s["generation_seed"] for s in registry["sources"]] == SOURCE_IDS
    rows = registry["rows"]
    assert len(rows) == 16 and {r["case_id"] for r in rows} == {
        f"{s}_event{j}" for s in SOURCE_IDS for j in range(2)
    }
    for r in rows:
        for key in ["parent", "target"]:
            if key in r:
                checked(Path(r[key]["path"]), r[key]["sha256"])
        if "failure" not in r:
            expected = (
                r["endpoints_preserved"]
                and all(
                    r[g][k] for g in ["neutral_gate", "target_gate"] for k in ["q0_pass", "q1_pass"]
                )
                and r["neutral_route"]["validity_class"]
                == r["target_route"]["validity_class"]
                == "valid_straight"
            )
            assert r["qualified"] == expected
        else:
            assert not r["qualified"]
    assert registry["qualified_targets"] == sum(r["qualified"] for r in rows)
    assert manifest["reference_gates_pass"] == all(r["qualified"] for r in rows)
    result.update(
        manifest=artifact(root / "manifest.json"),
        registry=registry,
        qualified_targets=registry["qualified_targets"],
        status="qualified" if manifest["reference_gates_pass"] else "reference_gate_failed",
        predictions="pending" if manifest["reference_gates_pass"] else "unassessed",
    )
    return result


def plot(result, acq, path):
    if result is None:
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        rows = acq.get("registry", {}).get("rows", [])
        counts = [sum(r["qualified"] for r in rows if r["carrier_seed"] == s) for s in SOURCE_IDS]
        ax.bar([str(s) for s in SOURCE_IDS], counts, color="#418678")
        ax.set_ylim(0, 2.3)
        ax.set_yticks([0, 1, 2])
        ax.set_ylabel("Qualified derivatives / 2")
        ax.set_title("Fresh-source acquisition stopped: inference predictions unassessed")
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), layout="constrained")
        x = np.arange(8)
        for k, (arm, label, color) in enumerate(
            [
                ("gradient", "Original gradient", "#2772b8"),
                ("pattern", "Learned pattern", "#27604d"),
                ("uniform_pattern", "Uniform pattern", "#af823c"),
            ]
        ):
            counts = [
                next(
                    r["valid"]
                    for r in result["summary"]
                    if r["arm"] == arm and r["carrier_seed"] == s
                )
                for s in SOURCE_IDS
            ]
            bars = axes[0].bar(x + (k - 1) * 0.25, counts, width=0.24, label=label, color=color)
            axes[0].bar_label(bars, fontsize=8)
        axes[0].set_xticks(x, [str(s) for s in SOURCE_IDS])
        axes[0].set_ylim(0, 58)
        axes[0].set_ylabel("Accepted outputs / 48")
        axes[0].legend(loc="upper right", ncols=3, fontsize=9)
        axes[0].set_title("Eight new motion sources; fixed checkpoints and search budgets")
        costs = [sum(r["search_seconds"] for r in result["rows"] if r["arm"] == a) for a in ARMS]
        bars = axes[1].bar(np.arange(7), costs, color="#418678")
        axes[1].bar_label(bars, fmt="%.1f s", fontsize=9)
        axes[1].set_xticks(np.arange(7), LABELS, rotation=12)
        axes[1].set_ylim(0, max(costs) * 1.25)
        axes[1].set_ylabel("Search time across 48 jobs (s)")
        axes[1].set_title(
            "Each assisted arm: 4,624 search queries per job; setup and audit excluded"
        )
        for ax in axes:
            ax.grid(axis="y", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "docs/motion2scene")
    args = parser.parse_args()
    acq = acquisition(args.root)
    result = verify(args.root / "result.json") if acq["status"] == "qualified" else None
    if result is not None:
        acq["predictions"] = result["predicates"]
    else:
        assert not (args.root / "search").exists(), "inference ran on failed acquisition"
    outputs = []
    exports = [
        ("fresh-source-acquisition", acq),
        ("fresh-source-registration", json.loads((args.root / "registration.json").read_text())),
        ("fresh-source-inventory", json.loads((args.root / "inventory.json").read_text())),
    ]
    if result is not None:
        exports.append(("fresh-source", result))
    for name, value in exports:
        path = args.out / "evidence" / (name + ".json")
        write_new(path, portable(value))
        outputs.append(artifact(path))
    path = args.out / "assets/fresh-source.svg"
    if path.exists():
        raise FileExistsError(path)
    plot(result, acq, path)
    outputs.append(artifact(path))
    write_new(
        args.out / "assets/fresh-source-manifest.json",
        {
            "source": artifact(args.root / ("result.json" if result else "generation_result.json")),
            "acquisition": artifact(args.root / "generation_result.json"),
            "renderer": artifact(Path(__file__)),
            "outputs": outputs,
        },
    )
    print(
        table(result)
        if result
        else json.dumps({k: acq[k] for k in ["status", "qualified_targets", "predictions"]})
    )
    if result:
        print(json.dumps(result["predicates"], indent=2))


if __name__ == "__main__":
    main()
