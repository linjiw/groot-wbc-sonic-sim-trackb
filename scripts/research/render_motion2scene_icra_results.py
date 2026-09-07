#!/usr/bin/env python3
# ruff: noqa: E501 -- Report prose and table text.
"""Publish completed labeling and a dated partial policy panel without extrapolating it."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import numpy as np

DATA = ROOT.parent / "research-data/groot-wbc"
DOC = ROOT / "docs/motion2scene"
STUDY = DATA / "m2s-icra-v1"
LEARN = DATA / "m2s-icra-learning-v1"
METHODS = [
    "uniform",
    "analytic",
    "no_contrast",
    "motion2scene",
    "scripted_rays",
    "privileged_geometry",
]
NAMES = [
    "Uniform",
    "Analytic",
    "Target-only",
    "M2S: background only",
    "Scripted rays",
    "Privileged geometry",
]


def main():
    # Use the newest summary rather than a pinned snapshot, so the final panel renders
    # with the same code. The narrative assertions below fail loudly if the data stops
    # supporting the prose, which then has to be rewritten rather than silently reused.
    comparison = max(LEARN.glob("comparison_*.json"), key=lambda q: int(q.stem.split("_")[1]))
    c = json.loads(comparison.read_text())
    fit = json.loads((LEARN / "fit.json").read_text())
    done = c["completed"]
    conditions = len({(r["source"], r["layout"], r["physics_seed"]) for r in c["rows"]})
    assert c["completed"] + c["pending"] == c["assigned"]
    public = DOC / f"evidence/icra-results-{done}-20260907"
    public.mkdir(exist_ok=False)
    exports = []

    def export(source, name):
        raw = source.read_bytes()
        dest = public / name
        if source.suffix not in (".npz", ".pt"):
            raw = (
                raw.decode()
                .replace(str(DATA), "research-data")
                .replace(str(ROOT), "repository")
                .encode()
            )
        dest.write_bytes(raw)
        exports.append(
            {"source": artifact(source), "public": name, "public_sha256": artifact(dest)["sha256"]}
        )

    for source in [
        LEARN / "fit.json",
        comparison,
        LEARN / f"mechanism_{done}.json",
        LEARN / "evaluation_master.json",
    ]:
        export(source, source.name)
    for source in sorted((LEARN / "models").iterdir()):
        export(source, source.name)
    costs = {"labels": 0.0, "evaluation": 0.0}
    for name, index in [
        ("labels", STUDY / "prepared.json"),
        ("evaluation", LEARN / "evaluation_master.json"),
    ]:
        for b in json.loads(index.read_text())["batches"]:
            folder = Path(b["directory"])
            ap = folder / "admission.json"
            if not ap.exists():
                continue
            a = json.loads(ap.read_text())
            assert a["admitted"]
            result = json.loads(
                checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text()
            )
            costs[name] += result["actual_contended_gpu_hours"]
            for file in ["admission.json", "result.json", "run_record.json"]:
                export(folder / file, folder.name + "-" + file)
    (public / "exports.json").write_text(
        json.dumps(exports, indent=2)
        .replace(str(DATA), "research-data")
        .replace(str(ROOT), "repository")
        + "\n"
    )
    folds = [f for f in fit["folds"] if f["arm"] == "analytic"]
    m = next(f for f in fit["fits"] if f["arm"] == "motion2scene")
    with np.load(m["linear"]["path"]) as a, np.load(folds[0]["model"]["path"]) as b:
        equality = {k: bool(np.array_equal(a[k], b[k])) for k in a.files}
    assert all(equality.values()) and m["training_ids"] == folds[0]["train_ids"]
    (public / "contrast-removal-check.json").write_text(
        json.dumps(
            {
                "arrays_equal": equality,
                "training_ids_equal": True,
                "primary": m["linear"],
                "fold": folds[0]["model"],
                "scope": "Existing registered refit; no new fitting or policy rollout",
            },
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 7.8), constrained_layout=True)
    cmap = colors.ListedColormap(["#dadeda", "#b66a59", "#25817c"])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)
    for ax, source in zip(axes, [41001, 41002, 41003]):
        values = np.full((6, 24), -1.0)
        for r in c["rows"]:
            if r["source"] != source:
                continue
            j = 2 * int(r["layout"].split("_")[1]) + (r["physics_seed"] == 8512)
            i = METHODS.index(r["arm"])
            values[i, j] = int(r["pass"])
            if r["readout"]["requested_action"] == 1:
                ax.text(j, i, "D", ha="center", va="center", fontsize=8, color="white")
        ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(range(6), NAMES, fontsize=9)
        ax.set_xticks(range(0, 24, 2), [f"L{i:02d}" for i in range(12)], fontsize=9)
        ax.set_title(f"Carrier {source}: every assigned layout/seed shown", loc="left", fontsize=11)
    fig.suptitle(
        f"{c['traversal_completed']}/{c['traversal_assigned']} traversal executions admitted; "
        f"{c['pending']} runs pending\n"
        "Green: task pass | red: task fail | gray: pending | D: actual d040 request",
        fontsize=13,
    )
    fig.savefig(DOC / f"assets/icra-{done}-outcomes.png", dpi=160)
    fig.savefig(DOC / f"assets/icra-{done}-outcomes.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    x = np.arange(4)
    generated = [len([i for i in f["training_ids"] if "shared" not in i]) for f in fit["fits"]]
    useful = [f["outcome_counts"].get("(False, True)", 0) for f in fit["fits"]]
    # The prose below states one useful contrast overall, from analytic construction.
    assert sum(useful) == 1 and useful[METHODS.index("analytic")] == 1, useful
    axs[0].bar(x - 0.22, [18] * 4, 0.22, label="Requested generated")
    axs[0].bar(x, generated, 0.22, label="Complete generated labels")
    axs[0].bar(x + 0.22, useful, 0.22, label="Useful physical contrasts")
    axs[0].set_xticks(x, ["Uniform", "Analytic", "Target-only", "M2S"])
    axs[0].set_ylim(0, 22)
    axs[0].legend(fontsize=8)
    axs[0].set_title("Acquisition: all 82 commands completed")
    rates = [np.mean(list(c["per_arm"][a]["carrier_passage"].values())) * 100 for a in METHODS]
    axs[1].barh(
        range(6), rates, color=["#8d9898", "#27817c", "#8d9898", "#8d9898", "#bd8740", "#7c94a6"]
    )
    axs[1].set_yticks(range(6), NAMES, fontsize=9)
    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 80)
    for i, a in enumerate(METHODS):
        axs[1].text(
            rates[i] + 1, i, f"{c['per_arm'][a]['pass']}/{conditions}", va="center", fontsize=9
        )
    axs[1].set_xlabel("Carrier-averaged passage (%)")
    axs[1].set_title("Partial panel; unequal training counts")
    fig.savefig(DOC / f"assets/icra-{done}-yield.png", dpi=160)
    fig.savefig(DOC / f"assets/icra-{done}-yield.pdf")
    plt.close(fig)
    table = "\n".join(
        f"| {name} | {c['per_arm'][arm]['pass']}/{conditions} | "
        f"{c['per_arm'][arm]['d040_requests']}/{conditions} | "
        f"{c['per_arm'][arm]['refusals']}/{conditions} |"
        for arm, name in zip(METHODS, NAMES)
    )
    corpus = "\n".join(
        f"| {f['arm']} | {len(f['training_ids'])}/24 | {f['outcome_counts']['(True, True)']} | {f['outcome_counts']['(False, True)']} | {f['outcome_counts']['(True, False)']} | {f['outcome_counts']['(False, False)']} |"
        for f in fit["fits"]
    )
    report = f"""# Completed paired labels and the first 366 fixed-policy evaluations

**September 7 snapshot:** all **82/82 labeling runs** complete and all 41 unique
encounter pairs pass measurement admission. Four primary deterministic learners and
24 prescribed leave-four-out refits complete. **366/540 evaluation runs** are admitted:
61 of 72 traversal conditions, each with all six policies. The remaining 66 traversal
and 108 background runs are pending. This is an ordered partial panel, not completion.

## Data quality and the failed acquisition quota

| Data arm | Complete / requested groups | Both pass | d040 only | Walk only | Both fail |
| --- | --- | --- | --- | --- | --- |
{corpus}

The six shared background pairs are counted in each arm's fitting set but acquired
once. Generated groups are uniform 18/18, analytic 1/18, target-only 16/18 and M2S 0/18.
Only the analytic generated group is physically useful. Uniform's generated pairs
contain six both-pass and twelve both-fail cases; all sixteen target-only generated
pairs are both-pass. Thus the registered uniform <=1 useful and target-only >=12/18
both-pass predictions hold, while the prediction of at least one useful generated
contrast per carrier for analytic and M2S fails. Three carriers qualify, but useful
contrast acquisition remains confined to one analytic scene on 41001.

The equal-24-complete-label goal fails in three arms. These are outcomes at equal
requested slots with unequal acquired data and costs. **The M2S-arm model is trained
only on shared backgrounds.** Its later performance cannot establish the quality of
accepted M2S-generated training scenes, because no such scenes were acquired.

## Actual policy executions, with paired conditions preserved

| Policy / data arm | Passage | d040 requests | Refusals followed by walking |
| --- | --- | --- | --- |
{table}

Analytic versus uniform has 21 both-pass, nine analytic-only, zero uniform-only and
31 both-fail conditions. The carrier-averaged difference is +14.603 percentage points
on this completed slice. Per-carrier extra passes are five on 41001 and two each on
41002/41003. The evaluated denominator is 21 conditions for 41001 and twenty each for
the other carriers; repeats and layout variants do not create independent ancestors.
All nine analytic-only passages requested d040 with 0 N measured beam force; their
walking comparators record 385.1–1685.6 N. These are matched policy executions under
the recorded-state and source-bank audit, not counterfactual outcome predictions.

The script passes six conditions that analytic misses, with no reverse difference.
The privileged forecast and analytic disagree in both directions (three analytic-only,
two privileged-only). Privileged means a known-scene achieved-capsule forecast, not
an optimal transition-time oracle. The script's 36/61 is therefore a stronger observed
baseline than the analytic learner in this slice. Uniform, target-only and background-
only M2S all execute walking, although uniform also reports 24 refusals. Three of those
refusals subsequently pass by walking; they are not protective stopping behavior.

Thirty-eight conditions include actual executions of both commands across the six
policies. Their outcomes agree whenever the issued command agrees. Fifteen of those
conditions show d040-only success, and none shows walking-only success. The other
23 conditions have walking observations only; their unexecuted d040 outcomes remain
unknown. Do not infer that d040 could rescue every failed walk.

## One useful contrast explains the adaptation requests in the registered refits

The analytic fitting set is the six shared backgrounds plus one useful generated
contrast. Its first leave-four-assignment-out fold removes that one available label
and three already refused assignments. The remaining six training IDs and all weight,
bias and scale arrays equal the background-only M2S primary fit exactly. On the 61
recorded inputs, this refit requests d040 zero times; the full analytic fit requests
it 22 times. All five other analytic folds retain the same 22 requests. Two folds
remove no available labels and are explicitly counted as such.

This is a controlled data-removal diagnostic of the fixed learner. It is not a new
physical evaluation of the fold models, and it does not establish reliable learning
from one example in a population of sources. It does connect one physical contrast
to changed learned requests, while the main admitted policy runs measure nine passage
improvements. [Exact equality check](evidence/icra-results-366-20260907/contrast-removal-check.json).

## Cost, current limit and remaining experiments

Label acquisition costs {costs['labels']:.6f} contended GPU h. The 366 admitted
policy executions cost {costs['evaluation']:.6f} h. These exclude separately recorded
generator/teacher costs and shared bank acquisition; no complete end-to-end cost claim
is made from physics time alone. Full query records and shared prior costs remain in
the acquisition registration and source-bank record.

The next six-cell batch reserves 0.625 GPU h. At the budget check, rolling daily use
is 7.409639 h, so 8.034639 h would exceed the standing 8 h/day envelope. The supervisor
correctly waits; it has neither lowered the reservation nor counted the pending 174
runs as failures. The first relevant cost expires at approximately 14:39:32 UTC on
September 7. Later batches remain subject to fresh rolling-window checks.

Finish the unchanged 174 evaluation assignments, including all background controls,
before the final paired result and September 10 claim decision. The current finding
is analytic data benefit on an inspected partial panel and learned-construction
acquisition failure. It establishes neither outcome A nor outcome B from the guidance,
both of which expected M2S to outperform untargeted data. The manuscript must retain
this distinction and the unequal-data shortfall.

[All outcomes](assets/icra-366-outcomes.pdf) · [Yield and passage](assets/icra-366-yield.pdf) ·
[Full records and 28 saved primary/refit models](evidence/icra-results-366-20260907/exports.json) ·
[Working manuscript](ICRA_MANUSCRIPT.pdf).
"""
    (DOC / "M2S_ICRA_366_RESULT.md").write_text(report)
    print(json.dumps({"exports": len(exports), "costs": costs, "completed": 366}))


if __name__ == "__main__":
    main()
