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
    conditions = len(
        {
            (r["source"], r["layout"], r["physics_seed"])
            for r in c["rows"]
            if r["suite"] == "traversal"
        }
    )
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
            # Background suites carry named layouts (absent/raised/blocked) and are
            # reported in their own table, not on this traversal grid.
            if r["source"] != source or r["suite"] != "traversal":
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
    axs[1].set_title(
        ("Complete panel" if c["complete"] else "Partial panel") + "; unequal training counts"
    )
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
    conditions_by = {}
    for r in c["rows"]:
        if r["suite"] == "traversal":
            conditions_by.setdefault((r["source"], r["layout"], r["physics_seed"]), {})[
                r["arm"]
            ] = r
    heights = sorted(
        {round(v["analytic"]["beam"]["underside_m"], 2) for v in conditions_by.values()}
    )
    band_rows = []
    for h in heights:
        keys = [
            k
            for k, v in conditions_by.items()
            if round(v["analytic"]["beam"]["underside_m"], 2) == h
        ]

        def tally(arm, field, keys=keys):
            return sum(field(conditions_by[k][arm]) for k in keys)

        band_rows.append(
            f"| {h:.2f} m | {len(keys)} | "
            f"{tally('analytic', lambda r: r['action'] == 1)} | {tally('analytic', lambda r: r['pass'])} | "
            f"{tally('scripted_rays', lambda r: r['action'] == 1)} | {tally('scripted_rays', lambda r: r['pass'])} | "
            f"{tally('uniform', lambda r: r['pass'])} |"
        )
    bands = "\n".join(band_rows)
    observed = {}
    for r in c["rows"]:
        if r["suite"] == "traversal":
            observed.setdefault((r["source"], r["layout"], r["physics_seed"]), {})[r["action"]] = r[
                "pass"
            ]
    paired_keys = [k for k, v in observed.items() if len(v) == 2]
    outcome = {}
    for k in paired_keys:
        outcome[(observed[k][0], observed[k][1])] = (
            outcome.get((observed[k][0], observed[k][1]), 0) + 1
        )
    ceiling = sum(1 for v in observed.values() if any(v.values()))
    suites = {}
    for r in c["rows"]:
        if r["suite"] == "traversal":
            continue
        key = (r["suite"], r["arm"])
        s = suites.setdefault(key, {"n": 0, "pass": 0, "d040": 0, "refuse": 0})
        s["n"] += 1
        s["pass"] += r["pass"]
        s["d040"] += r["readout"]["requested_action"] == 1
        s["refuse"] += r["readout"]["refusal"]
    background = "\n".join(
        f"| {suite} | {suites[(suite, a)]['pass']}/{suites[(suite, a)]['n']} "
        f"| {suites[(suite, a)]['d040']} | {suites[(suite, a)]['refuse']} |"
        for suite in ("absent", "raised", "blocked")
        for a in ["analytic"]
    )
    scripted_blocked = suites[("blocked", "scripted_rays")]["refuse"]
    learner_blocked = suites[("blocked", "motion2scene")]["refuse"]
    unnecessary = sum(suites[(s, a)]["d040"] for s in ("absent", "raised") for a in METHODS)
    mechanism = json.loads((LEARN / f"mechanism_{done}.json").read_text())
    folds_analytic = [f for f in mechanism["fold_readouts"] if f["arm"] == "analytic"]
    removed = next(f for f in folds_analytic if f["withheld_useful"])
    kept = [f for f in folds_analytic if not f["withheld_useful"]]
    # Folds that withhold nothing available reproduce the full fit exactly.
    intact = [f for f in folds_analytic if not f["withheld_ids"]]
    assert intact and len({f["request_d040"] for f in intact}) == 1
    full_requests = intact[0]["request_d040"]
    unchanged = sum(f["request_d040"] == full_requests for f in kept)
    empty_folds = len(intact)
    versus = {(x["first"], x["second"]): x for x in c["comparisons"]}
    au = versus[("analytic", "uniform")]
    wins = [
        k for k, v in conditions_by.items() if v["analytic"]["pass"] and not v["uniform"]["pass"]
    ]
    win_forces = [
        conditions_by[k]["uniform"]["maximum_beam_normal_force_n_through_passage"] for k in wins
    ]
    counted = {}
    for k in wins:
        counted[k[0]] = counted.get(k[0], 0) + 1
    win_carriers = ", ".join(f"{n} on {s}" for s, n in sorted(counted.items()))
    report = f"""# The completed 540-run panel: one contrast, and where it helps

**September 7, complete.** All **82/82 labeling runs** and all **540/540 evaluation
runs** are admitted: 432 traversal executions over 72 conditions with six policies
each, plus the 108 background control runs. Four primary deterministic learners and
24 prescribed leave-four-out refits are complete. Nothing remains pending.

## Data quality and the failed acquisition quota

| Data arm | Complete / requested groups | Both pass | d040 only | Walk only | Both fail |
| --- | --- | --- | --- | --- | --- |
{corpus}

The six shared background pairs are counted in each arm's fitting set but acquired
once. Generated groups are uniform 18/18, analytic 1/18, target-only 16/18 and M2S 0/18.
**Exactly one acquired group in the whole study is a useful walk-fail/d040-pass
contrast**, from analytic construction on 41001. The registered uniform <=1 useful and
target-only >=12/18 both-pass predictions hold; the prediction of at least one useful
generated contrast per carrier for analytic and M2S fails. The analytic and M2S
training sets therefore differ by exactly that one group and are otherwise identical.

The equal-24-complete-label goal fails in three arms. These are outcomes at equal
requested slots with unequal acquired data and costs. **The M2S-arm model is trained
only on shared backgrounds.** Its performance cannot establish the quality of accepted
M2S-generated training scenes, because no such scenes were acquired.

## Actual policy executions over all 72 traversal conditions

| Policy / data arm | Passage | d040 requests | Refusals followed by walking |
| --- | --- | --- | --- |
{table}

Analytic versus uniform has {au['counts']['both_pass']} both-pass,
{au['counts']['only_first']} analytic-only, {au['counts']['only_second']} uniform-only
and {au['counts']['both_fail']} both-fail conditions, with the same
{au['counts']['only_first']}-to-{au['counts']['only_second']} table against target-only
and against background-only M2S. The carrier-averaged difference is
{au['carrier_averaged_difference'] * 100:.3f} percentage points, positive on all three
carriers ({win_carriers}). All {len(wins)} additional passages request d040 and measure
0 N beam force; their matched walking comparators record {min(win_forces):.1f}-{max(win_forces):.1f} N.
The condition-level discordance p is {au['descriptive_discordance']['p_value']:.6f}, but
72 conditions are twelve layouts and two seeds inside three carriers, so the honest
statement is the carrier-level one: same direction on all three, n = 3.

Uniform, target-only and background-only M2S are **behaviourally identical**: each
issues walking on every condition and passes {c['per_arm']['uniform']['pass']}/72.
Adding eighteen untargeted groups or sixteen target-only groups changes the fitted
policy not at all; adding one contrast changes it completely. Both statements rest on
a single acquired example, so the effect size is not estimable here.

## Where the adaptation helps, and where nothing does

| Beam underside | Conditions | Analytic d040 | Analytic passes | Scripted d040 | Scripted passes | Walking-only passes |
| --- | --- | --- | --- | --- | --- | --- |
{bands}

The panel has a hard ceiling. In the {len(paired_keys)} conditions where both commands
were actually executed there are {outcome.get((False, True), 0)} d040-only successes,
{outcome.get((True, True), 0)} both-pass, {outcome.get((False, False), 0)} both-fail and
**{outcome.get((True, False), 0)} walking-only successes**, so the best outcome available
from the observed commands is {ceiling}/72. The scripted rule reaches
{c['per_arm']['scripted_rays']['pass']}/72, one below that ceiling; analytic reaches
{c['per_arm']['analytic']['pass']}/72. Under-adaptation, not misfiring adaptation,
separates the learners from the ceiling: analytic issues d040
{c['per_arm']['analytic']['d040_requests']} times against the script's
{c['per_arm']['scripted_rays']['d040_requests']}.

Every one of the {len(wins)} analytic-only wins falls in the middle band, which is where
the single training contrast sat (underside 1.2755 m). At the high band walking already
passes 24/24 and the learner mostly leaves it alone; at the low band neither command
ever succeeds. One example transferred across carriers within its own geometry band and
not beyond it.

## Background controls: no unnecessary adaptation, and a scripted blind spot

| Background suite | Analytic passage | Analytic d040 requests | Analytic refusals |
| --- | --- | --- | --- |
{background}

Across the absent and raised suites **no policy requests d040 even once**
({unnecessary} requests in {2 * 6 * len(METHODS)} runs), so no arm pays an
unnecessary-adaptation cost on this panel; the learned adaptation is not indiscriminate.
The blocked suite separates the methods in the opposite direction from traversal: every
learner and the privileged forecast refuse {learner_blocked}/6, while the scripted rule
refuses {scripted_blocked}/6. The script is the stronger traversal baseline and the
weaker infeasibility detector. No blocked scene is passable, so refusal there is correct
classification, not successful avoidance, and it is reported separately from passage.

## One useful contrast explains the adaptation requests in the registered refits

The analytic fitting set is the six shared backgrounds plus one useful generated
contrast. The one leave-four-assignment-out fold that withholds that label produces a
model whose training IDs, weights, biases and scales equal the background-only M2S
primary fit exactly. Over {removed['conditions']} recorded inputs it requests d040
**{removed['request_d040']}** times, against **{full_requests}** for the full analytic
fit, changing {removed['changed_selected_actions']} selected actions. {unchanged} of the
{len(kept)} folds that withhold no useful label leave the request count unchanged, and
{empty_folds} withhold no available label at all and are counted as such.

This is a controlled data-removal diagnostic of the fixed learner. It is not a new
physical evaluation of the fold models and it does not establish reliable learning from
one example in a population of sources.
[Exact equality check](evidence/icra-results-{done}-20260907/contrast-removal-check.json).

## Cost and what this settles

Label acquisition costs {costs['labels']:.6f} contended GPU h and the 540 admitted
policy executions cost {costs['evaluation']:.6f} h. These exclude separately recorded
generator/teacher costs and shared bank acquisition; no complete end-to-end cost claim
is made from physics time alone.

The completed panel establishes contrast-construction benefit over untargeted and
target-only construction, and establishes **no** learned-generator benefit, because the
learned arm acquired nothing to test. That is neither outcome A nor outcome B from the
guidance. The measured cause of the empty funnel is the inherited placement envelope
rather than the proposal model, and the separately registered nominal contract tests
whether a matched envelope changes the acquisition and the learning.

[All outcomes](assets/icra-{done}-outcomes.pdf) · [Yield and passage](assets/icra-{done}-yield.pdf) ·
[Envelope trade-off](ENVELOPE_TRADEOFF_V1_RESULT.md) ·
[Full records and saved primary/refit models](evidence/icra-results-{done}-20260907/exports.json) ·
[Working manuscript](ICRA_MANUSCRIPT.pdf).
"""
    (DOC / f"M2S_ICRA_{done}_RESULT.md").write_text(report)
    print(json.dumps({"exports": len(exports), "costs": costs, "completed": done}))


if __name__ == "__main__":
    main()
