#!/usr/bin/env python3
# ruff: noqa: E501 -- Research report and HTML paragraphs.
"""Publish diagnostic evidence without modifying any frozen evaluation artifact."""

import json
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked

DATA = ROOT.parent / "research-data/groot-wbc"
DOC = ROOT / "docs/motion2scene"
P0 = DATA / "m2s-selector-input-diagnostic-v2"
P1 = DATA / "m2s-selector-breakpoint-v1"
P2 = DATA / "m2s-transition-forecast-v1"
P3 = DATA / "m2s-shared-linear-control-v2"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")


def main():
    p0, p1, p2, p3 = [json.loads((p / "result.json").read_text()) for p in (P0, P1, P2, P3)]
    for p in (P1, P3):
        admission = json.loads((p / "admission.json").read_text())
        assert admission["admitted"]
        for k in ("result", "capture_audit"):
            checked(Path(admission[k]["path"]), admission[k]["sha256"])
    public = DOC / "evidence/selector-breakpoint-20260907"
    public.mkdir(exist_ok=False)
    receipts = []
    folders = (
        (DATA / "m2s-selector-input-diagnostic-v1", "p0-failed"),
        (P0, "p0"),
        (P1, "p1"),
        (P2, "p2"),
        (DATA / "m2s-shared-linear-control-v1", "p3-preflight-failed"),
        (P3, "p3"),
    )
    for folder, tag in folders:
        for src in sorted(folder.iterdir()):
            if src.is_file() and src.suffix in (".json", ".log", ".npz", ".py", ".md"):
                target = public / f"{tag}-{src.name}"
                raw = src.read_bytes()
                if src.suffix not in (".npz",):
                    raw = (
                        raw.decode()
                        .replace(str(DATA), "research-data")
                        .replace(str(ROOT), "repository")
                        .encode()
                    )
                target.write_bytes(raw)
                receipts.append(
                    {
                        "source": artifact(src),
                        "public": target.name,
                        "public_sha256": artifact(target)["sha256"],
                    }
                )
    for src in sorted((P3 / "models").glob("*.pt")):
        target = public / src.name
        shutil.copyfile(src, target)
        receipts.append(
            {
                "source": artifact(src),
                "public": target.name,
                "public_sha256": artifact(target)["sha256"],
            }
        )
    pause = DATA / "m2s-independent-layout-v1/operational_pause_breakpoint_v1.json"
    (public / "original-v1-pause.json").write_text(
        pause.read_text().replace(str(DATA), "research-data")
    )
    (public / "exports.json").write_text(
        json.dumps(receipts, indent=2)
        .replace(str(DATA), "research-data")
        .replace(str(ROOT), "repository")
        + "\n"
    )
    paired_table = [
        "| Height / physics seed | Walk | d040 | Peak walk / d040 force (N) | Resets walk / d040 |",
        "| --- | --- | --- | --- | --- |",
    ]
    pair_values = []
    for block in range(6):
        rr = sorted(
            [r for r in p1["rows"] if r["original_block"] == block], key=lambda r: r["action"]
        )
        a, b = rr
        pair_values.append([int(a["pass"]), int(b["pass"])])
        paired_table.append(
            f"| {a['beam']['underside_m']:.2f} m / {a['physics_seed']} | {'Pass' if a['pass'] else 'Fail'} | {'Pass' if b['pass'] else 'Fail'} | {a['maximum_beam_normal_force_n_through_passage']:.3f} / {b['maximum_beam_normal_force_n_through_passage']:.3f} | {a['reset_count']} / {b['reset_count']} |"
        )
    control_table = [
        "| Training-data arm | Actual passage | d040 requests | Refusals → walk | Returns / d040 requests |",
        "| --- | --- | --- | --- | --- |",
    ]
    summaries = {}
    for arm in ARMS:
        rr = [r for r in p3["rows"] if r["arm"] == arm]
        summaries[arm] = dict(
            passed=sum(r["pass"] for r in rr),
            requests=sum(r["readout"]["requested_action"] for r in rr),
            refusals=sum(r["readout"]["refusal"] for r in rr),
            returns=sum(r["return_logged"] for r in rr),
        )
        a = summaries[arm]
        control_table.append(
            f"| {arm} | {a['passed']}/6 | {a['requests']}/6 | {a['refusals']}/6 | {a['returns']}/{a['requests']} |"
        )
    assert summaries["analytic"]["passed"] == 4 and summaries["analytic"]["requests"] == 2
    report = f"""# Selector breakpoint: measured diagnosis and a shared learner control

**Result, September 6–7:** the targeted study completes 12/12 paired-command executions
and 24/24 common linear-control executions in Isaac Lab. All pairs and controls pass
the registered direct-input, command, geometry, bank and contact admission checks.
On the inspected middle-height seed-8512 encounter, the analytic-trained linear
control requests d040, executes the legal switch and passes with 0 N recorded beam
force; matched walking contacts at 57.947 N. This establishes one learned request
that realizes a measured action advantage. It does **not** establish Motion2Scene's
learning benefit, multi-source generalization or a normalization-only causal effect.

The original evaluation remains **120/600 admitted, 480 operationally paused**.
Its twenty checkpoints, inputs, assignment and scores are unchanged. The six inspected
conditions remain tests of those original models and development evidence for revisions.
[Pause record](evidence/selector-breakpoint-20260907/original-v1-pause.json).

## P0: why the original readouts collapse

Every arm's eleven-example fitting subset has all 70 non-ray dimensions clamped to
the stored 0.001 normalization scale. The largest evaluation displacement is 1.953
rad/s in left ankle-roll joint velocity, yielding an absolute normalized value of
1952.75. Previously constant ray-hit channels also reach approximately 1000
normalized units; this is not exclusively a state-scaling issue. Analytic logits range from −2134.02 to −50.19 on the completed wave.

All 9,240 donor/group interventions are retained. Replacing the full non-ray group
with a training donor restores a d040 request in 12/30 analytic readouts for at least
one donor; replacing only rays restores it in 0/30. None of the smaller individual
state groups restores an analytic request. Mixed vectors are diagnostic, potentially
unphysical inputs. These results support state extrapolation and interactions, not
proof that perception is ignored or that one feature alone causes the failure.

The low/middle beams are already visible at the 0.30 s decision: five/three upper
beam hits and fifteen/nine conditional lower queries, respectively, for each seed.
The high beam has no upper beam hit. Sensor visibility is not the only capability
limit: both supported commands fail under the low beam.

**The alias correction holds.** Across all 38 complete corpus pairs, the finite
observation-only action ceiling equals the paired-action ceiling: 19/38, gap zero.
The thirteen-scene mixed-outcome group has both ceilings 11/13. This is outcome-label
ambiguity without measured action-relevant ambiguity, not a population guarantee.
The uniform BCE infimum is 0.34718955, approximately 0.0001 below the five measured
training losses. More fitting updates are not the priority.

[P0 complete feature/logit/intervention record](evidence/selector-breakpoint-20260907/p0-result.json)
(about 7 MB) · [registered study](SELECTOR_BREAKPOINT_STUDY_V1.md).

## P1: all twelve newly executed command comparators

{chr(10).join(paired_table)}

Walking passes 3/6, d040 4/6; the paired-command ceiling is 4/6. There are three
both-success conditions, one d040-only success and two both-fail conditions. All six
new walking captures exactly reproduce their historical walking trajectories and
outcomes. All six command pairs match recorded pre-decision states/actions/token
histories and features. This does not claim a complete hidden simulator snapshot.

All six d040 entries are legal. Five log a return; the low-height seed-8511 trial
resets and has no return. Low-height seed 8512 logs a return but still contacts at
5283.938 N. Scoring requires force <=1 N, all body origins crossing by 0.1 m and
0.3 s upright first-episode stability. Return and later recovery are separate; the
body-origin endpoint is not native-extent clearance or continuous-time certification.
[P1 result](evidence/selector-breakpoint-20260907/p1-result.json) ·
[admission](evidence/selector-breakpoint-20260907/p1-admission.json).

## P3: common learner change, actually deployed in all four arms

Four deterministic regularized logistic controls use the same eleven original
examples per arm, zero initialization, fixed physical scales, Adam 0.01, 2000 full
batch updates and 0.01 mean squared weight penalty. No P1 labels entered fitting.
The threshold, walking preference, refusal fallback, 214 features, source bank and
transition manager are unchanged. The shared control jointly changes model capacity,
normalization and regularization. It is not a repair applied only to Motion2Scene.

{chr(10).join(control_table)}

The analytic control requests d040 at both middle-height conditions and walking at
both high-height conditions. It refuses at low height, which still executes walking
and fails. One middle-height crouch is unnecessary because walking also passes;
the other is the measured rescue. Uniform and Motion2Scene refuse low and walk
middle/high; no contrast walks throughout. Every actual feature vector, action,
full recorded trajectory and outcome matches its P1 command comparator. All 24
model readouts and capture audits pass. These are six conditions on one observed
source, one fitted control per arm; no inferential policy ranking is justified.

Within each seed the recorded non-ray state is identical across the three layouts;
the analytic control changes its request with the measured ray packet. The useful
request therefore connects observed geometry to the legal executed command in this
development panel. Its success strengthens the analytic comparator. It does not
repair the original Motion2Scene corpus's lack of successful generated adaptations.
[Execution protocol](LINEAR_SELECTOR_EXECUTION_V2.md) ·
[P3 full result](evidence/selector-breakpoint-20260907/p3-result.json) ·
[admission](evidence/selector-breakpoint-20260907/p3-admission.json).

## P2 recorded-data mechanism: forecast the deployed transition

The already measured empty-scene walk and 0.30 s d040 command runs have matched
pre-decision histories, no resets and legal switches. Their achieved body trajectories
provide a proposal forecast for all 35 existing generated training scenes. No new
scene or label is admitted in this CPU audit.

| Forecast | False clear despite physical contact, out of 70 commands | Missed contact-qualified commands | Clear screen but failed task |
| --- | --- | --- | --- |
| Complete reference, sampled capsules | 13 | 2 | 15 |
| Achieved empty transition, sampled capsules | 0 | 4 | 2 |

Nine reference false-clear commands are in Motion2Scene and four in analytic data.
The achieved screen catches all thirteen but misses two additional contact-qualified
analytic commands. A screen clearance of 10 mm is a proposal flag; it is not the
physical <=1 N endpoint. These 29-capsule screens use native 30/50 Hz input samples,
not native meshes or continuous time. The first achieved root x is 0.0249 m versus
reference x=0; authored coordinates are retained without silent realignment. The
comparison includes timing, tracking and sampling differences and is not an isolated
transition-timing ablation. One empty-scene seed cannot predict arbitrary contact
responses. [Protocol](TRANSITION_FORECAST_AUDIT_V1.md) ·
[complete forecast errors and inputs](evidence/selector-breakpoint-20260907/p2-result.json).

## Cost, failures and reproducibility

The twelve P1 runs cost {p1['actual_contended_gpu_hours']:.6f} contended GPU h and the
24 P3 runs cost {p3['actual_contended_gpu_hours']:.6f} h, measured by the hardened driver.
These are diagnostic costs, not the generator's total acquisition cost. P0 took
{p0['seconds']:.2f} CPU wall seconds; the forecast took {p2['cpu_seconds']:.2f} seconds.
Shared source acquisition and original fitting remain separately charged in their
existing ledgers. No pending original assignment is replaced by an offline prediction.

Two numerical failures are retained. Original P0 batch-of-six inference exceeded its
1e-6 comparison tolerance (max 5.36e-6); deployment-shaped single-example inference
matches exactly. [Numerical revision](SELECTOR_DIAGNOSIS_NUMERICAL_REVISION_V2.md).
The linear embedding preflight exceeded 1e-6 by at most 0.52e-6; all 158 action choices
per arm agreed. Before physics, v2 registered a 2e-6 representation tolerance. Actual
deployment auditing retains 1e-6. No threshold, weight or original policy was changed.
Both failures and their earlier protocols/scripts are included in the exports.

[Export hashes](evidence/selector-breakpoint-20260907/exports.json) bind public path-normalized
records to original artifact hashes. The four NPZ controls and four deployed PT
containers are included beside the records. Original exact fitting inputs and twenty
MLPs remain in the [previous portable bundle](evidence/selector-bundle-20260906/README.md).
[Updated source archive](evidence/selector-breakpoint-source-20260907/research-source.tar.gz).
[All 24 execution replays](assets/selector-breakpoint.mp4) use recorded Isaac states;
MuJoCo calls mj_forward for visuals only, never mj_step. All six conditions and full
capture intervals, including resets, are retained. Visual meshes differ from Isaac
collision assets; contact values come from Isaac.

## What remains for the paper

The next main experiment is [transition-aware construction on multiple development
carriers](TRANSITION_AWARE_NEXT_STAGE.md), with identical transition information for
the strong analytic and learned methods. Stabilize the common learner using a bounded
development factorial, then freeze five data arms at the existing 24/48/96 budgets.
Final candidate IDs are reserved only; acquisition, duplicate checks and switching
qualification remain undone. The current evidence is a measured mechanism and one
learned rescue, not a complete ICRA learning contribution or a hardware safety result.
"""
    (DOC / "SELECTOR_BREAKPOINT_RESULT.md").write_text(report)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    groups = [
        "Rays",
        "Phase/skill/age",
        "Gravity",
        "Root velocity",
        "Joint position",
        "Joint velocity",
    ]
    spans = [(0, 144), (144, 147), (147, 150), (150, 156), (156, 185), (185, 214)]
    features = next(m for m in p0["models"] if m["arm"] == "analytic")["features"]
    values = [max(f["evaluation_max_abs_normalized"] for f in features[a:b]) for a, b in spans]
    axes[0].barh(groups, values, color="#277d79")
    axes[0].set_xscale("symlog", linthresh=1)
    axes[0].set_xlabel("Maximum |normalized input| (symlog)")
    axes[0].set_title("Frozen analytic MLP: input range")
    axes[1].imshow(pair_values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axes[1].set_xticks([0, 1], ["Walk", "d040"])
    axes[1].set_yticks(
        range(6), [f"{h:.2f} / {s}" for h in (1.18, 1.27, 1.36) for s in (8511, 8512)]
    )
    for i, row in enumerate(pair_values):
        for j, val in enumerate(row):
            axes[1].text(j, i, "PASS" if val else "FAIL", ha="center", va="center", fontsize=10)
    axes[1].set_title("12 paired Isaac commands")
    axes[2].bar(
        range(4),
        [summaries[a]["passed"] for a in ARMS],
        color=["#849998", "#277d79", "#849998", "#849998"],
    )
    axes[2].set_xticks(range(4), ["Uniform", "Analytic", "No contrast", "M2S"], rotation=25)
    axes[2].set_ylim(0, 6)
    axes[2].set_ylabel("Actual passes / 6 conditions")
    axes[2].set_title("24 shared-linear-control executions")
    fig.suptitle(
        "Development diagnosis: one observed carrier, no Motion2Scene advantage claim", fontsize=12
    )
    for suffix in ("png", "pdf"):
        fig.savefig(DOC / f"assets/selector-breakpoint.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "p0": p0["aggregates"],
        "paired_outcomes": pair_values,
        "controls": summaries,
        "transition_forecast": p2["summary"],
        "original_v1": {"admitted": 120, "assigned": 600, "paused": 480},
    }
    (public / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["controls"]))


if __name__ == "__main__":
    main()
