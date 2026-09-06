#!/usr/bin/env python3
# ruff: noqa: E501 -- Complete HTML/Markdown paragraphs preserve their boundaries.
"""Publish action-label readiness and learning priority, including failed audits."""

import json
import re

from motion2scene_timing_diagnostic import ROOT
from render_motion2scene_icra_progress import export

DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT / "docs/motion2scene"


def main():
    exports = [
        ("m2s-learning-contract-v1/registration.json", "learning-contract"),
        ("m2s-learning-contract-v2/registration.json", "learning-contract-v2"),
        ("m2s-learning-input-audit-v1/registration.json", "learning-input-audit-v1-registration"),
        ("m2s-learning-input-audit-v1/failure.json", "learning-input-audit-v1-failure"),
        ("m2s-learning-input-audit-v2/registration.json", "learning-input-audit-v2-registration"),
        ("m2s-learning-input-audit-v2/result.json", "learning-input-audit-v2"),
        ("m2s-learning-input-audit-v3/registration.json", "learning-input-audit-v3-registration"),
        ("m2s-learning-input-audit-v3/result.json", "learning-input-audit-v3"),
        ("m2s-decision-binding-v1/registration.json", "decision-binding-registration"),
        ("m2s-decision-binding-v1/result.json", "decision-binding"),
        ("m2s-action-label-completion-v1/manifest.json", "action-label-manifest"),
        (
            "m2s-action-label-completion-v1/analysis_loader_revision_v2.json",
            "action-label-analysis-revision",
        ),
    ]
    exports += [
        ("m2s-no-contrast-baseline-v1/registration.json", "no-contrast-baseline-registration"),
        ("m2s-no-contrast-baseline-v1/result.json", "no-contrast-baseline"),
    ]
    reservation = DATA / "m2s-final-transfer-reservation-v1/reservation.json"
    if reservation.exists():
        exports.append(
            ("m2s-final-transfer-reservation-v1/reservation.json", "final-transfer-reservation")
        )
    receipts = []
    for path, name in exports:
        _, receipt = export(DATA / path, name, OUT)
        receipts.append(receipt)
    binding = json.loads((DATA / "m2s-decision-binding-v1/result.json").read_text())
    worst = max(r["origin_error_m"] for r in binding["rows"])
    (OUT / "LEARNING_INPUT_AUDIT_RESULT.md").write_text(
        f"""# Learning input binding: result and retained failures

The first 0.20 s decision packet is bound to its **same-index** recorded post-physics
robot state in all 42 prior cells. Worst reconstructed upper-ray-origin discrepancy:
**{worst:.3g} m**. These are paired sensing/state measurements, not policy performance.
The command's reference phase has advanced when sensing runs; it must not be used as
an array index into the next physical state. The frozen planning note's proposed
next-index alignment is superseded by this audit.

Requested and realized delay are measured separately. Every delivered packet has the
registered monotonic capture age: 100 ms → 5 frames → 100 ms; 250 ms → 13 frames →
260 ms; 500 ms → 25 frames → 500 ms. Both index and elapsed-timestamp checks pass all
42 cells, including unavailable warmup packets. The four 250/500 ms cells have no
available sensor packet at the fixed first decision; no future observation is imputed.

The initial audit aborted because the single-episode dataset loader rejected a
reset-spanning failure capture. The reset-aware adapter validates every segment,
including short failed first episodes and tails, while retaining all raw frames.
It does not reinterpret a reset as success or select a better episode.

The second audit's next-index physical-state prediction fails. The third audit's
whole-capture same-index prediction also fails at reset/terminal boundaries where
command and recorder lifecycle do not expose the same state. Those attempts remain
available. The fourth audit tests the **already prespecified first decision** only;
it does not claim whole-capture alignment. Future sequential decisions require their
own capture contract. No robot-data policy fit occurred.

[Decision binding](evidence/decision-binding.json) · [Decision registration](evidence/decision-binding-registration.json) · [Delay and failed next-index audit](evidence/learning-input-audit-v2.json) · [Failed whole-capture audit](evidence/learning-input-audit-v3.json) · [Initial loader failure](evidence/learning-input-audit-v1-failure.json)
"""
    )
    batch = DATA / "m2s-action-label-completion-v1"
    record = json.loads((batch / "run_record.json").read_text())
    count = sum(c["status"] == "completed" for c in record["cells"].values())
    status = f"{count}/12 registered label/timing executions completed; {record['status']}. Predictions remain unadjudicated until the panel completes."
    table = ""
    if (batch / "result.json").exists():
        result, receipt = export(batch / "result.json", "action-label-completion", OUT)
        receipts.append(receipt)
        complete = sum(r["label"]["feasible"] is not None for r in result["labels"])
        status = f"All twelve label/timing executions completed; {complete}/16 scene–seed groups have complete, matched command-outcome labels. These remain development examples, not a training corpus."
        lines = [
            "# Action labels and timing: complete result",
            "",
            status,
            "",
            f"Registered predictions: `{json.dumps(result['predictions'],sort_keys=True)}`.",
            "",
            "| Seed | Condition | Walk outcome | d040 command outcome | Pair valid |",
            "| --- | --- | --- | --- | --- |",
        ]
        for r in result["labels"]:
            f = r["label"]["feasible"]
            lines.append(
                f"| {r['seed']} | {r['condition']} | {f[0] if f is not None else 'unknown'} | {f[1] if f is not None else 'unknown'} | {r['pairing']['valid']} |"
            )
        lines += [
            "",
            "The label compares committing to neutral with a request to enter d040 at 0.20 s plus shared return. Exact recorded pre-decision root/joint state, velocities, applied actions, motion tokens and references match; this is not a hidden-state snapshot. All four outcome combinations are allowed, even if this small panel does not exhibit every combination.",
            "",
            "| Seed | Requested time (s) | Command executed | Passage | Peak beam force (N) | Contact duration (s) |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        timing = [r for r in result["rows"] if r["role"] == "timing"]
        for r in timing:
            lines.append(
                f"| {r['seed']} | {r['decision_time_s']:.2f} | {r['command_executed']} | {r['pass']} | {r['maximum_beam_normal_force_n_through_passage']:.3f} | {r['physics_contact_duration_s']:.3f} |"
            )
        table = (
            '<div class="scroll"><table><caption>Earlier-beam development timing test. Explicit d040 request, shared scene-independent guard.</caption><thead><tr><th>Seed</th><th>Request (s)</th><th>Executed</th><th>Passage</th><th>Peak force (N)</th></tr></thead><tbody>'
            + "".join(
                f"<tr><td>{r['seed']}</td><td>{r['decision_time_s']:.2f}</td><td>{r['command_executed']}</td><td>{r['pass']}</td><td>{r['maximum_beam_normal_force_n_through_passage']:.3f}</td></tr>"
                for r in timing
            )
            + "</tbody></table></div>"
        )
        rescued = [
            r["decision_time_s"]
            for r in timing
            if r["seed"] == 8042 and r["pass"] and r["command_executed"]
        ]
        finding = (
            f"Seed 8042 passes at tested times {rescued}."
            if rescued
            else "No tested legal request time rescues seed 8042. This does not prove impossibility at every time or with another skill."
        )
        table += f"<p><strong>{finding}</strong> The old 0.20 s failure remains unchanged. The scripted privileged baseline was already at the earliest legal time.</p>"
        lines += [
            "",
            finding,
            "",
            f"New actual cost: {result['new_contended_gpu_hours']:.6f} contended GPU h. The six missing controls and six timing runs are distinct cells; repeated source/seed/interface checks do not add independent ancestors.",
            "",
            "The reset-aware analysis-loader revision is explicit and hash-bound; physics manifest, thresholds, commands and predictions remain unchanged. Refusal is not stopping. No new policy-learning benefit, third skill or full collision certificate is claimed.",
            "",
            "Validation: 187 impacted tests passed and one optional-dependency test skipped. All 292 unique registered artifact references matched their recorded hashes. The frozen failed v3 input-audit script retains one E501 prose-line exemption; other checked sources pass Black and Ruff. [Completion audit](evidence/action-completion-audit.json).",
            "",
            "[Protocol](ACTION_LABEL_COMPLETION_V1.md) · [All rows](evidence/action-label-completion.json) · [Analysis revision](evidence/action-label-analysis-revision.json) · [Learning comparison plan](LEARNING_UTILITY_PLAN_V1.md)",
        ]
        (OUT / "ACTION_LABEL_COMPLETION_V1_RESULT.md").write_text("\n".join(lines) + "\n")
        table += '<p><a href="ACTION_LABEL_COMPLETION_V1_RESULT.md">All paired labels, timing results and predictions →</a></p>'
    portable_record = (
        json.dumps(record, indent=2)
        .replace(str(DATA), "research-data")
        .replace(str(ROOT), "repository")
    )
    (OUT / "evidence/action-label-run-record.json").write_text(portable_record + "\n")
    (OUT / "evidence/learning-stage-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    section = f"""<!-- learning-utility:start -->
<section id="learning-utility"><div class="wrap"><p class="eyebrow">Current priority · 06 September 2026 · transition-aware learning data</p>
<h2>Test whether the scenes<br>teach useful decisions.</h2><p class="intro">{status}</p>
{table}
<div class="grid"><div><h3>The comparison is now the priority</h3><p>Freeze the existing generator and SONIC. Compare Motion2Scene, strong analytic construction, uniform placement and a model whose full pipeline omits the contrast objective. The primary question is data quality at equal complete-label counts; acquisition cost is a separate comparison. <a href="LEARNING_CONTRACT_V2.md">The v2 contract</a> selects one common 0.30 s decision for newly acquired data in all arms. Old 0.20 s labels remain separate.</p><a href="LEARNING_UTILITY_PLAN_V1.md">Concrete learner, evaluation and cost plan →</a></div><div><h3>Inputs and labels have explicit meaning</h3><p>The shared learner predicts both supported command outcomes from 214 causal sensor/state values. Both-fail and missing outcomes stay distinct. Twelve test layouts are reserved independently before fitting. No robot-data policy has been trained yet. The no-contrast generator baseline has completed its matched 1,200-update CPU fit; this is preparation for the comparison, not a learning-utility result. <a href="NO_CONTRAST_BASELINE_V1_RESULT.md">Baseline construction record</a>.</p><a href="LEARNING_INPUT_AUDIT_RESULT.md">Timestamp, reset and state-binding audits →</a></div></div>
<p>Final transfer ancestors remain unreserved after the bounded ancestry inventory timed out; no new-source freshness is claimed. <a href="evidence/transfer-reservation-failure.json">Reservation attempt record</a>.</p>
<p>The registered 250 ms delay is measured as thirteen 50 Hz frames: 260 ms. The first decision matches its recorded physical state in all 42 prior runs. Broader failed alignment attempts remain in the audit ledger. <a href="evidence/learning-contract-v2.json">Current generator and learner freeze</a> · <a href="evidence/learning-stage-source-20260906/research-source.tar.gz">Updated source snapshot</a>.</p>
</div></section><!-- learning-utility:end -->"""
    page = OUT / "index.html"
    s = page.read_text()
    s = re.sub(
        r"<!-- learning-utility:start -->.*?<!-- learning-utility:end -->\n?", "", s, flags=re.S
    )
    s = s.replace("<main>", "<main>\n" + section, 1)
    s = re.sub(r'href="#[^"]+">Current progress', 'href="#learning-utility">Current progress', s)
    page.write_text(s)
    print(status)


if __name__ == "__main__":
    main()
