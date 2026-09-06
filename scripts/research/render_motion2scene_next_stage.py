#!/usr/bin/env python3
# ruff: noqa: E501 -- HTML/Markdown prose templates keep paragraph boundaries.
"""Publish native-crossing audit, variation status and bounded completion assessment."""

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import numpy as np
from render_motion2scene_icra_progress import export, save

DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT / "docs/motion2scene"


def main():
    native, receipt = export(
        DATA / "m2s-native-crossing-audit-v1/result.json", "native-crossing-audit", OUT
    )
    for row in native["rows"]:
        for key in ("trajectory", "physics_contacts", "support"):
            ref = row[key]
            checked(Path(ref["path"]), ref["sha256"])
    export(
        DATA / "m2s-native-crossing-audit-v1/registration.json", "native-crossing-registration", OUT
    )
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    rows = native["rows"]
    x = np.arange(len(rows))
    ax.bar(
        x,
        [1000 * r["finish_delay_s"] for r in rows],
        color=["#2a8c87" if r["native_outer_pass"] else "#b85a45" for r in rows],
    )
    ax.set_xticks(
        x,
        [f"{r['condition']}/{r['mode']}\n{r['seed']}" for r in rows],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set(
        ylabel="Later completion with native outer envelope (ms)",
        title="All 14 outcomes unchanged: 10 pass, 4 fail; authored outer shapes at 50 Hz",
        ylim=(0, 50),
    )
    receipt["figures"] = save(fig, OUT, "native-crossing-audit")
    (OUT / "evidence/native-crossing-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lines = [
        "# Native outer-envelope crossing: result",
        "",
        "The registered CPU audit preserves **all 14 classifications: 10 passes and four failures**. Requiring the complete authored outer envelope to cross shifts completion **20–40 ms later** than the body-origin criterion. Measured 200 Hz contact through the later horizon preserves the original verdicts.",
        "",
        "| Cell | Body-origin finish frame | Outer-envelope finish frame | Delay (ms) | Outer-envelope passage | Peak force (N) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['cell_id']} | {r['body_origin_finish_frame']} | {r['native_outer_finish_frame']} | {1000*r['finish_delay_s']:.0f} | {'pass' if r['native_outer_pass'] else 'fail'} | {r['peak_normal_force_n_through_native_horizon']:.3f} |"
        )
    lines += [
        "",
        f"All 45 native outer shapes are included. CPU time: {native['cpu_seconds']:.3f} s; no simulator execution or GPU spend. Asset-layer hashes were checked before and after computation. This is a **conditional authored-envelope audit at recorded 50 Hz poses**, not cooked-mesh equivalence or continuous collision detection. The old result remains immutable.",
        "",
        "[Protocol](NATIVE_CROSSING_AUDIT_V1.md) · [Full evidence](evidence/native-crossing-audit.json) · [Hash receipt](evidence/native-crossing-receipt.json)",
        "",
        "![Native crossing delay](assets/native-crossing-audit.svg)",
    ]
    lines += [
        "",
        "Supplementary quaternion-normalization sensitivity bounds endpoint displacement by 3.33e-7 m, below the 0.00200001 m outward allowance. [Supplement](evidence/native-crossing-quaternion-supplement.json).",
    ]
    (OUT / "NATIVE_CROSSING_AUDIT_V1_RESULT.md").write_text("\n".join(lines) + "\n")
    batch = DATA / "m2s-overhang-variation-v1"
    record = json.loads((batch / "run_record.json").read_text())
    count = sum(c["status"] == "completed" for c in record["cells"].values())
    export(batch / "manifest.json", "overhang-variation-manifest", OUT)
    portable_record = (
        json.dumps(record, indent=2)
        .replace(str(DATA), "research-data")
        .replace(str(ROOT), "repository")
    )
    (OUT / "evidence/overhang-variation-run-record.json").write_text(portable_record + "\n")
    runtime_label = (
        "paused at the registered GPU memory gate"
        if record["status"] == "yielded_gpu_contention"
        else record["status"]
    )
    status = f"{count}/42 registered variation executions completed; {runtime_label}. Full-panel predictions remain unadjudicated."
    panel = ""
    if (batch / "result.json").exists():
        result, var_receipt = export(batch / "result.json", "overhang-variation", OUT)
        (OUT / "evidence/overhang-variation-status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "runtime_completed": count,
                    "registered_cells": 42,
                    "predictions": result["predictions"],
                    "actual_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
                    "result": "overhang-variation.json",
                    "historical_partial_snapshot": "variation-publication-lifecycle.json",
                },
                indent=2,
            )
            + "\n"
        )
        status = f"All 42 registered Isaac Lab executions completed. {sum(r['pass'] for r in result['rows'])}/42 pass the original contact-free passage criterion; the denominator includes blind and blocked controls."
        conditions = list(dict.fromkeys(r["condition"] for r in result["rows"]))
        table = []
        for condition in conditions:
            values = []
            for mode in ("reactive", "blind", "oracle"):
                subset = [
                    r for r in result["rows"] if r["condition"] == condition and r["mode"] == mode
                ]
                values.append(
                    f"{sum(r['pass'] for r in subset)}/{len(subset)}"
                    if subset
                    else "nominal reused" if condition.startswith("delay") else "not run"
                )
            table.append((condition, *values))
        panel = (
            '<div class="scroll"><table><caption>Contact-free passage, two seeds per executed cell. Delayed rows reuse nominal blind/oracle controls.</caption><thead><tr><th>Condition</th><th>Reactive</th><th>Blind</th><th>Oracle</th></tr></thead><tbody>'
            + "".join("<tr>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>" for row in table)
            + "</tbody></table></div>"
        )
        report = [
            "# Overhang variation v1: complete Isaac result",
            "",
            status,
            "",
            f"Registered predictions: `{json.dumps(result['predictions'],sort_keys=True)}`.",
            "",
            "| Condition | Reactive | Blind | Oracle |",
            "| --- | --- | --- | --- |",
        ]
        report += ["| " + " | ".join(row) + " |" for row in table]
        report += [
            "",
            "**The pose-separation prediction fails.** Reactive and oracle each pass 9/10 pose trials; all ten blind trials contact the beam. At station −15 cm in seed 8042, reactive and oracle both record 429.159 N in one 5 ms sample. They cross without a recorded fall, but fail contact-free passage. This is a limit of the executed reference/scene pairing, not a sensor-only error.",
            "",
            "All six negative controls report zero overhang detections and no switch. Absent/raised pass 4/4; the two blocked controls collide, fail to cross and reset. Both 100 ms and realized 260 ms delay trials pass with zero measured force. Both 500 ms trials refuse switching and still contact (579.134 / 494.174 N). Refusal does not solve obstacle avoidance.",
            "",
            f"Actual new cost: {result['new_contended_gpu_hours']:.6f} contended GPU h. Capacity yields resume the same manifest and skip completed cells; no scientific replacement. See the complete 42-row JSON for each force peak/duration/impulse, reset/fall, packet, switch and bank audit.",
            "",
            "A denied command is not successful traversal. These are five poses and two physics seeds on one previously observed source; no downstream policy was trained. The 250 ms request is delivered at 260 ms because sensing runs at 50 Hz. The obstacle in the blocked-underpass control is a floor-to-beam-top cuboid.",
            "",
            "[Protocol](OVERHANG_VARIATION_V1.md) · [Results](evidence/overhang-variation.json) · [Manifest](evidence/overhang-variation-manifest.json) · [Next research gates](PROJECT_COMPLETION_STATUS.md)",
        ]
        (OUT / "OVERHANG_VARIATION_V1_RESULT.md").write_text("\n".join(report) + "\n")
        (OUT / "evidence/overhang-variation-receipt.json").write_text(
            json.dumps(var_receipt, indent=2) + "\n"
        )
        panel += '<p><a href="OVERHANG_VARIATION_V1_RESULT.md">Complete variation result and predictions →</a></p>'
        panel += "<p><strong>Pose robustness fails its registered prediction:</strong> reactive and oracle each pass 9/10 pose trials. Both fail at station −15 cm in seed 8042 with a 429.159 N contact. All ten blind trials contact. The 100 ms and 260 ms delays pass 2/2 each; 500 ms refuses and fails 2/2. Negative specificity passes 6/6, including blocked controls that still collide.</p>"
    else:
        pending = {
            "status": record["status"],
            "runtime_completed": count,
            "registered_cells": 42,
            "predictions": "unadjudicated",
            "actual_gpu_hours": record["budget"]["actual_contended_gpu_hours"],
            "manifest": artifact(batch / "manifest.json"),
            "run_record": artifact(batch / "run_record.json"),
        }
        (OUT / "evidence/overhang-variation-status.json").write_text(
            json.dumps(pending, indent=2) + "\n"
        )
        panel = '<p><a href="evidence/overhang-variation-status.json">Runtime status snapshot</a> · <a href="OVERHANG_VARIATION_V1.md">Registered protocol</a></p>'
    section = f"""<!-- next-stage-update:start -->
<section id="next-stage"><div class="wrap">
<p class="eyebrow">Current progress · 06 September 2026 · Isaac Lab + MuJoCo replay</p>
<h2>From a working traversal<br>to a measured operating range.</h2>
<p class="intro">{status} The next-stage code adds causal observation delay and a solid blocked-underpass control. The previous completed fourteen-run batch also survives a stricter native outer-envelope crossing audit.</p>
{panel}
<figure><video style="aspect-ratio:3" controls playsinline preload="metadata" poster="assets/overhang-variation-demo.jpg" aria-label="Recorded Isaac blind, reactive and delayed-observation trials rendered with MuJoCo"><source src="assets/overhang-variation-demo.mp4" type="video/mp4"><a href="assets/overhang-variation-demo.mp4">Download recorded-state replay</a></video><figcaption><strong>Variation panel, frozen first seed 8041.</strong> Nominal blind walking, reactive d040 and 500 ms delayed sensing shown together. The failed earlier-beam trial in seed 8042 is retained in the result table. Contact is measured in Isaac at 200 Hz. MuJoCo only renders recorded states with <code>mj_forward</code>; it does not execute new dynamics. The visual mesh differs from the Isaac collision asset. <a href="assets/overhang-variation-demo.json">Replay provenance</a>. <a href="assets/overhang-evaluation-demo.mp4">Previous 8031 replay</a>.</figcaption></figure>
<div class="grid"><div><h3>A stricter crossing check</h3><p>All fourteen previous classifications remain unchanged: ten passes and four failures. The 45 authored native outer shapes finish crossing 20–40 ms later than body origins. This is a recorded-pose envelope audit, with the old result retained.</p><a href="NATIVE_CROSSING_AUDIT_V1_RESULT.md">Audit and all fourteen outcomes →</a></div><div><h3>The central question is still open</h3><p>We have evidence for inverse generation and bounded embodied feasibility. We have no result showing that Motion2Scene environments improve policy learning over random or analytic generation. That comparison determines whether the intended contribution holds.</p><a href="#completion-status">How far from the complete project →</a></div></div>
</div></section>
<section id="completion-status"><div class="wrap">
<p class="eyebrow">Project assessment · evidence gates, not a completion percentage</p>
<h2>An embodied prototype.<br>An unfinished learning contribution.</h2>
<p class="intro">The project is beyond an offline geometry demo. It is not yet submission-ready: the decisive downstream comparison could still reject our thesis. More successful replays cannot substitute for that experiment.</p>
<div class="scroll"><table><thead><tr><th>Workstream</th><th>Current status</th><th>What remains</th></tr></thead><tbody>
<tr><td>Inverse scene generation</td><td>Strong offline evidence: 384/384 proxy-audited requests</td><td>Earn the learned generator's cost against the strong analytic method.</td></tr>
<tr><td>Geometry and continuous time</td><td>384/384 conditional native outer-envelope time bounds at nominal placements; separate static proxy pose domains</td><td>Joint pose/time uncertainty coverage, cooked-shape validation and numerical enclosure guarantees.</td></tr>
<tr><td>Closed-loop execution</td><td>Demonstrated: 18/24 requested generated-source slots separate</td><td>Fresh-source transfer, complete action vocabulary and disturbance testing.</td></tr>
<tr><td>Perception and commands</td><td>Complete 42-cell ideal-ray pose/delay stress panel</td><td>Missing alternative labels, noise/depth and any third-reference qualification.</td></tr>
<tr><td>Downstream policy learning</td><td>Not demonstrated</td><td>Four matched generator arms, five optimizer seeds, frozen source-level tests and learning curves.</td></tr>
<tr><td>Paper and release</td><td>Notebook, protocols, failure ledger and demos available</td><td>Comparative figures, final claim audit and reproducible benchmark.</td></tr>
</tbody></table></div>
<p><strong>Next decisions:</strong> complete the negative-control alternative labels; freeze the supported sensor/action and measurement contract; run the matched learning-data comparison; then test fresh sources and sensor/physics disturbances. Preserve failures at every stage. Hardware validation remains a separate choice tied to the final deployment claim.</p>
<div class="evidence"><a href="PROJECT_COMPLETION_STATUS.md">Detailed completion assessment</a><a href="ICRA_COMPLETION_PLAN.md">Claim–evidence ledger</a><a href="DOWNSTREAM_SENSOR_POLICY_PILOT_DESIGN.md">Learning comparison design</a><a href="OVERHANG_VARIATION_V1.md">42-cell registered protocol</a><a href="evidence/native-source-20260906/research-source.tar.gz">Latest source snapshot</a></div>
</div></section>
<!-- next-stage-update:end -->"""
    page = OUT / "index.html"
    html = page.read_text()
    html = re.sub(
        r"<!-- next-stage-update:start -->.*?<!-- next-stage-update:end -->\n?",
        "",
        html,
        flags=re.S,
    )
    html = (
        html.replace("<main>", "<main>\n" + section, 1)
        .replace(
            'href="#overhang-interface">Current progress', 'href="#next-stage">Current progress'
        )
        .replace('href="#watch">Motion replays', 'href="#next-stage">Latest demos')
    )
    html = html.replace(
        'class="wide" controls playsinline preload="metadata" poster="assets/overhang-evaluation-demo.jpg"',
        'style="aspect-ratio:3" controls playsinline preload="metadata" poster="assets/overhang-evaluation-demo.jpg"',
    )
    page.write_text(html)
    print(status)


if __name__ == "__main__":
    main()
