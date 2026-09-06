#!/usr/bin/env python3
# ruff: noqa: E501 -- Complete report paragraphs and HTML.
"""Publish the acquisition funnel without promoting geometry to learning utility."""

import hashlib
import json
import re

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT

DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT / "docs/motion2scene"


def main():
    source = DATA / "m2s-comparative-acquisition-v1"
    p = json.loads((source / "proposals.json").read_text())
    reg = json.loads((source / "registration.json").read_text())
    record = json.loads((source / "run_record.json").read_text())
    completed = sum(c["status"] == "completed" for c in record["cells"].values())
    rows = []
    receipts = []
    for name in (
        "registration",
        "proposals",
        "manifest",
        "budget_preflight",
        "run_record",
        "capture_audit_registration",
        "prelaunch_validation",
        "result",
        "capture_audit",
    ):
        path = source / f"{name}.json"
        if not path.exists():
            continue
        raw = path.read_bytes()
        portable = raw.decode().replace(str(DATA), "research-data").replace(str(ROOT), "repository")
        target = OUT / "evidence" / f"comparative-acquisition-{name}.json"
        target.write_text(portable)
        receipts.append(
            {
                "source": str(path).replace(str(DATA), "research-data"),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "public_file": target.name,
                "public_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
    reservation = DATA / "m2s-final-transfer-reservation-v3/reservation.json"
    if reservation.exists():
        for name in ("reservation", "inventory"):
            path = reservation.parent / f"{name}.json"
            raw = path.read_bytes()
            target = OUT / "evidence" / f"final-transfer-v3-{name}.json"
            target.write_text(
                raw.decode().replace(str(DATA), "research-data").replace(str(ROOT), "repository")
            )
        transfer = "Eight final-transfer ancestor IDs are now reserved after the local inventory; none has been generated, qualified or fitted. [Reservation](evidence/final-transfer-v3-reservation.json)."
    else:
        transfer = "Final transfer ancestors remain unreserved; the third bounded inventory exceeded its 180 s ceiling. [Retained failure](evidence/final-transfer-v3-failure.json)."
    for arm in reg["arms"]:
        selected = [r for r in p["rows"] if r["arm"] == arm and r["assigned"]]
        rows.append(
            (
                arm,
                len(selected),
                sum(r["eligible"] for r in selected),
                sum(r["critical_geometry"] for r in selected),
                min(r["station"] for r in selected),
                max(r["station"] for r in selected),
            )
        )
    fig, axes = plt.subplots(
        1, 4, figsize=(13, 3.5), sharex=True, sharey=True, layout="constrained"
    )
    for ax, (arm, *_) in zip(axes, rows):
        for layout in reg["layouts"]:
            ax.add_patch(
                Rectangle(
                    (layout["station"] - 0.01, layout["underside_m"] - 0.005),
                    0.02,
                    0.01,
                    facecolor="#dddddd",
                    edgecolor="#aaaaaa",
                    lw=0.5,
                )
            )
        assigned = [r for r in p["rows"] if r["arm"] == arm and r["assigned"]]
        for r in assigned:
            ax.scatter(
                r["station"],
                r["underside_m"],
                c="#167a78" if r["eligible"] else "#bd4337",
                marker="o" if r["eligible"] else "x",
                s=40,
            )
        ax.set(
            title=arm.replace("_", " "),
            xlim=(0.1, 0.9),
            ylim=(1.1, 1.45),
            xlabel="Normalized route station",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Beam underside (m)")
    fig.suptitle(
        "d040-bound proposal assignments — geometry only; shaded boxes are reserved test neighborhoods"
    )
    fig.savefig(OUT / "evidence/comparative-acquisition-proposals.png", dpi=160)
    fig.savefig(OUT / "evidence/comparative-acquisition-proposals.svg")
    plt.close(fig)
    table = [
        "| Arm | Assigned generated slots | Geometry-admitted | Contrast geometry passes | Station range |",
        "| --- | --- | --- | --- | --- |",
    ]
    for a, n, valid, critical, lo, hi in rows:
        table.append(f"| {a} | {n} | {valid} | {critical} | {lo:.3f}–{hi:.3f} |")
    result = (
        json.loads((source / "result.json").read_text())
        if (source / "result.json").exists()
        else None
    )
    status = (
        f"{completed}/20 first-slice physics runs complete; recorded status `{record['status']}`."
    )
    if result is None:
        physical = f'{status} No new paired learning labels or selector fits are claimed. The latest GPU gate yielded at {next(iter(record["cells"].values())).get("free_gpu_mib","unreported")} MiB free against the fixed 7500 MiB floor.'
    else:
        physical = f'{status} Complete valid pairs: {sum(p["valid"] for p in result["pairs"])}/{len(result["pairs"])}. Predictions: `{json.dumps(result["predictions"],sort_keys=True)}`. This is still an incomplete development corpus; selector fits remain zero.'
    lines = [
        "# Comparative acquisition v1: current result",
        "",
        physical,
        "",
        "The frozen generator and no-contrast baseline now receive the actual deployed d040 CSV geometry. Reconstructing both SONIC entries gives maximum root-array difference 1.4901161e-8 m and joint/axis-angle difference 3.3651304e-11, within the declared 1e-7 absolute tolerance; FPS and field sets match exactly. No d055 reference is loaded. Legacy helper slot names are explicitly mapped to d040 arrays.",
        "",
        *table,
        "",
        "These are reference-geometry outcomes, not action labels. Motion2Scene concentrates its assigned scenes near station 0.53–0.56; the analytic arm covers 0.40–0.68. The analytic slot overlapping a reserved test neighborhood is retained as a charged rejection. This concentration could matter for training coverage, but no downstream comparison has tested that hypothesis.",
        "",
        "All 64 generator outputs and costs are retained. Thirty-six generated slots are assigned, with three common background scenes reused across four arms for 48 arm-assigned groups (39 unique scenes before rejection). The first slice contains only the first two fixed slots per arm when eligible plus absent and blocked controls. Remaining slots and raised controls require a subsequent spend manifest; no outcome-dependent refill is allowed.",
        "",
        "Direct capture records the 214 features, delivered rays, joint order, robot state, reference phase and clocks before the actual command. An independent registered audit will compare that capture with the recorder, recompute features and check the loaded reference banks. Old 0.20 s labels remain separate.",
        "",
        transfer,
        "",
        "Validation: 190 impacted tests passed, one optional-dependency test skipped. All 229 registered artifact hashes matched. An AST comparison confirms the frozen command logic is unchanged except for the direct pre-command capture block. This is software validation, not an execution result. [Validation record](evidence/comparative-acquisition-prelaunch_validation.json).",
        "",
        "The main 24/48/96-data-budget comparison, robot-data fitting, independent-layout evaluation and final source transfer remain unfinished. Geometry admission is not proof that these examples teach a useful selector.",
        "",
        "[Protocol](COMPARATIVE_ACQUISITION_V1.md) · [Frozen proposals](evidence/comparative-acquisition-proposals.json) · [Run record](evidence/comparative-acquisition-run_record.json) · [Source snapshot](evidence/comparative-acquisition-source-20260906/research-source.tar.gz)",
    ]
    (OUT / "COMPARATIVE_ACQUISITION_V1_RESULT.md").write_text("\n".join(lines) + "\n")
    htmlrows = "".join(
        f"<tr><td>{a}</td><td>{n}</td><td>{valid}</td><td>{critical}</td></tr>"
        for a, n, valid, critical, *_ in rows
    )
    html = f"""<!-- comparative-acquisition:start -->
<section id="comparative-acquisition"><div class="wrap"><p class="eyebrow">Current acquisition · 06 September 2026</p><h2>Four data sources.<br>One deployed action contract.</h2>
<p class="intro">The proposal pipeline is bound to the actual neutral/d040 bank. {completed}/20 first-slice Isaac Lab runs are complete. No new selector-learning benefit is claimed.</p>
<div class="scroll"><table><caption>Assigned generated slots; reference geometry only.</caption><thead><tr><th>Arm</th><th>Assigned</th><th>Admitted</th><th>Contrast passes</th></tr></thead><tbody>{htmlrows}</tbody></table></div>
<img src="evidence/comparative-acquisition-proposals.png" alt="Four proposal distributions conditioned on the deployed d040 bank, with reserved test neighborhoods shaded" style="width:100%;height:auto">
<p>The first slice measures both commands at 0.30 s with direct pre-command sensing and state capture. Current execution status: <strong>{record['status']}</strong>. Rejected proposals and unfinished physics remain in the denominator. The old 0.20 s development labels stay separate.</p>
<p><a href="COMPARATIVE_ACQUISITION_V1_RESULT.md">Input binding, complete funnel, cost and remaining work →</a> · <a href="COMPARATIVE_ACQUISITION_V1.md">Before-run protocol</a></p></div></section><!-- comparative-acquisition:end -->"""
    page = OUT / "index.html"
    s = page.read_text()
    s = re.sub(
        r"<!-- comparative-acquisition:start -->.*?<!-- comparative-acquisition:end -->\n?",
        "",
        s,
        flags=re.S,
    )
    s = s.replace("<main>", "<main>\n" + html, 1)
    s = re.sub(
        r'href="#[^"]+">Current progress', 'href="#comparative-acquisition">Current progress', s
    )
    page.write_text(s)
    (OUT / "evidence/comparative-acquisition-publication-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(physical)


if __name__ == "__main__":
    main()
