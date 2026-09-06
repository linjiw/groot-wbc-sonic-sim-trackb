#!/usr/bin/env python3
# ruff: noqa: E501 -- prose and table templates keep paragraph boundaries.
"""Publish the complete native batch, capped failure and independent verification."""

import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, checked
import numpy as np
from render_motion2scene_icra_progress import export, save

DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT / "docs/motion2scene"


def main():
    first, r1 = export(
        DATA / "m2s-fresh-native-temporal-v1/result.json", "fresh-native-temporal-v1", OUT
    )
    second, r2 = export(
        DATA / "m2s-fresh-native-uncapped-v2/result.json", "fresh-native-uncapped-v2", OUT
    )
    binding, r3 = export(
        DATA / "m2s-native-frame-binding-v1/result.json", "native-frame-binding", OUT
    )
    for result in [first, second, binding]:
        for row in result["rows"]:
            checked(Path(row["raw"]["path"]), row["raw"]["sha256"])
    for folder, name in [
        ("m2s-fresh-native-temporal-v1", "fresh-native-temporal-registration"),
        ("m2s-fresh-native-uncapped-v2", "fresh-native-uncapped-registration"),
        ("m2s-native-frame-binding-v1", "native-frame-binding-registration"),
    ]:
        export(DATA / folder / "registration.json", name, OUT)
    sources = sorted({r["source"] for r in second["rows"]})
    groups = [[r for r in second["rows"] if r["source"] == s] for s in sources]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(8)
    axes[0].bar(
        x - 0.18,
        [1000 * min(r["sampled_target_minimum_m"] for r in g) for g in groups],
        0.36,
        label="120 Hz sampled",
        color="#a4c6c3",
    )
    axes[0].bar(
        x + 0.18,
        [1000 * min(r["interval_target_minimum_m"] for r in g) for g in groups],
        0.36,
        label="Uncapped interval bound",
        color="#2a8c87",
    )
    axes[0].axhline(10, color="black", ls="--", label="Required 10 mm")
    axes[0].set_xticks(x, sources, rotation=35)
    axes[0].set(
        ylabel="Worst target clearance / lower bound (mm)",
        title="All 48 fixed placements per source; authored outer shapes",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar([0, 1, 2], [384, 0, 384], color=["#2a8c87", "#b85a45", "#2a8c87"])
    axes[1].set_xticks(
        [0, 1, 2],
        [
            "Native sampled\n30 / 120 Hz",
            "Capped interval\noriginal v1",
            "Uncapped interval\nverification v2",
        ],
    )
    axes[1].set(
        ylim=(0, 425),
        ylabel="Passing nominal requests / 384",
        title="Preserved failure and independent measurement check",
    )
    for i, v in enumerate([384, 0, 384]):
        axes[1].text(i, v + 8, str(v), ha="center")
    fig.suptitle("Fixed nominal beams · declared reference interpolant · no new physics or fitting")
    receipts = {
        "original_native": r1,
        "uncapped_verification": r2,
        "frame_binding": r3,
        "figures": save(fig, OUT, "fresh-native-batch"),
    }
    (OUT / "evidence/fresh-native-batch-receipt.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    report = [
        "# Full native nominal audit: result and bound verification",
        "",
        "**All 384 frozen placements retain sampled native separation at 30 and 120 Hz.** An independently registered uncapped check certifies the declared target interpolant for all 384, with a worst lower bound of **21.623 mm** against the 10 mm requirement. These are authored native enclosures at nominal placements, not obstacle-present executions, cooked-mesh CCD, or the original 113-offset uncertainty domain.",
        "",
        "## Original failure retained",
        "",
        "The first native audit passes P1 (sampled separation) and P3 (proxy reproduction), but **fails P2: 0/384 pass its capped interval bound**. It clips all target clearances to 20 mm; subtracting motion bounds then produces minima from −2.253 to +3.463 mm. These weak bounds do not establish a collision. All target minima are saturated, so the original capped witness label is not an uncapped nearest-shape claim.",
        "",
        "The separately registered U1/U2 check computes uncapped distances with the same scene coordinates, shapes, displacement formula and 1e-5 m allowance. Reapplying the 20 mm cap reproduces **every stored target array exactly**. Both new predictions pass; the original failed P2 remains unchanged. The tightest uncapped witnesses are all on torso_link. No generator is trained or selected on these results; 430xx remains permanently excluded from fitting.",
        "",
        "## Complete source denominator",
        "",
        "| Source | Requested | Native sampled passes, 30/120 Hz | Uncapped bound passes | Worst target sampled (mm) | Worst interval bound (mm) | Weakest neutral primitive witness (mm) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for source, g in zip(sources, groups):
        report.append(
            f"| {source} | 48 | 48 / 48 | {sum(r['pass'] for r in g)} | {1000*min(r['sampled_target_minimum_m'] for r in g):.3f} | {1000*min(r['interval_target_minimum_m'] for r in g):.3f} | {1000*max(r['neutral_native_minimum_m'] for r in g):.3f} |"
        )
    report += [
        "",
        "Eight motion ancestors, two events and three model seeds produce 48 eight-output jobs. The 384 requests are not 384 independent sources. Target clearance uses 45 native outer shapes; neutral interference uses the 27 native primitives. Mesh spheres provide an outer enclosure, so their overlap alone is never called a mesh collision.",
        "",
        "## Frame binding and cost",
        "",
        f"Before this extension, all fourteen measured-qpos captures pass the frame-binding prediction over {binding['frames']} recorded frames. Maximum position discrepancy is {max(r['maximum_position_error_m'] for r in binding['rows'])*1e6:.3f} micrometres; maximum orientation discrepancy is {max(r['maximum_orientation_error_deg'] for r in binding['rows']):.6f} degree. This empirically checks the reference FK/Isaac frame contract; it is not a universal equivalence proof.",
        "",
        "| Query setting | Whole-motion queries | Query/bound time (s) |",
        "| --- | --- | --- |",
    ]
    for name, seconds in first["query_seconds"].items():
        report.append(
            f"| {name}, target + neutral | {first['whole_motion_queries'][name]} | {seconds:.4f} |"
        )
    report.append(f"| Uncapped native 120 Hz, target only | 384 | {second['query_seconds']:.4f} |")
    report += [
        "",
        f"Original audit total: {first['elapsed_seconds']:.3f} s, including {first['preprocessing_seconds']:.3f} s loading and {first['fk_geometry_cache_seconds']:.3f} s FK/geometry preparation. Uncapped verification total: {second['elapsed_seconds']:.3f} s, including {second['setup_seconds']:.3f} s setup. Both use CPU only. The uncapped target-only workload is not a matched timing comparison against the paired capped workloads.",
        "",
        "## What remains",
        "",
        "The interval result applies to the declared linear-position/shortest-SLERP body interpolant under a stated 1e-5 m allowance. It does not recover unrecorded dynamics. Native cooking/inflation equivalence, the full placement uncertainty domain, noisy sensing, fresh-source closed-loop transfer and downstream learning benefit remain open. Continue the frozen 42-cell interface panel, then prioritize the matched learning-data comparison.",
        "",
        "[Original protocol](FRESH_NATIVE_TEMPORAL_V1.md) · [Uncapped verification protocol](FRESH_NATIVE_UNCAPPED_V2.md) · [Frame-binding protocol](NATIVE_FRAME_BINDING_V1.md) · [All 384 original rows](evidence/fresh-native-temporal-v1.json) · [All 384 uncapped rows](evidence/fresh-native-uncapped-v2.json) · [Hash receipt](evidence/fresh-native-batch-receipt.json)",
        "",
        "![Native batch and retained cap failure](assets/fresh-native-batch.svg)",
    ]
    (OUT / "FRESH_NATIVE_TEMPORAL_V1_RESULT.md").write_text("\n".join(report) + "\n")
    section = """<!-- native-batch:start -->
<section id="native-batch"><div class="wrap"><p class="eyebrow">Latest CPU geometry extension · 06 September 2026</p>
<h2>All 384 fixed beams survive<br>the native nominal screen.</h2>
<p class="intro">The authored native outer shapes retain target clearance, and native primitives retain neutral interference, at 30 and 120 Hz. A separately registered uncapped interval check passes all 384 with a worst target bound of 21.62 mm. This tests the declared reference interpolant at fixed nominal placements; it is not new physics, cooked-mesh CCD or the full placement-uncertainty domain.</p>
<div class="cards"><div class="card"><div class="value good">384/384</div><p>Native sampled separation at both rates. Eight motion ancestors; outputs are not independent sources.</p></div><div class="card"><div class="value warning">0/384</div><p>Original capped interval checker. Its failed prediction is retained; an unresolved bound is not a collision.</p></div><div class="card"><div class="value good">21.62 mm</div><p>Worst uncapped conditional target bound; required clearance 10 mm. All 384 pass.</p></div><div class="card"><div class="value">0 GPU h</div><p>CPU measurement audit. No fitting, new scenes or obstacle-present executions.</p></div></div>
<figure><img src="assets/fresh-native-batch.svg" alt="All eight source groups retain native nominal clearance; the capped-bound failure is shown beside uncapped verification."><figcaption>Uncapped arrays reproduce every original capped target array exactly after applying its 20 mm cap. The correction concerns measurement conservatism, not generator performance. The 430xx sources remain excluded from fitting.</figcaption></figure>
<div class="evidence"><a href="FRESH_NATIVE_TEMPORAL_V1_RESULT.md">Complete result, failure and timings</a><a href="FRESH_NATIVE_UNCAPPED_V2.md">Independent verification protocol</a><a href="NATIVE_FRAME_BINDING_V1.md">Frame-binding check</a><a href="#completion-status">Remaining ICRA evidence</a></div>
</div></section>
<!-- native-batch:end -->"""
    page = OUT / "index.html"
    html = page.read_text()
    html = re.sub(
        r"<!-- native-batch:start -->.*?<!-- native-batch:end -->\n?", "", html, flags=re.S
    )
    html = html.replace("<main>", "<main>\n" + section, 1).replace(
        'href="#next-stage">Current progress', 'href="#native-batch">Current progress'
    )
    page.write_text(html)


if __name__ == "__main__":
    main()
