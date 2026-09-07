#!/usr/bin/env python3
# ruff: noqa: E501 -- Research report and HTML paragraphs.
"""Publish completed independent-layout blocks while preserving all pending assignments."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_comparative_acquisition import ARMS
from motion2scene_timing_diagnostic import ROOT, artifact, checked, write_new
import numpy as np

OUT = ROOT / "docs/motion2scene"
DATA = ROOT.parent / "research-data/groot-wbc"


def summarize(master, results):
    rows = [row for result in results for row in result["rows"]]
    ids = [r["cell_id"] for r in rows]
    assert len(ids) == len(set(ids))
    traversal = [r for r in rows if r["suite"] == "traversal"]
    per_fit = []
    for seed in range(8501, 8506):
        for arm in ARMS:
            selected = [r for r in traversal if r["arm"] == arm and r["optimizer_seed"] == seed]
            per_fit.append(
                {
                    "arm": arm,
                    "optimizer_seed": seed,
                    "passed": sum(r["pass"] for r in selected),
                    "measured": len(selected),
                    "assigned": 24,
                    "requests": sum(r["readout"]["requested_action"] == 1 for r in selected),
                    "refusals": sum(r["readout"]["refusal"] for r in selected),
                }
            )
    paired = []
    for seed in range(8501, 8506):
        a = next(r for r in per_fit if r["arm"] == "analytic" and r["optimizer_seed"] == seed)
        m = next(r for r in per_fit if r["arm"] == "motion2scene" and r["optimizer_seed"] == seed)
        assert a["measured"] == m["measured"]
        paired.append(
            {
                "optimizer_seed": seed,
                "matched_encounters": a["measured"],
                "motion2scene_minus_analytic_passes": m["passed"] - a["passed"],
                "percentage_points_on_completed_blocks": (
                    (100 * (m["passed"] - a["passed"]) / a["measured"]) if a["measured"] else None
                ),
            }
        )
    return {
        "assigned": master["assigned_cells"],
        "admitted_measurements": len(rows),
        "pending_or_unadmitted": master["assigned_cells"] - len(rows),
        "traversal_measured": len(traversal),
        "traversal_assigned": 480,
        "control_measured": len(rows) - len(traversal),
        "control_assigned": 120,
        "primary_traversal_panel_complete": len(traversal) == 480,
        "per_fit": per_fit,
        "paired_optimizer_differences": paired,
        "scope": "fixed development learners; independent layouts; one observed source, not source-held-out transfer",
    }


def export(path, name):
    raw = path.read_bytes()
    target = OUT / "evidence" / name
    target.write_text(
        raw.decode().replace(str(DATA), "research-data").replace(str(ROOT), "repository")
    )
    return {
        "source": str(path).replace(str(DATA), "research-data"),
        "source_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        "public_file": name,
        "public_sha256": "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest(),
    }


def main(out):
    prepared = json.loads((out / "prepared.json").read_text())
    master = json.loads(
        checked(Path(prepared["master"]["path"]), prepared["master"]["sha256"]).read_text()
    )
    results, blocks, receipts, records = [], [], [], []
    receipts += [
        export(out / n, f"independent-layout-{n}") for n in ("master.json", "prepared.json")
    ]
    for block, refs in zip(master["blocks"], prepared["blocks"]):
        folder = Path(block["directory"])
        for ref in refs.values():
            checked(Path(ref["path"]), ref["sha256"])
        record = json.loads((folder / "run_record.json").read_text())
        records.append(record)
        status = {
            "index": block["index"],
            "layout": block["spec"],
            "physics_seed": block["physics_seed"],
            "completed": sum(r["status"] == "completed" for r in record["cells"].values()),
            "assigned": 20,
            "status": record["status"],
            "admitted": False,
            "arms": None,
        }
        if (folder / "admission.json").exists():
            admission = json.loads((folder / "admission.json").read_text())
            for k in ("result", "capture_audit"):
                checked(Path(admission[k]["path"]), admission[k]["sha256"])
            result = json.loads((folder / "result.json").read_text())
            status["admitted"] = admission["admitted"]
            if admission["admitted"]:
                assert len(result["rows"]) == 20
                results.append(result)
                status["arms"] = {}
                for arm in ARMS:
                    rows = [r for r in result["rows"] if r["arm"] == arm]
                    assert len(rows) == 5
                    status["arms"][arm] = {
                        "passed": sum(r["pass"] for r in rows),
                        "measured": 5,
                        "requests": sum(r["readout"]["requested_action"] == 1 for r in rows),
                        "refusals": sum(r["readout"]["refusal"] for r in rows),
                        "peak_force_n": max(
                            r["maximum_beam_normal_force_n_through_passage"] for r in rows
                        ),
                        "resets": sum(r["reset_count"] for r in rows),
                        "exactly_one_n": sum(
                            r["maximum_beam_normal_force_n_through_passage"] == 1 for r in rows
                        ),
                    }
        for n in (
            "manifest.json",
            "run_record.json",
            "capture_audit_registration.json",
            "result.json",
            "capture_audit.json",
            "admission.json",
        ):
            path = folder / n
            if path.exists():
                receipts.append(export(path, f'layout-b{block["index"]:02d}-{n}'))
        for path in folder.glob("launch_gate_*.json"):
            receipts.append(export(path, f'layout-b{block["index"]:02d}-{path.name}'))
        blocks.append(status)
    for name in ("readout_diagnostic_registration.json", "readout_diagnostic_result.json"):
        if (out / name).exists():
            receipts.append(export(out / name, "layout-" + name))
    summary = summarize(master, results)
    summary.update(
        blocks=blocks,
        physics_completed=sum(b["completed"] for b in blocks),
        actual_gpu_hours=sum(r["budget"]["actual_contended_gpu_hours"] for r in records),
        master=artifact(out / "master.json"),
        prepared=artifact(out / "prepared.json"),
    )
    snapshot = (
        out
        / f"summary-{summary['physics_completed']:03d}-{summary['admitted_measurements']:03d}.json"
    )
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == summary, "existing outcome snapshot differs"
    else:
        write_new(snapshot, summary)
    receipts.append(export(snapshot, "independent-layout-summary.json"))
    table = [
        "| Layout / seed | Height | Uniform | Analytic | No contrast | Motion2Scene |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b in blocks:
        layout = b["layout"]
        values = [f"{b['arms'][arm]['passed']}/5" if b["arms"] else "pending" for arm in ARMS]
        height = layout["beam"]["underside_m"]
        height_text = "no beam" if layout["suite"] == "absent" else f"{height:.3f} m"
        table.append(
            f"| {layout['id']} / {b['physics_seed']} | {height_text} | {' | '.join(values)} |"
        )
    lines = [
        "# Independent-layout evaluation: frozen learners",
        "",
        f"**{summary['physics_completed']}/600 assigned physics executions completed; {summary['admitted_measurements']} admitted after block audits.** The first wave assigns 120 executions at station 0.35. Remaining assignments are preserved, and this is not a completed-panel success rate.",
        "",
        f"Measured cost in this evaluation: **{summary['actual_gpu_hours']:.6f} contended GPU h**. Training remains fixed at eleven complete encounters per arm, twenty fitted policies and source 41002. Five optimizer seeds do not create five datasets or independent sources.",
        "",
        "The headline comparison is Motion2Scene versus the strong analytic data arm. All contact failures remain in the reported denominators. Pending or unadmitted cells remain distinct from measured failures.",
        "",
        "The frozen scorer flags force >1 N and therefore accepts <=1 N, with body-origin crossing plus 0.3 s upright stability in the first episode. Peak forces, resets, denied requests and return logs remain in the complete rows. This is not a native-geometry continuous-time safety certificate or a guarantee of later neutral recovery.",
        "",
        *table,
        "",
        "## Paired fit differences on completed traversal blocks only",
        "",
        "| Optimizer seed | Matched layout/physics encounters per policy | Motion2Scene minus analytic passes | Difference on completed blocks |",
        "| --- | --- | --- | --- |",
    ]
    measured_rows = [r for result in results for r in result["rows"]]
    decision_rows = []
    for row in measured_rows:
        ref = row["decision"]
        captured = json.loads(checked(Path(ref["path"]), ref["sha256"]).read_text())
        decision_rows.append(
            {
                "cell_id": row["cell_id"],
                "arm": row["arm"],
                "optimizer_seed": row["optimizer_seed"],
                "physics_seed": row["physics_seed"],
                "capture_sha256": ref["sha256"],
                "features": captured["features"],
                "readout": row["readout"],
                "executed_action_pass": row["pass"],
            }
        )
    (OUT / "evidence/independent-layout-decisions.json").write_text(
        json.dumps(
            {
                "rows": decision_rows,
                "master_sha256": prepared["master"]["sha256"],
                "scope": "Recorded causal inputs and selected-action outcomes; alternatives not executed here remain unknown",
            },
            indent=2,
        )
        + "\n"
    )
    command_note = (
        f"Among admitted executions, {sum(r['readout']['requested_action'] == 1 for r in measured_rows)} request d040; "
        f"{sum(r['readout']['refusal'] for r in measured_rows)} log refusal. "
        f"Walking succeeds in {sum(r['pass'] and r['readout']['refusal'] for r in measured_rows)} refusal cases, "
        "so those neither-feasible predictions are contradicted by the measured fallback outcome. "
        "Refusal is a classification output, not a qualified stopping action."
    )
    lines[6:6] = [command_note, ""]
    for r in summary["paired_optimizer_differences"]:
        value = (
            "pending"
            if r["percentage_points_on_completed_blocks"] is None
            else f"{r['percentage_points_on_completed_blocks']:+.1f} pp"
        )
        lines.append(
            f"| {r['optimizer_seed']} | {r['matched_encounters']} | {r['motion2scene_minus_analytic_passes']} | {value} |"
        )
    lines += [
        "",
        "## Commands and measured contact on admitted blocks",
        "",
        "Counts below use five fits per arm and block. A refusal executes the declared walking fallback; it does not stop the robot. When no learner requests d040, these executions leave its counterfactual physical outcome unmeasured.",
        "",
        "| Block | Arm | d040 requests | Refusals | Resets | Maximum beam force through passage |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for b in blocks:
        if not b["admitted"]:
            continue
        for arm, r in b["arms"].items():
            lines.append(
                f"| {b['index']:02d} | {arm} | {r['requests']}/5 | {r['refusals']}/5 | {r['resets']} | {r['peak_force_n']:.3f} N |"
            )
    lines += [
        "",
        f"Exactly-1 N maxima: {sum(r['exactly_one_n'] for b in blocks if b['arms'] for r in b['arms'].values())}. Complete per-run probabilities, contact traces and command legality results are referenced by each published block result.",
    ]
    lines += [
        "",
        "No significance or noninferiority claim follows from this partial, single-source development comparison. Absent/raised adaptation rates and blocked refusal rates belong to separate suites; refusal does not physically stop the robot.",
        "",
        "The remaining work is to finish this fixed panel, execute the scripted comparators and perceptual shortcut diagnostics, evaluate genuinely new source ancestors under the same transition contract, and acquire the larger separately fitted data budgets with full cost accounting.",
        "",
        "[Before-run protocol](INDEPENDENT_LAYOUT_EXECUTION_V1.md) · [Complete summary](evidence/independent-layout-summary.json) · [Master assignment](evidence/independent-layout-master.json) · [Learning comparison methods draft](LEARNING_COMPARISON_METHODS_DRAFT.md)",
    ]
    lines += [
        "",
        "## Reproduce the fixed learners",
        "",
        "The small [selector bundle](evidence/selector-bundle-20260906/selectors.tar.gz), [archive manifest](evidence/selector-bundle-20260906/selectors-manifest.json) and [instructions](evidence/selector-bundle-20260906/README.md) contain all twenty original checkpoints and the exact training inputs. The [CPU reproduction check](evidence/selector-bundle-20260906/selectors-reproduction.json) matches every weight, normalization value and final training loss. These verification fits do not add primary models or physics trials. SONIC and source motion assets remain external.",
        "",
        "The [recorded evaluation inputs](evidence/independent-layout-decisions.json) also support portable CPU verification of each evaluated request, refusal and probability. Run `scripts/research/verify_motion2scene_layout_readouts.py` from the matching source snapshot. This checks the readout, not the physics or an unexecuted alternative's outcome.",
        "",
        "With the source snapshot extracted at the working directory and the two selector archive/manifest files in `bundle/`, the CPU check needs NumPy and PyTorch (versions are recorded in the selector manifest):",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/research/verify_motion2scene_layout_readouts.py \\",
        "  --bundle bundle --decisions independent-layout-decisions.json",
        "```",
        "",
        "[Measured readout reproduction receipt](evidence/independent-layout-readout-reproduction.json).",
        "",
        "[Focused related-work positioning](RELATED_WORK_POSITIONING_20260906.md). The contribution under test is the training-data construction method; motion-based environment learning and perceptive humanoid skill selection already have relevant prior work.",
    ]
    diagnostic = out / "readout_diagnostic_result.json"
    if diagnostic.exists():
        d = json.loads(diagnostic.read_text())
        lines += [
            "",
            "## Post hoc readout diagnostics",
            "",
            "These CPU observation interventions were specified after the first completed block. They measure decision sensitivity; no baseline physics success is inferred.",
            "",
            "| Arm | Measured readouts | Request changes with absent rays | Refusal changes with absent rays | Constant-phase request agreement |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm, r in d["arms"].items():
            lines.append(
                f"| {arm} | {r['measured_readouts']} | {r['request_changed_by_absent_rays']} | {r['refusal_changed_by_absent_rays']} | {r['constant_request_agrees']} |"
            )
        changes = [
            abs(p - q)
            for r in d["rows"]
            for p, q in zip(r["original"]["probabilities"], r["absent_rays"]["probabilities"])
        ]
        lines += [
            "",
            f"Across both output heads, the largest absolute probability change is {max(changes):.6g}; the mean is {np.mean(changes):.6g}. Exact cross-layout non-ray state matches: {sum(r['non_ray_features_exact'] for r in d['cross_layout_inputs'])}/{len(d['cross_layout_inputs'])} block comparisons (including each seed's self-reference).",
            "",
            "[Diagnostic protocol](LAYOUT_READOUT_DIAGNOSTIC_V1.md) · [Complete readout evidence](evidence/layout-readout_diagnostic_result.json)",
        ]
    demo = out / "demo/independent-layout-wave1.mp4"
    demo_html = ""
    if demo.exists():
        for suffix in (".mp4", ".jpg"):
            shutil.copyfile(
                demo.with_suffix(suffix), OUT / "assets" / demo.with_suffix(suffix).name
            )
        receipts.append(export(demo.with_suffix(".json"), "independent-layout-demo.json"))
        caption = "Fixed subset: all four data arms, optimizer seed 8501, both physics seeds 8511/8512, all three first-station heights (24 of 120 executions). Recorded Isaac Lab states and measured forces; MuJoCo renders the replay without integrating dynamics. Full capture intervals retain resets. One observed motion source; visual meshes are not the collision audit."
        demo_html = f'<figure><video controls preload="metadata" poster="assets/independent-layout-wave1.jpg" style="width:100%;height:auto"><source src="assets/independent-layout-wave1.mp4" type="video/mp4"></video><figcaption>{caption}</figcaption></figure>'
        lines += [
            "",
            "## Execution replay",
            "",
            caption,
            "",
            "[Video](assets/independent-layout-wave1.mp4) · [Bound recording provenance](evidence/independent-layout-demo.json)",
        ]
    lines += [
        "",
        "[Download the research source](evidence/independent-layout-source-20260906/research-source.tar.gz) · [Source manifest](evidence/independent-layout-source-20260906/research-source-manifest.json)",
        "",
        "## Validation",
        "",
        "The focused CPU regression suite passes 175 tests; one native-geometry test module skips because this CPU environment lacks the optional USD `pxr` package. Isaac Lab supplies USD for the actual geometry/physics audits. Exact commands:",
        "",
        "```bash",
        "PYTHONPATH=. .venv_research/bin/pytest -q tests/dataset_generation/test_motion2scene*.py tests/dataset_generation/test_capsule_box*.py tests/dataset_generation/test_placement_certificate.py",
        "```",
    ]
    (OUT / "INDEPENDENT_LAYOUT_V1_RESULT.md").write_text("\n".join(lines) + "\n")
    # Completed blocks only; a missing block is visibly blank rather than a zero.
    shown = blocks[:6]
    z = np.array([[b["arms"][a]["passed"] if b["arms"] else np.nan for a in ARMS] for b in shown])
    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(z), vmin=0, vmax=5, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(4), ["Uniform", "Analytic", "No contrast", "Motion2Scene"])
    ax.set_yticks(
        range(6), [f"h={b['layout']['underside_m']:.2f} / seed {b['physics_seed']}" for b in shown]
    )
    for i in range(6):
        for j in range(4):
            ax.text(
                j,
                i,
                "pending" if np.isnan(z[i, j]) else f"{int(z[i,j])}/5",
                ha="center",
                va="center",
                color="white" if z[i, j] >= 3 else "black",
            )
    ax.set_title(
        "First reserved station (0.35): contact-qualified passage\nFive fixed fits per cell; one observed motion source"
    )
    fig.savefig(OUT / "assets/independent-layout-wave1.png", dpi=180)
    fig.savefig(OUT / "assets/independent-layout-wave1.pdf")
    plt.close(fig)
    section = f"""<!-- independent-layout:start --><section id="independent-layout"><div class="wrap"><p class="eyebrow">Current stage · independent-layout policy executions</p><h2>The frozen learners<br>meet reserved scenes.</h2><p class="intro">{summary['physics_completed']}/600 assigned executions completed; {summary['admitted_measurements']} admitted after measurement audits. The first wave covers all twenty learners at three heights, two physics seeds and the first reserved station.</p><p>This is a partial, single-source evaluation. Training, observations and transition rules remain frozen; all failures and pending assignments remain visible.</p><figure><img src="assets/independent-layout-wave1.png" alt="First-wave passage counts for all four data arms; five optimizer seeds per height and physics seed" style="width:100%;height:auto"><figcaption>Each cell contains five fixed fits evaluated on the same encounter. These are repeated fits, not five independent sources. Empty cells are pending.</figcaption></figure><p><a href="INDEPENDENT_LAYOUT_V1_RESULT.md">All assigned blocks, paired differences and limits →</a> · <a href="INDEPENDENT_LAYOUT_EXECUTION_V1.md">Frozen execution protocol</a> · <a href="assets/independent-layout-wave1.pdf">Figure PDF</a></p></div></section><!-- independent-layout:end -->"""
    page = OUT / "index.html"
    section = section.replace("<figure>", f"<p>{command_note}</p><figure>", 1)
    section = section.replace(
        "</div></section>",
        demo_html
        + '<p><a href="evidence/selector-bundle-20260906/selectors.tar.gz">Download all twenty selectors and exact training inputs</a> · <a href="evidence/selector-bundle-20260906/README.md">CPU reproduction instructions</a> · <a href="evidence/selector-bundle-20260906/selectors-reproduction.json">20/20 exact CPU reproductions</a> · <a href="evidence/independent-layout-source-20260906/research-source.tar.gz">Research source archive</a></p></div></section>',
    )
    html = re.sub(
        r"<!-- independent-layout:start -->.*?<!-- independent-layout:end -->\n?",
        "",
        page.read_text(),
        flags=re.S,
    )
    html = html.replace("<main>", "<main>\n" + section, 1)
    html = re.sub(
        r'href="#[^"]+">Current progress', 'href="#independent-layout">Current progress', html
    )
    page.write_text(html)
    (OUT / "evidence/independent-layout-publication-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("physics_completed", "admitted_measurements", "actual_gpu_hours")
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
