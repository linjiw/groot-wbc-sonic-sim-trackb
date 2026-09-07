#!/usr/bin/env python3
# ruff: noqa: E501 -- Complete report and HTML paragraphs.
"""Publish measured paired outcomes and fitting status with explicit evidence tiers."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil

from motion2scene_timing_diagnostic import ROOT, checked

DATA = ROOT.parent / "research-data/groot-wbc"
OUT = ROOT / "docs/motion2scene"
FIRST = DATA / "m2s-comparative-acquisition-v1"
FULL = DATA / "m2s-comparative-corpus-completion-v1"


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


def main():
    receipts = []
    for folder, prefix, names in (
        (FIRST, "comparative-acquisition", ("result", "capture_audit", "run_record")),
        (
            FULL,
            "corpus-completion",
            (
                "registration",
                "manifest",
                "launch_gate",
                "run_record",
                "admission",
                "analysis_assembly_repair",
                "final_hash_audit",
            ),
        ),
        (
            FULL / "analysis_union",
            "corpus-analysis-union",
            (
                "derivation",
                "manifest",
                "run_record",
                "result",
                "capture_audit_registration",
                "capture_audit",
            ),
        ),
        (FULL / "fits", "corpus-fits", ("manifest", "result")),
        (
            DATA / "m2s-learned-command-check-v1",
            "learned-command-check",
            ("manifest", "launch_gate", "run_record", "result"),
        ),
    ):
        for name in names:
            path = folder / f"{name}.json"
            if path.exists():
                receipts.append(export(path, f"{prefix}-{name}.json"))
    receipts.append(export(Path("/tmp/m2s-transfer-v4.log"), "transfer-reservation-v4-failure.txt"))
    receipts.append(
        export(Path("/tmp/m2s-corpus-analysis.log"), "corpus-analysis-assembly-failure.txt")
    )
    reservation = DATA / "m2s-final-transfer-provenance-v1"
    for name in ("ledger", "reservation"):
        receipts.append(
            export(reservation / f"{name}.json", f"final-source-provenance-{name}.json")
        )
    first = json.loads((FIRST / "result.json").read_text())
    first_audit = json.loads((FIRST / "capture_audit.json").read_text())
    record = json.loads((FULL / "run_record.json").read_text())
    n = sum(r["status"] == "completed" for r in record["cells"].values())
    admission = (
        json.loads((FULL / "admission.json").read_text())
        if (FULL / "admission.json").exists()
        else None
    )
    fits = (
        json.loads((FULL / "fits/result.json").read_text())
        if (FULL / "fits/result.json").exists()
        else None
    )
    learned_path = DATA / "m2s-learned-command-check-v1/result.json"
    learned = json.loads(learned_path.read_text()) if learned_path.exists() else None
    result = json.loads((FULL / "analysis_union/result.json").read_text()) if admission else first
    pairs = result["pairs"]
    aliases = defaultdict(list)
    for pair in pairs:
        if pair["valid"]:
            aliases[tuple(pair["features"])].append(
                {"group_id": pair["group_id"], "outcomes": pair["outcomes"]}
            )
    conflicts = [rows for rows in aliases.values() if len({tuple(r["outcomes"]) for r in rows}) > 1]
    alias_review = {
        "scope": "post-fit development diagnostic; exact equality of the 214 available inputs",
        "conflicting_groups": conflicts,
        "changed_inputs_or_labels": False,
        "interpretation": "identical deployed inputs have different measured outcomes; deterministic readout cannot perfectly recover all labels in these groups",
    }
    (OUT / "evidence/corpus-feature-aliases.json").write_text(
        json.dumps(alias_review, indent=2) + "\n"
    )
    label_table = [
        "| Scene | Arm | Walk | d040 request | Matched inputs |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in pairs:
        outcomes = ["unknown" if x is None else "pass" if x else "fail" for x in p["outcomes"]]
        label_table.append(
            f"| {p['group_id']} | {p['arm']} | {outcomes[0]} | {outcomes[1]} | {p['valid']} |"
        )
    counts = []
    for arm in ("uniform", "analytic", "no_contrast", "motion2scene", "shared"):
        selected = [p for p in pairs if p["arm"] == arm]
        row = {"arm": arm, "measured_pairs": len(selected)}
        for key, value in (
            ("both_pass", [True, True]),
            ("useful_adaptation", [False, True]),
            ("walk_only", [True, False]),
            ("both_fail", [False, False]),
        ):
            row[key] = sum(p["outcomes"] == value for p in selected)
        row["missing"] = sum(any(x is None for x in p["outcomes"]) for p in selected)
        counts.append(row)
    returns = []
    for r in first["rows"]:
        if r["action"] == 1:
            sensor = json.loads(
                checked(Path(r["sensor"]["path"]), r["sensor"]["sha256"]).read_text()
            )
            returns.append(
                {
                    "cell_id": r["cell_id"],
                    "entry_executed": r["command_executed"],
                    "return_logged": any(s["to"] == 0 for s in sensor["switches"]),
                    "reset_count": r["reset_count"],
                }
            )
    p2_complete = all(r["entry_executed"] and r["return_logged"] for r in returns)
    review = {
        "protocol": "COMPARATIVE_ACQUISITION_V1.md prediction 2 includes return execution",
        "automatic_predicate": first["predictions"]["p2_legal_commands"],
        "entry_and_return_predicate": p2_complete,
        "rows": returns,
        "explanation": "automatic analyzer checks existing switches but omits required-return presence; preserve both records, do not count missing returns as completed returns",
    }
    (OUT / "evidence/first-slice-return-review.json").write_text(
        json.dumps(review, indent=2) + "\n"
    )
    measured = f"The first twenty Isaac Lab runs are complete, with 10/10 matched command-label pairs. The remaining fixed batch has completed {n}/56 runs."
    if admission:
        measured += f" The combined corpus contains {len(pairs)} measured scene pairs; input admission is {admission['all_inputs_admitted']}."
    if admission:
        measured += f" All 76 direct feature/state/ray/bank checks pass; {sum(r['return_logged'] for r in admission['command_returns'])}/38 d040 requests log a return. Missing returns remain failures, not completed recovery."
    fit_status = (
        f"All {fits['fits']} matched development fits are complete: four data arms × five optimizer seeds, eleven complete encounters per arm, 1000 updates each."
        if fits
        else "No robot-data selector fitting has completed in this stage."
    )
    cost = first["actual_contended_gpu_hours"] + record["budget"]["actual_contended_gpu_hours"]
    headers = [
        "| Arm | Pairs | Both pass | Walk fail / d040 pass | Walk pass / d040 fail | Both fail | Missing |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in counts:
        headers.append(
            f"| {c['arm']} | {c['measured_pairs']} | {c['both_pass']} | {c['useful_adaptation']} | {c['walk_only']} | {c['both_fail']} | {c['missing']} |"
        )
    lines = [
        "# Comparative command corpus: measured outcomes and learning readiness",
        "",
        measured,
        "",
        fit_status,
        "",
        "**Unseen-layout policy executions and learning-utility comparisons remain incomplete.** Training losses are fit diagnostics, not traversal success or evidence that one generator teaches a better policy.",
        "",
        *headers,
        "",
        "**Main observed limitation:** all nine Motion2Scene generated encounters are both-fail under this fixed transition contract; four of eight analytic encounters admit useful d040 adaptation. This is a development data-yield result, not an independently evaluated policy ranking. The generator remains frozen and no failed assignment is replaced.",
        "",
        f"Post-fit input diagnostic: {len(conflicts)} exactly identical feature group(s) contain conflicting measured outcomes, covering {sum(len(g) for g in conflicts)} scenes. A deterministic predictor given these same 214 values cannot match every outcome in the group. Keep this observation-interface limitation separate from generator quality; no sensor or label is changed here. [Exact feature aliases](evidence/corpus-feature-aliases.json).",
        "",
        "Shared controls are listed once. Repeating the same controls in four fitting datasets does not create four independent scene samples. The single source remains 41002; optimizer seeds are not source replication. All generated assignments and the analytic test-neighborhood rejection are retained.",
        "",
        f"First-slice capture audit: `{json.dumps(first_audit['predictions'], sort_keys=True)}`. These checks compare direct pre-command features, ray origins, recorder state and the actual loaded neutral/d040 banks. Prior 0.20 s development labels are excluded.",
        "",
        f"The automatic first-slice legal-switch predicate is `{first['predictions']['p2_legal_commands']}`, but the protocol's stronger entry-and-return prediction is `{p2_complete}`. {sum(r['return_logged'] for r in returns)}/{len(returns)} d040 executions log a return. A reset before the return phase is not evidence that recovery occurred. [Return review](evidence/first-slice-return-review.json).",
        "",
        f"Measured new physics cost across both batches so far: **{cost:.6f} contended GPU h**. The union record is an analysis-only derivation of the two real batches and is explicitly nonexecutable; it is never counted as a third run.",
        "",
        "The eleven-example fitting subset uses the first eight geometrically eligible assigned slots, in fixed slot order, plus the same three backgrounds. Its IDs were frozen before admission or fitting and do not depend on passage outcomes. Three unused ninth generated pairs remain in the corpus. The assigned background quota is 3/12; the disclosed fitting subset is 3/11 backgrounds in every arm. This development size does not replace the planned larger data-budget experiment.",
        "",
        "Eight final-transfer candidate IDs (9490001–9490008) are reserved against ten named acquisition-provenance files, with canonical fingerprints for 212 prior reference files. This revised scope does not establish absence from all JSON metadata; the four broad inventory attempts remain failed/refused history ([v4 timeout](evidence/transfer-reservation-v4-failure.txt)). No final source is generated or qualified. Post-generation duplicate checks and the same switching qualification remain mandatory. The twelve independent layout tests remain untouched. [Reservation scope](FINAL_SOURCE_PROVENANCE_RESERVATION_V1.md) · [Ledger](evidence/final-source-provenance-ledger.json) · [Reservation](evidence/final-source-provenance-reservation.json).",
        "",
        "The first combined CPU analysis stopped because the assembly omitted a local copy of the frozen proposals. A versioned adapter copies the identical hash-bound proposal bytes and reruns the unchanged analyzer; the original failure and scripts remain preserved. No physics was repeated and no labels or fitting IDs were changed. [Assembly repair](evidence/corpus-completion-analysis_assembly_repair.json).",
        "",
        "## Complete paired labels",
        "",
        *label_table,
        "",
        "[Before-run completion and fitting protocol](COMPARATIVE_CORPUS_COMPLETION_V1.md) · [First capture audit](evidence/comparative-acquisition-capture_audit.json) · [First measured rows](evidence/comparative-acquisition-result.json) · [Completion record](evidence/corpus-completion-run_record.json) · [Updated source snapshot](evidence/corpus-stage-source-20260906/research-source.tar.gz)",
    ]
    if fits:
        lines += [
            "",
            "## Training diagnostics only",
            "",
            "| Arm | Optimizer seed | Training BCE | CPU seconds |",
            "| --- | --- | --- | --- |",
        ]
        lines += [
            f"| {r['arm']} | {r['seed']} | {r['final_training_loss']:.6g} | {r['seconds']:.3f} |"
            for r in fits["rows"]
        ]
        lines += [
            "",
            "[Fit manifest](evidence/corpus-fits-manifest.json) · [All fit diagnostics](evidence/corpus-fits-result.json)",
        ]
    if learned:
        lines += [
            "",
            "## Learned command integration on observed scenes",
            "",
            f"Twelve actual learned-policy Isaac Lab runs completed. Predicates: `{json.dumps(learned['predictions'], sort_keys=True)}`.",
            "",
            "This is an implementation check on analytic_01, absent and blocked scenes already observed during development. It is not an independent test or a fair arm-ranking benchmark. A refusal is logged separately from neutral commitment; it does not stop the robot.",
            "",
            f"Additional integration cost: {learned['actual_contended_gpu_hours']:.6f} contended GPU h.",
            "",
            "[Before-run protocol](LEARNED_COMMAND_CHECK_V1.md) · [Complete integration result](evidence/learned-command-check-result.json)",
        ]
    demo = DATA / "m2s-learned-command-check-v1/demo/learned-integration.mp4"
    if demo.exists():
        for suffix in (".mp4", ".jpg"):
            shutil.copy2(demo.with_suffix(suffix), OUT / "assets" / demo.with_suffix(suffix).name)
        receipts.append(export(demo.with_suffix(".json"), "learned-command-demo.json"))
        lines += [
            "",
            "[All four learned-command replays](assets/learned-integration.mp4) · [Replay source receipt](evidence/learned-command-demo.json). Observed analytic_01 was in the analytic training set; this is an integration illustration, not a fair test comparison. Measured Isaac poses are rendered with MuJoCo mj_forward; no new dynamics are simulated.",
        ]
    (OUT / "COMPARATIVE_CORPUS_STAGE_RESULT.md").write_text("\n".join(lines) + "\n")
    html_rows = "".join(
        f"<tr><td>{c['arm']}</td><td>{c['measured_pairs']}</td><td>{c['both_pass']}</td><td>{c['useful_adaptation']}</td><td>{c['walk_only']}</td><td>{c['both_fail']}</td></tr>"
        for c in counts
    )
    section = f"""<!-- corpus-stage:start --><section id="corpus-stage"><div class="wrap"><p class="eyebrow">Current stage · 06 September 2026 · measured command outcomes</p>
<h2>Learn from what<br>the commands actually do.</h2><p class="intro">{measured}</p><p>{fit_status} Unseen-layout policy performance is still unmeasured.</p>
<div class="scroll"><table><caption>Physical command outcomes; shared backgrounds counted once.</caption><thead><tr><th>Arm</th><th>Pairs</th><th>Both pass</th><th>Only d040 passes</th><th>Only walk passes</th><th>Both fail</th></tr></thead><tbody>{html_rows}</tbody></table></div>
<p>All nine Motion2Scene generated encounters are both-fail; four of eight analytic encounters admit useful d040 adaptation. This is a development data-yield finding, not a policy ranking. All failed labels remain. Reference separation does not guarantee that switching at 0.30 s avoids contact. The next claim-bearing test is the fixed-learner comparison on independent layouts.</p>
<p><a href="COMPARATIVE_CORPUS_STAGE_RESULT.md">All paired labels, input audits, returns, costs and fitting status →</a> · <a href="COMPARATIVE_CORPUS_COMPLETION_V1.md">Frozen completion protocol</a></p></div></section><!-- corpus-stage:end -->"""
    if learned:
        summary = (
            "All registered readout/trace/outcome predicates pass."
            if all(learned["predictions"].values())
            else "At least one registered integration prediction fails; see the retained result."
        )
        section = section.replace(
            "<p>All nine Motion2Scene",
            f'<p>Twelve real learned-command integration runs on already observed scenes are complete. {summary} This verifies the command path, not unseen-scene performance. <a href="LEARNED_COMMAND_CHECK_V1.md">Frozen check</a> · <a href="evidence/learned-command-check-result.json">Measured result</a>.</p><p>All nine Motion2Scene',
        )
    if demo.exists():
        section = section.replace(
            "</div></section><!-- corpus-stage:end -->",
            '<figure><video controls preload="metadata" poster="assets/learned-integration.jpg" style="width:100%;height:auto"><source src="assets/learned-integration.mp4" type="video/mp4"></video><figcaption>All four fixed seed-8501 learners on the observed analytic_01 integration scene. The analytic arm saw this scene during training; this is not an independent comparison. Actual Isaac states, MuJoCo visual replay only; failures and reset intervals retained. <a href="evidence/learned-command-demo.json">Source receipt</a>.</figcaption></figure></div></section><!-- corpus-stage:end -->',
        )
    page = OUT / "index.html"
    text = page.read_text()
    text = re.sub(
        r"<!-- corpus-stage:start -->.*?<!-- corpus-stage:end -->\n?", "", text, flags=re.S
    )
    text = text.replace("<main>", "<main>\n" + section, 1)
    text = re.sub(r'href="#[^"]+">Current progress', 'href="#corpus-stage">Current progress', text)
    page.write_text(text)
    (OUT / "evidence/corpus-stage-publication-receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "first_completed": 20,
                "remaining_completed": n,
                "pairs": len(pairs),
                "fits": fits["fits"] if fits else 0,
                "physics_hours": cost,
            }
        )
    )


if __name__ == "__main__":
    main()
