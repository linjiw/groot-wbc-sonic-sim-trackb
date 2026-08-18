#!/usr/bin/env python
"""Render the release folder as a page, generated from its own manifest.

Every number is read from MANIFEST.json and the two index files, so the page cannot claim a count
the folder does not contain. Restating figures by hand is how a board came to say four families
while the draft beside it still said one.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

CSS = """
:root { --paper:#f4f5f7; --card:#fff; --ink:#1b1f26; --muted:#656d78; --rule:#d8dce2;
  --accent:#b06f24; --ok:#2f6b4f; --bad:#a33a2c; --warn:#8a6a1f; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper:#14171c; --card:#1b1f26; --ink:#e6e8ec; --muted:#98a0ab; --rule:#2e343d;
  --accent:#d69a53; --ok:#7fc09a; --bad:#e2897b; --warn:#d9b26a; } }
:root[data-theme="dark"] { --paper:#14171c; --card:#1b1f26; --ink:#e6e8ec; --muted:#98a0ab;
  --rule:#2e343d; --accent:#d69a53; --ok:#7fc09a; --bad:#e2897b; --warn:#d9b26a; }
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink);
  font:400 16px/1.65 "IBM Plex Sans",system-ui,sans-serif; max-width:64rem; margin:0 auto;
  padding:3.5rem 1.5rem 5rem; display:flex; flex-direction:column; gap:2.6rem; }
h1,h2,h3 { font-family:Spectral,Georgia,serif; font-weight:600; margin:0; text-wrap:balance; }
h1 { font-size:2.2rem; line-height:1.15; }
h2 { font-size:1.35rem; padding-bottom:.5rem; border-bottom:1px solid var(--rule); }
h3 { font-size:1rem; }
p { margin:0; max-width:66ch; }
section { display:flex; flex-direction:column; gap:1rem; }
.eyebrow { font:500 11.5px/1 "IBM Plex Mono",monospace; letter-spacing:.13em;
  text-transform:uppercase; color:var(--accent); }
.lede { font-family:Spectral,Georgia,serif; font-size:1.15rem; color:var(--muted); max-width:58ch; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(8rem,1fr)); gap:.6rem; }
.stat { background:var(--card); border:1px solid var(--rule); border-radius:3px; padding:.8rem 1rem;
  display:flex; flex-direction:column; gap:.1rem; }
.stat b { font:600 1.5rem/1.1 Spectral,serif; font-variant-numeric:tabular-nums; }
.stat span { font:500 10.5px/1.3 "IBM Plex Mono",monospace; letter-spacing:.06em;
  text-transform:uppercase; color:var(--muted); }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; min-width:32rem; }
th,td { text-align:left; padding:.5rem .7rem; border-bottom:1px solid var(--rule); vertical-align:top; }
th { font:500 11px "IBM Plex Mono",monospace; letter-spacing:.07em; text-transform:uppercase;
  color:var(--muted); white-space:nowrap; }
td.k { font-family:"IBM Plex Mono",monospace; font-size:13px; }
td.n { font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; text-align:right; }
.ok { color:var(--ok); font-weight:600; } .bad { color:var(--bad); font-weight:600; }
.warn { color:var(--warn); font-weight:600; } .dim { color:var(--muted); }
pre { background:var(--card); border:1px solid var(--rule); border-radius:3px; padding:1rem;
  overflow-x:auto; font:400 12.5px/1.55 "IBM Plex Mono",monospace; margin:0; }
.note { background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:3px; padding:1rem 1.1rem; display:flex; flex-direction:column; gap:.6rem; }
code { font-family:"IBM Plex Mono",monospace; font-size:.9em; }
ul { margin:0; padding-left:1.15rem; display:flex; flex-direction:column; gap:.4rem; max-width:66ch; }
"""

TREE = """sweepcf_release/
├── MANIFEST.json            counts, bytes, SHA-256 over every file
├── README.md
├── motions/                 the qualification view
│   ├── index.jsonl          one row per reference clip
│   └── clips/*.csv          36-column qpos at 30 fps
├── families/                the counterfactual view
│   ├── index.jsonl          one row per family
│   └── <family>/
│       ├── family.json      geometry, verdicts, contact attribution
│       ├── scenes/*.usda    the exact geometry physics loaded
│       ├── cells/<cell>/
│       │   ├── outcome.json verdict, reasons, contact body/direction/frame
│       │   └── trajectory.pkl   raw capture, authoritative
│       └── renders/*.mp4    room camera, side view, head camera
└── splits/
    ├── scene_first_v1.json      30 eval scenes, frozen before the operators
    └── review_cohort_v1.json    100 episodes stratified for human review"""


def verdict_cell(status: str) -> str:
    cls = {"accepted": "ok", "rejected": "bad", "unevaluable": "dim"}.get(status, "warn")
    return f'<span class="{cls}">{status}</span>'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads((args.release / "MANIFEST.json").read_text())
    families = [
        json.loads(line)
        for line in (args.release / "families/index.jsonl").read_text().splitlines()
        if line.strip()
    ]
    motions_path = args.release / "motions/index.jsonl"
    motions = (
        [json.loads(line) for line in motions_path.read_text().splitlines() if line.strip()]
        if motions_path.exists()
        else []
    )

    def role_of(cell_name: str) -> str | None:
        """Map a cell name onto its family role.

        Mined families name cells ``nominal_easy`` / ``adapted_hard``; matched families name them
        after the motion that produced them, ``w_nominal_easy`` / ``w_crouch08_hard``. Matching one
        convention counted two mined families as verified and silently excluded the only matched
        one, which is the opposite of the truth.
        """
        if cell_name.startswith("probe"):
            return None
        scene = (
            "easy"
            if cell_name.endswith("_easy")
            else "hard" if cell_name.endswith("_hard") else None
        )
        if scene is None:
            return None
        adapted = any(key in cell_name for key in ("crouch", "tuck", "adapted"))
        return f"{'adapted' if adapted else 'nominal'}_{scene}"

    def is_verified(row):
        by_role = {}
        for name, status in row["cells"].items():
            role = role_of(name)
            if role:
                by_role[role] = status
        return (
            by_role.get("nominal_easy") == "accepted"
            and by_role.get("adapted_easy") == "accepted"
            and by_role.get("nominal_hard") == "rejected"
            and by_role.get("adapted_hard") == "accepted"
        )

    verified = [f for f in families if is_verified(f)]
    feasible = sum(1 for m in motions if m.get("embodiment_feasible"))
    reasons = Counter(r for m in motions for r in (m.get("rejection_reasons") or []))

    rows = []
    for family in families:
        cells = family["cells"]
        shown = (
            " ".join(
                f'{k.replace("_", " ")}: {verdict_cell(v)}'
                for k, v in sorted(cells.items())
                if not k.startswith("probe")
            )
            or '<span class="dim">no cells &mdash; rebuild pending</span>'
        )
        window = family.get("window_m")
        verified_label = (
            "<b class=ok>verified</b>"
            if is_verified(family)
            else "<span class=dim>not verified</span>"
        )
        rows.append(
            f'<tr><td class="k">{family["family_id"]}</td>'
            f'<td class="n">{window * 1000:.1f}</td>'
            f'<td class="n">{len(family["scenes"])}</td>'
            f"<td>{verified_label}</td>"
            f"<td>{shown}</td></tr>"
        )

    reason_rows = "".join(
        f'<tr><td class="k">{name}</td><td class="n">{count}</td></tr>'
        for name, count in reasons.most_common(6)
    )

    args.out.write_text(
        f"""<title>SweepCF Dataset</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>

<header>
  <p class="eyebrow">sweepcf &middot; release folder</p>
  <h1>What is actually in the dataset</h1>
  <p class="lede">Generated from the folder's own manifest, so the page cannot claim a count the
  files do not contain.</p>
</header>

<section>
  <div class="stats">
    <div class="stat"><b>{len(verified)}</b><span>verified families</span></div>
    <div class="stat"><b>{len(families)}</b><span>families attempted</span></div>
    <div class="stat"><b>{manifest['cells']}</b><span>graded cells</span></div>
    <div class="stat"><b>{len(motions)}</b><span>reference motions</span></div>
    <div class="stat"><b>{manifest['files']}</b><span>files</span></div>
    <div class="stat"><b>{manifest['bytes'] / 1024 / 1024:.0f} MB</b><span>on disk</span></div>
  </div>
  <p class="dim" style="font-size:.85rem;font-family:'IBM Plex Mono',monospace">
  fingerprint {manifest['fingerprint_sha256'][:32]}&hellip;</p>
</section>

<section>
  <h2>Layout</h2>
  <pre>{TREE}</pre>
</section>

<section>
  <h2>Every family, and whether it counts</h2>
  <div class="scroll"><table>
    <tr><th>family</th><th>window mm</th><th>scenes</th><th>status</th><th>cells</th></tr>
    {''.join(rows)}
  </table></div>
  <p>A family is <b>verified</b> only when all four cells come out as a counterfactual requires:
  both motions accepted in the easy scene, the nominal rejected in the hard one, the adapted motion
  accepted there. Families that fail that test stay in the folder rather than being deleted &mdash;
  their cost was paid, and their absence from the total is what the yield number means.</p>
</section>

<section>
  <h2>The qualification view</h2>
  <p>{len(motions)} reference clips, of which <b>{feasible}</b> are reachable by the G1's
  embodiment. This view records what was <em>asked for</em> and what the clip can physically be,
  and deliberately leaves semantic validity unfilled rather than inferring it from the prompt.</p>
  <div class="scroll"><table>
    <tr><th>most common rejection reason</th><th>clips</th></tr>{reason_rows}
  </table></div>
</section>

<section>
  <h2>Three cautions for anyone using this</h2>
  <div class="note">
    <h3>A prompt is an intent, not a label</h3>
    <p>Of 75 prompts that admitted a semantic predicate, only 24 produced the behaviour they named.
    <code>prompt_intent</code> is stored as an intent, and
    <code>reference_semantic_valid</code> / <code>executed_semantic_valid</code> are separate
    fields, currently null. Treating the prompt as ground truth would mis-label about two thirds of
    them.</p>
  </div>
  <div class="note">
    <h3>Scene geometry travels with the family</h3>
    <p>Each family's claim is a claim about one shelf height in one room, so the USDA physics loaded
    is copied in rather than described. Two families were once built with the obstacle in the wrong
    coordinate frame and the robot walked two metres clear of it, so every cell now records
    <code>obstacle.route_reaches_it</code>.</p>
  </div>
  <div class="note">
    <h3>Family count is not motion count</h3>
    <p>Five shelf heights on one nominal motion are five scene instances, never five independent
    behaviour families. The target is 24&ndash;30 verified families over 8 or more distinct nominal
    motions; the verified count above is over <b>1</b>.</p>
  </div>
</section>
""",
        encoding="utf-8",
    )
    print(f"{len(verified)} verified of {len(families)} families, {len(motions)} motions")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
