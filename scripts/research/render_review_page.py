#!/usr/bin/env python
"""Render the review cohort as a self-contained page a person can score and hand back.

Three constraints shaped this. A published artifact cannot hand the viewer a file -- the sandbox
blocks downloads -- so results come back through copy-to-clipboard rather than a saved CSV.
Progress is kept in localStorage, because a hundred episodes is more than one sitting. And the
automatic verdict is never shown: the cohort exists to measure the gates, so a reviewer who sees the
gate's answer first measures their agreement with an anchor rather than the episode.
"""

from __future__ import annotations

import argparse
import base64
import csv
from pathlib import Path

QUESTIONS = (
    ("traversed", "Got past the obstacle without touching it?"),
    ("upright", "Stayed upright and in control throughout?"),
    ("behaviour", "Performed the behaviour its name claims?"),
)


#: Names that actually claim a behaviour. A first reviewer pass answered "can't tell" to the
#: behaviour question across roughly half the cohort, correctly: `density_moderate`, `combo09` and
#: `01_single_text_prompt__factory_aisle__p0` claim nothing a person could check. Asking anyway
#: manufactures unanswerable questions and buries the cards where the question is real.
BEHAVIOUR_WORDS = ("crouch", "duck", "tuck", "nominal", "walk", "squat", "step", "bend", "side")


def claims_behaviour(episode_id: str) -> bool:
    return any(word in episode_id.lower() for word in BEHAVIOUR_WORDS)


def has_obstacle(episode_id: str, clearance) -> bool:
    """Probes run on a bare plane, so there is nothing to clear and the first question cannot
    apply. The same reviewer pass flagged every probe as unjudgeable for exactly this reason."""
    return clearance not in ("", None) and "probe" not in episode_id.lower()


#: Names that actually claim a behaviour. A first reviewer pass answered "can't tell" to the
#: behaviour question on roughly half the cohort, correctly: `density_moderate`, `combo09` and
#: `01_single_text_prompt__factory_aisle__p0` claim nothing a person could check. Asking anyway
#: manufactures unanswerable questions and buries the cards where the question is real.
BEHAVIOUR_WORDS = ("crouch", "duck", "tuck", "nominal", "walk", "squat", "step", "bend", "side")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    with open(args.pack / "cohort_evidence.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    cards, embedded, asked = [], 0, 0
    for index, row in enumerate(rows):
        if not row["sheet"]:
            continue
        image = args.pack / row["sheet"]
        if not image.exists():
            continue
        uri = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
        embedded += 1
        reduced = row.get("evidence", "").startswith("reduced")
        note = (
            '<p class="warn">Reduced evidence — only the root path and height were recorded, so '
            "you can judge balance and route but <b>not</b> whether it touched anything. Answer "
            "“can’t tell” for the first question.</p>"
            if reduced
            else ""
        )
        clearance = row.get("min_clearance_mm", "")
        gap = (
            f'<span class="chip">closest approach {clearance} mm</span>'
            if clearance not in ("", None)
            else ""
        )
        askable = {
            "traversed": has_obstacle(row["episode_id"], clearance) and not reduced,
            "upright": True,
            "behaviour": claims_behaviour(row["episode_id"]),
        }
        asked += sum(1 for v in askable.values() if v)
        parts = []
        for key, text in QUESTIONS:
            if askable[key]:
                parts.append(f"""<div class="q" data-ep="{row['episode_id']}" data-q="{key}">
              <span class="qt">{text}</span>
              <div class="opts">
                <button data-v="yes">yes</button>
                <button data-v="no">no</button>
                <button data-v="unsure">can’t tell</button>
              </div>
            </div>""")
            else:
                why = (
                    "no obstacle in this episode"
                    if key == "traversed"
                    else "the name states no behaviour to check"
                )
                parts.append(f"""<div class="q na"><span class="qt">{text}</span>
              <span class="chip">not asked — {why}</span></div>""")
        buttons = "".join(parts)
        cards.append(f"""
  <article class="card" id="ep{index}">
    <header class="cardhead">
      <span class="num">{embedded}</span>
      <code>{row['episode_id']}</code>
      {gap}
    </header>
    <img src="{uri}" alt="frames from {row['episode_id']}">
    {note}
    {buttons}
  </article>""")

    page = f"""<title>SweepCF Review</title>
<link rel="preconnect" href="https://fonts.googleapis.com">

  link rel="stylesheet"
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
  href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{ --paper:#f4f5f7; --card:#fff; --ink:#1b1f26; --muted:#656d78; --rule:#d8dce2;
  --accent:#b06f24; --yes:#2f6b4f; --no:#a33a2c; --unsure:#8a6a1f; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper:#14171c; --card:#1b1f26; --ink:#e6e8ec; --muted:#98a0ab; --rule:#2e343d;
  --accent:#d69a53; --yes:#7fc09a; --no:#e2897b; --unsure:#d9b26a; }} }}
:root[data-theme="dark"] {{ --paper:#14171c; --card:#1b1f26; --ink:#e6e8ec; --muted:#98a0ab;
  --rule:#2e343d; --accent:#d69a53; --yes:#7fc09a; --no:#e2897b; --unsure:#d9b26a; }}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink);
  font:400 16px/1.6 "IBM Plex Sans",system-ui,sans-serif; max-width:58rem; margin:0 auto;
  padding:3rem 1.2rem 8rem; display:flex; flex-direction:column; gap:1.6rem; }}
h1,h2,h3 {{ font-family:Spectral,Georgia,serif; font-weight:600; margin:0; text-wrap:balance; }}
h1 {{ font-size:2.1rem; line-height:1.15; }}
h2 {{ font-size:1.25rem; padding-bottom:.4rem; border-bottom:1px solid var(--rule); }}
p {{ margin:0; max-width:66ch; }}
.eyebrow {{ font:500 11.5px/1 "IBM Plex Mono",monospace; letter-spacing:.13em;
  text-transform:uppercase; color:var(--accent); }}
.lede {{ font-family:Spectral,Georgia,serif; font-size:1.15rem; color:var(--muted); max-width:56ch; }}
.intro {{ background:var(--card); border:1px solid var(--rule); border-left:3px solid var(--accent);
  border-radius:4px; padding:1.1rem 1.2rem; display:flex; flex-direction:column; gap:.7rem; }}
.card {{ background:var(--card); border:1px solid var(--rule); border-radius:4px; padding:.9rem;
  display:flex; flex-direction:column; gap:.6rem; scroll-margin-top:1rem; }}
.card.done {{ border-left:3px solid var(--yes); }}
.cardhead {{ display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; font-size:.85rem; }}
.num {{ font:600 12px "IBM Plex Mono",monospace; background:var(--accent); color:#fff;
  border-radius:99px; padding:.15rem .5rem; }}
code {{ font-family:"IBM Plex Mono",monospace; font-size:.82rem; color:var(--muted); }}
.chip {{ font:500 11px "IBM Plex Mono",monospace; border:1px solid var(--rule);
  border-radius:99px; padding:.15rem .55rem; color:var(--muted); }}
img {{ width:100%; border-radius:3px; background:#0001; }}
.warn {{ font-size:.85rem; color:var(--unsure); }}
.q.na {{ opacity:.55; }}
.q.na .qt {{ text-decoration:line-through; }}
.q {{ display:flex; align-items:center; justify-content:space-between; gap:.8rem;
  border-top:1px solid var(--rule); padding-top:.5rem; flex-wrap:wrap; }}
.qt {{ font-size:.9rem; }}
.opts {{ display:flex; gap:.35rem; }}
.opts button {{ font:500 13px "IBM Plex Sans",sans-serif; border:1px solid var(--rule);
  background:transparent; color:var(--ink); border-radius:99px; padding:.3rem .8rem;
  cursor:pointer; }}
.opts button:hover {{ border-color:var(--accent); }}
.opts button[aria-pressed="true"][data-v="yes"] {{ background:var(--yes); color:#fff; border-color:var(--yes); }}
.opts button[aria-pressed="true"][data-v="no"] {{ background:var(--no); color:#fff; border-color:var(--no); }}
.opts button[aria-pressed="true"][data-v="unsure"] {{
  background:var(--unsure); color:#fff; border-color:var(--unsure); }}
.bar {{ position:fixed; left:0; right:0; bottom:0; background:var(--card);
  border-top:1px solid var(--rule); padding:.7rem 1.2rem; display:flex; align-items:center;
  gap:.9rem; flex-wrap:wrap; z-index:9; }}
.bar progress {{ flex:1; min-width:8rem; height:.6rem; }}
.bar button {{ font:500 13px "IBM Plex Sans",sans-serif; border:1px solid var(--accent);
  background:var(--accent); color:#fff; border-radius:4px; padding:.4rem .9rem; cursor:pointer; }}
.bar button.ghost {{ background:transparent; color:var(--accent); }}
textarea {{ width:100%; min-height:9rem; font:400 12px/1.5 "IBM Plex Mono",monospace;
  background:var(--paper); color:var(--ink); border:1px solid var(--rule); border-radius:4px;
  padding:.6rem; }}
ol,ul {{ margin:0; padding-left:1.2rem; display:flex; flex-direction:column; gap:.35rem; max-width:66ch; }}
</style>

<header>
  <p class="eyebrow">sweepcf · reviewer pack · cohort v1</p>
  <h1>Does the automatic gate agree with a person?</h1>
  <p class="lede">You are scoring {embedded} simulated episodes of a humanoid walking past an
  obstacle. Your answers measure the software, not the robot.</p>
</header>

<section class="intro">
  <h2>What you are looking at</h2>
  <p>Each strip shows six moments from one episode, left to right, seen from the side. The
  <b>brown block</b> is a shelf or wall; the <b>reddish shape</b> is the robot's body, drawn to its
  true size. Grey band at the bottom is the floor.</p>
  <p><b>Why this matters.</b> Software already judged every one of these episodes automatically.
  We need to know how often it is wrong, and that only works if you judge independently — so the
  software's answer is deliberately hidden from you here.</p>
  <h3>The three questions</h3>
  <ol>
    <li><b>Got past without touching?</b> — did the body stay clear of the brown block? If a
    “closest approach” number is shown it is the measured gap; negative means overlap.</li>
    <li><b>Stayed upright?</b> — no falling, stumbling, or drifting off course.</li>
    <li><b>Performed the behaviour its name claims?</b> — a clip named <code>crouch</code> should
    visibly lower; one named <code>tuck</code> should draw its arms in; a plain walk should just
    walk.</li>
  </ol>
  <p><b>“Can’t tell” is a real answer</b> and more useful than a guess. Some episodes only recorded
  the robot's path, not its body — those are marked, and the first question is unanswerable for them.</p>
  <p>Progress saves in this browser automatically. When you finish, press <b>Copy results</b> at the
  bottom and paste them back to the team.</p>
</section>

{"".join(cards)}

<div class="bar">
  <span id="count">0 / {asked}</span>
  <progress id="prog" value="0" max="{asked}"></progress>
  <button id="copy">Copy results</button>
  <button id="show" class="ghost">Show results</button>
</div>

<section id="results" style="display:none">
  <h2>Results</h2>
  <p>Select all and copy, or use the button above.</p>
  <textarea id="out" readonly></textarea>
</section>

<script>
const KEY = "sweepcf_review_v1";
const answers = JSON.parse(localStorage.getItem(KEY) || "{{}}");
const total = {asked};

function render() {{
  document.querySelectorAll(".q").forEach(q => {{
    const k = q.dataset.ep + "|" + q.dataset.q;
    q.querySelectorAll("button").forEach(b =>
      b.setAttribute("aria-pressed", answers[k] === b.dataset.v ? "true" : "false"));
  }});
  document.querySelectorAll(".card").forEach(c => {{
    const qs = [...c.querySelectorAll(".q")];
    const done = qs.every(q => answers[q.dataset.ep + "|" + q.dataset.q]);
    c.classList.toggle("done", done);
  }});
  const n = Object.keys(answers).length;
  document.getElementById("count").textContent = n + " / " + total;
  document.getElementById("prog").value = n;
  document.getElementById("out").value = toCsv();
}}

function toCsv() {{
  const lines = ["episode_id,question,answer"];
  Object.keys(answers).sort().forEach(k => {{
    const [ep, q] = k.split("|");
    lines.push(ep + "," + q + "," + answers[k]);
  }});
  return lines.join("\\n");
}}

document.addEventListener("click", e => {{
  const b = e.target.closest(".opts button");
  if (!b) return;
  const q = b.closest(".q");
  answers[q.dataset.ep + "|" + q.dataset.q] = b.dataset.v;
  localStorage.setItem(KEY, JSON.stringify(answers));
  render();
}});

document.getElementById("show").addEventListener("click", () => {{
  const r = document.getElementById("results");
  r.style.display = r.style.display === "none" ? "block" : "none";
  if (r.style.display === "block") r.scrollIntoView({{behavior: "smooth"}});
}});

document.getElementById("copy").addEventListener("click", async () => {{
  const text = toCsv();
  const btn = document.getElementById("copy");
  try {{
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied ✓";
  }} catch (err) {{
    // Clipboard permission varies by host, so fall back to revealing the text for manual copy
    // rather than failing silently -- a reviewer who loses an hour of scoring will not do it twice.
    document.getElementById("results").style.display = "block";
    document.getElementById("out").select();
    btn.textContent = "Select and copy below";
  }}
  setTimeout(() => {{ btn.textContent = "Copy results"; }}, 2500);
}});

render();
</script>
"""
    args.out.write_text(page, encoding="utf-8")
    print(f"{embedded} episodes, {asked} questions")
    print(f"{args.out.stat().st_size / 1024 / 1024:.2f} MB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
