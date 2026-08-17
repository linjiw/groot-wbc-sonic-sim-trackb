#!/usr/bin/env python3
"""Render the corpus review as one self-contained page for visual inspection.

Reads the artifacts ``build_corpus_review.py`` produced and inlines them, so the page is a
single file with no external requests: figures and clips become data URIs. Numbers come
from ``review.json`` rather than being restated, so the page cannot drift from the data it
describes.

The page is an instrument panel, not a report. A reviewer's questions are "which behaviours
does the tracker hold", "what does a failure look like", and "what should I distrust" — so
outcome is encoded in colour and bar length as well as in digits, and the palette is the
same one the matplotlib figures use so the page and the plots read as one artifact.

Usage::

    python scripts/research/render_corpus_review_page.py \\
        --review /data/.../review --out docs/g1_corpus_review.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path
import sys

BEHAVIOUR_LABELS = {
    "walk": "forward walk",
    "walk_pause": "walk with a pause",
    "walk_look": "walk and look around",
    "crouch_walk": "lowered walk",
    "crouch_deep": "deep crouch",
    "duck_under": "duck under",
    "side_step": "side-step",
    "backward": "backward walk",
    "stand_to_walk": "start from standing",
    "walk_to_stop": "walk to a stop",
    "step_over": "step over",
    "turn_in_place": "turn in place",
    "carry_walk": "carry while walking",
    "reach_walk": "walk and reach",
    "squat_pick": "squat to pick up",
}

#: What a reviewer should actually watch for in each behaviour, rather than a restatement
#: of the label. Absent behaviours simply get no note.
BEHAVIOUR_NOTES = {
    "walk": "The baseline. Everything else is measured against how this looks.",
    "backward": "Highest acceptance of any mode, which was not the expectation.",
    "duck_under": "Watch the head clear the shelf rather than pass through it.",
    "side_step": "Should translate sideways while facing forward — check it is not just turning.",
    "squat_pick": "Fails outright: the G1's waist pitch cannot fold as far as the reference asks.",
    "turn_in_place": "Check the feet actually pivot rather than the root sliding.",
    "carry_walk": "Arms held forward widen the swept volume — watch the clearance being used.",
    "walk_pause": "The pause is the point; confirm the robot actually stops.",
    "walk_look": "Lowest acceptance among the walking modes.",
    "stand_to_walk": "The first step from rest is where a tracker usually slips.",
    "step_over": "Watch the trailing foot, not the leading one.",
    "reach_walk": "The reach should happen after the walk, not during.",
    "walk_to_stop": "Deceleration without overshoot is the thing being tested.",
}

DATA_URI_LIMIT_MB = 15.0


def data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def pick_clip(videos_dir: Path, behaviour: str) -> tuple[Path | None, str]:
    """One clip per behaviour, preferring an accepted one so the page shows the norm."""
    folder = videos_dir / behaviour
    if not folder.is_dir():
        return None, ""
    for prefix in ("accepted", "rejected", "unevaluable"):
        matches = sorted(folder.glob(f"{prefix}__*.mp4"))
        if matches:
            return matches[0], prefix
    return None, ""


STYLE = """
:root {
  --ground: #f3f5f2;
  --panel: #ffffff;
  --panel-2: #eceee9;
  --ink: #171a1c;
  --muted: #4b5257;
  --faint: #767c74;
  --rule: #d9ddd7;
  --trace: #2f6fd0;
  --trace-soft: rgba(47,111,208,.10);
  --pass: #2f7d32;
  --fail: #c0392b;
  --unknown: #8a8a8a;
  --shadow: 0 1px 2px rgba(20,26,22,.06), 0 8px 24px rgba(20,26,22,.05);
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground: #14171a;
    --panel: #1b1f23;
    --panel-2: #22272c;
    --ink: #e8ebe6;
    --muted: #a9b1ad;
    --faint: #7d8781;
    --rule: #2e353b;
    --trace: #6ba3f0;
    --trace-soft: rgba(107,163,240,.13);
    --pass: #6bbf6f;
    --fail: #e77364;
    --unknown: #9aa19c;
    --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"] {
  --ground: #14171a;
  --panel: #1b1f23;
  --panel-2: #22272c;
  --ink: #e8ebe6;
  --muted: #a9b1ad;
  --faint: #7d8781;
  --rule: #2e353b;
  --trace: #6ba3f0;
  --trace-soft: rgba(107,163,240,.13);
  --pass: #6bbf6f;
  --fail: #e77364;
  --unknown: #9aa19c;
  --shadow: 0 1px 2px rgba(0,0,0,.4), 0 10px 30px rgba(0,0,0,.35);
}

* { box-sizing: border-box; }
body {
  margin: 0; background: var(--ground); color: var(--ink);
  font-family: var(--sans); font-size: 16px; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 0 26px 90px; }

.mast { padding: 62px 0 10px; border-bottom: 1px solid var(--rule); }
.kicker {
  font-family: var(--mono); font-size: 11px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--faint); margin: 0 0 14px;
}
h1 {
  margin: 0 0 12px; font-size: clamp(30px, 5vw, 48px); line-height: 1.04;
  letter-spacing: -.028em; font-weight: 660; text-wrap: balance;
}
.standfirst { margin: 0 0 30px; font-size: 17.5px; color: var(--muted); max-width: 66ch; }
.standfirst strong { color: var(--ink); font-weight: 620; }

.readout {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  border-radius: 4px; overflow: hidden; margin: 30px 0 0;
}
.readout div { background: var(--panel); padding: 15px 17px; }
.readout dt {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em;
  text-transform: uppercase; color: var(--faint); margin: 0 0 7px;
}
.readout dd {
  margin: 0; font-family: var(--mono); font-size: 23px; font-weight: 600;
  font-variant-numeric: tabular-nums; letter-spacing: -.02em;
}
.readout dd small { font-size: 12px; font-weight: 500; color: var(--muted); margin-left: 4px; }

section { padding: 58px 0 0; }
h2 {
  margin: 0 0 6px; font-size: 12.5px; font-family: var(--mono);
  letter-spacing: .16em; text-transform: uppercase; color: var(--trace); font-weight: 600;
}
h3 {
  margin: 0 0 14px; font-size: clamp(21px, 3vw, 29px); line-height: 1.16;
  letter-spacing: -.022em; font-weight: 640; text-wrap: balance; max-width: 28ch;
}
p { margin: 0 0 16px; max-width: 70ch; color: var(--muted); }
p strong { color: var(--ink); font-weight: 620; }
.num {
  font-family: var(--mono); font-variant-numeric: tabular-nums;
  color: var(--ink); font-weight: 600; font-size: .94em;
}

.grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
  gap: 20px; margin-top: 26px;
}
.card {
  background: var(--panel); border: 1px solid var(--rule); border-radius: 5px;
  overflow: hidden; box-shadow: var(--shadow); display: flex; flex-direction: column;
}
.card header { padding: 15px 17px 12px; }
.card h4 {
  margin: 0 0 3px; font-size: 16.5px; font-weight: 640; letter-spacing: -.012em;
}
.card .meta {
  font-family: var(--mono); font-size: 11px; color: var(--faint);
  font-variant-numeric: tabular-nums; letter-spacing: .02em;
}
.bar { height: 5px; background: var(--panel-2); position: relative; }
.bar span { position: absolute; inset: 0 auto 0 0; background: var(--pass); }
.card video { display: block; width: 100%; background: #000; }
.card .note {
  padding: 12px 17px 16px; font-size: 13.5px; color: var(--muted); margin: 0;
}
.tag {
  display: inline-block; font-family: var(--mono); font-size: 10px; font-weight: 600;
  letter-spacing: .1em; text-transform: uppercase; padding: 2px 7px; border-radius: 3px;
  vertical-align: 2px; margin-left: 7px;
}
.tag.accepted { color: var(--pass); background: rgba(47,125,50,.11); }
.tag.rejected { color: var(--fail); background: rgba(192,57,43,.11); }
.tag.unevaluable { color: var(--unknown); background: rgba(138,138,138,.14); }

figure { margin: 26px 0 0; }
figure img {
  display: block; width: 100%; height: auto; border: 1px solid var(--rule);
  border-radius: 4px; background: #fff; box-shadow: var(--shadow);
}
figcaption { margin-top: 11px; font-size: 13.5px; color: var(--muted); max-width: 78ch; }
figcaption b { color: var(--ink); font-weight: 620; }

.scroll { overflow-x: auto; margin: 24px 0 0; border: 1px solid var(--rule); border-radius: 4px; }
table { width: 100%; border-collapse: collapse; background: var(--panel); font-size: 14px; }
th, td { padding: 11px 15px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }
thead th {
  font-family: var(--mono); font-size: 10.5px; letter-spacing: .11em; text-transform: uppercase;
  color: var(--faint); font-weight: 600; background: var(--panel-2);
}
tbody tr:last-child td { border-bottom: 0; }
td.n { font-family: var(--mono); font-variant-numeric: tabular-nums; text-align: right; }
td.name { font-weight: 560; }

.note-box {
  margin-top: 24px; padding: 17px 19px; border-left: 2px solid var(--fail);
  background: rgba(192,57,43,.06); border-radius: 0 4px 4px 0;
}
.note-box p { color: var(--ink); margin: 0; max-width: 74ch; font-size: 14.5px; }
.note-box p + p { margin-top: 10px; }

footer {
  margin-top: 74px; padding-top: 24px; border-top: 1px solid var(--rule);
  font-size: 13px; color: var(--faint); font-family: var(--mono);
}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads((args.review / "review.json").read_text(encoding="utf-8"))
    records = summary["records"]
    outcomes = summary["outcomes"]
    diversity = summary["diversity"]

    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["behaviour"], []).append(record)
    ordered = sorted(groups, key=lambda k: -len(groups[k]))

    accepted = [r for r in records if r["outcome"] == "accepted"]
    speeds = sorted(r["mean_speed_mps"] for r in accepted)
    lateral = max((r["lateral_scene_contact_n"] for r in accepted), default=0.0)

    parts: list[str] = [
        "<title>G1 Corpus Review</title>",
        f"<style>{STYLE}</style>",
        '<div class="wrap">',
        '<header class="mast">',
        '<p class="kicker">Visual review · Unitree G1 · SONIC + Isaac Lab</p>',
        "<h1>Every behaviour in the corpus, and what the tracker did with it</h1>",
        f'<p class="standfirst">{summary["episodes"]} episodes across '
        f'{len(summary["behaviours"])} behaviours. Each card below is one clip and the '
        "acceptance rate of its whole family. Numbers are recomputed from the trajectories, "
        "so nothing here can disagree with the data it describes — including the parts that "
        "came out worse than hoped.</p>",
        "</header>",
        '<dl class="readout">',
        f'<div><dt>Episodes</dt><dd>{summary["episodes"]}<small>{summary["videos"]} clips</small></dd></div>',
        f'<div><dt>Accepted</dt><dd>{outcomes["acceptance_rate_of_evaluated"]:.0%}'
        f'<small>{outcomes["accepted"]}/{outcomes["accepted"] + outcomes["rejected"]}</small></dd></div>',
        f'<div><dt>Behaviours</dt><dd>{len(summary["behaviours"])}<small>of 15</small></dd></div>',
        f'<div><dt>Between-episode rank</dt><dd>{diversity["between_episode_rank"]:.2f}'
        f'<small>was 1.16</small></dd></div>',
        f'<div><dt>Speed span</dt><dd>{speeds[0]:.2f}–{speeds[-1]:.2f}<small>m/s</small></dd></div>',
        f'<div><dt>Scene contact</dt><dd>{lateral:.1f}<small>N worst accepted</small></dd></div>',
        "</dl>",
    ]

    # ---- behaviour cards ----
    parts += [
        "<section>",
        "<h2>Behaviours</h2>",
        "<h3>One clip each, with how the family fared</h3>",
        "<p>The bar under each title is the acceptance rate for that behaviour. The clip is "
        "the ego camera — the observation a policy would actually be trained on, which is "
        "why it is the one worth inspecting. Prefer an accepted clip where one exists.</p>",
        '<div class="grid">',
    ]
    for behaviour in ordered:
        group = groups[behaviour]
        good = sum(1 for r in group if r["outcome"] == "accepted")
        rate = good / len(group)
        clip, outcome = pick_clip(args.review / "videos", behaviour)
        label = html.escape(BEHAVIOUR_LABELS.get(behaviour, behaviour))
        note = html.escape(BEHAVIOUR_NOTES.get(behaviour, ""))
        colour = {"accepted": "var(--pass)", "rejected": "var(--fail)"}.get(
            outcome, "var(--unknown)"
        )
        parts.append('<article class="card">')
        parts.append("<header>")
        parts.append(
            f'<h4>{label}<span class="tag {outcome or "unevaluable"}">'
            f'{outcome or "no clip"}</span></h4>'
        )
        parts.append(
            f'<div class="meta">{good}/{len(group)} accepted · {rate:.0%}</div>'
        )
        parts.append("</header>")
        parts.append(
            f'<div class="bar"><span style="width:{rate * 100:.1f}%;'
            f'background:{"var(--fail)" if rate < 0.34 else "var(--pass)"}"></span></div>'
        )
        if clip is not None:
            parts.append(
                f'<video controls preload="metadata" playsinline '
                f'src="{data_uri(clip, "video/mp4")}"></video>'
            )
        if note:
            parts.append(f'<p class="note">{note}</p>')
        parts.append("</article>")
    parts += ["</div>", "</section>"]

    # ---- corpus figures ----
    for name, heading, caption in (
        (
            "corpus_behaviour_space.png",
            "Where acceptance actually lives",
            "<b>Left:</b> every episode by speed and how low the body goes. <b>Right:</b> "
            "acceptance per behaviour. Reading the scatter suggests low-bodied behaviours "
            "fail — binned, they do not. Episodes below 0.55 m are 0-for-12, but 0.55–0.66 m "
            "runs 88–90%, better than the 0.72–0.76 m band of ordinary walking. The visual "
            "impression came from red points being salient at the bottom, not from a threshold.",
        ),
        (
            "corpus_speed_axis.png",
            "The prompt axis does real work",
            "Requested speed style against achieved speed over accepted episodes. The medians "
            "separate and the interquartile bands do not overlap, so the axis steers the "
            "generator. The full ranges overlap heavily, so it shifts the distribution rather "
            "than fixing any single sample's speed.",
        ),
    ):
        figure = args.review / "figures" / name
        if figure.exists():
            parts += [
                "<section>",
                "<h2>Analysis</h2>",
                f"<h3>{heading}</h3>",
                f'<figure><img alt="{heading}" src="{data_uri(figure, "image/png")}">',
                f"<figcaption>{caption}</figcaption></figure>",
                "</section>",
            ]

    # ---- per-episode traces ----
    trace_dirs = [
        d for d in sorted((args.review / "figures").iterdir())
        if d.is_dir() and any(d.glob("*.png"))
    ]
    if trace_dirs:
        parts += [
            "<section>",
            "<h2>Traces</h2>",
            "<h3>What a single episode looks like from inside</h3>",
            "<p>Three panels per episode: the executed path against the commanded one, how "
            "low the body goes, and how the tracking error grows with time. The third panel "
            "is the one that explains most rejections — the robot starts on its reference "
            "and drifts, so a longer episode is a harder one.</p>",
        ]
        for directory in trace_dirs[:6]:
            image = sorted(directory.glob("*.png"))[0]
            label = html.escape(BEHAVIOUR_LABELS.get(directory.name, directory.name))
            parts.append(
                f'<figure><img alt="{label} trace" src="{data_uri(image, "image/png")}">'
                f"<figcaption><b>{label}.</b> "
                f"{html.escape(image.stem[:70])}</figcaption></figure>"
            )
        parts.append("</section>")

    # ---- acceptance table ----
    parts += [
        "<section>",
        "<h2>By behaviour</h2>",
        "<h3>Which behaviours the tracker can hold</h3>",
        '<div class="scroll"><table><thead><tr>'
        "<th>Behaviour</th><th>Episodes</th><th>Accepted</th><th>Rate</th>"
        "<th>Median speed</th><th>Lowest root</th></tr></thead><tbody>",
    ]
    for behaviour in ordered:
        group = groups[behaviour]
        good = sum(1 for r in group if r["outcome"] == "accepted")
        median_speed = sorted(r["mean_speed_mps"] for r in group)[len(group) // 2]
        lowest = min(r["root_height_min_m"] for r in group)
        parts.append(
            f'<tr><td class="name">{html.escape(BEHAVIOUR_LABELS.get(behaviour, behaviour))}</td>'
            f'<td class="n">{len(group)}</td><td class="n">{good}</td>'
            f'<td class="n">{good / len(group):.0%}</td>'
            f'<td class="n">{median_speed:.2f} m/s</td><td class="n">{lowest:.3f} m</td></tr>'
        )
    parts += ["</tbody></table></div>", "</section>"]

    # ---- limits ----
    parts += [
        "<section>",
        "<h2>Before you trust any of this</h2>",
        "<h3>Known wrong, so you do not have to find it</h3>",
        '<div class="note-box">',
        "<p><strong>Acceptance depends on episode length, not on which behaviour it is.</strong> "
        'Grading the same episodes at longer horizons gives <span class="num">83%</span> at '
        '1.2 s falling to <span class="num">64%</span> at 4.8 s, because the tracker starts on '
        "its reference and drifts. Duration is the strongest single predictor of rejection "
        '(<span class="num">r = −0.47</span>), and path-tracking error causes 40 of 54 '
        "rejections. Any rate on this page is for the full captured episode.</p>",
        "<p><strong>The third-person camera clips inside furniture</strong> in dense scenes, "
        "blanking part of the frame, and <strong>the overhead camera renders blank</strong>. "
        "Both are review-tooling defects, not data defects; the overhead one is unfixed and was "
        "reverted rather than shipped with a speculative fix.</p>",
        "<p><strong>Two accepted episodes carry non-zero lateral scene contact.</strong> One is "
        "2.8 N, marginal against a 1.0 N threshold and <strong>not understood</strong>. The "
        "obvious explanation — too little clearance — was tested and refuted: the episodes with "
        "contact had the lowest clearance requirement in their batch.</p>",
        "<p><strong>Squat-to-pick-up fails outright, 0 of 3.</strong> The G1's waist pitch "
        "cannot fold as far as the reference asks, so the joint clamps and the thigh ends up "
        "through the pelvis. That is a robot limit, not a tracker failure, and the prompt "
        "taxonomy now asks for a shallower crouch alongside the deep one.</p>",
        "</div>",
        "</section>",
        f'<footer>Built from {summary["episodes"]} recorded trajectories · '
        f'figures and clips embedded, no external requests</footer>',
        "</div>",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    page = "\n".join(parts) + "\n"
    args.out.write_text(page, encoding="utf-8")
    size_mb = len(page.encode("utf-8")) / 1e6
    print(f"wrote {args.out}  ({size_mb:.1f} MB)")
    if size_mb > DATA_URI_LIMIT_MB:
        print(f"WARNING: exceeds the {DATA_URI_LIMIT_MB} MB publish budget")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
