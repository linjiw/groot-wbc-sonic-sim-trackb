"""Render the working draft as a page that can actually be read.

The draft is the argument's real home and it changes daily, so this converts rather than
reproduces: point it at the markdown and it emits the reading environment. Anything that has to
stay true of the paper -- the status line, the claim ladder's levels, the generated tables -- comes
through from the source, and the only things added here are navigation and typography.

    python scripts/research/render_paper_page.py --draft docs/paper/sweepcf_draft.md --out page.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import markdown

CSS = """
:root{
  --paper:#FCFCFB; --raised:#F3F4F2; --ink:#191D20; --ink-2:#3C444A; --muted:#6E787E;
  --line:#DEE1DE; --accent:#9A5F0B; --accent-soft:#F5EBDA;
  --good:#2F6B52; --bad:#9E3A31;
  --measure:68ch;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --paper:#12161A; --raised:#1A2026; --ink:#E9EDEF; --ink-2:#BCC5CB; --muted:#87949C;
    --line:#28323A; --accent:#E8A33D; --accent-soft:#241C0F;
    --good:#5FA184; --bad:#D4695F;
  }
}
:root[data-theme="dark"]{
  --paper:#12161A; --raised:#1A2026; --ink:#E9EDEF; --ink-2:#BCC5CB; --muted:#87949C;
  --line:#28323A; --accent:#E8A33D; --accent-soft:#241C0F; --good:#5FA184; --bad:#D4695F;
}
*{box-sizing:border-box}
body{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:Newsreader,Georgia,"Times New Roman",serif; font-weight:300;
  font-size:clamp(1.02rem,.98rem + .25vw,1.16rem); line-height:1.62;
  -webkit-font-smoothing:antialiased;
}
.shell{display:grid; grid-template-columns:minmax(0,1fr); max-width:82rem; margin:0 auto;
  padding:clamp(1.5rem,4vw,4rem) clamp(1.1rem,4vw,3rem)}
@media (min-width:1080px){ .shell{grid-template-columns:15rem minmax(0,1fr); gap:clamp(2rem,4vw,4.5rem)} }
h1,h2,h3,nav,.chip,thead th{font-family:Archivo,system-ui,sans-serif}
h1{font-size:clamp(2rem,1.4rem + 2.6vw,3.2rem); font-weight:800; letter-spacing:-.03em;
   line-height:1.02; margin:0 0 .6rem; text-wrap:balance}
h2{font-size:clamp(1.4rem,1.15rem + 1vw,1.9rem); font-weight:700; letter-spacing:-.02em;
   line-height:1.15; margin:3.4rem 0 .9rem; text-wrap:balance; scroll-margin-top:1.5rem}
h3{font-size:clamp(1.08rem,1rem + .45vw,1.3rem); font-weight:700; letter-spacing:-.01em;
   margin:2.2rem 0 .6rem; color:var(--ink); text-wrap:balance; scroll-margin-top:1.5rem}
h2::before{content:""; display:block; width:2rem; height:2px; background:var(--accent); margin-bottom:.9rem}
p,ul,ol{max-width:var(--measure)}
p{margin:0 0 1.05rem; color:var(--ink-2)}
strong{color:var(--ink); font-weight:400; font-family:Archivo,sans-serif; font-size:.94em}
em{font-style:italic}
a{color:var(--accent); text-underline-offset:2px}
code{font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.86em;
  background:var(--raised); border:1px solid var(--line); border-radius:2px; padding:.08em .34em}
blockquote{margin:1.4rem 0; padding:.9rem 1.2rem; background:var(--raised);
  border-left:2px solid var(--accent); max-width:var(--measure)}
blockquote p{margin:0; color:var(--ink); font-family:"JetBrains Mono",monospace; font-size:.86rem}
hr{border:0; border-top:1px solid var(--line); margin:2.5rem 0}
ul,ol{padding-left:1.2rem} li{margin:.35rem 0; color:var(--ink-2)}
.tablewrap{overflow-x:auto; margin:1.3rem 0; border:1px solid var(--line); background:var(--raised)}
table{border-collapse:collapse; width:100%; min-width:32rem;
  font-family:"JetBrains Mono",ui-monospace,monospace; font-size:.82rem; font-variant-numeric:tabular-nums}
th,td{text-align:left; padding:.55rem .75rem; border-bottom:1px solid var(--line); vertical-align:top}
thead th{font-family:Archivo,sans-serif; font-size:.74rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); font-weight:700; background:var(--paper)}
tbody tr:last-child td{border-bottom:0}
.status{background:var(--accent-soft); border:1px solid var(--accent); padding:1rem 1.2rem;
  margin:1.6rem 0 0; max-width:var(--measure)}
.status p{margin:0; color:var(--ink)}
.masthead{border-bottom:1px solid var(--line); padding-bottom:1.6rem; margin-bottom:.5rem}
.kicker{font-family:Archivo,sans-serif; font-size:.74rem; font-weight:700; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); margin:0 0 .9rem}
.meta{display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem}
.chip{font-size:.7rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
  border:1px solid var(--line); color:var(--muted); padding:.3rem .55rem; border-radius:2px}
nav{position:sticky; top:1.5rem; align-self:start; max-height:calc(100svh - 3rem); overflow-y:auto;
  font-size:.78rem; display:none}
@media (min-width:1080px){ nav{display:block} }
nav p{font-family:Archivo,sans-serif; font-size:.7rem; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); margin:0 0 .7rem}
nav a{display:block; padding:.28rem 0 .28rem .7rem; color:var(--muted); text-decoration:none;
  border-left:2px solid var(--line); line-height:1.35}
nav a:hover{color:var(--ink)}
nav a.sub{padding-left:1.4rem; font-size:.74rem}
nav a.on{color:var(--accent); border-left-color:var(--accent)}
:focus-visible{outline:2px solid var(--accent); outline-offset:3px}
@media (prefers-reduced-motion:reduce){ html{scroll-behavior:auto} }
html{scroll-behavior:smooth}
"""

SCRIPT = """
const links=[...document.querySelectorAll('nav a')];
const map=new Map(links.map(a=>[a.getAttribute('href').slice(1),a]));
const io=new IntersectionObserver(es=>{es.forEach(e=>{
  const a=map.get(e.target.id); if(!a) return;
  if(e.isIntersecting){links.forEach(l=>l.classList.remove('on')); a.classList.add('on');}
})},{rootMargin:'-10% 0px -80% 0px'});
document.querySelectorAll('h2[id],h3[id]').forEach(h=>io.observe(h));
"""


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    source = args.draft.read_text()
    # The generated-table markers are provenance for the refresh script, not content.
    source = re.sub(r"<!-- /?generated:[a-z-]+ -->\n?", "", source)

    title_line = source.splitlines()[0].lstrip("# ").strip()
    body_md = "\n".join(source.splitlines()[1:])

    html = markdown.markdown(body_md, extensions=["tables", "attr_list", "sane_lists"])

    # Anchor every heading and collect the contents list in one pass.
    toc: list[tuple[int, str, str]] = []

    def anchor(match: re.Match) -> str:
        level, text = int(match.group(1)), match.group(2)
        ident = slug(re.sub(r"<[^>]+>", "", text))
        toc.append((level, ident, re.sub(r"<[^>]+>", "", text)))
        return f'<h{level} id="{ident}">{text}</h{level}>'

    html = re.sub(r"<h([23])>(.*?)</h\1>", anchor, html, flags=re.DOTALL)
    html = html.replace("<table>", '<div class="tablewrap"><table>').replace(
        "</table>", "</table></div>"
    )

    # The status paragraph is the first thing a reader must not miss.
    html = re.sub(
        r"<p><strong>Status:(.*?)</p>",
        r'<div class="status"><p><strong>Status:\1</p></div>',
        html,
        count=1,
        flags=re.DOTALL,
    )

    nav = "\n".join(
        f'<a class="{"sub" if level == 3 else ""}" href="#{ident}">{text}</a>'
        for level, ident, text in toc
    )

    page = f"""<title>SweepCF Working Draft</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@700;800&family=Newsreader:ital,opsz,wght@0,6..72,300;0,6..72,400;1,6..72,300;1,6..72,400&family=JetBrains+Mono:wght@400;700&display=swap">
<style>{CSS}</style>
<div class="shell">
  <nav aria-label="Contents"><p>Contents</p>{nav}</nav>
  <main>
    <header class="masthead">
      <p class="kicker">Working draft · ICRA 2027 · not yet submission-formatted</p>
      <h1>{title_line}</h1>
      <div class="meta">
        <span class="chip">Unitree G1 · 29 DoF</span>
        <span class="chip">Frozen SONIC</span>
        <span class="chip">Isaac Lab 2.3.2</span>
        <span class="chip">Pre-registered</span>
      </div>
    </header>
    {html}
  </main>
</div>
<script>{SCRIPT}</script>
"""
    args.out.write_text(page)
    print(f"{len(toc)} headings, {args.out.stat().st_size / 1024:.0f} KB -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
