#!/usr/bin/env python3
"""Render the unfinished manuscript for reading, without claiming submission compliance."""

import html
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/motion2scene"


def inline(text):
    """Escape, then restore the small subset of inline markdown the draft uses."""
    out = html.escape(text)
    out = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', out)
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    return out


def table(lines):
    """Render a pipe table; the second line is the alignment rule and is dropped."""
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    head, body = rows[0], rows[2:]
    cells = "".join(f"<th>{inline(c)}</th>" for c in head)
    out = [f"<table><thead><tr>{cells}</tr></thead><tbody>"]
    for row in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in row) + "</tr>")
    return "".join(out) + "</tbody></table>"


def main():
    raw = (DOC / "ICRA_MANUSCRIPT.md").read_text()
    blocks = []
    for paragraph in raw.split("\n\n"):
        lines = [line for line in paragraph.splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) > 2 and all(line.lstrip().startswith("|") for line in lines):
            blocks.append(table(lines))
            continue
        text = " ".join(lines)
        heading = re.match(r"^(#{1,3}) (.*)", text)
        if heading:
            level = len(heading[1])
            blocks.append(f"<h{level}>{html.escape(heading[2])}</h{level}>")
        elif text.startswith("!["):
            blocks.append("<figure>" + inline(text) + "</figure>")
        else:
            blocks.append("<p>" + inline(text) + "</p>")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Motion2Scene — working manuscript</title>
<style>
body{font:17px/1.55 Georgia,serif;color:#182e36;max-width:940px;margin:40px auto;padding:0 24px}
h1{font-size:32px;line-height:1.2}h2{font-size:22px;margin-top:28px}h3{font-size:17px;margin-top:20px}
p{overflow-wrap:anywhere}
.notice{padding:16px;background:#fff0d5;font:14px/1.5 system-ui}
table{border-collapse:collapse;width:100%;table-layout:fixed;font:13px/1.4 system-ui;margin:14px 0}
th,td{border-bottom:1px solid #ccd6da;padding:5px 8px;text-align:left;vertical-align:top;
overflow-wrap:anywhere;hyphens:auto}
th{font-weight:600;background:#f2f6f7}
td:not(:first-child),th:not(:first-child){text-align:right;font-variant-numeric:tabular-nums}
figure{margin:14px 0}img{max-width:100%}
@page{size:letter;margin:0.7in}
@media print{body{max-width:none;margin:0;padding:0;font:10pt/1.25 "Times New Roman",serif;color:black}
article{columns:2;column-gap:0.25in}h1{font-size:18pt;column-span:all}h2{font-size:11pt;break-after:avoid}
h3{font-size:10pt;break-after:avoid}p{margin:0 0 8pt}
table{font-size:7pt;break-inside:avoid}th,td{padding:1.5pt 3pt}
figure{break-inside:avoid}
.notice{font-size:9pt;padding:8pt;margin-bottom:12pt}a{color:black}}
</style><div class="notice">Working draft: the 540-run panel is complete; the nominal-contract study and the
abstract are pending. This reading copy is not an ICRA template or a submission-compliance check.</div><article>"""
        + "".join(blocks)
        + "</article></html>"
    )
    target = DOC / "ICRA_MANUSCRIPT.html"
    target.write_text(page)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"]
        )
        tab = browser.new_page()
        tab.goto(target.as_uri())
        tab.pdf(
            path=str(DOC / "ICRA_MANUSCRIPT.pdf"),
            prefer_css_page_size=True,
            display_header_footer=True,
            header_template="<span></span>",
            footer_template=(
                '<div style="font-size:8px;width:100%;text-align:center">Working draft — '
                '<span class="pageNumber"></span> / <span class="totalPages"></span></div>'
            ),
        )
        browser.close()


if __name__ == "__main__":
    main()
