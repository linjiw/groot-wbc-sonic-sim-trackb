#!/usr/bin/env python3
"""Render the unfinished manuscript for reading, without claiming submission compliance."""

import html
from pathlib import Path
import re

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/motion2scene"


def main():
    raw = (DOC / "ICRA_MANUSCRIPT.md").read_text()
    blocks = []
    for paragraph in raw.split("\n\n"):
        text = " ".join(paragraph.splitlines())
        heading = re.match(r"^(#{1,3}) (.*)", text)
        if heading:
            level = len(heading[1])
            blocks.append(f"<h{level}>{html.escape(heading[2])}</h{level}>")
        else:
            blocks.append("<p>" + html.escape(text) + "</p>")
    page = (
        """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Motion2Scene — working manuscript</title>
<style>
body{font:17px/1.55 Georgia,serif;color:#182e36;max-width:940px;margin:40px auto;padding:0 24px}
h1{font-size:32px;line-height:1.2}h2{font-size:22px;margin-top:28px}p{overflow-wrap:anywhere}
.notice{padding:16px;background:#fff0d5;font:14px/1.5 system-ui}
@page{size:letter;margin:0.7in}
@media print{body{max-width:none;margin:0;padding:0;font:10pt/1.25 "Times New Roman",serif;color:black}
article{columns:2;column-gap:0.25in}h1{font-size:18pt;column-span:all}h2{font-size:11pt;break-after:avoid}
p{margin:0 0 8pt}.notice{font-size:9pt;padding:8pt;margin-bottom:12pt}a{color:black}}
</style><div class="notice">Working draft: partial 366/540 evaluation results; final results and abstract pending.
This reading copy is not an ICRA template or a submission-compliance check.</div><article>"""
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
