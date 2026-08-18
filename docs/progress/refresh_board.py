"""Refresh the progress board in place: replace the draft section rather than insert another.

The first version of this script inserted the section before "Recent work", which is fine once
and duplicates on every later run. Replacing a delimited block is idempotent, so the page can be
rebuilt whenever the draft changes.
"""
import re
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import md_render as render_md

REPO = Path(__file__).resolve().parents[2]
PAGE = REPO / "docs/progress/sweepcf_board.html"
DRAFT = REPO / "docs/paper/sweepcf_draft.md"

t = PAGE.read_text()
before = len(t)
body = render_md.render(DRAFT.read_text())
if not body.strip():
    raise SystemExit("renderer produced nothing -- refusing to publish an empty draft section")

section = (
    '<section id="draft">\n'
    '  <h2>Paper draft</h2>\n'
    '  <p class="lede" style="font-size:.98rem">Rendered from '
    '<code>docs/paper/sweepcf_draft.md</code>. A draft of an argument, not of a paper &mdash; '
    'the status beside each claim is part of the content.</p>\n'
    f'  <div class="draft">{body}</div>\n'
    '</section>'
)
new, n = re.subn(r'<section id="draft">.*?</section>', lambda _: section, t, flags=re.S)
if n == 0:
    # First run on a page that has no draft section yet.
    new, n = re.subn(r'(<section>\s*<h2>Recent work</h2>)', lambda m: section + "\n" + m.group(1),
                     t, count=1)
if n != 1:
    raise SystemExit(f"expected exactly one insertion point, found {n}")
PAGE.write_text(new)
print(f"draft section replaced: {before} -> {len(new)} bytes, {len(body)} bytes of rendered draft")
