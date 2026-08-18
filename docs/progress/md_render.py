"""Pure markdown->HTML helpers, split out of render_md.py.

Importing render_md rewrote the page as a side effect of the import, which added a second
draft section the moment anything imported it. The rendering functions carry no state, so
they belong in a module that does nothing when imported.
"""
import html as H
import re
from pathlib import Path

def render(md: str) -> str:
    out, i = [], 0
    lines = md.split("\n")
    while i < len(lines):
        line = lines[i]
        if line.startswith("### "):
            out.append(f"<h4>{inline(line[4:])}</h4>"); i += 1; continue
        if line.startswith("## "):
            out.append(f"<h3>{inline(line[3:])}</h3>"); i += 1; continue
        if line.startswith("# "):
            i += 1; continue                      # page already has its own title
        if line.startswith("---"):
            out.append("<hr>"); i += 1; continue
        if line.startswith("|"):
            table, j = [], i
            while j < len(lines) and lines[j].startswith("|"):
                table.append(lines[j]); j += 1
            out.append(make_table(table)); i = j; continue
        if line.startswith("- "):
            items, j = [], i
            while j < len(lines) and lines[j].startswith("- "):
                items.append(lines[j][2:]); j += 1
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>")
            i = j; continue
        if re.match(r"^\d+\.\s", line):
            items, j = [], i
            while j < len(lines) and re.match(r"^\d+\.\s", lines[j]):
                items.append(re.sub(r"^\d+\.\s", "", lines[j])); j += 1
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ol>")
            i = j; continue
        if line.strip():
            para, j = [line], i + 1     # always consume at least this line, so i advances
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("#", "|", "- ", "---")):
                para.append(lines[j]); j += 1
            out.append(f"<p>{inline(' '.join(para))}</p>"); i = j; continue
        i += 1
    return "\n".join(out)

def make_table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    body = [r for r in cells if not all(set(c) <= set("-: ") for c in r)]
    if not body:
        return ""
    head, rest = body[0], body[1:]
    th = "".join(f"<th>{inline(c)}</th>" for c in head)
    trs = "".join(
        "<tr>" + "".join(
            f'<td class="{"n" if re.match(r"^[-+0-9]", c) else ""}">{inline(c)}</td>'
            for c in r) + "</tr>"
        for r in rest)
    return f'<div class="scroll"><table><tr>{th}</tr>{trs}</table></div>'

def inline(s: str) -> str:
    s = H.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = s.replace("&times;", "&times;").replace("&mdash;", "&mdash;")
    return s

draft = Path("/home/robotixx/GR00T-WholeBodyControl/docs/paper/sweepcf_draft.md").read_text()
