"""Regenerate the paper's measured tables in place, so no figure is ever typed by hand.

A number written into prose is a number that can drift from the artefact it came from, and this
project has already published wrong figures twice that way. Each table below lives between markers
in the draft and is rewritten from the measurement artefacts on every run; the script exits non-zero
if a rewrite changes anything, so a stale draft fails rather than passes quietly.

    python scripts/research/refresh_paper_tables.py --check    # CI: fail if the draft is stale
    python scripts/research/refresh_paper_tables.py            # rewrite the tables

Adding a table means adding a builder here and a marker pair in the draft -- never a literal.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import pickle
import re
import statistics

import numpy as np

from gear_sonic.dataset_generation.trajectory_acceptance import evaluate_locomotion_trajectory
from gear_sonic.dataset_generation.trajectory_segments import best_evaluable_payload

SURVIVAL_CSV = Path("/data/robotixx/groot-wbc-kimodo-m0/operator_survival.csv")
MODEBANK = Path("/data/robotixx/groot-wbc-kimodo-m0/modebank/sweep_rollouts")

#: Clips in the transport-cost table, in the order the argument needs them.
TRANSPORT = ("w_nominal", "w_tuckcap30", "w_crouch05", "w_crouch08", "w_crouch11")


def operator_class(clip: str) -> str:
    if "combo" in clip:
        return "combo (hip + waist)"
    if "crouch" in clip:
        return "crouch (lower body)"
    return "tuck (upper body)"


def survival_table() -> str:
    if not SURVIVAL_CSV.exists():
        return "_survival artefact missing; run measure_operator_survival.py_"
    rows = list(csv.DictReader(open(SURVIVAL_CSV)))
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(operator_class(row["clip"]), []).append(100 * float(row["survival"]))
    lines = ["| operator | n | survival | median |", "|---|---|---|---|"]
    for name in ("crouch (lower body)", "combo (hip + waist)", "tuck (upper body)"):
        values = groups.get(name)
        if not values:
            continue
        median = statistics.median(values)
        emphasis = "**" if "combo" not in name else ""
        lines.append(
            f"| {name} | {len(values)} | {min(values):.0f}–{max(values):.0f}% | "
            f"{emphasis}{median:.0f}%{emphasis} |"
        )
    return "\n".join(lines)


def transport_table() -> str:
    lines = ["| clip | endpoint lag | verdict |", "|---|---|---|"]
    for clip in TRANSPORT:
        found = sorted((MODEBANK / clip).glob("trajectories/*.trajectory.pkl"))
        if not found:
            lines.append(f"| `{clip}` | _missing_ | _missing_ |")
            continue
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
        reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, :3]
        executed = np.asarray(payload["root_pos_w"], dtype=np.float64)
        lag = float(np.linalg.norm(executed[-1] - reference[-1]))
        report = evaluate_locomotion_trajectory(payload)
        if report.accepted:
            verdict = "accepted"
        elif "disallowed_robot_contact" in report.rejection_reasons:
            verdict = "rejected on contact, not tracking"
        else:
            verdict = "rejected"
        lines.append(f"| `{clip}` | {lag:.3f} m | {verdict} |")
    return "\n".join(lines)


BUILDERS = {"survival-table": survival_table, "transport-table": transport_table}


def rewrite(text: str) -> tuple[str, list[str]]:
    """Replace each marked block with freshly built content."""
    changed = []
    for name, build in BUILDERS.items():
        pattern = re.compile(
            rf"(<!-- generated:{name} -->\n).*?(\n<!-- /generated:{name} -->)",
            re.DOTALL,
        )
        if not pattern.search(text):
            changed.append(f"{name}: marker absent from draft")
            continue
        body = build()
        replacement = rf"\g<1>{body}\g<2>"
        updated = pattern.sub(replacement, text)
        if updated != text:
            changed.append(f"{name}: rewritten")
        text = updated
    return text, changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draft", type=Path, default=Path("docs/paper/sweepcf_draft.md"))
    ap.add_argument("--check", action="store_true", help="fail if the draft is stale")
    args = ap.parse_args()

    original = args.draft.read_text()
    updated, changed = rewrite(original)

    if args.check:
        if updated != original:
            print("draft is stale; run refresh_paper_tables.py")
            for note in changed:
                print(f"  {note}")
            return 1
        print("draft tables match the artefacts")
        return 0

    if updated != original:
        args.draft.write_text(updated)
    for note in changed or ["nothing to change"]:
        print(f"  {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
