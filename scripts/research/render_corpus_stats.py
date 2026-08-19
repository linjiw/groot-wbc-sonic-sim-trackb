"""Draw the corpus as it is: what it covers, what it refuses, and where it is thin.

A dataset page that shows only its best episodes is an advertisement. This draws the distributions
that decide whether the corpus is usable — behaviour coverage, why episodes are refused, how close
the accepted ones come to their obstacles, and how much of each commanded edit survived execution.
The thin parts are drawn at the same scale as the thick ones, because the tail is what a reader
needs in order to judge it.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, LINE = "#191D20", "#6C767C", "#DCDFDC"
ACCENT, ACCEPT, REJECT = "#9A5F0B", "#2F6B52", "#9E3A31"


def style(axis) -> None:
    axis.tick_params(labelsize=8, colors=MUTED)
    axis.grid(alpha=0.12, lw=0.6, axis="x")
    axis.set_axisbelow(True)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(LINE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True, help="release index directory")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    episodes = list(csv.DictReader(open(args.index / "episodes.csv")))
    motions = list(csv.DictReader(open(args.index / "motions.csv")))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), dpi=110)
    (behaviour_ax, reason_ax), (clearance_ax, survival_ax) = axes

    # 1. behaviour coverage, with the tail named rather than lumped into "other"
    counts = collections.Counter(m["behaviour_class"] for m in motions)
    order = sorted(counts, key=counts.get)
    behaviour_ax.barh(order, [counts[k] for k in order], color=ACCENT, height=0.62)
    for index, key in enumerate(order):
        behaviour_ax.text(counts[key] + 0.8, index, str(counts[key]), va="center",
                          fontsize=8, color=MUTED)
    behaviour_ax.set_title("Behaviour coverage across screened motions", fontsize=10,
                           color=INK, loc="left")
    behaviour_ax.set_xlabel("clips", fontsize=8)
    style(behaviour_ax)

    # 2. why episodes are refused -- the shape of the refusals is the shape of the gates
    reasons: collections.Counter = collections.Counter()
    for row in episodes:
        for reason in (row["rejection_reasons"] or "").split(";"):
            if reason:
                reasons[reason.replace("_", " ")] += 1
    if reasons:
        keys = sorted(reasons, key=reasons.get)
        reason_ax.barh(keys, [reasons[k] for k in keys], color=REJECT, height=0.6)
        for index, key in enumerate(keys):
            reason_ax.text(reasons[key] + 0.06, index, str(reasons[key]), va="center",
                           fontsize=8, color=MUTED)
    else:
        reason_ax.text(0.5, 0.5, "no refusals recorded", ha="center", color=MUTED, fontsize=9)
    reason_ax.set_title("Why episodes are refused", fontsize=10, color=INK, loc="left")
    reason_ax.set_xlabel("episodes", fontsize=8)
    style(reason_ax)

    # 3. how close the accepted episodes actually come
    gaps = [float(r["min_clearance_mm"]) for r in episodes if r["min_clearance_mm"]]
    if gaps:
        clearance_ax.hist(gaps, bins=12, color=ACCEPT, alpha=0.85)
        clearance_ax.axvline(0.0, color=REJECT, lw=1.4)
        clearance_ax.text(0.0, clearance_ax.get_ylim()[1] * 0.92, "  contact", fontsize=8,
                          color=REJECT)
    else:
        clearance_ax.text(0.5, 0.5, "no clearance measured", ha="center", color=MUTED, fontsize=9)
    clearance_ax.set_title("Closest approach, where it was measured", fontsize=10, color=INK,
                           loc="left")
    clearance_ax.set_xlabel("minimum clearance (mm)", fontsize=8)
    clearance_ax.set_ylabel("episodes", fontsize=8)
    style(clearance_ax)
    clearance_ax.grid(alpha=0.12, lw=0.6, axis="y")

    # 4. commanded against executed -- the gap is the dataset-quality claim
    pairs = [
        (float(r["commanded_amplitude_rad"]), float(r["executed_amplitude_rad"]), r["operator"])
        for r in episodes
        if r["commanded_amplitude_rad"] and r["executed_amplitude_rad"]
    ]
    if pairs:
        for operator, colour, label in (
            ("local_crouch", REJECT, "crouch (legs)"),
            ("local_arm_tuck", ACCEPT, "tuck (arms)"),
        ):
            subset = [(c, e) for c, e, o in pairs if o == operator]
            if subset:
                survival_ax.scatter(*zip(*subset), s=42, color=colour, alpha=0.85, label=label)
        other = [(c, e) for c, e, o in pairs if o not in ("local_crouch", "local_arm_tuck")]
        if other:
            survival_ax.scatter(*zip(*other), s=34, color=MUTED, alpha=0.7, label="other")
        top = max(max(c for c, _, _ in pairs), max(e for _, e, _ in pairs)) * 1.08
        survival_ax.plot([0, top], [0, top], color=LINE, lw=1.2, ls=(0, (4, 3)))
        survival_ax.text(top * 0.62, top * 0.92, "commanded = executed", fontsize=7.5, color=MUTED)
        survival_ax.legend(frameon=False, fontsize=8, loc="lower right")
    survival_ax.set_title("What was asked for, against what happened", fontsize=10, color=INK,
                          loc="left")
    survival_ax.set_xlabel("commanded amplitude (rad)", fontsize=8)
    survival_ax.set_ylabel("executed amplitude (rad)", fontsize=8)
    style(survival_ax)
    survival_ax.grid(alpha=0.12, lw=0.6, axis="y")

    fig.tight_layout(pad=1.1)
    fig.savefig(args.out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"{len(episodes)} episodes, {len(motions)} motions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
