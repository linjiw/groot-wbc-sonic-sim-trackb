"""Show how little room there is: the shelf drops, and only the edited motion still fits.

The counterfactual is a claim about millimetres, and millimetres do not survive prose. This draws
the two rooms side by side at true scale -- the shelf where it actually sits, the robot's tallest
point where it actually reaches -- and then the four headroom figures on one axis with the contact
line marked. A reader should be able to see that the hard room is not rhetorically hard but
physically 178 mm lower, and that the edited motion clears it by about the width of two fingers.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, LINE = "#191D20", "#6C767C", "#DCDFDC"
SHELF, ACCEPT, REJECT = "#8A6A45", "#2F6B52", "#9E3A31"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--family", default="duck_003")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = {
        r["episode_id"].split("/")[1]: r
        for r in csv.DictReader(open(args.index / "episodes.csv"))
        if r["episode_id"].startswith(args.family + "/")
    }
    cells = ("nominal_easy", "adapted_easy", "nominal_hard", "adapted_hard")
    missing = [c for c in cells if c not in rows]
    if missing:
        raise SystemExit(f"{args.family} is missing {missing}")

    gap = {c: float(rows[c]["min_clearance_mm"]) for c in cells}
    shelf = {c: float(rows[c]["obstacle_underside_m"]) for c in cells}
    verdict = {c: rows[c]["outcome"] for c in cells}

    fig, (geom, bars) = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=110,
                                     gridspec_kw={"width_ratios": [1.05, 1]})

    # Left: the two rooms at true scale, shelf and reach drawn where they are.
    for index, (room, label) in enumerate((("easy", "Easy room"), ("hard", "Hard room"))):
        base = index * 1.5
        underside = shelf[f"nominal_{room}"]
        geom.add_patch(Rectangle((base, underside), 1.0, 0.09, facecolor=SHELF, ec="#6b5d4a", lw=0.8))
        geom.text(base + 0.5, underside + 0.135, f"{underside:.3f} m", ha="center", fontsize=8,
                  color=MUTED)
        for offset, motion, colour in ((0.28, "nominal", REJECT), (0.72, "adapted", ACCEPT)):
            cell = f"{motion}_{room}"
            peak = underside - gap[cell] / 1000.0
            ok = verdict[cell] == "accepted"
            geom.plot([base + offset, base + offset], [0.0, peak],
                      color=colour if not ok else ACCEPT, lw=7, solid_capstyle="butt", alpha=.85)
            geom.annotate(
                f"{gap[cell]:+.1f} mm", (base + offset, peak), textcoords="offset points",
                xytext=(0, 7), ha="center", fontsize=8,
                color=REJECT if gap[cell] <= 0 else INK,
            )
        geom.text(base + 0.5, -0.1, label, ha="center", fontsize=9, color=INK)
    geom.axhline(0, color=LINE, lw=1.2)
    geom.set_ylim(-0.2, 1.62)
    geom.set_xlim(-0.15, 2.65)
    geom.set_ylabel("height (m)", fontsize=9)
    geom.set_xticks([])
    geom.set_title(
        f"{args.family}: the shelf drops {1000 * (shelf['nominal_easy'] - shelf['nominal_hard']):.0f} mm",
        fontsize=10, color=INK, loc="left",
    )

    # Right: the same four numbers on one axis, with contact marked.
    labels = ["as generated\neasy", "one edit\neasy", "as generated\nhard", "one edit\nhard"]
    values = [gap[c] for c in cells]
    colours = [ACCEPT if verdict[c] == "accepted" else REJECT for c in cells]
    bars.barh(labels, values, color=colours, height=0.55)
    bars.axvline(0, color=REJECT, lw=1.4)
    bars.text(2, 3.42, "contact", fontsize=8, color=REJECT)
    for index, value in enumerate(values):
        bars.text(value + (4 if value >= 0 else -4), index, f"{value:+.1f} mm",
                  va="center", ha="left" if value >= 0 else "right", fontsize=8.5, color=INK)
    bars.set_xlabel("clearance at the closest frame (mm)", fontsize=9)
    bars.set_xlim(min(values) - 40, max(values) + 55)
    bars.set_title("Headroom, per cell", fontsize=10, color=INK, loc="left")

    for axis in (geom, bars):
        axis.tick_params(labelsize=8, colors=MUTED)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color(LINE)
    bars.grid(alpha=0.12, lw=0.6, axis="x")
    bars.set_axisbelow(True)

    fig.tight_layout(pad=1.1)
    fig.savefig(args.out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"{args.family}: " + ", ".join(f"{c}={gap[c]:+.1f}mm" for c in cells) + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
