#!/usr/bin/env python3
"""Draw the placement-envelope price and the acquisition funnel it explains."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact
import numpy as np

DATA = ROOT.parent / "research-data/groot-wbc"
DOC = ROOT / "docs/motion2scene"
SWEEP = DATA / "m2s-envelope-tradeoff-v1/envelope-tradeoff.json"
STUDY = DATA / "m2s-icra-v1"
NOMINAL = DATA / "m2s-icra-nominal-v1"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")
LABELS = ("Uniform", "Analytic", "Target-only", "Motion2Scene")


def eligible(path):
    rows = json.loads((path / "proposals.json").read_text())["rows"]
    return [sum(r["eligible"] and r["arm"] == a for r in rows) for a in ARMS]


def main():
    sweep = json.loads(SWEEP.read_text())
    public = DOC / "evidence/envelope-tradeoff-20260907"
    public.mkdir(exist_ok=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    ax = axes[0]
    ends = []
    for source, colour in zip(("41001", "41002", "41003"), ("#1b6ca8", "#c1512b", "#3f7d3f")):
        rows = [r for r in sweep["rows"] if str(r["source"]) == source]
        x = [r.get("envelope_xy_mm", 0.0) for r in rows]
        y = [r["witnesses"] for r in rows]
        ax.plot(x, y, marker="o", ms=4, color=colour, label=f"carrier {source}")
        ends.append((y[-1], colour))
    ax.axvline(20, color="#666666", ls="--", lw=1)
    ax.set_xlim(-1.2, 27.5)
    top = ax.get_ylim()[1]
    for rank, (value, colour) in enumerate(sorted(ends, reverse=True)):
        ax.text(
            21.0, top * (0.30 - 0.09 * rank), str(value), color=colour, fontsize=10, va="center"
        )
    ax.text(19.2, top * 0.55, "inherited audit", rotation=90, ha="right", va="center", fontsize=9)
    ax.set_xlabel("placement envelope, horizontal half-width (mm)")
    ax.set_ylabel("10 mm contrast witnesses")
    ax.set_title("Contrast support versus placement robustness")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    width = 0.38
    idx = np.arange(len(ARMS))
    ax.bar(idx - width / 2, eligible(STUDY), width, color="#9aa7b0", label="inherited contract")
    ax.bar(idx + width / 2, eligible(NOMINAL), width, color="#1b6ca8", label="nominal contract")
    for i, (a, b) in enumerate(zip(eligible(STUDY), eligible(NOMINAL))):
        ax.text(i - width / 2, a + 0.8, str(a), ha="center", fontsize=9)
        ax.text(i + width / 2, b + 0.8, str(b), ha="center", fontsize=9)
    ax.set_xticks(idx)
    ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylabel("eligible generated proposals (of 48)")
    ax.set_ylim(0, 58)
    ax.set_title("Acquisition funnel under each contract")
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(DOC / "assets/envelope-tradeoff.png", dpi=160)
    fig.savefig(DOC / "assets/envelope-tradeoff.pdf")
    plt.close(fig)

    exports = []
    for source in (SWEEP, STUDY / "proposals.json", NOMINAL / "proposals.json"):
        name = source.parent.name + "-" + source.name
        raw = (
            source.read_text()
            .replace(str(DATA), "research-data")
            .replace(str(ROOT), "repository")
            .encode()
        )
        (public / name).write_bytes(raw)
        exports.append(
            {
                "source": artifact(source),
                "public": name,
                "public_sha256": artifact(public / name)["sha256"],
            }
        )
    (public / "exports.json").write_text(json.dumps(exports, indent=2) + "\n")
    print(
        json.dumps(
            {
                "inherited_eligible": dict(zip(ARMS, eligible(STUDY))),
                "nominal_eligible": dict(zip(ARMS, eligible(NOMINAL))),
                "figure": str(Path("assets/envelope-tradeoff.png")),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
