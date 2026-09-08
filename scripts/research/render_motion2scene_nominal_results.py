#!/usr/bin/env python3
"""Publish the completed nominal-contract study: equal labels, four arms, one seed."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from motion2scene_timing_diagnostic import ROOT, artifact, checked
import numpy as np

DATA = ROOT.parent / "research-data/groot-wbc"
DOC = ROOT / "docs/motion2scene"
STUDY = DATA / "m2s-icra-nominal-v1"
LEARN = DATA / "m2s-icra-nominal-learning-v1"
ARMS = ("uniform", "analytic", "no_contrast", "motion2scene")
NAMES = ("Uniform", "Analytic", "Target-only", "Motion2Scene")


def main():
    c = json.loads((LEARN / "comparison_144.json").read_text())
    fit = json.loads((LEARN / "fit.json").read_text())
    assert c["complete"] and c["completed"] == 144
    public = DOC / "evidence/nominal-results-20260908"
    public.mkdir(exist_ok=False)
    exports = []

    def export(source, name):
        raw = source.read_bytes()
        if source.suffix not in (".npz", ".pt"):
            raw = (
                raw.decode().replace(str(DATA), "research-data").replace(str(ROOT), "repository")
            ).encode()
        (public / name).write_bytes(raw)
        exports.append(
            {
                "source": artifact(source),
                "public": name,
                "public_sha256": artifact(public / name)["sha256"],
            }
        )

    for source in (
        LEARN / "fit.json",
        LEARN / "comparison_144.json",
        LEARN / "evaluation_master.json",
        STUDY / "proposals.json",
        STUDY / "registration.json",
    ):
        export(source, source.name if source.parent == LEARN else "construction-" + source.name)
    for source in sorted((LEARN / "models").iterdir()):
        export(source, source.name)
    cost = 0.0
    for index, key in (
        (STUDY / "prepared.json", "labels"),
        (LEARN / "evaluation_master.json", "eval"),
    ):
        for b in json.loads(index.read_text())["batches"]:
            folder = Path(b["directory"])
            a = json.loads((folder / "admission.json").read_text())
            assert a["admitted"]
            result = json.loads(
                checked(Path(a["result"]["path"]), a["result"]["sha256"]).read_text()
            )
            cost += result["actual_contended_gpu_hours"]
            for name in ("admission.json", "result.json", "run_record.json"):
                export(folder / name, folder.name + "-" + name)
    (public / "exports.json").write_text(json.dumps(exports, indent=2) + "\n")

    useful = [f["outcome_counts"].get("(False, True)", 0) for f in fit["fits"]]
    order = [next(f for f in fit["fits"] if f["arm"] == a) for a in ARMS]
    useful = [f["outcome_counts"].get("(False, True)", 0) for f in order]
    passage = [c["per_arm"][a]["pass"] for a in ARMS]
    requests = [c["per_arm"][a]["d040_requests"] for a in ARMS]

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(4)
    axs[0].bar(x - 0.2, [9] * 4, 0.4, color="#9aa7b0", label="generated groups acquired")
    axs[0].bar(x + 0.2, useful, 0.4, color="#1b6ca8", label="useful contrasts")
    for i, v in enumerate(useful):
        axs[0].text(i + 0.2, v + 0.2, str(v), ha="center", fontsize=10)
    axs[0].set_xticks(x, NAMES, fontsize=9)
    axs[0].set_ylim(0, 11)
    axs[0].set_ylabel("groups (of 9 generated per arm)")
    axs[0].set_title("Acquisition at equal labels: 15 groups per arm")
    axs[0].legend(fontsize=8, frameon=False)
    axs[0].spines[["top", "right"]].set_visible(False)

    colours = ["#9aa7b0", "#27817c", "#9aa7b0", "#1b6ca8"]
    axs[1].barh(x, passage, color=colours)
    for i, v in enumerate(passage):
        axs[1].text(v + 0.4, i, f"{v}/36  ({requests[i]} requests)", va="center", fontsize=9)
    axs[1].set_yticks(x, NAMES, fontsize=9)
    axs[1].invert_yaxis()
    axs[1].set_xlim(0, 36)
    axs[1].set_xlabel("contact-qualified passage (of 36 conditions)")
    axs[1].set_title("Held-out layouts, seed 8511")
    axs[1].spines[["top", "right"]].set_visible(False)
    fig.savefig(DOC / "assets/nominal-outcomes.png", dpi=160)
    fig.savefig(DOC / "assets/nominal-outcomes.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "useful": dict(zip(ARMS, useful)),
                "passage": dict(zip(ARMS, passage)),
                "cost_gpu_h": cost,
                "exports": len(exports),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
