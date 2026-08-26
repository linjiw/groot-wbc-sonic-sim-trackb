#!/usr/bin/env python3
"""Top-down map of where the binding face lands for a set of motions.

For each selected case this draws the executed root path, the binding station, and the face
footprint *oriented along the route tangent* -- which is where the face has to go, and where the
current axis-aligned authoring would instead put it. On a straight walk the two coincide; on a
curve they do not, and the gap is the argument for oriented faces.

Reference-side geometry only: no scene is authored here, because geometry is authored only from
accepted executed reaches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.local_adaptation import route_progress  # noqa: E402
from scripts.research.hallucination.prepare_probe_candidates import DATA_ROOT  # noqa: E402

STATION_FRACTION = 0.55
FACE_ALONG_M = 0.10
FACE_ACROSS_M = 1.6


def _rect(centre, yaw, along, across, **kwargs):
    corner = np.asarray(centre) - (
        np.asarray([math.cos(yaw), math.sin(yaw)]) * along / 2
        + np.asarray([-math.sin(yaw), math.cos(yaw)]) * across / 2
    )
    return patches.Rectangle(
        corner, along, across, angle=math.degrees(yaw), rotation_point="xy", **kwargs
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-study", type=Path, default=REPO_ROOT / "docs/hallucination/case_study.json"
    )
    parser.add_argument(
        "--source-dir", type=Path, default=DATA_ROOT / "sweepcf_release/motions/clips"
    )
    parser.add_argument("--index", type=int, action="append", required=True)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cases = {
        case["motion_index"]: case for case in json.loads(args.case_study.read_text())["cases"]
    }
    chosen = [cases[index] for index in args.index if index in cases]
    if not chosen:
        raise SystemExit("none of the requested indices are in the case study")

    columns = min(args.columns, len(chosen))
    rows = math.ceil(len(chosen) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 4.3 * rows), squeeze=False)
    for position, case in enumerate(chosen):
        axis = axes[position // columns][position % columns]
        index = case["motion_index"]
        path = sorted(args.source_dir.glob(f"{index:03d}_*.csv"))[0]
        root = np.loadtxt(path, delimiter=",")[:, :2]
        progress = route_progress(root)
        station_index = int(np.argmin(np.abs(progress - STATION_FRACTION)))
        station = root[station_index]

        low = max(0, station_index - 3)
        high = min(len(root) - 1, station_index + 3)
        tangent = root[high] - root[low]
        yaw = math.atan2(tangent[1], tangent[0])
        world_axis = "x" if np.ptp(root, axis=0)[0] >= np.ptp(root, axis=0)[1] else "y"
        world_yaw = 0.0 if world_axis == "x" else math.pi / 2

        axis.plot(root[:, 0], root[:, 1], color="0.25", lw=2.0, label="executed root path")
        axis.plot(*root[0], "o", color="0.25", ms=5)
        axis.plot(*station, "o", color="tab:red", ms=7, label="binding station")
        axis.add_patch(
            _rect(
                station,
                yaw,
                FACE_ALONG_M,
                FACE_ACROSS_M,
                facecolor="tab:blue",
                alpha=0.55,
                edgecolor="tab:blue",
                lw=1.5,
                label="face oriented along route",
            )
        )
        axis.add_patch(
            _rect(
                station,
                world_yaw,
                FACE_ALONG_M,
                FACE_ACROSS_M,
                facecolor="none",
                edgecolor="tab:orange",
                lw=1.8,
                ls="--",
                label="face as authored today (world axis)",
            )
        )
        window = case.get("best_engineering_window_mm")
        verdict = (
            f"propose, window {window:.0f} mm"
            if case["decision"] == "propose" and window is not None
            else "refuse: " + ", ".join(case["refusal_reasons"])
        )
        axis.set_title(
            f"{index:03d}  {case['body_mode']}  ({case['route']['turn_sign']})\n"
            f"misalignment {case['route']['face_misalignment_deg']:.1f} deg -- {verdict}",
            fontsize=9,
        )
        axis.set_aspect("equal")
        axis.grid(alpha=0.3)
        axis.set_xlabel("x (m)", fontsize=8)
        axis.set_ylabel("y (m)", fontsize=8)
        axis.tick_params(labelsize=7)
        if position == 0:
            axis.legend(fontsize=7, loc="best")
    for position in range(len(chosen), rows * columns):
        axes[position // columns][position % columns].axis("off")
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"wrote {args.out} ({len(chosen)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
