"""Draw the route view: where the robot went, what it passed, and where it was closest.

The camera views show what a traversal looked like. This shows what it *was* -- the executed route
against the commanded one, the obstacle to scale, and the frame of closest approach marked, which
is the one moment the whole episode is about. It is a diagram rather than a render on purpose: a
route is a claim about geometry, and geometry reads better drawn than photographed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

INK = "#191D20"
MUTED = "#8A9AA5"
ROUTE = "#9A5F0B"
REF = "#7C8A93"
HIT = "#9E3A31"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    found = sorted(args.cell.glob("trajectories/*.trajectory.pkl"))
    if not found:
        raise SystemExit(f"no trajectory under {args.cell}")
    with open(found[0], "rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))

    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    reference = np.asarray(payload["reference_g1_qpos"], dtype=np.float64)[:, :3]
    count = min(len(root), len(reference))

    names = list(payload.get("contact_body_names", []))
    forces = np.asarray(payload.get("robot_contact_force_w", np.zeros((count, 1, 3))))
    feet = set(payload.get("allowed_foot_contact_body_names", ()))
    keep = [i for i, name in enumerate(names) if name not in feet]
    strike = None
    if keep:
        magnitude = np.linalg.norm(forces[:, keep, :], axis=2).max(axis=1)
        if magnitude.max() > 50.0:
            strike = int(magnitude.argmax())

    fig, (plan, height) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), dpi=110, gridspec_kw={"height_ratios": [2.1, 1]}
    )

    plan.plot(reference[:count, 0], reference[:count, 1], color=REF, lw=1.4, ls=(0, (4, 3)),
              label="commanded route")
    plan.plot(root[:count, 0], root[:count, 1], color=ROUTE, lw=2.4, label="executed route")
    plan.scatter([root[0, 0]], [root[0, 1]], s=42, color=ROUTE, zorder=5)
    plan.annotate("start", (root[0, 0], root[0, 1]), textcoords="offset points",
                  xytext=(8, 6), fontsize=8, color=MUTED)
    if strike is not None:
        plan.scatter([root[strike, 0]], [root[strike, 1]], s=110, facecolor="none",
                     edgecolor=HIT, lw=2.0, zorder=6)
        plan.annotate(f"contact · frame {strike}", (root[strike, 0], root[strike, 1]),
                      textcoords="offset points", xytext=(10, -14), fontsize=8, color=HIT)
    plan.set_aspect("equal")
    plan.set_xlabel("x (m)", fontsize=9)
    plan.set_ylabel("y (m)", fontsize=9)
    plan.legend(frameon=False, fontsize=8, loc="best")
    plan.grid(alpha=0.12, lw=0.6)
    plan.set_title(args.title or args.cell.name, fontsize=10, color=INK, loc="left")

    height.plot(np.arange(count), reference[:count, 2], color=REF, lw=1.4, ls=(0, (4, 3)))
    height.plot(np.arange(count), root[:count, 2], color=ROUTE, lw=2.0)
    if strike is not None:
        height.axvline(strike, color=HIT, lw=1.2, alpha=0.7)
    height.set_xlabel("frame", fontsize=9)
    height.set_ylabel("pelvis height (m)", fontsize=9)
    height.grid(alpha=0.12, lw=0.6)

    for axis in (plan, height):
        axis.tick_params(labelsize=8, colors=MUTED)
        for spine in axis.spines.values():
            spine.set_color("#DEE1DE")

    fig.tight_layout(pad=0.8)
    fig.savefig(args.out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"{args.cell.name}: {count} frames, strike={strike} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
