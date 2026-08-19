#!/usr/bin/env python
"""Render the frozen review cohort as evidence a person can judge without seeing the verdict.

The point of the cohort is to measure the *gates*, so a reviewer who is shown the gate's answer
first is measuring their own agreement with an anchor rather than the episode. Each card therefore
leads with the evidence -- a strip of frames spanning the episode, the obstacle drawn to scale, the
running clearance -- and keeps the automatic verdict behind a disclosure the reviewer opens only
after deciding.

Contact sheets rather than video: a hundred clips is an hour of rendering and an hour of watching,
and a six-frame strip through the approach carries what a reviewer needs to see.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import pickle
import re
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "research"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import build_counterfactual_family as cf  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from gear_sonic.dataset_generation.scene_route_check import (  # noqa: E402
    capsule_box_clearance,
)
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

SCENES = REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual"
ROBOT = "#8a5a4a"
SHELF = "#b8a68f"

#: What a reviewer is asked. Deliberately three separable questions, because physics acceptance,
#: behaviour semantics and scene compatibility are different labels and a single "looks fine"
#: collapses them.
QUESTIONS = (
    ("traversed", "Did the robot get past the obstacle without hitting it?"),
    ("upright", "Did it stay upright and in control throughout?"),
    ("behaviour", "Did it perform the behaviour its name claims (walk / crouch / tuck)?"),
)


def scene_of(cell_dir: Path) -> str | None:
    for name in (f"{cell_dir.name}.runner.log", f"{cell_dir.name}.log"):
        log = cell_dir.parent / "logs" / name
        if log.exists():
            match = re.search(r"scene=(\S+)", log.read_text(errors="ignore")[:8000])
            if match:
                return match.group(1)
    return None


GHOST = "#9aa3ad"


def nominal_sibling(cell_dir: Path) -> Path | None:
    """The unadapted counterpart of a cell, by the layout conventions already in use.

    A reviewer asked whether a clip performs its named behaviour is comparing it against ordinary
    gait remembered from cards seen much earlier. Drawing the nominal underneath makes that
    comparison direct. Returns None when the cell has no counterpart, and the card then shows the
    adapted body alone, exactly as before.
    """
    name = cell_dir.name
    if name.startswith("adapted_"):
        candidate = cell_dir.parent / f"nominal_{name[len('adapted_'):]}"
        return candidate if candidate.is_dir() else None
    prefix = name.split("_")[0]
    candidate = cell_dir.parent / f"{prefix}_nominal"
    if candidate.is_dir() and candidate != cell_dir:
        return candidate
    return None


def ghost_capsules(cell_dir: Path):
    """Body capsules of the nominal counterpart, or None."""
    sibling = nominal_sibling(cell_dir)
    if sibling is None:
        return None
    found = sorted(sibling.glob("trajectories/*.trajectory.pkl"))
    if not found:
        return None
    try:
        with open(found[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except (SegmentError, Exception):  # noqa: BLE001
        return None
    if payload is None or "body_pos_w" not in payload:
        return None
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )
    return starts, ends, radii


def _draw_body(ax, starts, ends, radii, index: int, axis: int, colour: str, alpha: float) -> None:
    """One frame of capsules projected onto x and the given second axis."""
    index = min(index, starts.shape[0] - 1)
    for capsule in range(starts.shape[1]):
        a, b, r = starts[index, capsule], ends[index, capsule], radii[capsule]
        p0, p1 = np.array([a[0], a[axis]]), np.array([b[0], b[axis]])
        for point in (p0, p1):
            ax.add_patch(Circle(point, r, facecolor=colour, ec="none", alpha=alpha, zorder=2))
        d = p1 - p0
        n = float(np.hypot(*d))
        if n > 1e-9:
            normal = np.array([-d[1], d[0]]) / n * r
            ax.add_patch(
                Polygon(
                    [p0 + normal, p1 + normal, p1 - normal, p0 - normal],
                    closed=True,
                    facecolor=colour,
                    ec="none",
                    alpha=alpha,
                    zorder=2,
                )
            )


def _plan_view(ax, starts, ends, radii, root, box, index: int, ghost=None) -> None:
    """The same instant seen from above.

    The side elevation cannot show a lateral arm tuck at all: it projects onto x-z, and the tuck
    retracts the wrist along y by around 0.16 m. A reviewer shown only that view correctly reported
    no visible tuck on clips whose physics contains one. This row carries the missing axis.
    """
    if box is not None:
        ax.add_patch(
            Rectangle(
                (box[0], box[1]),
                box[3] - box[0],
                box[4] - box[1],
                facecolor=SHELF,
                edgecolor="#6b5d4a",
                lw=0.8,
                zorder=3,
            )
        )
    if ghost is not None:
        _draw_body(ax, ghost[0], ghost[1], ghost[2], index, 1, GHOST, 0.30)
    _draw_body(ax, starts, ends, radii, index, 1, ROBOT, 0.55)
    centre = float(root[index, 0])
    ax.set_xlim(centre - 1.0, centre + 1.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def strip(payload, box, out: Path, frames: int = 6, ghost=None) -> bool:
    """Side elevation and plan view of the episode, with the obstacle to scale.

    Where the cell has an unadapted counterpart it is drawn underneath in grey, so a reviewer
    judging whether the named behaviour happened compares two bodies in one panel rather than
    recalling ordinary gait from a card seen much earlier.
    """
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]),
        capsules=G1_COLLISION_CAPSULES,
    )
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    picks = np.linspace(0, len(root) - 1, frames).astype(int)

    fig, grid = plt.subplots(
        2, len(picks), figsize=(2.0 * len(picks), 4.6), dpi=95, squeeze=False
    )
    for ax, plan, index in zip(grid[0], grid[1], picks):
        _plan_view(plan, starts, ends, radii, root, box, index, ghost=ghost)
        ax.axhspan(-0.06, 0.0, facecolor="#c9ced6", zorder=0)
        if box is not None:
            ax.add_patch(
                Rectangle(
                    (box[0], box[2]),
                    box[3] - box[0],
                    box[5] - box[2],
                    facecolor=SHELF,
                    edgecolor="#6b5d4a",
                    lw=0.8,
                    zorder=3,
                )
            )
        if ghost is not None:
            _draw_body(ax, ghost[0], ghost[1], ghost[2], index, 2, GHOST, 0.30)
        _draw_body(ax, starts, ends, radii, index, 2, ROBOT, 0.55)
        centre = float(root[index, 0])
        ax.set_xlim(centre - 1.3, centre + 1.3)
        ax.set_ylim(-0.06, 1.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"f{index}", fontsize=7)
    grid[0][0].set_ylabel("from the side", fontsize=7)
    grid[1][0].set_ylabel("from above", fontsize=7)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return True


def root_only(payload, out: Path) -> bool:
    """Reduced evidence for captures that predate the body-pose fields.

    Root height and top-down path let a reviewer judge whether the robot stayed upright and held
    its route. They cannot show whether it touched an obstacle, because the body is not recorded --
    so cards built from this are marked, and their answers count only toward the questions the
    evidence supports. Dropping these instead would quietly shrink a frozen cohort to the episodes
    that happen to render.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), dpi=95)
    axes[0].plot(root[:, 2], color=ROBOT, lw=1.6)
    axes[0].set_title("root height (m)", fontsize=8)
    axes[0].set_xlabel("frame", fontsize=7)
    axes[0].tick_params(labelsize=6)
    axes[0].grid(alpha=0.15, lw=0.5)
    axes[1].plot(root[:, 0], root[:, 1], color=ROBOT, lw=1.6)
    axes[1].scatter([root[0, 0]], [root[0, 1]], s=18, color="#2f6b4f", zorder=3)
    axes[1].set_title("path from above (m)", fontsize=8)
    axes[1].set_aspect("equal")
    axes[1].tick_params(labelsize=6)
    axes[1].grid(alpha=0.15, lw=0.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text())
    episodes = cohort["episodes"][: args.limit] if args.limit else cohort["episodes"]
    sheets = args.out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)

    rows, rendered, skipped, reduced = [], 0, 0, 0
    for entry in episodes:
        path = Path(entry["path"])
        cell_dir = path.parents[1]
        episode_id = f"{cell_dir.parent.name}/{cell_dir.name}"
        record = {
            "episode_id": episode_id,
            "sheet": "",
            "automatic_outcome": entry.get("outcome", ""),
            "automatic_reasons": ";".join(entry.get("reasons", []) or []),
            "overhead_n": entry.get("overhead_n", ""),
            "lateral_n": entry.get("lateral_n", ""),
            "drift_mps": entry.get("drift_mps", ""),
            "min_clearance_mm": "",
            "evidence": "full",
        }
        try:
            with open(path, "rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))
        except (SegmentError, Exception):  # noqa: BLE001
            payload = None
        if payload is None:
            record["automatic_outcome"] = record["automatic_outcome"] or "unevaluable"
            rows.append(record)
            skipped += 1
            continue

        # Older captures predate the body-pose fields but still carry the root. Root height and
        # path let a reviewer judge uprightness and route, though not obstacle contact, so the card
        # is marked and its answers count only toward the questions the evidence supports.
        # Dropping them would quietly shrink a frozen cohort to the episodes that happen to render.
        if "body_pos_w" not in payload:
            name = episode_id.replace("/", "__") + ".png"
            if "root_pos_w" in payload and root_only(payload, sheets / name):
                record["sheet"] = f"sheets/{name}"
                record["evidence"] = "reduced: root only, obstacle contact not visible"
                reduced += 1
            else:
                record["evidence"] = "none"
                skipped += 1
            rows.append(record)
            continue

        box = None
        scene = scene_of(cell_dir)
        if scene and (SCENES / f"{scene}.usda").exists():
            try:
                box = cf.rendered_shelf_box(SCENES / f"{scene}.usda")
                starts, ends, radii, _ = body_capsules_world(
                    np.asarray(payload["body_pos_w"], dtype=np.float64),
                    np.asarray(payload["body_quat_w"], dtype=np.float64),
                    list(payload["body_names"]),
                    capsules=G1_COLLISION_CAPSULES,
                )
                gap, _f, _c = capsule_box_clearance(starts, ends, radii, box)
                record["min_clearance_mm"] = round(gap * 1000, 1)
            except Exception:  # noqa: BLE001
                box = None

        name = episode_id.replace("/", "__") + ".png"
        strip(payload, box, sheets / name, ghost=ghost_capsules(cell_dir))
        record["sheet"] = f"sheets/{name}"
        rows.append(record)
        rendered += 1

    for rater in ("a", "b"):
        target = args.out / f"review_rater_{rater}.csv"
        if target.exists():
            print(f"keeping existing {target.name}; a review in progress is not overwritten")
            continue
        with open(target, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["episode_id", *[q for q, _ in QUESTIONS], "note"])
            for row in rows:
                writer.writerow([row["episode_id"], "", "", "", ""])

    with open(args.out / "cohort_evidence.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{len(rows)} episodes: {rendered} full contact sheets, {reduced} reduced "
        f"(root only, obstacle contact not visible), {skipped} with no usable evidence"
    )
    print("blank rater sheets: review_rater_a.csv, review_rater_b.csv")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
