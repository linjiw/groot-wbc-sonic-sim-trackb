#!/usr/bin/env python
"""Turn screened nominals into a queue of family configurations, ranked by what they cost.

Screening answers whether the controller can hold a motion. This answers what that motion is worth:
how many banded configurations have a window wide enough to survive the predictor's error, which
operator each needs, and therefore how many families the motion can support.

Ranking matters because the budget is rollouts, not motions. A nominal offering four wide-window
configurations is worth four times one offering a single marginal one, and that is invisible until
the windows are measured.

Emits a plan only. Scenes are written by ``build_graded_scene.py`` and rollouts are a separate
batch, so a bad plan costs nothing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

#: Metres. A window narrower than this is not worth four rollouts: the predictor's calibrated error
#: on the one family where both boundaries were bracketed is 8 mm at the reference, and the realised
#: window ran 23-82% of the predicted one.
MIN_WINDOW_M = 0.020


def screened(directory: Path) -> str | None:
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        return None
    try:
        with open(paths[0], "rb") as handle:
            payload, _ = best_evaluable_payload(pickle.load(handle))
    except (SegmentError, Exception):  # noqa: BLE001
        return None
    if payload is None:
        return "unevaluable"
    return classify_episode(directory.name, payload).outcome


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screens", type=Path, required=True, help="directory of screen rollouts")
    ap.add_argument("--clips", type=Path, required=True, help="directory of reference CSVs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--python", default=str(Path(sys.executable)))
    args = ap.parse_args()

    plans, skipped = [], []
    for cell in sorted(p for p in args.screens.iterdir() if p.is_dir() and p.name.startswith("n_")):
        verdict = screened(cell)
        if verdict != "accepted":
            skipped.append((cell.name, verdict or "no rollout"))
            continue
        matches = sorted(args.clips.glob(f"{cell.name[2:]}*.csv"))
        if not matches:
            skipped.append((cell.name, "no reference csv"))
            continue

        for obstacle in ("ceiling", "wall"):
            target = args.out / f"{cell.name}_{obstacle}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    args.python,
                    str(REPO_ROOT / "scripts/research/plan_graded_families.py"),
                    "--nominal-rollout",
                    str(cell),
                    "--nominal-csv",
                    str(matches[0]),
                    "--obstacle",
                    obstacle,
                    "--json",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            if not target.exists():
                skipped.append((f"{cell.name}/{obstacle}", "planner produced nothing"))
                continue
            for entry in json.loads(target.read_text())["plans"]:
                if entry["window_m"] >= MIN_WINDOW_M:
                    plans.append({"nominal": cell.name, **entry})

    by_nominal: dict[str, int] = {}
    for entry in plans:
        by_nominal[entry["nominal"]] = by_nominal.get(entry["nominal"], 0) + 1

    print(
        f"{len(plans)} configurations with a window over {MIN_WINDOW_M * 1000:.0f} mm, "
        f"across {len(by_nominal)} nominals\n"
    )
    print(f"{'nominal':>10s}{'configs':>9s}   widest windows (mm)")
    for nominal, count in sorted(by_nominal.items(), key=lambda kv: -kv[1]):
        widths = sorted(
            (e["window_m"] * 1000 for e in plans if e["nominal"] == nominal), reverse=True
        )[:4]
        print(f"{nominal:>10s}{count:9d}   {', '.join(f'{w:.0f}' for w in widths)}")
    if skipped:
        print("\nnot planned:")
        for name, why in skipped:
            print(f"   {name:>16s}  {why}")

    # Four rollouts per family is the 2x2; the estimate is what makes the budget visible before it
    # is spent rather than after.
    print(f"\n{len(plans)} families would cost {len(plans) * 4} rollouts")
    (args.out / "batch.json").write_text(
        json.dumps({"min_window_m": MIN_WINDOW_M, "plans": plans, "skipped": skipped}, indent=2)
        + "\n"
    )
    print(f"wrote {args.out / 'batch.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
