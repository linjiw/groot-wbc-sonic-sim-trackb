#!/usr/bin/env python3
"""Report whether the corpus covers a behaviour space or repeats one behaviour.

This is the Phase B exit gate. It reports effective rank three ways -- pooled,
between-episode, and within-episode -- because they answer different questions and quoting
the wrong one understates coverage (see gear_sonic.dataset_generation.behaviour_diversity).

Usage::

    python scripts/research/report_behaviour_diversity.py \\
        --root /data/.../groot-wbc-kimodo-m0 --json diversity.json [--accepted-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from gear_sonic.dataset_generation.behaviour_diversity import (  # noqa: E402
    build_diversity_report,
    summarise_episode,
)
from gear_sonic.dataset_generation.episode_outcome import (  # noqa: E402
    OutcomeTally,
    classify_episode,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory to search")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="restrict to episodes that pass the acceptance gates",
    )
    args = parser.parse_args()

    # rglob, not a fixed-depth glob: rollout directories sit at different depths under the
    # work root (clutter_rollouts/<id>/, newprompt/rollouts/<id>/, ...), and a fixed pattern
    # silently reports diversity over whichever subset it happened to match.
    paths = sorted(args.root.rglob("*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectories under {args.root}")

    episodes, actions, skipped = [], [], 0
    tally = OutcomeTally()
    for path in paths:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
        except Exception as error:  # noqa: BLE001 - a corrupt file must not end the sweep
            print(f"  skip {path.parent.parent.name}: {type(error).__name__}: {error}")
            skipped += 1
            continue
        episode_id = path.parent.parent.name
        outcome = classify_episode(episode_id, payload)
        tally.add(outcome)
        # An unevaluable capture has no valid trajectory to measure. Including one poisons
        # every corpus-level statistic: a capture that reset 22 times in 199 frames read as
        # 10.16 m of path with 0.25 m of displacement, -106.9 rad of heading change and a
        # tortuosity of 41, which would have been reported as diversity.
        if not outcome.evaluated:
            continue
        if args.accepted_only and not outcome.accepted:
            continue
        # A reset-spanning capture holds two passes; measuring diversity over the stitched
        # file would sum a teleport into the path length. Use the recovered pass.
        if outcome.recovered_from_split:
            payload, _ = best_evaluable_payload(payload)
        episodes.append(summarise_episode(episode_id, payload))
        actions.append(np.asarray(payload["action_motion_token"], dtype=np.float64))

    if not episodes:
        raise SystemExit("no episodes matched the filter")

    report = build_diversity_report(episodes, actions)
    print(f"outcomes: {tally.summary()}")
    for line in report.summary_lines():
        print(line)
    if skipped:
        print(f"  ({skipped} unreadable trajectory file(s) skipped)")

    if args.json_path is not None:
        payload = {
            "episode_count": report.episode_count,
            "action_dim": report.action_dim,
            "pooled_rank": report.pooled_rank,
            "between_episode_rank": report.between_episode_rank,
            "within_episode_rank_mean": report.within_episode_rank_mean,
            "accepted_only": bool(args.accepted_only),
            "outcomes": {
                "accepted": tally.accepted,
                "rejected": tally.rejected,
                "unevaluable": tally.unevaluable,
                "recovered_from_split": tally.recovered_from_split,
                "acceptance_rate_of_evaluated": tally.acceptance_rate,
            },
            "spreads": report.spreads,
            "episodes": [
                {
                    "episode_id": e.episode_id,
                    "mean_speed_mps": e.mean_speed_mps,
                    "path_length_m": e.path_length_m,
                    "net_displacement_m": e.net_displacement_m,
                    "tortuosity": e.tortuosity,
                    "heading_change_rad": e.heading_change_rad,
                    "root_height_min_m": e.root_height_min_m,
                    "root_height_range_m": e.root_height_range_m,
                    "max_tilt_rad": e.max_tilt_rad,
                    "within_rank": e.within_rank,
                }
                for e in report.episodes
            ],
        }
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
