#!/usr/bin/env python3
"""Search the corpus for motion pairs that could form a counterfactual family.

Every family so far was built from a hand-picked pair, which does not scale and wastes what
already exists: 131 distinct executed trajectories are recorded, and the pair that separates
under some obstacle is probably among them. Asking the generator for a new adapted motion
each time is both slower and worse, because a generated motion still has to be rolled out
before anything is known about it, while these have been.

Two stages, cheap then expensive:

1. **signature** -- reduce each trajectory to the numbers pairing needs, and reject pairs
   that are not doing the same task. This is where most candidates die, and it should be:
   two motions that end up in different places are not a counterfactual whatever their
   swept volumes do.
2. **station** -- for surviving overhead pairs, scan the shared route for the position where
   the window is widest. This is what the family builder then uses, and it is far more
   honest than the global minimum: on the one measured family the global spread reads
   0.197 m where the station-aware value reads 0.052 m against a measured 0.053 m.

Usage::

    python scripts/research/mine_counterfactual_pairs.py \\
        --snapshot /data/.../snapshots/g1_motionbank_v0.4.json --regime overhead
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.motion_envelope import (  # noqa: E402
    best_overhead_station,
    compute_envelope,
    mine_pairs,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)


def load_signatures(snapshot: dict, limit: int | None) -> tuple[list, dict, dict]:
    """Envelope every accepted episode the snapshot points at, once per trajectory.

    Deduplicated by ``trajectory_fingerprint``. The corpus deliberately replays one motion
    through several clutter scenes -- that is visual augmentation, and the snapshot already
    records 107 distinct fingerprints among 132 accepted episodes. Without the dedup the
    ranking fills with the same pair under different scene names: ``density_dense``,
    ``density_moderate``, ``density_sparse`` and ``density_tight`` returned an identical
    0.088 m screen and 0.127 m refined window, because they are one trajectory.
    """
    episodes = [e for e in snapshot["episodes"] if e.get("outcome") == "accepted"]
    if limit:
        episodes = episodes[:limit]

    signatures, payloads = [], {}
    seen_fingerprints: dict[str, str] = {}
    stats = {"total": len(episodes), "duplicate": 0, "no_body_geometry": 0, "unreadable": 0}

    for episode in episodes:
        fingerprint = episode.get("trajectory_fingerprint")
        if fingerprint and fingerprint in seen_fingerprints:
            stats["duplicate"] += 1
            continue
        path = Path(episode["trajectory_path"])
        if not path.exists():
            stats["unreadable"] += 1
            continue
        try:
            with path.open("rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
        except (SegmentError, KeyError, ValueError):
            stats["unreadable"] += 1
            continue

        # Older recordings predate per-body capture and carry no geometry at all. That is a
        # different fact from an unreadable file and is counted separately, because it caps
        # how much of the corpus can ever be mined rather than pointing at a bug.
        if payload.get("body_pos_w") is None or "body_pos_w" not in payload:
            stats["no_body_geometry"] += 1
            continue

        try:
            signature = compute_envelope(
                payload, episode["episode_id"], episode.get("behaviour", "")
            )
        except (KeyError, ValueError):
            stats["unreadable"] += 1
            continue

        if fingerprint:
            seen_fingerprints[fingerprint] = episode["episode_id"]
        signatures.append(signature)
        payloads[episode["episode_id"]] = payload

    return signatures, payloads, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--regime", default="overhead",
                        choices=("overhead", "lateral", "floor"))
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="cap episodes, for a quick look")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    snapshot = json.loads(args.snapshot.read_text())
    print(f"snapshot {snapshot['snapshot']}  "
          f"{snapshot['counts']['accepted']} accepted episodes")

    signatures, payloads, stats = load_signatures(snapshot, args.limit or None)
    print(f"  {stats['duplicate']} skipped as trajectory duplicates")
    print(f"  {stats['no_body_geometry']} carry no per-body geometry (recorded before "
          "the recorder captured it)")
    if stats["unreadable"]:
        print(f"  {stats['unreadable']} unreadable")
    print(f"enveloped {len(signatures)} distinct episode(s) -- this, not the accepted "
          "count, is what can be mined\n")

    ranked = mine_pairs(signatures, args.regime, limit=args.top)
    if not ranked:
        print(f"no compatible {args.regime} pair in this corpus")
        print("  the compatibility check is doing its job: a wide geometric gap between two")
        print("  motions going to different places is not a counterfactual")
        return 1

    by_id = {signature.episode_id: signature for signature in signatures}
    print(f"top {len(ranked)} {args.regime} candidates, screened then station-refined:\n")
    print(f"{'nominal':22s}{'adapted':22s}{'screen':>9s}{'station':>9s}{'refined':>9s}")

    rows = []
    for candidate in ranked:
        station, refined = (float("nan"), float("nan"))
        if args.regime == "overhead":
            station, refined = best_overhead_station(
                payloads[candidate.nominal], payloads[candidate.adapted]
            )
        rows.append({
            "nominal": candidate.nominal,
            "adapted": candidate.adapted,
            "nominal_behaviour": by_id[candidate.nominal].behaviour,
            "adapted_behaviour": by_id[candidate.adapted].behaviour,
            "screen_spread_m": candidate.spread_m,
            "station_x_m": station,
            "refined_window_m": refined,
        })
        print(f"{candidate.nominal[:21]:22s}{candidate.adapted[:21]:22s}"
              f"{candidate.spread_m:9.3f}{station:9.2f}{refined:9.3f}")

    if args.regime == "overhead":
        usable = [row for row in rows if row["refined_window_m"] > 0.05]
        print(f"\n{len(usable)} of {len(rows)} survive station refinement above 0.05 m")
        print("  the screen is deliberately optimistic; this is the number that matters")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
