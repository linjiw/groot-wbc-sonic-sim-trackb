#!/usr/bin/env python3
"""Does choosing where to put the obstacle matter? Measure it against two baselines.

Obstacle position has been treated as a scene parameter -- something a room generator picks.
On the one family built so far it behaves like an algorithm parameter instead: the route
midpoint gave a 53 mm discriminative window and the station where the two motions actually
differ gave 178 mm, which is the difference between a 3 N graze and a decisive collision.

One family is an anecdote. This runs all three policies over every compatible pair in the
corpus, geometrically, with no GPU:

* **random** -- a station drawn uniformly along the shared route, averaged over draws
* **midpoint** -- the middle of the route, which is what the builder used to do
* **maximal separation** -- the station maximising the envelope gap, which is what it does now

The window is what the family is made of: below about 50 mm there is nowhere to put an
obstacle that reliably separates the two motions, so the fraction of pairs clearing that bar
is the yield the method actually delivers.

Usage::

    python scripts/research/ablate_station_choice.py \\
        --snapshot /data/.../g1_motionbank_v0.4.json --regime overhead
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
    DEFAULT_STATION_SPAN_M,
    compute_envelope,
    half_width_at_stations,
    mine_pairs,
    silhouette_at_stations,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    SegmentError,
    best_evaluable_payload,
)

#: Window below which no obstacle placement reliably separates two motions in physics. The
#: measured family sits at 178 mm; the abandoned midpoint build sat at 53 mm and produced a
#: 3 N graze that barely cleared the acceptance gate's 1 N threshold.
USABLE_WINDOW_M = 0.05

PROFILE = {"overhead": silhouette_at_stations, "lateral": half_width_at_stations}


def load_signatures(snapshot: dict) -> tuple[list, dict]:
    signatures, payloads, seen = [], {}, set()
    for episode in snapshot["episodes"]:
        if episode.get("outcome") != "accepted":
            continue
        fingerprint = episode.get("trajectory_fingerprint")
        if fingerprint in seen:
            continue
        path = Path(episode["trajectory_path"])
        if not path.exists():
            continue
        try:
            with path.open("rb") as handle:
                payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
            if payload.get("body_pos_w") is None:
                continue
            signature = compute_envelope(
                payload, episode["episode_id"], episode.get("behaviour", "")
            )
        except (SegmentError, KeyError, ValueError):
            continue
        seen.add(fingerprint)
        signatures.append(signature)
        payloads[episode["episode_id"]] = payload
    return signatures, payloads


def windows(nominal, adapted, regime: str, rng: np.random.Generator) -> dict | None:
    """Window width under each placement policy, for one ordered pair."""
    profile = PROFILE[regime]

    def route(payload):
        root = np.asarray(payload["root_pos_w"], dtype=np.float64)
        return float(root[:, 0].min()), float(root[:, 0].max())

    low = max(route(nominal)[0], route(adapted)[0])
    high = min(route(nominal)[1], route(adapted)[1])
    if high - low < DEFAULT_STATION_SPAN_M:
        return None

    stations = np.arange(low, high, 0.05)
    gap = profile(nominal, stations) - profile(adapted, stations)
    if not np.isfinite(gap).any():
        return None

    finite = np.where(np.isfinite(gap))[0]
    midpoint_index = int(np.argmin(np.abs(stations - 0.5 * (low + high))))
    if not np.isfinite(gap[midpoint_index]):
        midpoint_index = int(finite[len(finite) // 2])

    draws = rng.choice(finite, size=min(24, len(finite)), replace=True)
    return {
        "random": float(np.mean(gap[draws])),
        "midpoint": float(gap[midpoint_index]),
        "maximal": float(np.nanmax(gap)),
        "station_maximal_m": float(stations[int(np.nanargmax(gap))]),
        "station_midpoint_m": float(stations[midpoint_index]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--regime", default="overhead", choices=sorted(PROFILE))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    snapshot = json.loads(args.snapshot.read_text())
    signatures, payloads = load_signatures(snapshot)
    pairs = mine_pairs(signatures, args.regime, limit=400)
    print(f"{len(signatures)} distinct episodes, {len(pairs)} compatible {args.regime} pairs\n")

    rows = []
    for candidate in pairs:
        result = windows(
            payloads[candidate.nominal], payloads[candidate.adapted], args.regime, rng
        )
        if result is None:
            continue
        result.update({"nominal": candidate.nominal, "adapted": candidate.adapted})
        rows.append(result)

    if not rows:
        print("no pair had enough shared route to place an obstacle on")
        return 1

    print(f"{'placement policy':22s}{'median window':>15s}{'best':>9s}"
          f"{'pairs >= 50 mm':>16s}")
    for policy, label in (("random", "random station"), ("midpoint", "route midpoint"),
                          ("maximal", "maximal separation")):
        values = np.array([r[policy] for r in rows])
        usable = int((values >= USABLE_WINDOW_M).sum())
        print(f"{label:22s}{np.median(values)*1000:12.0f} mm{values.max()*1000:7.0f} mm"
              f"{usable:10d}/{len(rows)}")

    gain = np.array([r["maximal"] for r in rows]) - np.array([r["midpoint"] for r in rows])
    moved = np.array([abs(r["station_maximal_m"] - r["station_midpoint_m"]) for r in rows])
    print("\nchoosing the station over taking the midpoint:")
    print(f"  median gain {np.median(gain)*1000:+.0f} mm, best {gain.max()*1000:+.0f} mm")
    print(f"  the chosen station sits a median {np.median(moved):.2f} m from the midpoint")
    rescued = int(((np.array([r['maximal'] for r in rows]) >= USABLE_WINDOW_M)
                   & (np.array([r['midpoint'] for r in rows]) < USABLE_WINDOW_M)).sum())
    print(f"  {rescued} pair(s) become usable that the midpoint would have discarded")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
