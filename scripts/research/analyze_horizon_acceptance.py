#!/usr/bin/env python3
"""Measure how acceptance depends on how much of an episode is graded.

The tracker starts on its reference and drifts, so a longer episode is a harder one. That
makes episode length a hidden parameter of every acceptance rate this project has quoted --
and it was hidden, because captures were silently truncated at 59% of each motion.

This grades the *same* episodes at a range of horizons, which isolates the effect: no
difference in scene, motion, or seed, only in how much of the trajectory the gates see. The
result is a dose-response curve and, with it, a defensible answer to "how long should a
generated motion be?" rather than a number inherited from whichever clip came first.

Usage::

    python scripts/research/analyze_horizon_acceptance.py \\
        --root /data/.../batch_v2/rollouts [--json horizon.json]
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

from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Control rate of the environment, so frames convert to seconds.
CONTROL_HZ = 50.0

#: Horizons to grade at, in frames. The densest sampling is where the curve bends.
DEFAULT_HORIZONS = (60, 100, 148, 175, 200, 225, 240)


def truncate(payload: dict, frames: int) -> dict:
    """A view of the payload as if the recorder had stopped at ``frames``."""
    total = int(payload["total_frames"])

    def cut(value):
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == total:
            return value[:frames]
        if isinstance(value, dict):
            return {key: cut(inner) for key, inner in value.items()}
        return value

    out = {key: cut(value) for key, value in payload.items() if key != "total_frames"}
    out["total_frames"] = frames
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--horizons", type=int, nargs="*", default=list(DEFAULT_HORIZONS))
    args = parser.parse_args()

    episodes = []
    for path in sorted(args.root.rglob("*.trajectory.pkl")):
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
            payload, _ = best_evaluable_payload(payload)
        except Exception as error:  # noqa: BLE001 - a bad file must not end the sweep
            print(f"  skip {path.parent.parent.name}: {type(error).__name__}: {error}")
            continue
        episodes.append(payload)

    if not episodes:
        raise SystemExit(f"no usable trajectories under {args.root}")

    # Only horizons every episode can reach are comparable; a horizon that silently drops
    # the shorter episodes compares different populations at each row.
    shortest = min(int(e["total_frames"]) for e in episodes)
    horizons = [h for h in sorted(args.horizons) if h <= shortest]
    dropped = [h for h in sorted(args.horizons) if h > shortest]
    if not horizons:
        raise SystemExit(f"no requested horizon fits the shortest episode ({shortest} frames)")

    print(f"{len(episodes)} episode(s), shortest {shortest} frames "
          f"({shortest / CONTROL_HZ:.2f} s)")
    if dropped:
        print(f"  (horizons {dropped} exceed it and are not comparable, so they are omitted)")
    print()
    header = f"{'frames':>7} {'seconds':>8} {'accepted':>9} {'rate':>6} {'median p95 path err':>21}"
    print(header)
    print("-" * len(header))

    rows = []
    for horizon in horizons:
        accepted = 0
        graded = 0
        errors = []
        for payload in episodes:
            report = evaluate_locomotion_trajectory(truncate(payload, horizon))
            if report.errors:
                continue
            graded += 1
            accepted += bool(report.accepted)
            gates = {gate.name: gate.value for gate in report.gates}
            if "path_error_p95" in gates:
                errors.append(float(gates["path_error_p95"]))
        rate = accepted / graded if graded else 0.0
        median_error = float(np.median(errors)) if errors else float("nan")
        rows.append(
            {
                "frames": horizon,
                "seconds": horizon / CONTROL_HZ,
                "graded": graded,
                "accepted": accepted,
                "acceptance_rate": rate,
                "median_path_error_p95_m": median_error,
            }
        )
        print(
            f"{horizon:7d} {horizon / CONTROL_HZ:8.2f} {accepted:9d} {rate:6.0%} "
            f"{median_error:21.3f}"
        )

    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print(
            f"\nacceptance falls {first['acceptance_rate']:.0%} -> {last['acceptance_rate']:.0%} "
            f"between {first['seconds']:.1f} s and {last['seconds']:.1f} s, while median p95 "
            f"path error grows {first['median_path_error_p95_m']:.3f} -> "
            f"{last['median_path_error_p95_m']:.3f} m."
        )
        print("Episode length is therefore a parameter of any acceptance rate quoted here,")
        print("and a rate reported without its horizon is not comparable to another.")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {"episodes": len(episodes), "shortest_frames": shortest, "horizons": rows},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
