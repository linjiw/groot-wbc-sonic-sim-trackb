#!/usr/bin/env python3
"""Stage 2: did SONIC preserve the operator's effect while leaving everything else matched?

Acceptance is the wrong summary here. A matched operator holds root path, duration and gait
phase fixed in the *reference*, so the question is not whether the adapted clip was accepted
but whether the pair stayed matched through execution and how much of the intended envelope
change survived. Two clips can both be accepted while the adaptation has been tracked away to
nothing.

Two verdicts, kept apart:

    operator_trackable    the motion survived physics with the pair still matched
    family_eligible       its *executed* effect is large enough to build a family on

The second is what decides whether six more rollouts are worth spending, and it is measured
on the executed capsules rather than the reference, because what collides is the executed body.
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

from gear_sonic.dataset_generation.episode_outcome import classify_episode  # noqa: E402
from gear_sonic.dataset_generation.local_adaptation import (  # noqa: E402
    _silhouette,
    route_progress,
)
from gear_sonic.dataset_generation.self_intersection import DEFAULT_G1_MJCF  # noqa: E402
from gear_sonic.dataset_generation.swept_volume import (  # noqa: E402
    G1_COLLISION_CAPSULES,
    body_capsules_world,
)
from gear_sonic.dataset_generation.trajectory_segments import (  # noqa: E402
    best_evaluable_payload,
)

#: Executed effect below which a pair cannot anchor a family, in metres. Provisional: it is
#: the smallest window any verified family has been built on, not a calibrated threshold.
MIN_EXECUTED_EFFECT_M = 0.05


def executed(directory: Path) -> dict:
    paths = sorted(directory.glob("trajectories/*.trajectory.pkl"))
    if not paths:
        raise SystemExit(f"no trajectory in {directory}")
    with paths[0].open("rb") as handle:
        payload, _ = best_evaluable_payload(pickle.load(handle))  # noqa: S301
    return payload


def half_width_of(payload: dict) -> np.ndarray:
    """Per-frame half-width across the heading, for the lateral regime.

    A lateral operator moves width, not height. Measuring an arm tuck on the silhouette
    reports a reference effect of exactly 0.0 mm -- true, and about the wrong axis.
    """
    root = np.asarray(payload["root_pos_w"], dtype=np.float64)
    quat = np.asarray(payload["root_quat_w"], dtype=np.float64)
    w, x, y, z = (quat[:, i] for i in range(4))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    lateral = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]), capsules=G1_COLLISION_CAPSULES,
    )
    centres = 0.5 * (starts + ends)
    offsets = centres[:, :, :2] - root[:, None, :2]
    return (np.abs(np.einsum("tcd,td->tc", offsets, lateral)) + radii[None, :]).max(axis=1)


def silhouette_of(payload: dict) -> np.ndarray:
    starts, ends, radii, _ = body_capsules_world(
        np.asarray(payload["body_pos_w"], dtype=np.float64),
        np.asarray(payload["body_quat_w"], dtype=np.float64),
        list(payload["body_names"]), capsules=G1_COLLISION_CAPSULES,
    )
    return (np.maximum(starts[:, :, 2], ends[:, :, 2]) + radii[None, :]).max(axis=1)


def report(
    nominal_ref: np.ndarray, adapted_ref: np.ndarray,
    nominal_dir: Path, adapted_dir: Path, station: float, half_window: float = 0.08,
    regime: str = "overhead",
) -> dict:
    from gear_sonic.dataset_generation.local_adaptation import _half_width
    from gear_sonic.dataset_generation.reference_payload import payload_from_reference

    progress = route_progress(nominal_ref[:, :2])
    core = np.abs(progress - station) <= half_window
    if regime == "overhead":
        reference_effect = float(
            (_silhouette(nominal_ref, DEFAULT_G1_MJCF)
             - _silhouette(adapted_ref, DEFAULT_G1_MJCF))[core].min()
        )
        measure = silhouette_of
    elif regime == "lateral":
        reference_effect = float(
            (_half_width(nominal_ref, DEFAULT_G1_MJCF)
             - _half_width(adapted_ref, DEFAULT_G1_MJCF))[core].min()
        )
        measure = half_width_of
    else:
        raise ValueError(f"unknown regime {regime!r}; expected overhead or lateral")

    pn, pa = executed(nominal_dir), executed(adapted_dir)
    outcome_n = classify_episode(nominal_dir.name, {**pn, "total_frames": len(pn["root_pos_w"])})
    outcome_a = classify_episode(adapted_dir.name, {**pa, "total_frames": len(pa["root_pos_w"])})

    sn, sa = measure(pn), measure(pa)
    frames = min(len(sn), len(sa))
    # Executed clips can differ in length; compare over the shared prefix and say so.
    grid = np.linspace(0.0, 1.0, frames)
    inside = np.abs(grid - station) <= half_window
    executed_effect = float((sn[:frames] - sa[:frames])[inside].min()) if inside.any() else 0.0

    rn = np.asarray(pn["root_pos_w"], dtype=np.float64)[:frames]
    ra = np.asarray(pa["root_pos_w"], dtype=np.float64)[:frames]
    outside = ~inside
    return {
        "reference_effect_m": reference_effect,
        "executed_effect_m": executed_effect,
        "effect_retention_ratio": (
            executed_effect / reference_effect if reference_effect > 1e-9 else float("nan")
        ),
        "nominal_outcome": outcome_n.outcome,
        "adapted_outcome": outcome_a.outcome,
        "nominal_frames": int(len(sn)),
        "adapted_frames": int(len(sa)),
        "duration_matched": bool(len(sn) == len(sa)),
        "root_xy_p95_difference_m": float(
            np.percentile(np.linalg.norm(rn[:, :2] - ra[:, :2], axis=1), 95)
        ),
        "outside_window_envelope_difference_m": float(
            np.abs(sn[:frames] - sa[:frames])[outside].max() if outside.any() else 0.0
        ),
        "operator_trackable": bool(
            outcome_a.evaluated and outcome_n.evaluated
            and outcome_a.outcome == "accepted"
        ),
        "family_eligible": bool(executed_effect >= MIN_EXECUTED_EFFECT_M),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--nominal", required=True)
    parser.add_argument("--adapted", required=True)
    parser.add_argument("--station", type=float, default=0.55)
    parser.add_argument("--regime", choices=("overhead", "lateral"), default="overhead",
                        help="which axis the operator moves; measuring the wrong one "
                             "reports a reference effect of 0.0 mm")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = report(
        np.loadtxt(args.clips / f"{args.nominal}.csv", delimiter=","),
        np.loadtxt(args.clips / f"{args.adapted}.csv", delimiter=","),
        args.rollouts / args.nominal, args.rollouts / args.adapted, args.station,
        regime=args.regime,
    )
    print(f"pair: {args.nominal} vs {args.adapted}\n")
    print(f"  reference effect      {result['reference_effect_m']*1000:7.1f} mm")
    print(f"  executed  effect      {result['executed_effect_m']*1000:7.1f} mm")
    print(f"  retention             {result['effect_retention_ratio']:7.0%}")
    print(f"  outcomes              {result['nominal_outcome']} / {result['adapted_outcome']}")
    print(f"  frames                {result['nominal_frames']} / {result['adapted_frames']}"
          f"   matched={result['duration_matched']}")
    print(f"  root XY p95 diff      {result['root_xy_p95_difference_m']*1000:7.1f} mm")
    print(f"  outside-window drift  {result['outside_window_envelope_difference_m']*1000:7.1f} mm")
    print(f"\n  operator_trackable    {result['operator_trackable']}")
    print(f"  family_eligible       {result['family_eligible']}"
          f"   (needs >= {MIN_EXECUTED_EFFECT_M*1000:.0f} mm executed)")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
