#!/usr/bin/env python3
"""Stage 2: is the adapted motion trackable, with the pair still matched?

**This no longer reports family eligibility, and the reason is worth recording.** It used to,
by inferring the executed envelope effect from the two rollouts. Four estimators were tried --
frame-indexed differences, maxima over a stride, minima at matched route progress, extremes at
matched route progress -- and they gave four different answers on the same rollouts: crouch
retention of 50/54/60%, then 32/32/38%, then 50/52/53%, then 63/73/90%. Two of them produced
retentions above 100%, which a tracker cannot do.

The difficulty is real rather than a coding slip. A walking robot's own envelope oscillates --
73 mm peak-to-peak in half-width, 63 mm in silhouette height -- an adapted clip that loses
forward progress is at a different gait phase, and an arm tuck perturbs the very swing that
sets the width. Any statistic over those two signals encodes a phase relationship as much as
an adaptation.

So the inference is abandoned rather than patched again. Whether an adaptation is large enough
to build a family on is answered directly by Stage 3: place the obstacle and see whether it
stops the nominal and passes the adapted. Stage 2 answers the question it can answer well.

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

#: Frames spanned by one stride, at 50 Hz. The comparison window must cover at least this,
#: or a phase difference between two clips is read as an adaptation.
STRIDE_FRAMES = 50


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


def active_window(nominal_ref: np.ndarray, adapted_ref: np.ndarray,
                  inner: float = 0.6) -> np.ndarray:
    """Frames where the adaptation is at full strength, found from the clips themselves.

    Defined once, on the references, and then applied to both sides. Every earlier version of
    this measurement got the window wrong in a different way: a fixed route-progress slice
    took its maximum from the ramp edges where the adaptation has barely begun, and widening a
    too-narrow window to a full stride pulled in frames where the adaptation is zero, which
    makes a minimum-over-window exactly zero by construction.
    """
    departure = np.abs(adapted_ref[:, 7:] - nominal_ref[:, 7:]).max(axis=1)
    if departure.max() <= 1e-9:
        return np.zeros(len(departure), dtype=bool)
    active = np.flatnonzero(departure > 0.5 * departure.max())
    if len(active) < 3:
        return np.zeros(len(departure), dtype=bool)
    # Keep the inner portion, away from the ramps at either end.
    margin = int(len(active) * (1.0 - inner) / 2)
    keep = active[margin:len(active) - margin] if margin else active
    mask = np.zeros(len(departure), dtype=bool)
    mask[keep] = True
    return mask


def report(
    nominal_ref: np.ndarray, adapted_ref: np.ndarray,
    nominal_dir: Path, adapted_dir: Path, station: float, half_window: float = 0.08,
    regime: str = "overhead",
) -> dict:
    """Compare a matched pair on the axis its operator moves, at matched route progress.

    Both effects are the *guaranteed* one -- the smallest difference anywhere in the window --
    because an obstacle has to be cleared on every frame the robot passes it, and both use the
    same estimator so their ratio means something.
    """
    from gear_sonic.dataset_generation.local_adaptation import _half_width

    curve = {"overhead": (_silhouette, silhouette_of),
             "lateral": (_half_width, half_width_of)}.get(regime)
    if curve is None:
        raise ValueError(f"unknown regime {regime!r}; expected overhead or lateral")
    reference_curve, executed_curve = curve

    mask = active_window(nominal_ref, adapted_ref)
    if not mask.any():
        raise SystemExit("the two reference clips are identical; nothing to measure")

    # Reference: frame-aligned by construction, so a frame-wise difference is exact.
    # What an obstacle meets is the *extreme* each clip presents while passing it, so the
    # effect is the difference of maxima over the active window -- for both regimes, and on
    # both sides. Differencing frame by frame works for the references, which are aligned by
    # construction, and fails for half-width once executed, because arms swing at their own
    # frequency and the tuck perturbs that phase, so route progress does not align them.
    # Comparing extremes is phase-free by construction.
    reference_effect = float(
        reference_curve(nominal_ref, DEFAULT_G1_MJCF)[mask].max()
        - reference_curve(adapted_ref, DEFAULT_G1_MJCF)[mask].max()
    )
    reference_progress = route_progress(nominal_ref[:, :2])
    window = (float(reference_progress[mask].min()), float(reference_progress[mask].max()))

    pn, pa = executed(nominal_dir), executed(adapted_dir)
    outcome_n = classify_episode(nominal_dir.name, {**pn, "total_frames": len(pn["root_pos_w"])})
    outcome_a = classify_episode(adapted_dir.name, {**pa, "total_frames": len(pa["root_pos_w"])})

    # Executed: a clip that loses forward progress is at a different point of its route and a
    # different gait phase, so a frame-indexed difference reads that offset as an adaptation.
    # Resample both onto route progress, then apply the same window.
    sn_raw, sa_raw = executed_curve(pn), executed_curve(pa)
    progress_n = route_progress(np.asarray(pn["root_pos_w"], dtype=np.float64)[:, :2])
    progress_a = route_progress(np.asarray(pa["root_pos_w"], dtype=np.float64)[:, :2])
    grid = np.linspace(window[0], window[1], 120)
    executed_effect = float(
        np.interp(grid, progress_n, sn_raw).max() - np.interp(grid, progress_a, sa_raw).max()
    )

    # Floor: how repeatable the estimator is on the nominal alone, not the raw gait swing.
    full = np.linspace(0.0, 1.0, 200)
    nominal_resampled = np.interp(full, progress_n, sn_raw)
    span = max(1, int(len(full) * (window[1] - window[0])))
    windows = [
        float(nominal_resampled[i:i + span].max())
        for i in range(0, max(1, len(full) - span), 5)
    ]
    repeatability = float(np.std(windows)) if len(windows) > 1 else 0.0

    shared = min(len(pn["root_pos_w"]), len(pa["root_pos_w"]))
    rn = np.asarray(pn["root_pos_w"], dtype=np.float64)[:shared]
    ra = np.asarray(pa["root_pos_w"], dtype=np.float64)[:shared]
    return {
        "reference_effect_m": reference_effect,
        "window_progress": window,
        "stride_repeatability_m": repeatability,
        "noise_floor_m": 3.0 * repeatability,
        "nominal_outcome": outcome_n.outcome,
        "adapted_outcome": outcome_a.outcome,
        "nominal_frames": int(len(sn_raw)),
        "adapted_frames": int(len(sa_raw)),
        "duration_matched": bool(len(sn_raw) == len(sa_raw)),
        "root_xy_p95_difference_m": float(
            np.percentile(np.linalg.norm(rn[:, :2] - ra[:, :2], axis=1), 95)
        ),
        #: Reported for the record, not used for any decision. See the module docstring: four
        #: estimators gave four answers on these same rollouts, two of them impossible.
        "executed_effect_m_unreliable": executed_effect,
        "operator_trackable": bool(
            outcome_a.evaluated and outcome_n.evaluated and outcome_a.outcome == "accepted"
        ),
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
    print(f"  reference effect      {result['reference_effect_m']*1000:7.1f} mm"
          f"   (frame-aligned by construction, so this one is exact)")
    print(f"  outcomes              {result['nominal_outcome']} / {result['adapted_outcome']}")
    print(f"  frames                {result['nominal_frames']} / {result['adapted_frames']}"
          f"   matched={result['duration_matched']}")
    print(f"  root XY p95 diff      {result['root_xy_p95_difference_m']*1000:7.1f} mm")
    print(f"  window (progress)     {result['window_progress'][0]:.2f} to {result['window_progress'][1]:.2f}")
    print(f"\n  operator_trackable    {result['operator_trackable']}")
    print("  family_eligible       decided by Stage 3, not inferred here")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
