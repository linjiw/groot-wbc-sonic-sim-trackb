#!/usr/bin/env python3
"""Summarise a batch of Kimodo+SONIC rollouts: validation, acceptance, latent parity.

Each rollout directory is expected to contain ``trajectories/*.trajectory.pkl`` and
a ``success_manifest.json`` written by ``run_kimodo_sonic_rollout.sh``.

A recorded rollout usually runs past the end of its reference motion and loops, so
the motion clock resets. This tool slices each capture at the first reset, which is
exactly one motion pass, and evaluates that.

Usage::

    python scripts/research/analyze_kimodo_rollout_batch.py /path/to/rollouts_batch
    python scripts/research/analyze_kimodo_rollout_batch.py DIR --json report.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.latent_parity import (  # noqa: E402
    LatentParityReport,
    check_latent_parity,
)
from gear_sonic.dataset_generation.trajectory_acceptance import (  # noqa: E402
    evaluate_locomotion_trajectory,
)
from gear_sonic.dataset_generation.trajectory_export import (  # noqa: E402
    slice_trajectory_frames,
)
from gear_sonic.dataset_generation.trajectory_validation import (  # noqa: E402
    validate_sonic_trajectory,
)


def first_motion_pass(payload: dict) -> int:
    """Frame count of the first complete motion pass, i.e. up to the first clock reset."""
    times = np.asarray(payload["motion_time_s"])
    drops = np.flatnonzero(np.diff(times) < 0)
    return int(drops[0]) + 1 if drops.size else len(times)


def analyze_one(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local recorder artifact

    end = first_motion_pass(payload)
    clip = slice_trajectory_frames(payload, start_frame=0, end_frame=end)
    validation = validate_sonic_trajectory(clip)
    record: dict = {
        "captured_frames": int(payload["total_frames"]),
        "motion_pass_frames": end,
        "fps": float(payload["fps"]),
        "raw_valid": validation.ok,
        "raw_errors": list(validation.errors),
    }
    if not validation.ok:
        record["accepted"] = False
        record["rejection_reasons"] = ["raw_validation_failed"]
        return record

    acceptance = evaluate_locomotion_trajectory(clip)
    parity = check_latent_parity(np.asarray(clip["action_motion_token"]))
    record.update(
        {
            "accepted": bool(acceptance.accepted),
            "rejection_reasons": list(acceptance.rejection_reasons),
            "gates": {gate.name: gate.to_dict() for gate in acceptance.gates},
            "diagnostics": acceptance.to_dict()["diagnostics"],
            "latent_verdict": parity.verdict,
            "latent_residual_rms": parity.residual_rms,
            "latent_on_lattice_fraction": parity.on_lattice_fraction,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    args = parser.parse_args()

    rollouts = sorted(
        d for d in args.batch_dir.iterdir() if d.is_dir() and (d / "trajectories").is_dir()
    )
    if not rollouts:
        print(f"ERROR: no rollout directories under {args.batch_dir}", file=sys.stderr)
        return 2

    results: dict[str, dict] = {}
    for rollout in rollouts:
        paths = sorted(rollout.glob("trajectories/*.trajectory.pkl"))
        if not paths or not (rollout / "success_manifest.json").is_file():
            # An in-flight rollout is incomplete, not rejected. Counting it as a
            # rejection would overstate the failure rate of a running batch.
            results[rollout.name] = {"incomplete": True}
            continue
        record = analyze_one(paths[0])
        record["has_success_manifest"] = True
        record["incomplete"] = False
        results[rollout.name] = record

    header = f"{'rollout':44s} {'frames':>6s} {'acc':>4s}  {'latent':<26s} rejections"
    print(header)
    print("-" * len(header))
    for name, record in results.items():
        if record.get("incomplete"):
            status = "..."
            detail = "incomplete (still running or interrupted)"
        else:
            status = "YES" if record.get("accepted") else "no"
            detail = ",".join(record.get("rejection_reasons", [])) or "-"
        print(
            f"{name:44s} {record.get('motion_pass_frames', 0):6d} "
            f"{status:>4s}  {record.get('latent_verdict', '-'):<26s} {detail}"
        )

    complete = {n: r for n, r in results.items() if not r.get("incomplete")}
    incomplete = [n for n, r in results.items() if r.get("incomplete")]
    accepted = [n for n, r in complete.items() if r.get("accepted")]
    reasons = Counter(
        reason for r in complete.values() for reason in r.get("rejection_reasons", [])
    )
    verdicts = Counter(r.get("latent_verdict", "-") for r in complete.values())
    total_frames = sum(
        r.get("motion_pass_frames", 0) for n, r in complete.items() if r.get("accepted")
    )

    print()
    print(f"accepted: {len(accepted)}/{len(complete)} complete rollouts, {total_frames} frames")
    if incomplete:
        print(f"incomplete (excluded from the rate): {len(incomplete)} -> {incomplete}")
    print(f"latent verdicts: {dict(verdicts)}")
    if reasons:
        print("rejection reasons:")
        for reason, count in reasons.most_common():
            print(f"    {count:3d}  {reason}")

    non_equivalent = [
        name
        for name, record in results.items()
        if not record.get("incomplete")
        and record.get("latent_verdict") not in (LatentParityReport.ENCODER_EQUIVALENT, None)
    ]
    if non_equivalent:
        print(f"\nWARNING: non encoder-equivalent latents in: {non_equivalent}")

    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(
                {
                    "batch_dir": str(args.batch_dir.resolve()),
                    "accepted_rollouts": accepted,
                    "incomplete_rollouts": incomplete,
                    "complete_rollout_count": len(complete),
                    "accepted_frames": total_frames,
                    "rejection_reason_counts": dict(reasons),
                    "latent_verdict_counts": dict(verdicts),
                    "rollouts": results,
                },
                indent=2,
                sort_keys=True,
                default=float,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
