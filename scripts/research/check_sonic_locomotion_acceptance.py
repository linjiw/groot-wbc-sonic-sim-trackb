#!/usr/bin/env python3
"""Apply reason-coded pilot acceptance gates to a SONIC trajectory pickle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

from gear_sonic.dataset_generation.trajectory_acceptance import (
    LocomotionAcceptanceThresholds,
    evaluate_locomotion_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--min-reference-displacement-m", type=float, default=0.25)
    parser.add_argument("--endpoint-error-m", type=float, default=0.35)
    parser.add_argument("--path-error-p95-m", type=float, default=0.25)
    parser.add_argument("--min-root-height-m", type=float, default=0.50)
    parser.add_argument("--max-abs-roll-pitch-rad", type=float, default=0.60)
    parser.add_argument("--max-nonfoot-contact-force-n", type=float, default=1.0)
    parser.add_argument("--foot-support-force-n", type=float, default=10.0)
    parser.add_argument("--min-supported-fraction", type=float, default=0.90)
    args = parser.parse_args()

    with args.trajectory.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - trusted local recorder artifact.
    thresholds = LocomotionAcceptanceThresholds(
        min_frames=args.min_frames,
        min_reference_displacement_m=args.min_reference_displacement_m,
        endpoint_error_m=args.endpoint_error_m,
        path_error_p95_m=args.path_error_p95_m,
        min_root_height_m=args.min_root_height_m,
        max_abs_roll_pitch_rad=args.max_abs_roll_pitch_rad,
        max_nonfoot_contact_force_n=args.max_nonfoot_contact_force_n,
        foot_support_force_n=args.foot_support_force_n,
        min_supported_fraction=args.min_supported_fraction,
    )
    report = evaluate_locomotion_trajectory(payload, thresholds)
    result = report.to_dict()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if report.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
