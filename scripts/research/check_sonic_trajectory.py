#!/usr/bin/env python3
"""Validate a raw SONIC physics trajectory before dataset export."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.trajectory_validation import (  # noqa: E402
    validate_sonic_trajectory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument(
        "--allow-multiple-motions",
        action="store_true",
        help="Do not require a single motion ID (episode export still needs splitting)",
    )
    args = parser.parse_args()

    try:
        with args.trajectory.open("rb") as handle:
            payload = pickle.load(handle)  # noqa: S301 - local recorder artifact
    except (OSError, pickle.UnpicklingError, EOFError) as exc:
        print(f"ERROR: could not load trajectory: {exc}", file=sys.stderr)
        return 2

    report = validate_sonic_trajectory(
        payload, require_single_motion=not args.allow_multiple_motions
    )
    result = {
        "trajectory": str(args.trajectory.resolve()),
        **report.to_dict(),
    }
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    for note in report.notes:
        print(f"NOTE: {note}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print("PASS" if report.ok else "FAIL")
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
