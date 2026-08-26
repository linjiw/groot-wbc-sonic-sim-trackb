#!/usr/bin/env python3
"""Extract a measured LFH ConstraintSpec from one verified family."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.extract_spec import extract_spec  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=Path, required=True)
    parser.add_argument(
        "--scenes-root",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/scenes/g1_counterfactual",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = extract_spec(args.family, args.scenes_root)
    spec.save(args.out)
    print(
        f"{spec.spec_id}: {spec.axis_type} faces "
        f"{spec.easy_coordinate_m:.4f}/{spec.hard_coordinate_m:.4f} m -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
