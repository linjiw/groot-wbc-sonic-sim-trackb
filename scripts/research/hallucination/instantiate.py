#!/usr/bin/env python3
"""Instantiate a ConstraintSpec as deterministic easy/hard LFH scenes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import ConstraintSpec  # noqa: E402
from gear_sonic.dataset_generation.hallucination.instantiate import instantiate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--archetype", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = instantiate(ConstraintSpec.load(args.spec), args.archetype, args.seed, args.out)
    print(
        f"{result.spec_id}/{result.archetype_id}: "
        f"easy={result.easy.measurement.coordinate_m:.4f} m "
        f"hard={result.hard.measurement.coordinate_m:.4f} m -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
