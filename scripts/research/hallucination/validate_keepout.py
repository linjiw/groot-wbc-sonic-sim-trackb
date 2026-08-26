#!/usr/bin/env python3
"""Run the mandatory LFH CPU preflight for an instantiated scene pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.constraint_spec import ConstraintSpec  # noqa: E402
from gear_sonic.dataset_generation.hallucination.validate_keepout import (  # noqa: E402
    validate_pair,
    write_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--easy", type=Path, required=True)
    parser.add_argument("--hard", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate_pair(ConstraintSpec.load(args.spec), args.easy, args.hard)
    write_report(result, args.out)
    print(f"{'PASS' if result['ok'] else 'REFUSED'} -> {args.out}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
