#!/usr/bin/env python3
"""Build representation-blind source panels for the LACE RQ1 intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.panels import build_representation_blind_panels  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", default="D_curriculum")
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    manifest = build_representation_blind_panels(
        _load_json(args.split),
        partition=args.partition,
        panel_count=args.panel_count,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "panel_count": manifest["panel_count"],
                "panel_sha256": manifest["panel_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
