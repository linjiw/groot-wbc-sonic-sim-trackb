#!/usr/bin/env python3
"""Build a frozen, reconstruction-verified LACE RQ1 intervention plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.intervention_plan import (  # noqa: E402
    build_intervention_plan,
    validate_intervention_plan,
)


def _load_object(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} {path} must contain a JSON object")
    return payload


def _write_canonical(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w" if force else "x", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            handle.write("\n")
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing intervention plan {path}; pass --force explicitly"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--panels", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    split = _load_object(args.split, "split manifest")
    panels = _load_object(args.panels, "panel manifest")
    protocol = _load_object(args.protocol, "intervention protocol")
    plan = build_intervention_plan(split, panels, protocol)
    validate_intervention_plan(
        plan,
        split_manifest=split,
        panel_manifest=panels,
        protocol=protocol,
    )
    _write_canonical(args.output, plan, force=args.force)
    print(
        json.dumps(
            {
                "file_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "intervention_plan_sha256": plan["intervention_plan_sha256"],
                "motion_count": plan["motion_count"],
                "panel_count": plan["panel_count"],
                "target_kl_nats": plan["protocol"]["target_kl_nats"],
                "added_exposure_range": plan["dose_balance"]["added_panel_exposure_range"],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
