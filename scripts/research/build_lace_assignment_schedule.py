#!/usr/bin/env python3
"""Build a frozen assignment-only LACE probe schedule from a self-hashed spec."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.assignment_schedule import (  # noqa: E402
    build_assignment_probe_schedule,
    deep_validate_assignment_artifacts,
)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {name} JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} at {path} must contain a JSON object")
    return payload


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing schedule {path}; pass --force explicitly"
        )
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not force:
            raise FileExistsError(
                f"refusing to overwrite existing schedule {path}; pass --force explicitly"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen LACE split JSON")
    parser.add_argument(
        "--reference-length-inventory",
        type=Path,
        required=True,
        help="Full-partition assignment inventory",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Self-hashed schedule spec")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing output after successful deep validation",
    )
    args = parser.parse_args(argv)

    split_manifest = _load_json_object(args.split, "split")
    inventory = _load_json_object(args.reference_length_inventory, "assignment inventory")
    spec = _load_json_object(args.spec, "assignment schedule spec")
    schedule = build_assignment_probe_schedule(
        split_manifest,
        reference_length_inventory=inventory,
        spec=spec,
    )
    deep_validate_assignment_artifacts(
        split_manifest=split_manifest,
        reference_length_inventory=inventory,
        spec=spec,
        schedule=schedule,
    )
    _write_atomic(args.output, schedule, force=args.force)
    output = args.output.expanduser().resolve()
    print(
        json.dumps(
            {
                "assignment_only": True,
                "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "fit_normalizer": False,
                "motion_count": schedule["motion_count"],
                "output": str(output),
                "partition": schedule["partition"],
                "rollout_count": schedule["rollout_count"],
                "schedule_sha256": schedule["schedule_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
