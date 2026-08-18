#!/usr/bin/env python3
"""Build a full-partition, assignment-only LACE reference-length inventory."""

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
    build_assignment_reference_length_inventory,
    validate_assignment_reference_length_inventory,
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
            f"refusing to overwrite existing inventory {path}; pass --force explicitly"
        )
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid4().hex}"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() and not force:
            raise FileExistsError(
                f"refusing to overwrite existing inventory {path}; pass --force explicitly"
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen LACE split JSON")
    parser.add_argument(
        "--partition",
        choices=("D_atlas", "D_curriculum", "D_geometry", "D_controller", "D_test"),
        required=True,
    )
    parser.add_argument(
        "--final-open",
        action="store_true",
        help="Required acknowledgement for D_test; forbidden on every other partition",
    )
    parser.add_argument(
        "--motion-root",
        type=Path,
        help="Optional exact <motion_key>.pkl root; defaults to frozen split robot paths",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly replace an existing output after a successful rebuild",
    )
    args = parser.parse_args(argv)

    split_manifest = _load_json_object(args.split, "split")
    inventory = build_assignment_reference_length_inventory(
        split_manifest,
        partition=args.partition,
        final_open=args.final_open,
        motion_root=args.motion_root,
    )
    validate_assignment_reference_length_inventory(
        inventory,
        split_manifest=split_manifest,
        verify_digest=True,
        verify_source_files=True,
        deterministic_rebuild=True,
    )
    _write_atomic(args.output, inventory, force=args.force)
    output = args.output.expanduser().resolve()
    print(
        json.dumps(
            {
                "assignment_only": True,
                "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "fit_normalizer": False,
                "inventory_sha256": inventory["inventory_sha256"],
                "motion_count": inventory["motion_count"],
                "output": str(output),
                "partition": inventory["partition"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
