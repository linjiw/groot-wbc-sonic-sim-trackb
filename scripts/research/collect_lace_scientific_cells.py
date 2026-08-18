#!/usr/bin/env python3
"""Collect receipt-committed LACE cells into one scientific rollout artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.instrument_collection import (  # noqa: E402
    build_scientific_rollout_manifest,
    collect_scientific_rollouts,
)
from gear_sonic.research.lace.instrument_runtime import (  # noqa: E402
    _strict_json_loads,
    write_new_json,
)


def _load_object(path: Path, name: str) -> dict[str, Any]:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise ValueError(f"{name} path may not be a symlink: {absolute}")
    resolved = absolute.resolve()
    if absolute != resolved:
        raise ValueError(f"{name} path must be canonical and traverse no symlinks: {absolute}")
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {resolved}")
    raw = resolved.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"{name} must not be blank")
    value = _strict_json_loads(raw, name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_receipt_list(path: Path) -> list[Path]:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise ValueError(f"receipt-list path may not be a symlink: {absolute}")
    resolved = absolute.resolve()
    if absolute != resolved or not resolved.is_file():
        raise ValueError(f"receipt-list must be a canonical regular file: {absolute}")
    raw = resolved.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError("receipt-list must not be blank")
    value = _strict_json_loads(raw, "receipt-list")
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError("receipt-list must be a non-empty JSON array of explicit paths")
    return [Path(item) for item in value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    receipt_group = parser.add_mutually_exclusive_group(required=True)
    receipt_group.add_argument("--receipt", type=Path, action="append")
    receipt_group.add_argument(
        "--receipt-list",
        type=Path,
        help="JSON array of explicit receipt paths; no glob expansion is performed",
    )
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference-length-inventory", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--analysis-protocol-lock", type=Path, required=True)
    parser.add_argument("--expected-analysis-protocol-lock-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.suffix != ".json":
        raise ValueError("--output must use the .json suffix")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rollout collection: {output}")
    schedule = _load_object(args.schedule, "schedule")
    split = _load_object(args.split, "split")
    inventory = _load_object(
        args.reference_length_inventory,
        "reference-length inventory",
    )
    protocol_path = args.analysis_protocol.expanduser().absolute()
    protocol = _load_object(protocol_path, "analysis protocol")
    collection = collect_scientific_rollouts(
        receipt_paths=(
            args.receipt if args.receipt is not None else _load_receipt_list(args.receipt_list)
        ),
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        analysis_protocol=protocol,
        analysis_protocol_path=protocol_path,
        analysis_protocol_lock_path=args.analysis_protocol_lock.expanduser().absolute(),
        expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
        repo_root=args.repo_root,
    )
    manifest = build_scientific_rollout_manifest(
        collection,
        schedule_manifest=schedule,
        split_manifest=split,
        reference_length_inventory=inventory,
        analysis_protocol=protocol,
        analysis_protocol_path=protocol_path,
        analysis_protocol_lock_path=args.analysis_protocol_lock.expanduser().absolute(),
        expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
        repo_root=args.repo_root,
    )
    write_new_json(output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
