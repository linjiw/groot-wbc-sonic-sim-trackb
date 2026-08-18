#!/usr/bin/env python3
"""Build a deterministic LACE atlas JSON from probe-rollout and split manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.atlas import build_atlas_manifest  # noqa: E402
from gear_sonic.research.lace.instrument_collection import (  # noqa: E402
    ROLLOUT_COLLECTION_KIND,
    build_scientific_rollout_manifest,
    validate_rollout_collection,
)
from gear_sonic.research.lace.instrument_runtime import (  # noqa: E402
    _strict_json_loads,
    write_new_json,
)


def _load_json(path: Path) -> dict[str, Any]:
    expanded = path.expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    if absolute.is_symlink():
        raise ValueError(f"input path may not be a symlink: {absolute}")
    resolved = absolute.resolve()
    if absolute != resolved or not resolved.is_file():
        raise ValueError(f"input path must be a canonical regular file: {absolute}")
    raw = resolved.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"{resolved} must not be blank")
    payload = _strict_json_loads(raw, str(resolved))
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument(
        "--schedule",
        type=Path,
        help="Frozen rollout schedule; required for scientific rollouts and forbidden for smoke",
    )
    parser.add_argument(
        "--reference-length-inventory",
        type=Path,
        help=(
            "Frozen inventory bound by a schema-v2 schedule; required for scientific "
            "rollouts and forbidden for smoke"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--analysis-protocol",
        type=Path,
        help="Required exact external protocol for scientific rollouts; forbidden for smoke",
    )
    parser.add_argument(
        "--analysis-protocol-lock",
        type=Path,
        help="Required pre-outcome protocol commitment lock for scientific rollouts",
    )
    parser.add_argument("--expected-analysis-protocol-lock-sha256")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)

    rollouts = _load_json(args.rollouts)
    split = _load_json(args.split)
    schedule = _load_json(args.schedule) if args.schedule is not None else None
    inventory = (
        _load_json(args.reference_length_inventory)
        if args.reference_length_inventory is not None
        else None
    )
    protocol_path = (
        args.analysis_protocol.expanduser().absolute()
        if args.analysis_protocol is not None
        else None
    )
    protocol = _load_json(protocol_path) if protocol_path is not None else None
    protocol_lock_path = (
        args.analysis_protocol_lock.expanduser().absolute()
        if args.analysis_protocol_lock is not None
        else None
    )
    scientific = (
        rollouts.get("kind") == ROLLOUT_COLLECTION_KIND
        or rollouts.get("artifact_mode") == "scientific"
    )
    if scientific and (
        protocol is None
        or protocol_lock_path is None
        or args.expected_analysis_protocol_lock_sha256 is None
    ):
        raise ValueError(
            "scientific atlas build requires --analysis-protocol and --analysis-protocol-lock"
        )
    if not scientific and (protocol is not None or protocol_lock_path is not None):
        raise ValueError("contract-smoke atlas build forbids analysis protocol inputs")
    if rollouts.get("kind") == ROLLOUT_COLLECTION_KIND:
        if schedule is None or inventory is None:
            raise ValueError("scientific collection verification requires schedule and inventory")
        validate_rollout_collection(
            rollouts,
            schedule_manifest=schedule,
            split_manifest=split,
            reference_length_inventory=inventory,
            analysis_protocol=protocol,
            analysis_protocol_path=protocol_path,
            analysis_protocol_lock_path=protocol_lock_path,
            expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
            verify_artifacts=True,
            repo_root=args.repo_root,
        )
        rollouts = build_scientific_rollout_manifest(
            rollouts,
            schedule_manifest=schedule,
            split_manifest=split,
            reference_length_inventory=inventory,
            analysis_protocol=protocol,
            analysis_protocol_path=protocol_path,
            analysis_protocol_lock_path=protocol_lock_path,
            expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
            repo_root=args.repo_root,
        )
    elif rollouts.get("artifact_mode") == "scientific":
        collection = rollouts.get("rollout_collection")
        if not isinstance(collection, dict):
            raise ValueError("scientific rollouts require rollout_collection")
        if schedule is None or inventory is None:
            raise ValueError("scientific collection verification requires schedule and inventory")
        validate_rollout_collection(
            collection,
            schedule_manifest=schedule,
            split_manifest=split,
            reference_length_inventory=inventory,
            analysis_protocol=protocol,
            analysis_protocol_path=protocol_path,
            analysis_protocol_lock_path=protocol_lock_path,
            expected_analysis_protocol_lock_sha256=(args.expected_analysis_protocol_lock_sha256),
            verify_artifacts=True,
            repo_root=args.repo_root,
        )
    atlas = build_atlas_manifest(
        rollouts,
        split,
        schedule,
        inventory,
        repo_root=args.repo_root,
    )
    output = args.output.expanduser()
    output = output if output.is_absolute() else Path.cwd() / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite atlas artifact: {output}")
    write_new_json(output, atlas)
    print(
        json.dumps(
            {
                "output": str(output),
                "atlas_sha256": atlas["atlas_sha256"],
                "motion_count": atlas["motion_count"],
                "rollout_count": atlas["rollout_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
