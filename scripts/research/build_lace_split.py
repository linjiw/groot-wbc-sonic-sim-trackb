#!/usr/bin/env python3
"""Build and validate a frozen LACE source-disjoint split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.schema import validate_split_manifest  # noqa: E402
from gear_sonic.research.lace.split import build_source_disjoint_split  # noqa: E402


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _attach_materialized_paths(
    motions: list[dict[str, Any]], dataset_root: Path, *, require_files: bool
) -> None:
    for motion in motions:
        motion_key = motion["motion_key"]
        robot_path = dataset_root / "robot_filtered" / f"{motion_key}.pkl"
        smpl_path = dataset_root / "smpl_filtered" / f"{motion_key}.pkl"
        if require_files:
            missing = [str(path) for path in (robot_path, smpl_path) if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"materialized files missing for {motion_key}: {missing}")
        motion["robot_path"] = str(robot_path)
        motion["smpl_path"] = str(smpl_path)
        motion["available_modalities"] = ["g1", "smpl"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True, help="Metadata cohort JSON")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Materialized cohort root")
    parser.add_argument(
        "--materialized-manifest",
        type=Path,
        help="Optional paired dataset manifest whose motion keys must match the cohort",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Build a planning manifest without requiring every materialized PKL",
    )
    args = parser.parse_args()

    cohort = _load_json(args.cohort)
    motions = cohort.get("motions")
    if not isinstance(motions, list) or not motions:
        raise ValueError("cohort.motions must be a non-empty list")
    mutable_motions = [dict(record) for record in motions]
    dataset_root = args.dataset_root.resolve()
    _attach_materialized_paths(
        mutable_motions,
        dataset_root,
        require_files=not args.allow_missing_files,
    )

    dataset_metadata: dict[str, Any] = {
        "cohort_manifest": str(args.cohort.resolve()),
        "cohort_manifest_sha256": _sha256_file(args.cohort),
        "dataset_root": str(dataset_root),
        "motion_count": len(mutable_motions),
    }
    if args.materialized_manifest is not None:
        materialized = _load_json(args.materialized_manifest)
        materialized_keys = materialized.get("output", {}).get("motion_keys")
        cohort_keys = [record["motion_key"] for record in mutable_motions]
        if materialized_keys != cohort_keys:
            raise ValueError("materialized manifest motion keys/order do not match cohort")
        dataset_metadata.update(
            {
                "materialized_manifest": str(args.materialized_manifest.resolve()),
                "materialized_manifest_sha256": _sha256_file(args.materialized_manifest),
                "paired_dataset_sha256": materialized.get("output", {}).get(
                    "paired_dataset_sha256"
                ),
            }
        )

    manifest = build_source_disjoint_split(
        mutable_motions,
        seed=args.seed,
        dataset=dataset_metadata,
    )
    validate_split_manifest(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"output": str(args.output), **manifest["partition_summary"]}, indent=2))
    print(f"split_sha256={manifest['split_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
