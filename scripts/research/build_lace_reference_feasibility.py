#!/usr/bin/env python3
"""Build a canonical CPU-only LACE reference-feasibility manifest."""

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

from gear_sonic.research.lace.reference_feasibility import (  # noqa: E402
    ReferenceFeasibilityThresholds,
)
from gear_sonic.research.lace.reference_feasibility_manifest import (  # noqa: E402
    build_reference_feasibility_manifest,
    load_live_robot_evidence,
)


def _write_canonical_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if force else "x"
    try:
        with path.open(mode, encoding="utf-8") as handle:
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
            f"refusing to overwrite existing feasibility artifact {path}; pass --force explicitly"
        ) from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen LACE split JSON")
    parser.add_argument(
        "--reference-inventory",
        type=Path,
        required=True,
        help="Frozen reference-length inventory JSON",
    )
    parser.add_argument(
        "--cohort-manifest",
        type=Path,
        required=True,
        help="Pinned official metadata cohort JSON",
    )
    parser.add_argument(
        "--g1-config",
        type=Path,
        default=REPO_ROOT / "gear_sonic/envs/manager_env/robots/g1.py",
        help="Pinned SONIC G1 articulation configuration",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf",
        help="Pinned URDF referenced by the G1 configuration",
    )
    parser.add_argument(
        "--motion-command-config",
        type=Path,
        default=REPO_ROOT / "gear_sonic/config/manager_env/commands/terms/motion.yaml",
        help="Pinned SONIC motion command config that selects the reference MJCF",
    )
    parser.add_argument(
        "--motion-mjcf",
        type=Path,
        default=(REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"),
        help="Pinned MJCF used by SONIC MotionLibRobot forward kinematics",
    )
    parser.add_argument(
        "--live-robot-data",
        type=Path,
        help=(
            "Optional recorder JSON object or strict JSONL containing byte-equivalent "
            "robot_contract_readback values and canonical digests. Omit for an explicitly "
            "unverified CPU artifact."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="New canonical manifest JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow replacement of an existing output file",
    )
    args = parser.parse_args(argv)

    live_robot_data = (
        None if args.live_robot_data is None else load_live_robot_evidence(args.live_robot_data)
    )
    manifest = build_reference_feasibility_manifest(
        split_manifest_path=args.split,
        reference_inventory_path=args.reference_inventory,
        cohort_manifest_path=args.cohort_manifest,
        g1_config_path=args.g1_config,
        urdf_path=args.urdf,
        motion_command_config_path=args.motion_command_config,
        motion_mjcf_path=args.motion_mjcf,
        thresholds=ReferenceFeasibilityThresholds(),
        live_robot_data=live_robot_data,
    )
    _write_canonical_json(args.output, manifest, force=args.force)
    print(
        json.dumps(
            {
                "file_sha256": _sha256_file(args.output),
                "hard_corruption_exclusion_count": manifest["summary"][
                    "hard_corruption_exclusion_count"
                ],
                "manifest_sha256": manifest["manifest_sha256"],
                "motion_count": manifest["motion_count"],
                "output": str(args.output),
                "scientific_use": manifest["scientific_use"],
                "scientific_use_status": manifest["scientific_use_status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
