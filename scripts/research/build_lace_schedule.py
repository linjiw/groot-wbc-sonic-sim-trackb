#!/usr/bin/env python3
"""Build a frozen, CPU-only LACE atlas rollout schedule from a JSON spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.reference_lengths import (  # noqa: E402
    REFERENCE_LENGTH_DIGEST_FIELD,
    validate_reference_length_inventory,
)
from gear_sonic.research.lace.schedule import build_rollout_schedule  # noqa: E402

_REQUIRED_SPEC_FIELDS = {
    "schema_version",
    "reference_length_inventory_sha256",
    "probe_policies",
    "domain_randomization_seeds",
    "phase_targets",
    "repeats",
}
_OPTIONAL_SPEC_FIELDS = {
    "rollout_id_prefix",
}


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} at {path} must contain a JSON object")
    return payload


def _validate_spec_fields(spec: Mapping[str, Any]) -> None:
    missing = sorted(_REQUIRED_SPEC_FIELDS - set(spec))
    if missing:
        raise ValueError(f"schedule spec is missing required fields: {missing}")
    unknown = sorted(set(spec) - _REQUIRED_SPEC_FIELDS - _OPTIONAL_SPEC_FIELDS)
    if unknown:
        raise ValueError(f"schedule spec contains unknown fields: {unknown}")
    if spec.get("schema_version") != 2:
        raise ValueError("schedule spec schema_version must be 2")


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
            f"refusing to overwrite existing schedule {path}; pass --force explicitly"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, required=True, help="Frozen LACE split JSON")
    parser.add_argument(
        "--reference-length-inventory",
        type=Path,
        required=True,
        help="Validated frozen inventory from which every reference step is derived",
    )
    parser.add_argument("--spec", type=Path, required=True, help="Rollout schedule spec JSON")
    parser.add_argument("--output", type=Path, required=True, help="New canonical schedule JSON")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Explicitly allow replacement of an existing output file",
    )
    args = parser.parse_args(argv)

    split_manifest = _load_json_object(args.split, "split")
    reference_length_inventory = _load_json_object(
        args.reference_length_inventory,
        "reference-length inventory",
    )
    spec = _load_json_object(args.spec, "schedule spec")
    _validate_spec_fields(spec)
    validate_reference_length_inventory(
        reference_length_inventory,
        split_manifest=split_manifest,
        verify_digest=True,
    )
    if (
        spec["reference_length_inventory_sha256"]
        != reference_length_inventory[REFERENCE_LENGTH_DIGEST_FIELD]
    ):
        raise ValueError(
            "schedule spec reference_length_inventory_sha256 does not match the supplied inventory"
        )
    manifest = build_rollout_schedule(
        split_manifest,
        reference_length_inventory=reference_length_inventory,
        probe_policies=spec["probe_policies"],
        domain_randomization_seeds=spec["domain_randomization_seeds"],
        phase_targets=spec["phase_targets"],
        repeats=spec["repeats"],
        rollout_id_prefix=spec.get("rollout_id_prefix", "lace-rollout"),
    )
    _write_canonical_json(args.output, manifest, force=args.force)
    print(
        json.dumps(
            {
                "artifact_mode": manifest["artifact_mode"],
                "motion_count": manifest["motion_count"],
                "output": str(args.output),
                "rollout_count": manifest["rollout_count"],
                "schedule_sha256": manifest["schedule_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
