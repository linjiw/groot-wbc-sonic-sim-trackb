#!/usr/bin/env python3
"""Build a report-only LACE cross-policy transport analysis artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.schema import validate_split_manifest  # noqa: E402
from gear_sonic.research.lace.transport import (  # noqa: E402
    build_cross_policy_transport_report,
    validate_transport_protocol,
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


def _artifact(payload: Mapping[str, Any], name: str) -> tuple[list[Mapping[str, Any]], str]:
    signatures = payload.get("signatures")
    if not isinstance(signatures, list) or not signatures:
        raise ValueError(f"{name}.signatures must be a non-empty list")
    digest = payload.get("atlas_sha256", payload.get("assignment_sha256"))
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name} must expose atlas_sha256 or assignment_sha256")
    return signatures, digest


def _sha_arg(value: str) -> str:
    if len(value) != 64:
        raise argparse.ArgumentTypeError("value must be a 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a hexadecimal SHA-256") from error
    if value != value.lower():
        raise argparse.ArgumentTypeError("SHA-256 must use lowercase hexadecimal")
    return value


def _write(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
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
            f"refusing to overwrite existing transport report {path}; pass --force explicitly"
        ) from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--trainee-artifact", type=Path, required=True)
    parser.add_argument("--trainee-policy-id", required=True)
    parser.add_argument("--parent-normalizer-sha256", type=_sha_arg, required=True)
    parser.add_argument("--measurement-protocol-sha256", type=_sha_arg, required=True)
    parser.add_argument("--condition-grid-sha256", type=_sha_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    protocol = _load_object(args.protocol, "transport protocol")
    split = _load_object(args.split, "split manifest")
    reference_artifact = _load_object(args.reference_artifact, "reference artifact")
    trainee_artifact = _load_object(args.trainee_artifact, "trainee artifact")
    validate_transport_protocol(protocol)
    validate_split_manifest(split, verify_digest=True)
    if protocol["split_selection_sha256"] != split["selection_sha256"]:
        raise ValueError("transport protocol does not bind the supplied split")
    reference_signatures, reference_digest = _artifact(
        reference_artifact,
        "reference artifact",
    )
    trainee_signatures, trainee_digest = _artifact(trainee_artifact, "trainee artifact")
    source_groups = {
        str(record["motion_key"]): str(record["source_group_id"])
        for record in split["motions"]
        if record["partition"] == protocol["partition"]
    }
    report = build_cross_policy_transport_report(
        reference_signatures,
        trainee_signatures,
        trainee_policy_id=args.trainee_policy_id,
        source_group_by_motion=source_groups,
        artifact_bindings={
            "reference_artifact_sha256": reference_digest,
            "trainee_artifact_sha256": trainee_digest,
            "parent_normalizer_sha256": args.parent_normalizer_sha256,
            "measurement_protocol_sha256": args.measurement_protocol_sha256,
            "condition_grid_sha256": args.condition_grid_sha256,
        },
        protocol=protocol,
    )
    _write(args.output, report, force=args.force)
    print(
        json.dumps(
            {
                "coverage_passes": report["coverage"]["passes"],
                "motion_count_common_support": report["num_motion_keys_common_support"],
                "output": str(args.output.resolve()),
                "reference_policy_id": report["reference_policy_id"],
                "trainee_policy_id": report["trainee_policy_id"],
                "transport_report_sha256": report["transport_report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
