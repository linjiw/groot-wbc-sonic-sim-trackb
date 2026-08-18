#!/usr/bin/env python3
"""Exclusively freeze a pre-outcome LACE analysis protocol commitment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.analysis_protocol_lock import (  # noqa: E402
    build_analysis_protocol_lock,
)
from gear_sonic.research.lace.instrument_runtime import write_new_json  # noqa: E402


def _checkpoint_config_bindings(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        policy_id, separator, path_value = value.partition("=")
        if not separator or not policy_id or not path_value:
            raise ValueError("--checkpoint-config must use PROBE_POLICY_ID=/absolute/config.yaml")
        if policy_id in result:
            raise ValueError(f"duplicate --checkpoint-config policy id: {policy_id}")
        result[policy_id] = Path(path_value).expanduser().absolute()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument(
        "--analysis-protocol-preflight-receipt",
        type=Path,
        required=True,
        help=(
            "Receipt-last zero-outcome live capture that deterministically produced "
            "the exact analysis protocol bytes."
        ),
    )
    parser.add_argument("--schedule-lock", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-config",
        action="append",
        default=[],
        metavar="PROBE_POLICY_ID=PATH",
        help=(
            "Exact companion config.yaml for each frozen probe policy. Repeat once "
            "per policy; these bytes are committed before outcomes."
        ),
    )
    parser.add_argument(
        "--execution-root",
        type=Path,
        required=True,
        help=(
            "Dedicated existing directory whose deterministic per-cell children are "
            "the only permitted scientific attempts. It must be empty when locked."
        ),
    )
    parser.add_argument(
        "--runtime-storage-root",
        type=Path,
        required=True,
        help="Existing non-overlapping /data root for deterministic per-cell caches.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().absolute()
    execution_root = args.execution_root.expanduser().absolute()
    runtime_storage_root = args.runtime_storage_root.expanduser().absolute()
    if output.is_relative_to(execution_root) or output.is_relative_to(runtime_storage_root):
        raise ValueError(
            "analysis protocol lock output must be outside the dedicated execution/cache roots"
        )
    lock = build_analysis_protocol_lock(
        analysis_protocol_path=args.analysis_protocol.expanduser().absolute(),
        analysis_protocol_preflight_receipt_path=(
            args.analysis_protocol_preflight_receipt.expanduser().absolute()
        ),
        schedule_lock_path=args.schedule_lock.expanduser().absolute(),
        checkpoint_config_paths=_checkpoint_config_bindings(args.checkpoint_config),
        execution_root=execution_root,
        runtime_storage_root=runtime_storage_root,
        repo_root=REPO_ROOT,
    )
    published = write_new_json(output, lock)
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
