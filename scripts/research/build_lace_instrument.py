#!/usr/bin/env python3
"""Build a source-bound LACE scientific instrument without launching a simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.research.lace.instrument import (  # noqa: E402
    INSTRUMENT_DIGEST_FIELD,
    build_scientific_instrument,
    episode_instrument_sha256,
)


def _load_json_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-sha256", required=True)
    parser.add_argument("--probe-thresholds", type=Path, required=True)
    parser.add_argument("--recorder-config", type=Path, required=True)
    parser.add_argument("--resolved-hydra-config", type=Path, required=True)
    parser.add_argument("--environment-fingerprint", type=Path, required=True)
    parser.add_argument("--sensor-semantics", type=Path, required=True)
    parser.add_argument("--termination-predicates", type=Path, required=True)
    parser.add_argument("--score-window-config", type=Path, required=True)
    parser.add_argument("--domain-randomization-config", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--source-path",
        "--source",
        dest="source_paths",
        type=Path,
        action="append",
        required=True,
        help="Repository source file to byte-bind; repeat for every compute source",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    instrument = build_scientific_instrument(
        schedule_sha256=args.schedule_sha256,
        probe_thresholds=_load_json_mapping(args.probe_thresholds),
        recorder_config=_load_json_mapping(args.recorder_config),
        resolved_hydra_config=_load_json_mapping(args.resolved_hydra_config),
        environment_fingerprint=_load_json_mapping(args.environment_fingerprint),
        sensor_semantics=_load_json_mapping(args.sensor_semantics),
        termination_predicates=_load_json_mapping(args.termination_predicates),
        score_window_config=_load_json_mapping(args.score_window_config),
        domain_randomization_config=_load_json_mapping(args.domain_randomization_config),
        git_commit=args.git_commit,
        repo_root=args.repo_root,
        source_paths=args.source_paths,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(instrument, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                INSTRUMENT_DIGEST_FIELD: instrument[INSTRUMENT_DIGEST_FIELD],
                "episode_instrument_sha256": episode_instrument_sha256(instrument),
                "source_file_count": len(instrument["source_bundle"]["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
