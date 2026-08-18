#!/usr/bin/env python3
"""Build a pre-outcome, self-hashed LACE atlas analysis protocol on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from gear_sonic.research.lace.analysis_protocol import build_analysis_protocol
from gear_sonic.research.lace.instrument_runtime import _strict_json_loads, write_new_json


def _load_object(path: Path, name: str) -> dict[str, Any]:
    candidate = path.expanduser().absolute()
    if candidate.is_symlink():
        raise ValueError(f"{name} may not be a symlink")
    resolved = candidate.resolve()
    if resolved != candidate or not resolved.is_file():
        raise ValueError(f"{name} must be an existing canonical regular file")
    payload = _strict_json_loads(resolved.read_text(encoding="utf-8"), name)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return dict(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--reference-length-inventory", type=Path, required=True)
    parser.add_argument("--probe-thresholds", type=Path, required=True)
    parser.add_argument("--score-window-config", type=Path, required=True)
    parser.add_argument("--sensor-semantics", type=Path, required=True)
    parser.add_argument("--termination-predicates", type=Path, required=True)
    parser.add_argument("--domain-randomization-config", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--source-bundle-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.expanduser().absolute()
    if output.is_symlink() or output.exists():
        raise FileExistsError(f"refusing to overwrite analysis protocol: {output}")
    protocol = build_analysis_protocol(
        schedule_manifest=_load_object(args.schedule, "schedule"),
        split_manifest=_load_object(args.split, "split"),
        reference_length_inventory=_load_object(
            args.reference_length_inventory,
            "reference-length inventory",
        ),
        probe_thresholds=_load_object(args.probe_thresholds, "probe thresholds"),
        score_window_config=_load_object(args.score_window_config, "score-window config"),
        sensor_semantics=_load_object(args.sensor_semantics, "sensor semantics"),
        termination_predicates=_load_object(
            args.termination_predicates,
            "termination predicates",
        ),
        domain_randomization_config=_load_object(
            args.domain_randomization_config,
            "domain-randomization config",
        ),
        git_commit=args.git_commit,
        source_bundle_sha256=args.source_bundle_sha256,
    )
    write_new_json(output, protocol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
