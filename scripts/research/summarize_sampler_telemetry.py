#!/usr/bin/env python3
"""Summarize adaptive-sampler telemetry (Env/adp_samp/*) from SONIC training logs.

The PPO trainer prints every ``Env/adp_samp/<key>`` scalar once per learning
iteration (``%.4f``). This tool extracts the full per-iteration series for every
emitted key — last-value summaries cannot distinguish an under-active sampler
from a saturating one — plus first/last/min/max and a least-squares slope over
the final iterations.

Parsing is block-based: the log is split on ``Learning iteration N`` headers and
each telemetry value is recorded as an ``[iteration, value]`` pair. This keeps
series aligned even when keys appear late (the ``prob_*`` keys are emitted only
after the first probability recompute) or drop out for an iteration.

Uniform-sampling logs contain no adp_samp keys; they yield
``adaptive_telemetry_present: false`` with an empty key map — never a warning.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
# %.4f renders non-finite tensors as nan/inf; capture them so they cannot
# silently drop an iteration from a series.
_VALUE = rf"(?:{_NUMBER}|nan|-?inf)"
# Iteration headers carry ANSI bold codes: " \033[1m Learning iteration N  \033[0m "
_ITERATION_PATTERN = re.compile(r"Learning iteration\s+(\d+)")
_TELEMETRY_PATTERN = re.compile(rf"Env/adp_samp/([A-Za-z0-9_]+):\s*({_VALUE})")

_SLOPE_WINDOW = 20


def _parse_value(text: str) -> float | None:
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _least_squares_slope(points: list[tuple[int, float]]) -> float | None:
    """Slope of value vs iteration over the given points (None if degenerate)."""
    if len(points) < 2:
        return None
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denominator = sum((x - mean_x) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in points)
    return numerator / denominator


def _key_summary(series: list[list[Any]], non_finite_count: int) -> dict[str, Any]:
    # A truncated/rotated log can carry telemetry lines before the first iteration
    # header; those points keep a null iteration and are excluded from the slope.
    finite = [
        (int(iteration), float(value))
        for iteration, value in series
        if value is not None and iteration is not None
    ]
    finite_values = [float(value) for iteration, value in series if value is not None]
    values = finite_values
    summary: dict[str, Any] = {
        "series": series,
        "num_points": len(series),
        "non_finite_count": non_finite_count,
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "slope_final_iters": _least_squares_slope(finite[-_SLOPE_WINDOW:]),
        "slope_window": _SLOPE_WINDOW,
    }
    return summary


def parse_sampler_telemetry(text: str) -> dict[str, Any]:
    """Parse Env/adp_samp/* series from one training-log text."""
    iterations: list[int] = []
    raw_series: dict[str, list[list[Any]]] = {}
    non_finite_counts: dict[str, int] = {}

    current_iteration: int | None = None
    for line in text.splitlines():
        iteration_match = _ITERATION_PATTERN.search(line)
        if iteration_match:
            current_iteration = int(iteration_match.group(1))
            iterations.append(current_iteration)
            continue
        for key, value_text in _TELEMETRY_PATTERN.findall(line):
            value = _parse_value(value_text)
            if value is None:
                non_finite_counts[key] = non_finite_counts.get(key, 0) + 1
            raw_series.setdefault(key, []).append([current_iteration, value])

    keys = {
        key: _key_summary(series, non_finite_counts.get(key, 0))
        for key, series in sorted(raw_series.items())
    }
    return {
        "kind": "sampler_telemetry_log",
        "adaptive_telemetry_present": bool(keys),
        "iteration_count": len(iterations),
        "iteration_first": iterations[0] if iterations else None,
        "iteration_last": iterations[-1] if iterations else None,
        "telemetry_key_count": len(keys),
        "keys": keys,
    }


def summarize_telemetry_logs(log_paths: list[Path]) -> dict[str, Any]:
    """Summarize one record per log; logs are never concatenated into one series."""
    logs: list[dict[str, Any]] = []
    for path in log_paths:
        record = parse_sampler_telemetry(path.read_text(encoding="utf-8", errors="replace"))
        record["log_path"] = str(path)
        logs.append(record)
    return {
        "schema_version": 1,
        "kind": "sampler_telemetry_summary",
        "log_count": len(logs),
        "logs": logs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-log",
        type=Path,
        action="append",
        required=True,
        help="Training log. Repeat per log.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    summary = summarize_telemetry_logs(args.train_log)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    for record in summary["logs"]:
        status = "adaptive" if record["adaptive_telemetry_present"] else "no adp_samp telemetry"
        print(f"{record['log_path']}: {status}, {record['telemetry_key_count']} keys")
    print(f"wrote sampler telemetry summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
