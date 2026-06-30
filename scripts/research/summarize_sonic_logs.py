#!/usr/bin/env python3
"""Summarize SONIC IsaacLab training/eval logs into paper-friendly artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_ERROR_PATTERN = re.compile(r"Traceback|Error executing job|RuntimeError|Exception|ModuleNotFoundError")


_FIELD_PATTERNS: dict[str, str] = {
    "learning_iteration": rf"Learning iteration\s+({_NUMBER})",
    "mean_rewards": rf"Mean rewards:\s*({_NUMBER})",
    "mean_length": rf"Mean length:\s*({_NUMBER})",
    "error_anchor_pos": rf"error_anchor_pos:\s*({_NUMBER})",
    "error_body_pos": rf"error_body_pos:\s*({_NUMBER})",
    "total_episodes": rf"Total episodes:\s*({_NUMBER})",
    "total_timesteps": rf"Total timesteps:\s*({_NUMBER})",
    "iteration_time_s": rf"Iteration time:\s*({_NUMBER})s",
    "total_time_s": rf"Total time:\s*({_NUMBER})s",
}

_INT_FIELDS = {"learning_iteration", "total_episodes", "total_timesteps"}


def _last_number(text: str, pattern: str, *, as_int: bool = False) -> int | float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = float(matches[-1])
    return int(value) if as_int else value


def _error_count(text: str) -> int:
    return len(_ERROR_PATTERN.findall(text))


def parse_training_log(text: str) -> dict[str, Any]:
    """Parse the final reported metrics from a SONIC training log."""
    summary: dict[str, Any] = {
        "kind": "sonic_training_log",
        "ok": _error_count(text) == 0,
        "traceback_count": _error_count(text),
    }
    for key, pattern in _FIELD_PATTERNS.items():
        value = _last_number(text, pattern, as_int=key in _INT_FIELDS)
        if value is not None:
            summary[key] = value
    return summary


def _parse_metric_line(text: str, prefix: str) -> dict[str, float]:
    matches = re.findall(rf"(?:^|\n){re.escape(prefix)}:\s*([^\n\r]+)", text)
    if not matches:
        return {}
    line = matches[-1]
    return {key: float(value) for key, value in re.findall(rf"([A-Za-z0-9_]+):\s*({_NUMBER})", line)}


def parse_eval_log(text: str) -> dict[str, Any]:
    """Parse final MPJPE-style metrics from a SONIC eval log."""
    summary: dict[str, Any] = {
        "kind": "sonic_eval_log",
        "ok": _error_count(text) == 0 and bool(_parse_metric_line(text, "All")),
        "traceback_count": _error_count(text),
        "all": _parse_metric_line(text, "All"),
        "succ": _parse_metric_line(text, "Succ"),
    }
    terminated = _last_number(text, rf"Terminated:\s*({_NUMBER})", as_int=True)
    success_rate = _last_number(text, rf"Succ rate:\s*({_NUMBER})")
    if terminated is not None:
        summary["terminated_final"] = terminated
    if success_rate is not None:
        summary["success_rate_final"] = success_rate
    return summary


def summarize_logs(train_log: Path | None = None, eval_log: Path | None = None) -> dict[str, Any]:
    """Summarize selected logs into a JSON-serializable record."""
    summary: dict[str, Any] = {"schema_version": 1}
    if train_log is not None:
        train_text = train_log.read_text(encoding="utf-8", errors="replace")
        train_summary = parse_training_log(train_text)
        train_summary["log_path"] = str(train_log)
        summary["train"] = train_summary
    if eval_log is not None:
        eval_text = eval_log.read_text(encoding="utf-8", errors="replace")
        eval_summary = parse_eval_log(eval_text)
        eval_summary["log_path"] = str(eval_log)
        summary["eval"] = eval_summary
    return summary


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return "`" + json.dumps(value, sort_keys=True) + "`"
    return str(value)


def _write_section(f, title: str, data: dict[str, Any]) -> None:
    f.write(f"## {title}\n\n")
    f.write("| Metric | Value |\n")
    f.write("|---|---:|\n")
    for key, value in data.items():
        if isinstance(value, dict):
            continue
        f.write(f"| `{key}` | {_format_value(value)} |\n")
    for key, value in data.items():
        if isinstance(value, dict):
            f.write(f"\n### {title} `{key}` metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|---|---:|\n")
            for subkey, subvalue in value.items():
                f.write(f"| `{subkey}` | {_format_value(subvalue)} |\n")
    f.write("\n")


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact Markdown report for a SONIC log summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SONIC Log Summary\n\n")
        if "train" in summary:
            _write_section(f, "Training", summary["train"])
        if "eval" in summary:
            _write_section(f, "Evaluation", summary["eval"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--eval-log", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    if args.train_log is None and args.eval_log is None:
        parser.error("at least one of --train-log or --eval-log is required")

    summary = summarize_logs(args.train_log, args.eval_log)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    if args.output_md:
        write_markdown(args.output_md, summary)
    print(f"wrote SONIC log summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
