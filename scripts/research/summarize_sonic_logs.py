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
_ERROR_PATTERN = re.compile(
    r"Traceback|Error executing job|RuntimeError|Exception|ModuleNotFoundError"
)


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

# Adaptive-sampler telemetry printed per iteration as Env/adp_samp/<key>. Only the
# final value lands here; full series live in summarize_sampler_telemetry.py.
_ADP_SAMP_KEYS = (
    "num_episodes_min",
    "num_episodes_max",
    "num_episodes_mean",
    "num_failures_min",
    "num_failures_max",
    "num_failures_mean",
    "failure_rate_min",
    "failure_rate_max",
    "failure_rate_mean",
    "prob_max",
    "prob_min",
    "prob_mean",
    "prob_max_over_uniform",
    "effective_num_bins",
    "num_concentrated_bins",
    "episodes_max_over_mean",
)
# Subset used by the SIM-M4 classification rule; single source of truth for the
# telemetry columns in compare_sonic_manifests and aggregate_sonic_comparisons.
ADP_SAMP_CLASSIFICATION_KEYS = (
    "prob_max_over_uniform",
    "num_concentrated_bins",
    "effective_num_bins",
    "episodes_max_over_mean",
    "failure_rate_mean",
    "failure_rate_max",
)
# %.4f can print nan/inf; match them so a non-finite FINAL value is reported as
# non-finite instead of silently falling back to an earlier finite iteration.
_ADP_SAMP_VALUE = rf"(?:{_NUMBER}|nan|-?inf)"
_FIELD_PATTERNS.update(
    {f"adp_samp_{key}": rf"Env/adp_samp/{key}:\s*({_ADP_SAMP_VALUE})" for key in _ADP_SAMP_KEYS}
)

_INT_FIELDS = {"learning_iteration", "total_episodes", "total_timesteps"}


def _last_number(text: str, pattern: str, *, as_int: bool = False) -> int | float | None:
    matches = re.findall(pattern, text)
    if not matches:
        return None
    value = float(matches[-1])
    # A non-finite FINAL value is reported as absent rather than falling back to
    # an earlier finite iteration (which would misreport stale telemetry) or
    # emitting NaN into JSON output.
    if value != value or value in (float("inf"), float("-inf")):
        return None
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
    return {
        key: float(value) for key, value in re.findall(rf"([A-Za-z0-9_]+):\s*({_NUMBER})", line)
    }


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
    success_rate = _last_number(text, rf"Success Rate:\s*({_NUMBER})")
    if success_rate is None:
        success_rate = _last_number(text, rf"Succ rate:\s*({_NUMBER})")
    progress_rate = _last_number(text, rf"Progress Rate:\s*({_NUMBER})")
    if terminated is not None:
        summary["terminated_final"] = terminated
    if success_rate is not None:
        summary["success_rate_final"] = success_rate
    if progress_rate is not None:
        summary["progress_rate_final"] = progress_rate
    return summary


# Retention metric (research_plan_zpd_teacher.md §3.4.3, decision D8): mean
# per-motion metrics over the easiest decile of a FROZEN difficulty ranking
# (produced at SIM-D1 time, easiest first). The direct analogue of the
# curriculum-MaxRL H6 forgetting probe: frontier sampling must not degrade
# mastered motions. Preregistered non-inferiority bound lives in the gates,
# not here — this is plumbing only.
_EASY_DECILE_FRACTION = 0.1
_EASY_DECILE_METRIC_KEYS = ("mpjpe_g", "mpjpe_l", "mpjpe_pa", "pa_mpjpe")


def load_difficulty_ranking(path: Path) -> list[str]:
    """Load a frozen difficulty ranking: motion keys, easiest first.

    Accepts either a bare JSON list or an object with a ``ranking`` field
    (the SIM-D1 artifact shape, which also carries dataset provenance).
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ranking = data.get("ranking") if isinstance(data, dict) else data
    if not isinstance(ranking, list) or not all(isinstance(k, str) for k in ranking):
        raise ValueError(f"difficulty ranking in {path} must be a list of motion keys")
    if len(ranking) != len(set(ranking)):
        raise ValueError(f"difficulty ranking in {path} contains duplicate motion keys")
    return ranking


def compute_easy_decile_metrics(
    metrics_eval: dict[str, Any],
    ranking: list[str],
    *,
    decile_fraction: float = _EASY_DECILE_FRACTION,
) -> dict[str, Any]:
    """Easy-decile retention metrics from a per-motion ``metrics_eval.json`` dict.

    ``metrics_eval`` is the eval callback's saved JSON (``save_metrics_eval``):
    per-motion arrays live under ``eval/all_metrics_dict`` keyed alongside
    ``motion_keys``. Only motions present in BOTH the ranking's easiest decile
    and the eval are aggregated; the record reports the counts so a decile
    poorly covered by the eval is visible rather than silently thin.
    """
    per_motion = metrics_eval.get("eval/all_metrics_dict")
    if not isinstance(per_motion, dict):
        raise ValueError("metrics_eval has no 'eval/all_metrics_dict' per-motion table")
    motion_keys = per_motion.get("motion_keys")
    if not isinstance(motion_keys, list) or not motion_keys:
        raise ValueError("per-motion table has no 'motion_keys'")

    n_easy = max(1, int(len(ranking) * decile_fraction))
    easy_set = set(ranking[:n_easy])
    indices = [i for i, key in enumerate(motion_keys) if key in easy_set]

    record: dict[str, Any] = {
        "decile_fraction": decile_fraction,
        "ranking_size": len(ranking),
        "decile_size": n_easy,
        "evaluated_in_decile": len(indices),
        "motion_keys": sorted(motion_keys[i] for i in indices),
    }
    if not indices:
        record["ok"] = False
        return record

    for key in _EASY_DECILE_METRIC_KEYS:
        values = per_motion.get(key)
        if isinstance(values, list) and len(values) == len(motion_keys):
            picked = [float(values[i]) for i in indices]
            record[key] = sum(picked) / len(picked)
    if "terminated" in per_motion and len(per_motion["terminated"]) == len(motion_keys):
        terminated = [float(per_motion["terminated"][i]) for i in indices]
        record["success_rate"] = 1.0 - sum(terminated) / len(terminated)
    record["ok"] = any(key in record for key in _EASY_DECILE_METRIC_KEYS)
    return record


def summarize_logs(
    train_log: Path | None = None,
    eval_log: Path | None = None,
    *,
    metrics_eval_json: Path | None = None,
    difficulty_ranking_json: Path | None = None,
) -> dict[str, Any]:
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
    if metrics_eval_json is not None and difficulty_ranking_json is not None:
        with metrics_eval_json.open("r", encoding="utf-8") as f:
            metrics_eval = json.load(f)
        ranking = load_difficulty_ranking(difficulty_ranking_json)
        easy_decile = compute_easy_decile_metrics(metrics_eval, ranking)
        easy_decile["metrics_eval_path"] = str(metrics_eval_json)
        easy_decile["difficulty_ranking_path"] = str(difficulty_ranking_json)
        summary.setdefault("eval", {"kind": "sonic_eval_log"})["easy_decile"] = easy_decile
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
    parser.add_argument(
        "--metrics-eval-json",
        type=Path,
        help="Eval callback's metrics_eval.json (per-motion table) for the easy-decile "
        "retention metric; requires --difficulty-ranking-json.",
    )
    parser.add_argument(
        "--difficulty-ranking-json",
        type=Path,
        help="Frozen SIM-D1 difficulty ranking (motion keys, easiest first).",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    if args.train_log is None and args.eval_log is None:
        parser.error("at least one of --train-log or --eval-log is required")
    if (args.metrics_eval_json is None) != (args.difficulty_ranking_json is None):
        parser.error("--metrics-eval-json and --difficulty-ranking-json must be given together")

    summary = summarize_logs(
        args.train_log,
        args.eval_log,
        metrics_eval_json=args.metrics_eval_json,
        difficulty_ranking_json=args.difficulty_ranking_json,
    )
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
