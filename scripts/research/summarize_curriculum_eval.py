#!/usr/bin/env python3
"""Summarize CG-WBC stage-wise manifest/eval results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.curriculum_sampler import competence_score, hard_gates_pass  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f).get("curriculum", {})


def rate(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if bool(record.get(key, False))) / len(records)


def mean(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record and record[key] is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_manifest(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    stages = {stage["id"]: stage for stage in config.get("stages", [])}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("stage", "unknown"))].append(record)

    summaries: dict[str, dict[str, Any]] = {}
    for stage_id, stage_records in sorted(grouped.items()):
        metrics: dict[str, Any] = {
            "num_episodes": len(stage_records),
            "eval_episodes": len(stage_records),
            "success_rate": rate(stage_records, "success"),
            "fall_rate": rate(stage_records, "fall"),
            "drop_rate": rate(stage_records, "drop"),
            "schema_valid": all(bool(record.get("schema_valid", False)) for record in stage_records),
            "schema_valid_rate": rate(stage_records, "schema_valid"),
        }
        avg_frames = mean(stage_records, "num_frames")
        if avg_frames is not None:
            metrics["avg_num_frames"] = avg_frames
        avg_difficulty = mean(stage_records, "difficulty")
        if avg_difficulty is not None:
            metrics["avg_difficulty"] = avg_difficulty

        gates = stages.get(stage_id, {}).get("gates", {})
        metrics["competence"] = competence_score(metrics, gates) if gates else None
        metrics["hard_gates_pass"] = hard_gates_pass(metrics, gates) if gates else None
        summaries[stage_id] = metrics

    return {"stage_metrics": summaries}


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Curriculum Eval Summary\n\n")
        f.write("| Stage | Episodes | Success | Fall | Drop | Schema | Competence | Gates |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for stage_id, metrics in sorted(summary["stage_metrics"].items()):
            competence = metrics.get("competence")
            competence_text = "" if competence is None else f"{competence:.3f}"
            gates = metrics.get("hard_gates_pass")
            gates_text = "" if gates is None else str(bool(gates)).lower()
            f.write(
                "| {stage} | {episodes} | {success:.3f} | {fall:.3f} | {drop:.3f} | "
                "{schema:.3f} | {competence} | {gates} |\n".format(
                    stage=stage_id,
                    episodes=metrics["num_episodes"],
                    success=metrics["success_rate"],
                    fall=metrics["fall_rate"],
                    drop=metrics["drop_rate"],
                    schema=metrics["schema_valid_rate"],
                    competence=competence_text,
                    gates=gates_text,
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    summary = summarize_manifest(records, load_config(args.config))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.output_md:
        write_markdown(args.output_md, summary)

    print(f"wrote summary to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
