#!/usr/bin/env python3
"""Build a CG-WBC curriculum manifest from a SONIC/VLA LeRobot dataset."""

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

from scripts.research.check_sonic_vla_dataset import (  # noqa: E402
    ValidationReport,
    check_info_and_videos,
    check_modality,
    check_parquet,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_stage_ids(config_path: Path) -> set[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    stages = config.get("curriculum", {}).get("stages", [])
    return {stage["id"] for stage in stages}


def dataset_schema_valid(dataset_path: Path, max_rows: int) -> tuple[bool, list[str], list[str]]:
    report = ValidationReport()
    check_modality(dataset_path, report)
    check_info_and_videos(dataset_path, report)
    check_parquet(dataset_path, report, max_rows)
    return not report.errors, report.errors, report.warnings


def load_labels(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    labels: dict[int, dict[str, Any]] = {}
    for record in read_jsonl(path):
        if "episode_index" in record:
            labels[int(record["episode_index"])] = record
            continue
        episode_id = str(record.get("episode_id", ""))
        if episode_id.startswith("episode_"):
            labels[int(episode_id.removeprefix("episode_"))] = record
    return labels


def load_task_prompts(dataset_path: Path) -> dict[int, str]:
    prompts: dict[int, str] = {}
    for record in read_jsonl(dataset_path / "meta" / "tasks.jsonl"):
        task_index = record.get("task_index", record.get("index"))
        task = record.get("task", record.get("task_description", record.get("text")))
        if task_index is not None and task is not None:
            prompts[int(task_index)] = str(task)
    return prompts


def episodes_from_metadata(dataset_path: Path) -> list[dict[str, Any]]:
    records = read_jsonl(dataset_path / "meta" / "episodes.jsonl")
    episodes: list[dict[str, Any]] = []
    for record in records:
        index = record.get("episode_index", record.get("index"))
        if index is None:
            continue
        tasks = record.get("tasks", record.get("task", []))
        if isinstance(tasks, str):
            tasks = [tasks]
        episodes.append(
            {
                "episode_index": int(index),
                "num_frames": int(record.get("length", record.get("num_frames", 0))),
                "tasks": tasks,
            }
        )
    return episodes


def episodes_from_parquet(dataset_path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "meta/episodes.jsonl is missing or empty, and pandas is required to derive episodes "
            "from parquet files"
        ) from exc

    groups: dict[int, dict[str, Any]] = defaultdict(lambda: {"num_frames": 0, "task_indices": set()})
    for parquet_path in sorted((dataset_path / "data").glob("*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["episode_index", "task_index"])
        for episode_index, rows in df.groupby("episode_index"):
            group = groups[int(episode_index)]
            group["num_frames"] += int(len(rows))
            group["task_indices"].update(int(v) for v in rows["task_index"].unique())

    episodes = []
    for episode_index, values in sorted(groups.items()):
        episodes.append(
            {
                "episode_index": episode_index,
                "num_frames": values["num_frames"],
                "task_indices": sorted(values["task_indices"]),
                "tasks": [],
            }
        )
    return episodes


def load_episodes(dataset_path: Path) -> list[dict[str, Any]]:
    episodes = episodes_from_metadata(dataset_path)
    if episodes:
        return episodes
    return episodes_from_parquet(dataset_path)


def load_discarded(dataset_path: Path) -> set[int]:
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        return set()
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    return {int(idx) for idx in info.get("discarded_episode_indices", [])}


def infer_prompt(episode: dict[str, Any], prompts: dict[int, str], fallback: str) -> str:
    tasks = episode.get("tasks") or []
    if tasks:
        return str(tasks[0])
    task_indices = episode.get("task_indices") or []
    for task_index in task_indices:
        if int(task_index) in prompts:
            return prompts[int(task_index)]
    return fallback


def build_manifest(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_path = args.dataset_path.resolve()
    stage_ids = load_stage_ids(args.config)
    if args.default_stage not in stage_ids:
        raise ValueError(f"default stage {args.default_stage!r} is not present in {args.config}")

    schema_valid, validation_errors, validation_warnings = dataset_schema_valid(
        dataset_path,
        args.max_validation_rows,
    )
    if validation_errors and not args.allow_invalid_schema:
        joined = "\n".join(f"- {error}" for error in validation_errors)
        raise RuntimeError(f"dataset validation failed:\n{joined}")

    labels = load_labels(args.episode_labels)
    prompts = load_task_prompts(dataset_path)
    episodes = load_episodes(dataset_path)
    discarded = load_discarded(dataset_path)

    manifest: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        label = labels.get(episode_index, {})
        stage = label.get("stage", args.default_stage)
        if stage not in stage_ids:
            raise ValueError(f"episode {episode_index} references unknown stage {stage!r}")

        success = bool(label.get("success", args.default_success)) and episode_index not in discarded
        record = {
            "episode_id": f"episode_{episode_index:06d}",
            "episode_index": episode_index,
            "dataset_path": str(dataset_path),
            "stage": stage,
            "task_family": label.get("task_family", args.task_family),
            "prompt": label.get("prompt", infer_prompt(episode, prompts, args.prompt)),
            "success": success,
            "fall": bool(label.get("fall", False)),
            "drop": bool(label.get("drop", False)),
            "object": label.get("object", args.object),
            "target": label.get("target", args.target),
            "difficulty": float(label.get("difficulty", args.difficulty)),
            "num_frames": int(label.get("num_frames", episode.get("num_frames", 0))),
            "schema_valid": bool(schema_valid),
            "discarded": episode_index in discarded,
        }
        if validation_warnings:
            record["validation_warnings"] = validation_warnings
        manifest.append(record)

    return manifest


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-labels", type=Path, help="Optional JSONL per-episode labels")
    parser.add_argument("--default-stage", default="S4")
    parser.add_argument("--task-family", default="fetch_place")
    parser.add_argument("--prompt", default="pick up the red cup and place it on the tray")
    parser.add_argument("--object", default="red_cup")
    parser.add_argument("--target", default="tray")
    parser.add_argument("--difficulty", type=float, default=0.25)
    parser.add_argument("--default-success", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-invalid-schema", action="store_true")
    parser.add_argument("--max-validation-rows", type=int, default=500)
    args = parser.parse_args()

    manifest = build_manifest(args)
    write_jsonl(args.output, manifest)
    print(f"wrote {len(manifest)} manifest records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
