#!/usr/bin/env python3
"""Update CG-WBC curriculum state from stage-wise metrics."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.curriculum_sampler import (  # noqa: E402
    StageState,
    competence_score,
    hard_gates_pass,
    update_difficulty,
)


def load_json(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path, {})
    if "stage_metrics" in data:
        return {str(stage_id): dict(metrics) for stage_id, metrics in data["stage_metrics"].items()}
    if isinstance(data, list):
        return {str(record["stage"]): dict(record) for record in data}
    return {str(stage_id): dict(metrics) for stage_id, metrics in data.items()}


def load_graph(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["curriculum"]


def initial_state(graph: dict[str, Any]) -> dict[str, Any]:
    stages = {}
    for stage in graph["stages"]:
        stage_id = stage["id"]
        stages[stage_id] = asdict(
            StageState(
                stage_id=stage_id,
                unlocked=stage.get("unlock") == "always",
            )
        )
    return {
        "curriculum_name": graph["name"],
        "updated_at": None,
        "stages": stages,
        "history": [],
    }


def prerequisites_pass(stage: dict[str, Any], state: dict[str, Any]) -> bool:
    prerequisites = stage.get("prerequisites", [])
    return all(state["stages"].get(stage_id, {}).get("gate_passed", False) for stage_id in prerequisites)


def update_state(
    graph: dict[str, Any],
    state: dict[str, Any],
    metrics_by_stage: dict[str, dict[str, Any]],
    ema_alpha: float,
) -> dict[str, Any]:
    competence_threshold = float(graph.get("competence_threshold", 0.85))
    min_episodes = int(graph.get("minimum_eval_episodes_per_stage", 50))

    stage_defs = {stage["id"]: stage for stage in graph["stages"]}

    for stage_id, stage_def in stage_defs.items():
        state["stages"].setdefault(stage_id, asdict(StageState(stage_id=stage_id)))
        stage_state = state["stages"][stage_id]
        metrics = metrics_by_stage.get(stage_id)
        if metrics is None:
            continue

        gates = stage_def.get("gates", {})
        competence = competence_score(metrics, gates)
        gate_passed = hard_gates_pass(metrics, gates)
        eval_episodes = int(metrics.get("eval_episodes", metrics.get("num_episodes", 0)))

        prev_ema = float(stage_state.get("ema_success", 0.0))
        success = float(metrics.get("success_rate", 0.0))
        stage_state["prev_ema_success"] = prev_ema
        stage_state["ema_success"] = ema_alpha * success + (1.0 - ema_alpha) * prev_ema
        stage_state["competence"] = competence
        stage_state["difficulty"] = update_difficulty(float(stage_state.get("difficulty", 0.0)), metrics)
        stage_state["eval_episodes"] = eval_episodes
        stage_state["gate_passed"] = (
            bool(stage_state.get("unlocked", False))
            and gate_passed
            and competence >= competence_threshold
            and eval_episodes >= min_episodes
        )
        stage_state["last_metrics"] = metrics

    changed = True
    while changed:
        changed = False
        for stage in graph["stages"]:
            stage_id = stage["id"]
            stage_state = state["stages"][stage_id]
            if stage_state.get("unlocked", False):
                continue
            if prerequisites_pass(stage, state):
                stage_state["unlocked"] = True
                changed = True

    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    state.setdefault("history", []).append(
        {
            "updated_at": state["updated_at"],
            "metrics_stages": sorted(metrics_by_stage),
            "unlocked_stages": [
                stage_id for stage_id, values in state["stages"].items() if values.get("unlocked")
            ],
        }
    )
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--state-in", type=Path)
    parser.add_argument("--state-out", type=Path, required=True)
    parser.add_argument("--ema-alpha", type=float, default=0.30)
    args = parser.parse_args()

    graph = load_graph(args.config)
    state = load_json(args.state_in, initial_state(graph))
    metrics_by_stage = load_metrics(args.metrics)
    state = update_state(graph, state, metrics_by_stage, args.ema_alpha)

    args.state_out.parent.mkdir(parents=True, exist_ok=True)
    with args.state_out.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote curriculum state to {args.state_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
