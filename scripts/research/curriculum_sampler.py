#!/usr/bin/env python3
"""CG-WBC v0 curriculum scoring, difficulty, and sampling utilities."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import dataclasses
import json
import math
from pathlib import Path
import random
import sys
from typing import Any


@dataclass
class StageState:
    """Mutable curriculum state for one stage."""

    stage_id: str
    unlocked: bool = False
    competence: float = 0.0
    difficulty: float = 0.0
    ema_success: float = 0.0
    prev_ema_success: float = 0.0
    gate_passed: bool = False
    eval_episodes: int = 0


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def learning_progress(state: StageState) -> float:
    return abs(state.ema_success - state.prev_ema_success)


def metric_name_from_gate(gate_key: str) -> tuple[str, str] | None:
    """Return (metric_name, direction) for a gate key.

    Direction is ``"min"`` or ``"max"``. Unknown gate keys return ``None`` so
    callers can keep config metadata near gates without affecting evaluation.
    """

    if gate_key.endswith("_min"):
        return gate_key[: -len("_min")], "min"
    if gate_key.endswith("_max"):
        return gate_key[: -len("_max")], "max"
    return None


def update_difficulty(difficulty: float, metrics: dict[str, Any]) -> float:
    success = float(metrics.get("success_rate", 0.0))
    fall = float(metrics.get("fall_rate", 0.0))
    drop = float(metrics.get("drop_rate", 0.0))

    unsafe = fall > 0.08 or drop > 0.25

    if unsafe:
        difficulty -= 0.10
    elif success > 0.90:
        difficulty += 0.05
    elif success < 0.50:
        difficulty -= 0.05

    return clamp(difficulty, 0.0, 1.0)


def hard_gates_pass(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    for key, threshold_raw in gates.items():
        parsed = metric_name_from_gate(key)
        if parsed is None:
            continue

        metric_name, direction = parsed
        threshold = float(threshold_raw)
        if direction == "min":
            if float(metrics.get(metric_name, 0.0)) < threshold:
                return False
        elif float(metrics.get(metric_name, float("inf"))) > threshold:
            return False

    return True


def _max_gate_credit(metric_value: float, threshold: float) -> float:
    if threshold <= 0:
        return 1.0 if metric_value <= threshold else 0.0
    return clamp(1.0 - metric_value / threshold, 0.0, 1.0)


def competence_score(metrics: dict[str, Any], gates: dict[str, Any]) -> float:
    """Compute the conservative CG-WBC v0 stage competence score.

    The weighted score is normalized over the terms active for a stage. This
    keeps stages with fewer gates, such as schema-only S0, comparable to richer
    manipulation stages.
    """

    score = 0.0
    possible = 0.0

    if "success_rate_min" in gates:
        threshold = float(gates["success_rate_min"])
        if threshold > 0:
            possible += 0.50
            score += 0.50 * clamp(
                float(metrics.get("success_rate", 0.0)) / threshold,
                0.0,
                1.0,
            )

    if "fall_rate_max" in gates:
        possible += 0.20
        score += 0.20 * _max_gate_credit(
            float(metrics.get("fall_rate", 1.0)),
            float(gates["fall_rate_max"]),
        )

    if "drop_rate_max" in gates:
        possible += 0.15
        score += 0.15 * _max_gate_credit(
            float(metrics.get("drop_rate", 1.0)),
            float(gates["drop_rate_max"]),
        )

    if "action_jerk_max" in gates:
        possible += 0.10
        score += 0.10 * _max_gate_credit(
            float(metrics.get("action_jerk", 1.0)),
            float(gates["action_jerk_max"]),
        )

    possible += 0.05
    if metrics.get("schema_valid", False):
        score += 0.05

    if possible <= 0:
        return 0.0
    return clamp(score / possible, 0.0, 1.0)


def stage_sampling_weights(
    states: dict[str, StageState],
    anchor_stage_ids: list[str],
    beta: float = 5.0,
    anchor_mass: float = 0.20,
) -> dict[str, float]:
    """Return normalized sampling weights over unlocked stages."""

    anchor_mass = clamp(anchor_mass, 0.0, 1.0)
    unlocked = [state for state in states.values() if state.unlocked]
    if not unlocked:
        raise ValueError("No unlocked curriculum stages.")

    raw = {
        state.stage_id: math.exp(beta * learning_progress(state))
        for state in unlocked
    }
    total_raw = sum(raw.values())
    if total_raw <= 0:
        raw = {state.stage_id: 1.0 for state in unlocked}
        total_raw = float(len(raw))

    weights = {
        stage_id: (1.0 - anchor_mass) * value / total_raw
        for stage_id, value in raw.items()
    }

    valid_anchors = [stage_id for stage_id in anchor_stage_ids if stage_id in weights]
    if valid_anchors:
        per_anchor = anchor_mass / len(valid_anchors)
        for stage_id in valid_anchors:
            weights[stage_id] += per_anchor

    total = sum(weights.values())
    return {stage_id: weight / total for stage_id, weight in weights.items()}


def sample_stage(weights: dict[str, float], rng: random.Random | None = None) -> str:
    rng = rng or random
    stage_ids = list(weights.keys())
    probs = [weights[stage_id] for stage_id in stage_ids]
    return rng.choices(stage_ids, weights=probs, k=1)[0]


def states_from_json(data: dict[str, Any]) -> dict[str, StageState]:
    raw_stages = data.get("stages", data)
    allowed_fields = {field.name for field in dataclasses.fields(StageState)}
    states: dict[str, StageState] = {}
    for stage_id, values in raw_stages.items():
        state_data = {key: value for key, value in dict(values).items() if key in allowed_fields}
        state_data.setdefault("stage_id", stage_id)
        states[stage_id] = StageState(**state_data)
    return states


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Curriculum state JSON")
    parser.add_argument(
        "--anchor-stage",
        action="append",
        default=[],
        help="Stage id to receive anchor replay mass. Can be repeated.",
    )
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--anchor-mass", type=float, default=0.20)
    parser.add_argument("--sample", action="store_true", help="Also print one sampled stage")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with args.state.open("r", encoding="utf-8") as f:
        states = states_from_json(json.load(f))

    weights = stage_sampling_weights(
        states,
        anchor_stage_ids=args.anchor_stage,
        beta=args.beta,
        anchor_mass=args.anchor_mass,
    )

    payload: dict[str, Any] = {"weights": weights}
    if args.sample:
        payload["sampled_stage"] = sample_stage(weights, random.Random(args.seed))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
