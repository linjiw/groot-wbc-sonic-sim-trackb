#!/usr/bin/env python3
"""Dump the adaptive sampler's trained state from SONIC checkpoints.

Checkpoints save the sampler's per-bin ``adp_samp_num_episodes`` and
``adp_samp_num_failures`` under ``env_state_dict['motion_lib']``. Eval never
restores this state (eval-time ``sampling_prob`` is the uniform init), so this
dump is the only artifact holding the trained sampled-bin distribution — any
wrong-target or prior-domination test must read it from here.

Run in the training environment (robotixx ``env_isaaclab``): unpickling the full
checkpoint (``weights_only=False``) needs the trainer's classes importable. The
script's own imports are torch-only.

Caveats recorded in the output:
- Length-based bin weights are not checkpointed, so the recomputed distribution
  omits them (``recomputed_prob_unweighted``). The training-time weighted
  distribution is in the ``Env/adp_samp/prob_*`` telemetry; do not compare the
  unweighted recompute against telemetry thresholds.
- The recompute assumes ``use_failure_rate_decay=false`` (the release default);
  with decay enabled, failures/episodes is not the rate the sampler used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_CAVEATS = [
    "recomputed_prob_unweighted omits length-based bin weights (not checkpointed); "
    "training-time weighted probabilities are in Env/adp_samp/prob_* telemetry.",
    "failure_rate assumes use_failure_rate_decay=false (release default).",
    "the clip bound uses the mean failure rate over ALL bins; the sampler clips "
    "against the mean over the ACTIVE bin subset, which differs when motions are "
    "loaded in batches (all sample_data bins fit in one batch, so identical there).",
]


def extract_sampler_state(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint and pull out the motion-lib sampler tensors as lists."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    env_state = checkpoint.get("env_state_dict") or {}
    motion_lib_state = env_state.get("motion_lib") or {}
    state: dict[str, Any] = {"checkpoint_path": str(checkpoint_path)}
    trainer_state = checkpoint.get("state")
    if trainer_state is not None and hasattr(trainer_state, "global_step"):
        state["global_step"] = int(trainer_state.global_step)
    if "adp_samp_num_episodes" in motion_lib_state:
        state["adp_samp_num_episodes"] = [
            float(v) for v in motion_lib_state["adp_samp_num_episodes"].cpu().tolist()
        ]
        state["adp_samp_num_failures"] = [
            float(v) for v in motion_lib_state["adp_samp_num_failures"].cpu().tolist()
        ]
    return state


def _recompute_prob_unweighted(
    failure_rates: list[float], *, failure_rate_cap: float, uniform_sampling_rate: float
) -> list[float]:
    """Clip -> normalize -> blend with uniform, mirroring the sampler minus bin weights."""
    mean_rate = sum(failure_rates) / len(failure_rates)
    upper_bound = mean_rate * failure_rate_cap
    clipped = [min(max(rate, 0.0), upper_bound) for rate in failure_rates]
    clipped_sum = sum(clipped)
    n = len(failure_rates)
    uniform = 1.0 / n
    if clipped_sum == 0.0:
        failure_based = [uniform] * n
    else:
        failure_based = [value / clipped_sum for value in clipped]
    return [
        fb * (1.0 - uniform_sampling_rate) + uniform * uniform_sampling_rate for fb in failure_based
    ]


def build_sampler_state_summary(
    state: dict[str, Any],
    *,
    init_num_failures: float = 1.0,
    failure_rate_cap: float = 200.0,
    uniform_sampling_rate: float = 0.1,
    prior_domination_threshold: float = 2.0,
) -> dict[str, Any]:
    """Pure summary of one checkpoint's sampler state (unit-testable without torch)."""
    summary: dict[str, Any] = {
        "checkpoint_path": state.get("checkpoint_path"),
        "global_step": state.get("global_step"),
        "adaptive_state_present": "adp_samp_num_episodes" in state,
        "params": {
            "init_num_failures": init_num_failures,
            "failure_rate_cap": failure_rate_cap,
            "uniform_sampling_rate": uniform_sampling_rate,
            "prior_domination_threshold": prior_domination_threshold,
        },
        "caveats": _CAVEATS,
    }
    if not summary["adaptive_state_present"]:
        return summary

    episodes = state["adp_samp_num_episodes"]
    failures = state["adp_samp_num_failures"]
    if len(episodes) != len(failures) or not episodes:
        raise ValueError(
            f"episode/failure length mismatch or empty: {len(episodes)} vs {len(failures)}"
        )
    n = len(episodes)
    observed_failures = [f - init_num_failures for f in failures]
    observed_episodes = [e - init_num_failures for e in episodes]
    failure_rates = [f / e if e > 0 else 0.0 for f, e in zip(failures, episodes)]
    prob = _recompute_prob_unweighted(
        failure_rates,
        failure_rate_cap=failure_rate_cap,
        uniform_sampling_rate=uniform_sampling_rate,
    )
    uniform = 1.0 / n
    prior_dominated = [obs <= prior_domination_threshold for obs in observed_failures]

    summary["num_bins"] = n
    summary["bins"] = [
        {
            "bin": i,
            "num_episodes": episodes[i],
            "num_failures": failures[i],
            "observed_failures": observed_failures[i],
            "observed_episodes": observed_episodes[i],
            "failure_rate": failure_rates[i],
            "recomputed_prob_unweighted": prob[i],
            "prior_dominated": prior_dominated[i],
        }
        for i in range(n)
    ]
    summary["aggregates"] = {
        "observed_failures_total": sum(observed_failures),
        "observed_failures_max": max(observed_failures),
        "observed_episodes_total": sum(observed_episodes),
        "failure_rate_mean": sum(failure_rates) / n,
        "failure_rate_max": max(failure_rates),
        "prior_dominated_bin_count": sum(prior_dominated),
        "prior_dominated_fraction": sum(prior_dominated) / n,
        "prob_max_over_uniform_unweighted": max(prob) / uniform,
        "effective_num_bins_unweighted": 1.0 / sum(p * p for p in prob),
        "num_concentrated_bins_unweighted": sum(1 for p in prob if p > 10.0 * uniform),
    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint .pt. Repeat per seed.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--init-num-failures", type=float, default=1.0)
    parser.add_argument(
        "--failure-rate-cap",
        type=float,
        default=200.0,
        help="adp_samp_failure_rate_max_over_mean (release: 200).",
    )
    parser.add_argument("--uniform-sampling-rate", type=float, default=0.1)
    parser.add_argument(
        "--prior-domination-threshold",
        type=float,
        default=2.0,
        help="Bins with observed failures <= this are prior-dominated.",
    )
    args = parser.parse_args()

    summaries = [
        build_sampler_state_summary(
            extract_sampler_state(path),
            init_num_failures=args.init_num_failures,
            failure_rate_cap=args.failure_rate_cap,
            uniform_sampling_rate=args.uniform_sampling_rate,
            prior_domination_threshold=args.prior_domination_threshold,
        )
        for path in args.checkpoint
    ]
    output = {
        "schema_version": 1,
        "kind": "sampler_checkpoint_state_dump",
        "checkpoint_count": len(summaries),
        "checkpoints": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, sort_keys=True)
        f.write("\n")
    for summary in summaries:
        label = summary.get("checkpoint_path")
        if summary["adaptive_state_present"]:
            aggregates = summary["aggregates"]
            print(
                f"{label}: {summary['num_bins']} bins, "
                f"prior-dominated {aggregates['prior_dominated_fraction']:.0%}, "
                f"observed failures total {aggregates['observed_failures_total']:.2f}"
            )
        else:
            print(f"{label}: no adaptive sampler state")
    print(f"wrote sampler checkpoint state dump to {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
