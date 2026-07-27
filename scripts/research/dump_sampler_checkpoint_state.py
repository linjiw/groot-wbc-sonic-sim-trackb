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
- Legacy/failure-rate checkpoints do not contain bin weights, so their recomputed
  distribution remains ``recomputed_prob_unweighted``. New ZPD checkpoints expose
  weights plus cumulative actual selected-target-bin draws. ``sampled_fraction``
  is authoritative mass across global selection and active-batch conditioning,
  measured before the random frame and ``pre_failure_sample_window`` shift; it is
  not executed-start-bin mass.
- The failure-rate recompute assumes ``use_failure_rate_decay=false`` (the release default);
  with decay enabled, failures/episodes is not the rate the sampler used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

_CAVEATS = [
    "failure_rate assumes use_failure_rate_decay=false (release default).",
    "the clip bound uses the mean failure rate over ALL bins; the sampler clips "
    "against the mean over the ACTIVE bin subset, which differs when motions are "
    "loaded in batches (all sample_data bins fit in one batch, so identical there).",
]

_ZPD_SAMPLING_MASS_SEMANTICS = "selected_target_bin_pre_window_shift"


def extract_sampler_state(checkpoint_path: Path) -> dict[str, Any]:
    """Load a checkpoint and pull out the motion-lib sampler tensors as lists."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    env_state = checkpoint.get("env_state_dict") or {}
    motion_lib_state = env_state.get("motion_lib") or {}
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    state: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest.hexdigest(),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
    }
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
    if "adp_samp_bin_motion_ids" in motion_lib_state:
        state["adp_samp_bin_motion_ids"] = [
            int(v) for v in motion_lib_state["adp_samp_bin_motion_ids"].cpu().tolist()
        ]
        state["adp_samp_motion_data_keys"] = [
            str(v) for v in motion_lib_state.get("adp_samp_motion_data_keys", [])
        ]
    if "adp_samp_bin_ranges" in motion_lib_state:
        state["adp_samp_bin_ranges"] = [
            [int(value) for value in row]
            for row in motion_lib_state["adp_samp_bin_ranges"].cpu().tolist()
        ]
    if "adp_samp_bin_weights" in motion_lib_state:
        state["adp_samp_bin_weights"] = [
            float(v) for v in motion_lib_state["adp_samp_bin_weights"].cpu().tolist()
        ]
    if "adp_samp_bin_draw_counts" in motion_lib_state:
        state["adp_samp_bin_draw_counts"] = [
            int(v) for v in motion_lib_state["adp_samp_bin_draw_counts"].cpu().tolist()
        ]
    for key in ("adp_samp_zpd_schema", "adp_samp_zpd_config"):
        if key in motion_lib_state:
            state[key] = dict(motion_lib_state[key])
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
        "checkpoint_sha256": state.get("checkpoint_sha256"),
        "checkpoint_size_bytes": state.get("checkpoint_size_bytes"),
        "global_step": state.get("global_step"),
        "adaptive_state_present": "adp_samp_num_episodes" in state,
        "params": {
            "init_num_failures": init_num_failures,
            "failure_rate_cap": failure_rate_cap,
            "uniform_sampling_rate": uniform_sampling_rate,
            "prior_domination_threshold": prior_domination_threshold,
        },
        "caveats": list(_CAVEATS),
    }
    if "adp_samp_zpd_schema" in state:
        summary["sampler_schema"] = state["adp_samp_zpd_schema"]
    if "adp_samp_zpd_config" in state:
        summary["sampler_config"] = state["adp_samp_zpd_config"]
    if not summary["adaptive_state_present"]:
        return summary

    episodes = state["adp_samp_num_episodes"]
    failures = state["adp_samp_num_failures"]
    if len(episodes) != len(failures) or not episodes:
        raise ValueError(
            f"episode/failure length mismatch or empty: {len(episodes)} vs {len(failures)}"
        )
    n = len(episodes)
    weights = state.get("adp_samp_bin_weights")
    draw_counts = state.get("adp_samp_bin_draw_counts")
    bin_ranges = state.get("adp_samp_bin_ranges")
    if weights is not None:
        if len(weights) != n:
            raise ValueError(f"bin-weight length mismatch: {len(weights)} vs {n} sampler bins")
        if any(
            not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0
            for weight in weights
        ):
            raise ValueError("bin weights must be finite nonnegative numbers")
        summary["caveats"].append(
            "ZPD bin weights are checkpointed; realized sampled_fraction is preferred "
            "because sampling also includes global motion selection and active-batch conditioning."
        )
    else:
        summary["caveats"].append(
            "recomputed_prob_unweighted omits length-based bin weights (not checkpointed in "
            "legacy/failure-rate state); training-time weighted probabilities are in telemetry."
        )
    if draw_counts is not None:
        if len(draw_counts) != n:
            raise ValueError(
                f"bin-draw-count length mismatch: {len(draw_counts)} vs {n} sampler bins"
            )
        if any(not isinstance(count, int) or count < 0 for count in draw_counts):
            raise ValueError("bin draw counts must be nonnegative integers")
    if bin_ranges is not None:
        if len(bin_ranges) != n or any(len(row) != 2 for row in bin_ranges):
            raise ValueError(f"bin-range shape mismatch: expected {n} rows of [start, end]")

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
    sampled_total = sum(draw_counts) if draw_counts is not None else None
    sampled_fractions = (
        [count / sampled_total for count in draw_counts]
        if sampled_total
        else ([0.0] * n if draw_counts is not None else None)
    )

    summary["num_bins"] = n
    if "adp_samp_bin_motion_ids" in state:
        motion_ids = state["adp_samp_bin_motion_ids"]
        motion_keys = state.get("adp_samp_motion_data_keys", [])
        if len(motion_ids) != n:
            raise ValueError(
                f"bin-motion id length mismatch: {len(motion_ids)} vs {n} sampler bins"
            )
        if any(motion_id < 0 or motion_id >= len(motion_keys) for motion_id in motion_ids):
            raise ValueError("bin-motion id is outside adp_samp_motion_data_keys")
        summary["bin_motion_keys"] = [motion_keys[motion_id] for motion_id in motion_ids]
    summary["bins"] = []
    for i in range(n):
        bin_summary = {
            "bin": i,
            "num_episodes": episodes[i],
            "num_failures": failures[i],
            "observed_failures": observed_failures[i],
            "observed_episodes": observed_episodes[i],
            "failure_rate": failure_rates[i],
            "recomputed_prob_unweighted": prob[i],
            "prior_dominated": prior_dominated[i],
        }
        if weights is not None:
            bin_summary["bin_weight"] = weights[i]
        if draw_counts is not None:
            bin_summary["sampled_count"] = draw_counts[i]
            bin_summary["sampled_fraction"] = sampled_fractions[i]
        if bin_ranges is not None:
            bin_summary["bin_start"] = bin_ranges[i][0]
            bin_summary["bin_end"] = bin_ranges[i][1]
        summary["bins"].append(bin_summary)
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
    if sampled_total is not None:
        summary["aggregates"]["sampled_count_total"] = sampled_total
        summary["aggregates"]["sampled_fraction_sum"] = sum(sampled_fractions)
        summary["sampling_mass_source"] = (
            "empirical_selected_target_bin_draw_counts_pre_window_shift"
        )
        summary["sampling_mass_semantics"] = _ZPD_SAMPLING_MASS_SEMANTICS
        summary["caveats"].append(
            "sampled_count/sampled_fraction are selected-target-bin mass before the "
            "pre_failure_sample_window shift, not executed-start-bin mass."
        )
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
        "schema_version": 2,
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
