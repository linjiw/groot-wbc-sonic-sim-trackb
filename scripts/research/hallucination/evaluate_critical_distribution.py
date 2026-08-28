"""Evaluate q_LFH conditional inference without spending physics rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.hallucination.critical_distribution import (  # noqa: E402
    compatible_archetypes,
    infer_critical_scene_distribution,
    load_q_lfh_atoms,
)


def sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def evaluate(q_path: Path, samples_per_source: int = 100) -> dict[str, object]:
    atoms = load_q_lfh_atoms(q_path)
    losses: list[float] = []
    top3_hits = 0
    labels = 0
    geometry_samples = 0
    geometry_valid = 0
    source_results: list[dict[str, object]] = []
    for heldout in atoms:
        distribution = infer_critical_scene_distribution(
            heldout.support,
            atoms,
            excluded_source_pair_id=heldout.source_pair_id,
        )
        ranked = sorted(
            distribution.archetype_probabilities,
            key=distribution.archetype_probabilities.get,
            reverse=True,
        )
        per_label = {}
        for archetype_id in heldout.verified_archetypes:
            probability = distribution.archetype_probabilities[archetype_id]
            losses.append(-math.log(probability))
            labels += 1
            top3_hits += int(archetype_id in ranked[:3])
            per_label[archetype_id] = probability
        for index in range(samples_per_source):
            proposal = distribution.sample(index)
            geometry_samples += 1
            geometry_valid += int(
                heldout.support.lower_m <= proposal.hard_coordinate_m <= heldout.support.upper_m
            )
        excluded = set(heldout.verified_archetypes)
        novelty = infer_critical_scene_distribution(
            heldout.support,
            atoms,
            excluded_source_pair_id=heldout.source_pair_id,
            excluded_archetypes=excluded,
        )
        novelty_top = max(
            novelty.archetype_probabilities,
            key=novelty.archetype_probabilities.get,
        )
        source_results.append(
            {
                "source_pair_id": heldout.source_pair_id,
                "heldout_verified_probabilities": per_label,
                "top3": ranked[:3],
                "novelty_top": novelty_top,
                "novelty_probabilities": dict(novelty.archetype_probabilities),
                "self_source_excluded": heldout.source_pair_id
                not in distribution.evidence_source_weights,
            }
        )

    mean_log_loss = sum(losses) / len(losses)
    uniform_log_loss = math.log(len(compatible_archetypes(atoms[0].support.axis_type)))
    result = {
        "schema_version": "q_lfh_conditional_eval_v1",
        "source_q_lfh": str(q_path),
        "source_q_lfh_sha256": sha256(q_path),
        "sources": len(atoms),
        "heldout_verified_labels": labels,
        "mean_heldout_log_loss": mean_log_loss,
        "uniform_compatible_log_loss": uniform_log_loss,
        "log_loss_improvement_over_uniform": uniform_log_loss - mean_log_loss,
        "top3_recall": top3_hits / labels,
        "geometry_support_valid_rate": geometry_valid / geometry_samples,
        "geometry_samples": geometry_samples,
        "source_results": source_results,
    }
    result["decision"] = (
        "keep"
        if result["log_loss_improvement_over_uniform"] > 0
        and result["top3_recall"] >= 0.8
        and result["geometry_support_valid_rate"] == 1.0
        else "discard"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=Path, default=Path("docs/hallucination/q_lfh_v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-per-source", type=int, default=100)
    args = parser.parse_args()
    report = evaluate(args.q, args.samples_per_source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "decision",
                    "mean_heldout_log_loss",
                    "uniform_compatible_log_loss",
                    "log_loss_improvement_over_uniform",
                    "top3_recall",
                    "geometry_support_valid_rate",
                )
            },
            indent=2,
        )
    )
    return 0 if report["decision"] == "keep" else 2


if __name__ == "__main__":
    raise SystemExit(main())
