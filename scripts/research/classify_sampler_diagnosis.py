#!/usr/bin/env python3
"""Mechanize the preregistered SIM-M4b sampler-diagnosis classification rule.

This is preregistration-in-code: the rule from ``fable-next.md`` §Phase 1 is
frozen here as tested logic BEFORE the robotixx SIM-M3 logs/checkpoints are
synced, so the eventual diagnosis cannot be gamed post-hoc. It reads the two
committed SIM-M4a artifacts and emits a verdict plus decision routing.

Inputs (exact SIM-M4a tool schemas):
- ``sampler_telemetry.json`` from ``summarize_sampler_telemetry.py``: the true
  training-time (weighted) distribution. Its ``logs[].keys[<k>].{first,last}``
  are authoritative for the flatness thresholds (``prob_max_over_uniform``,
  ``num_concentrated_bins``, ``effective_num_bins``).
- ``sampler_checkpoint_state.json`` from ``dump_sampler_checkpoint_state.py``:
  the only source of per-bin observed-failure counts. Its
  ``checkpoints[].aggregates.prior_dominated_fraction`` is authoritative for the
  signal-starvation check. Its recomputed probabilities are UNWEIGHTED and are
  NOT used for the flatness thresholds (that would double-count the caveat).

Preregistered rule (majority over the adaptive seeds):
- ``flat``    := prob_max_over_uniform < 5 AND num_concentrated_bins == 0
                AND effective_num_bins >= 0.9 * num_bins   (telemetry, final iter)
- ``starved`` := prior_dominated_fraction >= 0.9           (checkpoint)
- per-seed verdict:
    flat AND starved       -> compound_under_active_signal_starved  (SIM-D1 -> SIM-M5b)
    flat AND NOT starved   -> plain_under_active                    (SIM-M5a)
    NOT flat, targeting known:
        rank-disagree       -> wrong_target                         (SIM-D1 -> SIM-M5b)
        rank-agree + gate fail -> active_but_not_useful             (SIM-D1 -> SIM-M5b)
    NOT flat, targeting unknown -> active_targeting_undetermined     (SIM-D1 -> SIM-M5b)

``num_bins`` comes from the checkpoint dump when present, else from the telemetry
proxy ``round(effective_num_bins.first)`` (the distribution is uniform at init,
so effective bins == bin count there). The wrong-target branch needs a
per-motion difficulty ranking that does not exist at M4b time; it is optional
(``--per-motion-difficulty``) and, when absent while a seed is NOT flat, that
seed's targeting is ``undetermined`` — which still routes to SIM-D1 -> SIM-M5b,
so the overall decision is unaffected under the working hypothesis.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

# Preregistered thresholds (fable-next.md §Phase 1). Exposed as CLI flags so the
# frozen defaults are auditable, not to invite tuning.
_PROB_MAX_OVER_UNIFORM_FLAT = 5.0
# The prose freezes "effective_num_bins >= 0.9 x 70" as a LITERAL floor, not a
# fraction of the observed bin count — sample_data is "~70 bins" but the actual
# checkpoint count may be 69/71, and substituting it would silently move the
# threshold. Use the frozen absolute floor; record the real num_bins for audit.
_PREREGISTERED_BIN_COUNT = 70
_EFFECTIVE_NUM_BINS_FRACTION = 0.9
_EFFECTIVE_NUM_BINS_FLOOR = _EFFECTIVE_NUM_BINS_FRACTION * _PREREGISTERED_BIN_COUNT
# The prior-domination fraction the checkpoint dump must have been generated with
# (num_failures - init <= 2). The coupling only holds at this threshold.
_PRIOR_DOMINATED_FRACTION = 0.9
_PRIOR_DOMINATION_THRESHOLD = 2.0

_SEED_RE = re.compile(r"seed[_\-]?(\d+)")

# Per-seed verdicts and their preregistered routing.
_ROUTING = {
    "plain_under_active": "SIM-M5a",
    "compound_under_active_signal_starved": "SIM-D1 -> SIM-M5b",
    "under_active_starvation_unconfirmed": "SIM-D1 -> SIM-M5b",
    "wrong_target": "SIM-D1 -> SIM-M5b",
    "active_but_not_useful": "SIM-D1 -> SIM-M5b",
    "active_targeting_undetermined": "SIM-D1 -> SIM-M5b",
    "incomplete": "insufficient data — resync artifacts",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _seed_from_path(path: str | None) -> int | None:
    if not path:
        return None
    match = _SEED_RE.search(str(path))
    return int(match.group(1)) if match else None


def _telemetry_by_seed(telemetry: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Adaptive-only telemetry logs keyed by seed extracted from log_path."""
    out: dict[int, dict[str, Any]] = {}
    for record in telemetry.get("logs", []):
        if not isinstance(record, dict) or not record.get("adaptive_telemetry_present"):
            continue
        seed = _seed_from_path(record.get("log_path"))
        if seed is None:
            raise ValueError(
                f"cannot extract seed from telemetry log_path {record.get('log_path')!r}; "
                "expected a 'seedN' token"
            )
        if seed in out:
            raise ValueError(f"duplicate adaptive telemetry for seed {seed}")
        out[seed] = record
    return out


def _checkpoints_by_seed(dump: dict[str, Any]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for record in dump.get("checkpoints", []):
        if not isinstance(record, dict) or not record.get("adaptive_state_present"):
            continue
        seed = _seed_from_path(record.get("checkpoint_path"))
        if seed is None:
            raise ValueError(
                f"cannot extract seed from checkpoint_path {record.get('checkpoint_path')!r}; "
                "expected a 'seedN' token"
            )
        if seed in out:
            raise ValueError(f"duplicate checkpoint state for seed {seed}")
        out[seed] = record
    return out


def _last(record: dict[str, Any], key: str) -> float | None:
    entry = (record.get("keys") or {}).get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("last")
    return float(value) if isinstance(value, (int, float)) else None


def _first(record: dict[str, Any], key: str) -> float | None:
    entry = (record.get("keys") or {}).get(key)
    if not isinstance(entry, dict):
        return None
    value = entry.get("first")
    return float(value) if isinstance(value, (int, float)) else None


def classify_seed(
    telemetry: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    *,
    prob_max_over_uniform_flat: float = _PROB_MAX_OVER_UNIFORM_FLAT,
    effective_num_bins_floor: float = _EFFECTIVE_NUM_BINS_FLOOR,
    prior_dominated_fraction: float = _PRIOR_DOMINATED_FRACTION,
    prior_domination_threshold: float = _PRIOR_DOMINATION_THRESHOLD,
    effect_gate_failed: bool = True,
    difficulty_rank_agrees: bool | None = None,
) -> dict[str, Any]:
    """Classify one adaptive seed. Returns a per-seed evidence + verdict record.

    ``difficulty_rank_agrees`` is None when no per-motion difficulty ranking is
    available (the usual M4b state); it only matters on the NOT-flat branch.
    """
    evidence: dict[str, Any] = {
        "has_telemetry": telemetry is not None,
        "has_checkpoint": checkpoint is not None,
    }
    if telemetry is None:
        return {
            "verdict": "incomplete",
            "reason": "no adaptive telemetry for seed",
            "evidence": evidence,
        }

    prob_max_over_uniform = _last(telemetry, "prob_max_over_uniform")
    num_concentrated_bins = _last(telemetry, "num_concentrated_bins")
    effective_num_bins = _last(telemetry, "effective_num_bins")

    # num_bins is recorded for audit only; the flatness floor is the frozen
    # absolute value (_EFFECTIVE_NUM_BINS_FLOOR), not a fraction of the observed
    # bin count, so a 69/71-bin dataset cannot silently move the threshold.
    num_bins: float | None = None
    num_bins_source = None
    if checkpoint is not None and isinstance(checkpoint.get("num_bins"), (int, float)):
        num_bins = float(checkpoint["num_bins"])
        num_bins_source = "checkpoint"

    evidence.update(
        {
            "prob_max_over_uniform_last": prob_max_over_uniform,
            "num_concentrated_bins_last": num_concentrated_bins,
            "effective_num_bins_last": effective_num_bins,
            "effective_num_bins_floor": effective_num_bins_floor,
            "num_bins": num_bins,
            "num_bins_source": num_bins_source,
        }
    )

    if prob_max_over_uniform is None or num_concentrated_bins is None or effective_num_bins is None:
        return {
            "verdict": "incomplete",
            "reason": "missing required telemetry key(s) at final iteration",
            "evidence": evidence,
        }

    flat = (
        prob_max_over_uniform < prob_max_over_uniform_flat
        and num_concentrated_bins == 0
        and effective_num_bins >= effective_num_bins_floor
    )
    evidence["flat"] = flat

    # Starvation check requires per-bin observed failures — checkpoint-only. Verify
    # the dump was generated at the preregistered prior-domination threshold, or the
    # num_failures - init <= 2 coupling does not hold and the fraction is unusable.
    prior_dominated = None
    starved = None
    threshold_ok = True
    if checkpoint is not None:
        params = checkpoint.get("params") or {}
        dump_threshold = params.get("prior_domination_threshold")
        if isinstance(dump_threshold, (int, float)) and dump_threshold != prior_domination_threshold:
            threshold_ok = False
        aggregates = checkpoint.get("aggregates") or {}
        prior_dominated = aggregates.get("prior_dominated_fraction")
        if threshold_ok and isinstance(prior_dominated, (int, float)):
            starved = prior_dominated >= prior_dominated_fraction
    evidence["prior_dominated_fraction"] = prior_dominated
    evidence["starved"] = starved
    evidence["prior_domination_threshold_ok"] = threshold_ok

    if not threshold_ok:
        return {
            "verdict": "incomplete",
            "reason": (
                "checkpoint dump prior_domination_threshold != "
                f"{prior_domination_threshold}; starvation fraction is not comparable"
            ),
            "evidence": evidence,
        }

    if flat:
        if starved is True:
            return {"verdict": "compound_under_active_signal_starved", "evidence": evidence}
        if starved is False:
            return {"verdict": "plain_under_active", "evidence": evidence}
        # flat but no checkpoint to confirm starvation: cannot distinguish plain
        # (M5a) from compound (D1->M5b). Route to the safer D1->M5b branch, but do
        # NOT assert starvation in the label (it was never measured).
        return {
            "verdict": "under_active_starvation_unconfirmed",
            "reason": "flat but starvation unconfirmed (no checkpoint); routed conservatively to D1->M5b",
            "evidence": evidence,
        }

    # NOT flat -> concentrated. Distinguish wrong-target from not-useful.
    evidence["effect_gate_failed"] = effect_gate_failed
    evidence["difficulty_rank_agrees"] = difficulty_rank_agrees
    if difficulty_rank_agrees is None:
        return {
            "verdict": "active_targeting_undetermined",
            "reason": "no per-motion difficulty ranking available to test targeting",
            "evidence": evidence,
        }
    if difficulty_rank_agrees is False:
        return {"verdict": "wrong_target", "evidence": evidence}
    if effect_gate_failed:
        return {"verdict": "active_but_not_useful", "evidence": evidence}
    return {
        "verdict": "active_targeting_undetermined",
        "reason": "concentrated, correctly targeted, but effect gate did not fail",
        "evidence": evidence,
    }


def classify_diagnosis(
    telemetry: dict[str, Any],
    checkpoint_dump: dict[str, Any],
    *,
    prob_max_over_uniform_flat: float = _PROB_MAX_OVER_UNIFORM_FLAT,
    effective_num_bins_floor: float = _EFFECTIVE_NUM_BINS_FLOOR,
    prior_dominated_fraction: float = _PRIOR_DOMINATED_FRACTION,
    prior_domination_threshold: float = _PRIOR_DOMINATION_THRESHOLD,
    effect_gate_failed: bool = True,
    difficulty_rank_agrees: bool | None = None,
) -> dict[str, Any]:
    """Apply the preregistered rule across all adaptive seeds (majority verdict)."""
    tele_by_seed = _telemetry_by_seed(telemetry)
    ckpt_by_seed = _checkpoints_by_seed(checkpoint_dump)
    seeds = sorted(set(tele_by_seed) | set(ckpt_by_seed))
    if not seeds:
        raise ValueError("no adaptive seeds found in telemetry or checkpoint dump")

    per_seed: dict[str, Any] = {}
    for seed in seeds:
        per_seed[str(seed)] = classify_seed(
            tele_by_seed.get(seed),
            ckpt_by_seed.get(seed),
            prob_max_over_uniform_flat=prob_max_over_uniform_flat,
            effective_num_bins_floor=effective_num_bins_floor,
            prior_dominated_fraction=prior_dominated_fraction,
            prior_domination_threshold=prior_domination_threshold,
            effect_gate_failed=effect_gate_failed,
            difficulty_rank_agrees=difficulty_rank_agrees,
        )

    verdicts = [rec["verdict"] for rec in per_seed.values()]
    scored = [v for v in verdicts if v != "incomplete"]
    counts = Counter(scored)
    n_seeds = len(seeds)
    n_scored = len(scored)
    # Preregistered: majority over ALL adaptive seeds, not just the ones that
    # parsed. A verdict needs strictly more than half of the adaptive seeds, so
    # 1-of-3 (with 2 incomplete) is NOT a majority — that routes to insufficient_data.
    majority_verdict = None
    if scored:
        top_verdict, top_count = counts.most_common(1)[0]
        if top_count * 2 > n_seeds:
            majority_verdict = top_verdict

    if majority_verdict is None:
        # No verdict reaches a majority of the adaptive seeds. Distinguish "not
        # enough coverage" from "covered but seeds disagree".
        if n_scored * 2 <= n_seeds:
            overall = "insufficient_data"
            routing = "insufficient data — too few seeds scored; resync artifacts"
        else:
            overall = "no_majority"
            routing = "manual review — seeds disagree"
    else:
        overall = majority_verdict
        routing = _ROUTING[majority_verdict]

    return {
        "schema_version": 1,
        "kind": "sampler_diagnosis_classification",
        "overall_verdict": overall,
        "decision_routing": routing,
        "seed_count": len(seeds),
        "scored_seed_count": n_scored,
        "verdict_counts": dict(counts),
        "thresholds": {
            "prob_max_over_uniform_flat": prob_max_over_uniform_flat,
            "effective_num_bins_floor": effective_num_bins_floor,
            "prior_dominated_fraction": prior_dominated_fraction,
            "prior_domination_threshold": prior_domination_threshold,
            "effect_gate_failed": effect_gate_failed,
            "difficulty_rank_agrees": difficulty_rank_agrees,
        },
        "per_seed": per_seed,
        "preregistration_note": (
            "Rule frozen in scripts/research/classify_sampler_diagnosis.py before the "
            "robotixx SIM-M3 artifacts were synced; see fable-next.md Phase 1."
        ),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# SIM-M4b Sampler Diagnosis\n\n")
        f.write("| Field | Value |\n|---|---|\n")
        for key in ["overall_verdict", "decision_routing", "seed_count", "scored_seed_count"]:
            f.write(f"| `{key}` | {result.get(key)} |\n")
        f.write(
            f"| `verdict_counts` | `{json.dumps(result.get('verdict_counts', {}), sort_keys=True)}` |\n"
        )
        f.write("\n## Per-seed evidence\n\n")
        f.write(
            "| Seed | Verdict | flat | starved | prob_max/uniform | num_concentrated | eff_bins/num_bins |\n"
        )
        f.write("|---|---|---|---|---:|---:|---|\n")
        for seed, rec in sorted(result.get("per_seed", {}).items()):
            ev = rec.get("evidence", {})
            nb = ev.get("num_bins")
            eff = ev.get("effective_num_bins_last")
            eff_ratio = (
                f"{eff:.4g}/{nb:g}"
                if isinstance(eff, (int, float)) and isinstance(nb, (int, float))
                else ""
            )
            pmax = ev.get("prob_max_over_uniform_last")
            conc = ev.get("num_concentrated_bins_last")
            f.write(
                f"| {seed} | `{rec.get('verdict')}` | {ev.get('flat')} | {ev.get('starved')} | "
                f"{pmax} | {conc} | {eff_ratio} |\n"
            )
        f.write(f"\n> {result.get('preregistration_note', '')}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--telemetry-json",
        type=Path,
        required=True,
        help="Output of summarize_sampler_telemetry.py.",
    )
    parser.add_argument(
        "--checkpoint-state-json",
        type=Path,
        required=True,
        help="Output of dump_sampler_checkpoint_state.py.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--prob-max-over-uniform-flat", type=float, default=_PROB_MAX_OVER_UNIFORM_FLAT
    )
    parser.add_argument(
        "--effective-num-bins-floor",
        type=float,
        default=_EFFECTIVE_NUM_BINS_FLOOR,
        help="Frozen absolute floor (0.9 x 70 = 63), not a fraction of observed bins.",
    )
    parser.add_argument("--prior-dominated-fraction", type=float, default=_PRIOR_DOMINATED_FRACTION)
    parser.add_argument(
        "--prior-domination-threshold",
        type=float,
        default=_PRIOR_DOMINATION_THRESHOLD,
        help="The checkpoint dump must have been generated at this threshold (default 2.0).",
    )
    parser.add_argument(
        "--effect-gate-passed",
        action="store_true",
        help="Set only if the SIM-M3 effect gate PASSED (it failed on record, so default is failed).",
    )
    args = parser.parse_args()

    result = classify_diagnosis(
        _load_json(args.telemetry_json),
        _load_json(args.checkpoint_state_json),
        prob_max_over_uniform_flat=args.prob_max_over_uniform_flat,
        effective_num_bins_floor=args.effective_num_bins_floor,
        prior_dominated_fraction=args.prior_dominated_fraction,
        prior_domination_threshold=args.prior_domination_threshold,
        effect_gate_failed=not args.effect_gate_passed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    if args.output_md:
        write_markdown(args.output_md, result)
    print(f"verdict: {result['overall_verdict']} -> {result['decision_routing']}")
    print(f"wrote diagnosis to {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
