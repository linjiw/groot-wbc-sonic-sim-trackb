#!/usr/bin/env python3
"""Run a multi-seed paired SONIC experiment from one seed-templated spec.

Replaces the untracked per-milestone ``run_sim_m*.py`` orchestrators: one
tracked spec template with literal ``{seed}`` placeholders is rendered per seed,
each seed goes through ``run_sonic_paired_experiment.materialize_paired_experiment``
(same dry-run/execute semantics), and the resulting seed-level comparisons are
aggregated with the preregistered effect gate and paired statistics.

The template must contain ``{seed}`` in at least one string value (commands and
exp_var names must differ per seed) — a template without it would silently run
identical seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.aggregate_sonic_comparisons import (  # noqa: E402
    build_aggregate_comparison,
    write_aggregate_json,
    write_aggregate_table_markdown,
)
from scripts.research.run_sonic_paired_experiment import (  # noqa: E402
    _load_json,
    _write_json,
    materialize_paired_experiment,
)

_SEED_PLACEHOLDER = "{seed}"


def _render(value: Any, seed: int) -> Any:
    if isinstance(value, str):
        return value.replace(_SEED_PLACEHOLDER, str(seed))
    if isinstance(value, dict):
        return {key: _render(item, seed) for key, item in value.items()}
    if isinstance(value, list):
        return [_render(item, seed) for item in value]
    return value


def _template_has_placeholder(template: dict[str, Any]) -> bool:
    # Rendering changes the template iff a {seed} placeholder exists somewhere.
    return _render(template, 0) != template


def render_spec_for_seed(template: dict[str, Any], seed: int) -> dict[str, Any]:
    """Render one seed's spec: substitute {seed} in strings, set the seed field."""
    spec = _render(template, seed)
    spec["seed"] = seed
    return spec


def run_multiseed_experiment(
    template: dict[str, Any],
    *,
    seeds: list[int],
    output_dir: Path,
    dry_run: bool,
    repo_root: Path,
    variant_a: str,
    variant_b: str,
    effect_metric: str | None,
    a_minus_b_threshold: float,
    min_improved_seeds: int,
) -> dict[str, Any]:
    """Materialize every seed then aggregate the seed-level comparisons."""
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {seeds}")
    if not _template_has_placeholder(template):
        raise ValueError(
            "spec template must contain a literal {seed} placeholder in at least one "
            "string value; otherwise all seeds would run identical commands"
        )

    template_variants = {
        str(variant.get("name"))
        for variant in template.get("variants", [])
        if isinstance(variant, dict) and variant.get("name")
    }
    unmatched = {variant_a, variant_b} - template_variants
    if template_variants and unmatched:
        raise ValueError(
            f"variant names {sorted(unmatched)} not found in spec template variants "
            f"{sorted(template_variants)}; pass matching --variant-a/--variant-b"
        )

    seed_results: list[dict[str, Any]] = []
    comparison_paths: list[Path] = []
    missing_comparison_seeds: list[int] = []
    for seed in seeds:
        seed_dir = output_dir / f"seed{seed}"
        spec = render_spec_for_seed(template, seed)
        _write_json(seed_dir / "spec.json", spec)
        plan = materialize_paired_experiment(
            spec, output_dir=seed_dir, dry_run=dry_run, repo_root=repo_root
        )
        seed_results.append(
            {
                "seed": seed,
                "output_dir": str(seed_dir),
                "comparison_json": plan.get("comparison_json"),
                "ok_for_causal_comparison": plan.get("ok_for_causal_comparison"),
            }
        )
        if plan.get("comparison_json"):
            comparison_paths.append(Path(plan["comparison_json"]))
        else:
            missing_comparison_seeds.append(seed)

    aggregate: dict[str, Any] | None = None
    if comparison_paths:
        aggregate = build_aggregate_comparison(
            comparison_paths,
            variant_a=variant_a,
            variant_b=variant_b,
            effect_metric=effect_metric,
            a_minus_b_threshold=a_minus_b_threshold,
            min_improved_seeds=min_improved_seeds,
        )
        write_aggregate_json(output_dir / "aggregate_comparison.json", aggregate)
        write_aggregate_table_markdown(output_dir / "aggregate_table.md", aggregate)

    run = {
        "schema_version": 1,
        "kind": "sonic_multiseed_run",
        "dry_run": dry_run,
        "seeds": seeds,
        "variant_a": variant_a,
        "variant_b": variant_b,
        "output_dir": str(output_dir),
        "seed_results": seed_results,
        "missing_comparison_seeds": missing_comparison_seeds,
        "aggregate_json": str(output_dir / "aggregate_comparison.json") if aggregate else None,
        # A seed that produced no comparison invalidates the whole run: the
        # aggregate (and its effect gate) would otherwise silently cover a
        # subset of the preregistered seeds.
        "ok_for_causal_comparison": (
            aggregate.get("ok_for_causal_comparison") is True and not missing_comparison_seeds
            if aggregate
            else False
        ),
    }
    _write_json(output_dir / "multiseed_run.json", run)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec-template",
        type=Path,
        required=True,
        help="Paired-experiment spec JSON with literal {seed} placeholders.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--variant-a", default="adaptive_sampling_micro", help="Treatment variant name."
    )
    parser.add_argument(
        "--variant-b", default="uniform_sampling_micro", help="Control variant name."
    )
    parser.add_argument("--effect-metric", default="eval.all.mpjpe_g")
    parser.add_argument("--a-minus-b-threshold", type=float, default=-0.5)
    parser.add_argument("--min-improved-seeds", type=int, default=2)
    args = parser.parse_args()

    template = _load_json(args.spec_template)
    try:
        run = run_multiseed_experiment(
            template,
            seeds=args.seeds,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            repo_root=args.repo_root,
            variant_a=args.variant_a,
            variant_b=args.variant_b,
            effect_metric=args.effect_metric,
            a_minus_b_threshold=args.a_minus_b_threshold,
            min_improved_seeds=args.min_improved_seeds,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"wrote multiseed run to {args.output_dir / 'multiseed_run.json'}")
    if run.get("aggregate_json"):
        print(f"wrote aggregate to {run['aggregate_json']}")
    if run.get("missing_comparison_seeds"):
        print(
            "WARNING: seed(s) "
            f"{run['missing_comparison_seeds']} produced no comparison.json; "
            "the aggregate covers a subset and ok_for_causal_comparison=false"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
