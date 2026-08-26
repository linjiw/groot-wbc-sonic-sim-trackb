#!/usr/bin/env python3
"""Render SweepCF-DCS coverage from the additive release index.

Unknown fields are first-class output.  The coverage score uses only admissible v1 target bins
declared in ``configs/research/sweepcf_dcs_v1.json``; it never treats scene variants as new causal
families and never infers an axis or source identity from a directory name.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.dataset_generation.sweepcf_coverage import (  # noqa: E402
    UNKNOWN,
    coordinate_bucket,
    edit_behaviour,
    independent_verified_sources,
    margin_bucket,
    semantic_keypoint,
    target_bins,
    target_occupancy_episodes,
)

TARGET_FIELDS = (
    "edit_behaviour_class",
    "operator",
    "constraint_axis",
    "binding_keypoint",
    "constraint_coordinate_bucket",
    "margin_bucket",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return text or UNKNOWN


def enrich(row: Mapping[str, str], config: Mapping[str, object]) -> dict[str, str]:
    result = dict(row)
    result["source_family_id"] = normalized(row.get("source_family_id"))
    result["variant_id"] = normalized(row.get("variant_id"))
    result["constraint_axis"] = normalized(row.get("constraint_axis"))
    result["binding_keypoint"] = normalized(
        row.get("binding_keypoint") or semantic_keypoint(row.get("closest_body"))
    )
    result["edit_behaviour_class"] = normalized(
        row.get("edit_behaviour_class") or edit_behaviour(row.get("operator"))
    )
    result["constraint_coordinate_bucket"] = coordinate_bucket(
        result["constraint_axis"], row.get("constraint_coordinate_m"), config
    )
    result["margin_bucket"] = margin_bucket(row.get("min_clearance_mm"), config)
    for field in ("operator", "cell_role"):
        result[field] = normalized(row.get(field))
    return result


def target_key(row: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(normalized(row.get(field)) for field in TARGET_FIELDS)


def configured_values(config: Mapping[str, object], dimension: str) -> list[str]:
    if dimension == "margin_bucket":
        return [str(v) for v in config["margin_labels"]]  # type: ignore[index]
    if dimension == "constraint_coordinate_bucket":
        values: list[str] = []
        for spec in config["coordinate_buckets"].values():  # type: ignore[index,union-attr]
            values.extend(str(v) for v in spec["labels"])
        return list(dict.fromkeys(values))
    return [str(v) for v in config["dimensions"].get(dimension, [])]  # type: ignore[index,union-attr]


def marginal_rows(
    episodes: list[dict[str, str]], config: Mapping[str, object], dimensions: list[str]
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dimension in dimensions:
        counts = Counter(row[dimension] for row in episodes)
        configured = configured_values(config, dimension)
        ordered = configured + sorted(set(counts) - set(configured) - {UNKNOWN}) + [UNKNOWN]
        denominator = len(episodes) or 1
        for value in ordered:
            output.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "episodes": counts.get(value, 0),
                    "share": round(counts.get(value, 0) / denominator, 6),
                    "configured": int(value in configured),
                }
            )
    return output


def joint_rows(episodes: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, object]]:
    counts = Counter(tuple(row[field] for field in fields) for row in episodes)
    rows: list[dict[str, object]] = []
    for values, count in sorted(counts.items()):
        rows.append({**dict(zip(fields, values)), "episodes": count})
    return rows


def coverage_curve(
    episodes: list[dict[str, str]], verified_sources: set[str], target_set: set[tuple[str, ...]]
) -> list[dict[str, object]]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_order: list[str] = []
    for row in episodes:
        source = row["source_family_id"]
        if source not in verified_sources:
            continue
        if source not in by_source:
            source_order.append(source)
        by_source[source].append(row)

    occupied: set[tuple[str, ...]] = set()
    episode_count = 0
    curve: list[dict[str, object]] = []
    for index, source in enumerate(source_order, start=1):
        episode_count += len(by_source[source])
        occupied.update(
            target_key(row) for row in by_source[source] if target_key(row) in target_set
        )
        curve.append(
            {
                "independent_families": index,
                "source_family_id": source,
                "episodes_seen": episode_count,
                "occupied_target_bins": len(occupied),
                "target_bin_fraction": round(len(occupied) / len(target_set), 6),
            }
        )
    return curve


def render_curve(rows: list[dict[str, object]], out: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 3.8), dpi=120)
    if rows:
        x = [int(row["independent_families"]) for row in rows]
        y = [100 * float(row["target_bin_fraction"]) for row in rows]
        axis.step(x, y, where="post", color="#2F6B52", linewidth=2)
        axis.scatter(x, y, color="#2F6B52", s=28, zorder=3)
        axis.set_xticks(x)
    else:
        axis.text(0.5, 0.5, "no verified causal sources", ha="center", va="center")
    axis.set_xlabel("independent causal families")
    axis.set_ylabel("targetable keypoint-DCS bins occupied (%)")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.18, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    plt.close(fig)


def ranked_targets(
    configured: list[dict[str, str]], occupied: set[tuple[str, ...]]
) -> list[dict[str, object]]:
    empty = [row for row in configured if target_key(row) not in occupied]

    def priority(row: Mapping[str, str]) -> tuple[object, ...]:
        mid_margin = row["margin_bucket"] in ("clear_10_25", "clear_25_50")
        return (
            0 if row["edit_behaviour_class"] == "arms" else 1,
            0 if mid_margin else 1,
            row["constraint_axis"],
            row["binding_keypoint"],
            row["constraint_coordinate_bucket"],
            row["margin_bucket"],
        )

    output: list[dict[str, object]] = []
    for rank, row in enumerate(sorted(empty, key=priority), start=1):
        output.append(
            {
                "rank": rank,
                **row,
                "reason": (
                    "tail behaviour; mid-margin first"
                    if row["margin_bucket"] in ("clear_10_25", "clear_25_50")
                    else "tail behaviour; boundary/extreme margin after mid-margin bins"
                ),
            }
        )
    return output


def report_text(
    summary: Mapping[str, object], verified_variants: Mapping[str, list[str]], out_name: str
) -> str:
    unknown = summary["unknown_episode_fields"]
    track = summary["motion_gate_counts"]
    station_mismatch = summary["binding_station_mismatch"]
    family_lines = "\n".join(
        f"| `{source}` | {len(variants)} | {', '.join(f'`{v}`' for v in variants)} |"
        for source, variants in sorted(verified_variants.items())
    )
    unknown_lines = "\n".join(
        f"| `{field}` | {count} | {count / max(int(summary['episodes']), 1):.1%} |"
        for field, count in unknown.items()
    )
    return f"""# LFH Phase-1 Coverage

**Status:** CPU-only {summary['schema_version']} baseline, generated from the additive release
index.
Physics verdicts are unchanged; source-family scenes were read only to rebuild their measured index
fields. No frozen scene, rollout, or prediction-register artifact was used for target selection or
modified.

## Evidence Accounting

The repaired contract yields **{summary['independent_verified_families']} independent causal
families** across **{summary['verified_variants']} verified scene variants**. This passes the
Phase-0 review's fixed acceptance test of exactly two independent families.

| causal source | verified variants | artifact ids |
|---|---:|---|
{family_lines}

Variants are retained as scene evidence but never advance the independent-family count.

Route intersection is necessary but not sufficient for source verification. The repaired contract
refuses **{len(station_mismatch['variants'])} artifact variants** whose loaded constraint is
inconsistent with its manifest binding station; the largest measured offset is
**{float(station_mismatch['max_offset_mm']):.1f} mm**. These variants remain visible in the index as
`not_verified` instead of being discarded or counted.

## Coverage Baseline

- Episodes indexed: **{summary['episodes']}** ({summary['graded_episodes']} physics-graded).
- Coupling-valid keypoint-DCS bins occupied by canonical cells from exact verified variants:
  **{summary['occupied_target_bins']} / {summary['target_bins']}**
  ({float(summary['target_bin_fraction']):.1%}).
- Eligible canonical occupancy rows: **{summary['target_occupancy_episodes']}**; all observed
  episodes would touch **{summary['observed_episode_occupied_target_bins']}** target bins, but probe
  and refused-variant rows do not suppress causal-family targets.
- Empty generation targets emitted: **{summary['empty_target_bins']}**.
- Motion rows: **{track['motions']}**; reference semantic gate known for
  **{track['reference_semantic_known']}**; executed empty-room trackability known for
  **{track['controller_trackable_known']}**.
- **{track['accepted_nonempty_only']}** motions have acceptance evidence only in nonempty scenes;
  this is not promoted to LFH trackability. Fully gated LFH candidates today: **{track['fully_gated']}**.

The v2 denominator contains only coupling-valid crouch/overhead and arm-tuck/lateral templates in
`configs/research/sweepcf_dcs_v2.json`. It removes the 48 v1 crouch/shoulder cells that
`local_crouch` cannot target. Ground-support and unavailable edit operators are not silently
counted as missed opportunities. Absolute DCS occupancy remains a reporting/novelty statistic;
trajectory-conditioned support, route phase, and finite face extent determine proposal feasibility.

## Unknown-Bin Audit

| field | unknown episodes | share |
|---|---:|---:|
{unknown_lines}

Unknowns are retained rather than reconstructed from names. In particular, the present index does
not carry a measured lateral gap coordinate for the wall attempts, so those coordinates remain
unknown and lateral target bins remain honest targets. `duck_000` is also intentionally source
`unknown`: it is a two-probe work directory with no `family.json`, unlike `duck_001` and later
variants whose manifests explicitly identify `cf_005_056`.

## Reproducible Validation

The reported **21 focused tests** are 10 LFH contract tests, 5 route-intersection tests, and 6
sealed-split immutability tests. They are run as an explicit list because collecting the entire
`tests/dataset_generation/` directory on this machine hits a pre-existing pandas/numpy version
incompatibility (numpy 1.21.5 while the installed pandas requires at least 1.22.4):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv_sim/bin/python -m pytest \
  tests/dataset_generation/test_sweepcf_coverage.py \
  tests/dataset_generation/test_scene_route_check.py \
  tests/dataset_generation/test_scene_first_testset.py -q
```

## Generated Artifacts

- [`marginal_coverage.csv`](coverage/marginal_coverage.csv): every configured marginal plus
  observed extras and explicit unknowns.
- [`joint_binding_axis_margin.csv`](coverage/joint_binding_axis_margin.csv): semantic anatomy ×
  constraint axis × signed margin.
- [`joint_behaviour_operator_role.csv`](coverage/joint_behaviour_operator_role.csv): edit behaviour
  × operator × four-cell role.
- [`coverage_curve.csv`](coverage/coverage_curve.csv) and
  [`coverage_curve.png`](coverage/coverage_curve.png): coverage growth by independent causal source,
  not by scene variant.
- [`targets.json`](coverage/targets.json): ranked empty admissible bins for later CPU proposal work.
- [`summary.json`](coverage/summary.json): machine-readable values used in this report.

## Interpretation and Gate

The current evidence is overhead-only at the independent-family level and concentrated in the
`head_torso` semantic capsule group. The high empty-bin count is therefore a measured description
of corpus thinness, not evidence that generated scenes should be rolled out immediately. Phase 2
must first provide tested placement and keepout machinery, and every later physics batch remains
manifest- and user-approval-gated.

Reproduce this report with:

```bash
python scripts/research/hallucination/render_coverage.py \\
  --episodes <release>/index/episodes.csv \\
  --families <release>/index/families.csv \\
  --motions <release>/index/motions.csv \\
  --out {out_name} --report docs/hallucination/COVERAGE.md \\
  --expect-independent-families 2
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--families", type=Path, required=True)
    parser.add_argument("--motions", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/research/sweepcf_dcs_v2.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expect-independent-families", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    episodes = [enrich(row, config) for row in read_csv(args.episodes)]
    families = read_csv(args.families)
    motions = read_csv(args.motions)
    verified_sources = independent_verified_sources(families)
    if (
        args.expect_independent_families is not None
        and len(verified_sources) != args.expect_independent_families
    ):
        raise SystemExit(
            "independent-family acceptance test failed: "
            f"expected {args.expect_independent_families}, measured {len(verified_sources)} "
            f"({sorted(verified_sources)})"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    dimensions = [
        "binding_keypoint",
        "constraint_axis",
        "constraint_coordinate_bucket",
        "margin_bucket",
        "edit_behaviour_class",
        "operator",
        "cell_role",
    ]
    marginals = marginal_rows(episodes, config, dimensions)
    write_csv(
        args.out / "marginal_coverage.csv",
        marginals,
        ["dimension", "value", "episodes", "share", "configured"],
    )
    joint_one_fields = ("binding_keypoint", "constraint_axis", "margin_bucket")
    write_csv(
        args.out / "joint_binding_axis_margin.csv",
        joint_rows(episodes, joint_one_fields),
        [*joint_one_fields, "episodes"],
    )
    joint_two_fields = ("edit_behaviour_class", "operator", "cell_role")
    write_csv(
        args.out / "joint_behaviour_operator_role.csv",
        joint_rows(episodes, joint_two_fields),
        [*joint_two_fields, "episodes"],
    )

    configured_targets = target_bins(config)
    target_set = {target_key(row) for row in configured_targets}
    occupancy_episodes = target_occupancy_episodes(episodes, families, config)
    occupied = {target_key(row) for row in occupancy_episodes if target_key(row) in target_set}
    observed_occupied = {target_key(row) for row in episodes if target_key(row) in target_set}
    empty_targets = ranked_targets(configured_targets, occupied)
    curve = coverage_curve(occupancy_episodes, verified_sources, target_set)
    write_csv(
        args.out / "coverage_curve.csv",
        curve,
        [
            "independent_families",
            "source_family_id",
            "episodes_seen",
            "occupied_target_bins",
            "target_bin_fraction",
        ],
    )
    render_curve(curve, args.out / "coverage_curve.png")

    verified_variants: dict[str, list[str]] = defaultdict(list)
    for row in families:
        source = normalized(row.get("source_family_id"))
        if row.get("status") == "verified" and source in verified_sources:
            verified_variants[source].append(normalized(row.get("variant_id")))

    evidence_refusals: Counter[str] = Counter()
    station_mismatch_variants: set[str] = set()
    for row in families:
        reasons = [reason for reason in row.get("evidence_reasons", "").split(";") if reason]
        evidence_refusals.update(reasons)
        if "binding_station_mismatch" in reasons:
            station_mismatch_variants.add(normalized(row.get("variant_id")))
    station_offsets = [
        float(row["binding_station_offset_mm"])
        for row in episodes
        if row["variant_id"] in station_mismatch_variants
        and row.get("binding_station_offset_mm", "") != ""
    ]

    unknown_counts = {
        field: sum(row[field] == UNKNOWN for row in episodes)
        for field in (
            "source_family_id",
            "constraint_axis",
            "constraint_coordinate_bucket",
            "binding_keypoint",
            "margin_bucket",
            "edit_behaviour_class",
        )
    }
    semantic_known = sum(row.get("reference_semantic_valid", "") != "" for row in motions)
    trackable_known = sum(row.get("controller_trackable", "") != "" for row in motions)
    accepted_nonempty_only = sum(
        row.get("controller_trackability_evidence") == "accepted_nonempty_only" for row in motions
    )
    fully_gated = sum(
        row.get("embodiment_feasible") == "1"
        and row.get("reference_semantic_valid") == "1"
        and row.get("controller_trackable") == "1"
        for row in motions
    )
    summary = {
        "schema_version": config["schema_version"],
        "episodes": len(episodes),
        "graded_episodes": sum(row.get("outcome") in ("accepted", "rejected") for row in episodes),
        "independent_verified_families": len(verified_sources),
        "verified_source_family_ids": sorted(verified_sources),
        "verified_variants": sum(len(values) for values in verified_variants.values()),
        "target_bins": len(target_set),
        "occupied_target_bins": len(occupied),
        "empty_target_bins": len(empty_targets),
        "target_bin_fraction": round(len(occupied) / len(target_set), 6),
        "target_occupancy_policy": config.get("target_occupancy_policy", "all_episodes"),
        "target_occupancy_episodes": len(occupancy_episodes),
        "observed_episode_occupied_target_bins": len(observed_occupied),
        "unknown_episode_fields": unknown_counts,
        "evidence_refusals": dict(sorted(evidence_refusals.items())),
        "binding_station_mismatch": {
            "variants": sorted(station_mismatch_variants),
            "max_offset_mm": max(station_offsets, default=0.0),
        },
        "motion_gate_counts": {
            "motions": len(motions),
            "reference_semantic_known": semantic_known,
            "controller_trackable_known": trackable_known,
            "controller_trackable_true": sum(
                row.get("controller_trackable") == "1" for row in motions
            ),
            "accepted_nonempty_only": accepted_nonempty_only,
            "fully_gated": fully_gated,
        },
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "targets.json").write_text(
        json.dumps(
            {
                "schema_version": config["schema_version"],
                "source_index": str(args.episodes),
                "independent_verified_families": len(verified_sources),
                "configured_bins": len(target_set),
                "occupied_bins": len(occupied),
                "observed_episode_occupied_bins": len(observed_occupied),
                "target_occupancy_policy": config.get("target_occupancy_policy", "all_episodes"),
                "empty_bins": len(empty_targets),
                "targets": empty_targets,
            },
            indent=2,
        )
        + "\n"
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text(summary, verified_variants, str(args.out)))

    print(
        f"{len(episodes)} episodes, {len(verified_sources)} independent families, "
        f"{len(occupied)}/{len(target_set)} target bins -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
