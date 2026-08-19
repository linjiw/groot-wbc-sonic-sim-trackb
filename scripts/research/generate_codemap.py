#!/usr/bin/env python
"""Generate the code map from the modules themselves, so it cannot drift.

A hand-written index of 154 files is stale the week it is written. This reads each file's first
docstring line and files it under a pipeline stage, so the map is regenerated rather than
maintained. Anything that does not match a stage is listed as unclassified rather than dropped --
a map that silently omits files is worse than no map.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Pipeline stages in the order data flows through them. Each entry is (title, why it exists,
#: patterns matching file stems). Order matters: the first matching stage wins.
STAGES = (
    (
        "Motion generation and qualification",
        "Producing reference clips and deciding which the robot's body can hold at all.",
        (
            "kimodo",
            "prompt",
            "taxonomy",
            "motion_prefilter",
            "reference_gate",
            "screen_",
            "generate_",
            "encode_",
            "retarget",
            "behaviour_predicates",
            "episode_semantics",
        ),
    ),
    (
        "Scene design and construction",
        "Choosing what an obstacle tests and placing it to an exact margin.",
        (
            "criticality",
            "scene_route_check",
            "clutter_scene",
            "scene_route_sampler",
            "eval_scene",
            "build_graded",
            "plan_graded",
            "route_placement",
            "scene_asset",
            "scene_build_sheet",
            "build_clutter",
            "build_eval_scenes",
            "sample_scene",
        ),
    ),
    (
        "Adaptation operators",
        "Constructing an adapted motion from a nominal one, locally.",
        (
            "local_adaptation",
            "motion_envelope",
            "swept_volume",
            "self_clearance",
            "self_intersection",
            "adaptation_cost",
            "retarget_reference",
        ),
    ),
    (
        "Physics verification and gating",
        "Deciding whether an episode is evidence, and of what.",
        (
            "trajectory_acceptance",
            "episode_outcome",
            "contact_decomposition",
            "gate_policy",
            "trajectory_validation",
            "trajectory_segments",
            "perception_timing",
            "latent_parity",
            "audit_",
            "verify_",
            "check_",
            "analyze_gate",
        ),
    ),
    (
        "Counterfactual families",
        "Assembling and certifying the unit of counterfactual evidence.",
        (
            "counterfactual_family",
            "build_counterfactual",
            "build_matched",
            "build_window",
            "mine_counterfactual",
            "report_family",
            "build_family",
            "analyze_family",
            "report_operator",
            "report_dataset",
        ),
    ),
    (
        "Dataset assembly and release",
        "Turning verified families into something another person can use.",
        (
            "build_dataset_release",
            "render_release",
            "corpus_snapshot",
            "runtime_manifest",
            "trajectory_export",
            "reference_payload",
            "schemas",
            "build_corpus",
            "sample_review",
            "build_scene_build_sheet",
            "freeze_",
        ),
    ),
    (
        "Learning and evaluation",
        "The selector, its controls, and the experiments that decide the claims.",
        (
            "compatibility_selector",
            "evaluate_selector",
            "gpu_capacity",
            "behaviour_diversity",
            "report_behaviour",
            "analyze_generation",
            "report_tracking",
            "stage2",
        ),
    ),
    (
        "Rendering and presentation",
        "Making the result visible to a person.",
        (
            "render_family_views",
            "render_room_camera",
            "render_multiview",
            "visualize_",
            "render_corpus",
            "build_dataset_gallery",
            "generate_codemap",
        ),
    ),
    # Everything below is in this repository but is *not* SweepCF. Kept separate deliberately: a
    # map that mixes them invites a LACE throughput script to be read as part of this pipeline, and
    # invites this pipeline's file count to be quoted as if it were all one project.
    (
        "Adjacent: LACE",
        "A separate line of work sharing this repository. Not part of SweepCF.",
        ("lace",),
    ),
    (
        "Adjacent: SONIC training, curriculum and sampler",
        "Controller-side training and diagnosis. SweepCF consumes the trained policy, it does "
        "not train it.",
        ("sonic", "sampler", "curriculum", "paired_stats", "stall_diagnostic"),
    ),
    (
        "Adjacent: earlier phases and other cohorts",
        "Superseded phases and unrelated dataset cohorts, kept for provenance.",
        (
            "bones_seed",
            "sim_d1",
            "sim_d2",
            "m5_",
            "g0_",
            "mvp_",
            "horizon_acceptance",
            "ablate_station",
            "groot",
            "synthetic_g1",
            "benchmark_pipeline",
            "open_loop",
        ),
    ),
)


def summary(path: Path) -> str:
    """First sentence of the module docstring, or the first comment line for shell scripts."""
    if path.suffix == ".sh":
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("#!"):
                return stripped.lstrip("# ").strip()
        return ""
    try:
        tree = ast.parse(path.read_text(errors="ignore"))
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
    return re.sub(r"\s+", " ", first)[:150]


def stage_of(stem: str) -> str | None:
    for title, _why, patterns in STAGES:
        if any(pattern in stem for pattern in patterns):
            return title
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    roots = [REPO_ROOT / "gear_sonic/dataset_generation", REPO_ROOT / "scripts/research"]
    files = [
        p
        for root in roots
        for p in sorted(root.iterdir())
        if p.suffix in (".py", ".sh") and not p.name.startswith("__")
    ]

    grouped: dict[str, list[tuple[Path, str]]] = {title: [] for title, _, _ in STAGES}
    unclassified: list[tuple[Path, str]] = []
    for path in files:
        stage = stage_of(path.stem)
        (grouped[stage] if stage else unclassified).append((path, summary(path)))

    lines = [
        "# Code map",
        "",
        "Generated by `scripts/research/generate_codemap.py` from the modules' own docstrings, so",
        "it cannot drift from the code. Regenerate rather than edit.",
        "",
        f"{len(files)} files across `gear_sonic/dataset_generation/` and `scripts/research/`,",
        "grouped by the stage of the pipeline they serve.",
        "",
    ]
    for title, why, _ in STAGES:
        entries = grouped[title]
        if not entries:
            continue
        lines += [f"## {title}", "", f"*{why}*", "", "| file | what it does |", "|---|---|"]
        for path, text in entries:
            rel = path.relative_to(REPO_ROOT)
            lines.append(f"| `{rel}` | {text or '—'} |")
        lines.append("")

    if unclassified:
        lines += [
            "## Unclassified",
            "",
            "*Not matched by any stage pattern. Listed rather than dropped: a map that silently",
            "omits files is worse than no map.*",
            "",
            "| file | what it does |",
            "|---|---|",
        ]
        for path, text in unclassified:
            lines.append(f"| `{path.relative_to(REPO_ROOT)}` | {text or '—'} |")
        lines.append("")

    args.out.write_text("\n".join(lines))
    classified = sum(len(v) for v in grouped.values())
    print(f"{len(files)} files: {classified} classified, {len(unclassified)} unclassified")
    for title, _, _ in STAGES:
        if grouped[title]:
            print(f"   {len(grouped[title]):3d}  {title}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
