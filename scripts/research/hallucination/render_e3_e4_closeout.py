#!/usr/bin/env python3
"""Render the reproducible E3 funnel and E4 before/after coverage closeout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lateral-cpu", type=Path, required=True)
    parser.add_argument("--lateral-pilot", type=Path, required=True)
    parser.add_argument("--crouch-screen", type=Path, required=True)
    parser.add_argument("--crouch-calibration", type=Path, required=True)
    parser.add_argument("--coverage-before", type=Path, required=True)
    parser.add_argument("--coverage-after", type=Path, required=True)
    parser.add_argument("--e3-out", type=Path, required=True)
    parser.add_argument("--e4-out", type=Path, required=True)
    args = parser.parse_args()

    lateral_cpu = json.loads(args.lateral_cpu.read_text())
    lateral = json.loads(args.lateral_pilot.read_text())
    crouch_screen = json.loads(args.crouch_screen.read_text())
    crouch = json.loads(args.crouch_calibration.read_text())
    before = json.loads(args.coverage_before.read_text())
    after = json.loads(args.coverage_after.read_text())

    physics_rollouts = len(lateral["cells"]) + 8
    gpu_hours = lateral["actual_contended_gpu_hours"] + crouch["actual_contended_gpu_hours"]
    arms_row = (
        f"| arms / lateral gap | {lateral_cpu['funnel']['trials']} trials -> "
        f"{lateral_cpu['funnel']['selected']} candidate | 1 scene pair | "
        "3/4 predicted outcomes | 0 | adapted-hard binding contact; no engineering-margin in-bin retry |"
    )
    crouch_row = (
        f"| crouch / overhead | {crouch_screen['source_motions']} motions -> "
        f"{crouch_screen['trials']} settings -> "
        f"{len(crouch_screen['calibration_recommendations'])} calibrations -> "
        f"{crouch['finite_face_trials']} face trials | {crouch['accepted_pairs']} accepted pairs, "
        f"{crouch['scene_candidates']} CPU candidates, 0 instantiated | 8 calibration rollouts | "
        "0 | DCS-v1 suppression repaired; physics verification pending |"
    )
    cpu_trials = lateral_cpu["funnel"]["trials"] + crouch_screen["trials"]
    selected = 1 + len(crouch_screen["calibration_recommendations"])
    crouch_window = crouch["maximum_eligible_raw_window_mm"]
    e3 = f"""# E3 Coverage-Targeted Tail-Synthesis Report

**Final-audit correction:** E3 produced **0 verified families**, but the earlier conclusion that
the crouch branch had zero scene candidates was wrong. SweepCF-DCS v1 let calibration `probe` rows
fill causal targets and included structurally impossible crouch/shoulder bins. V2 restricts target
occupancy to canonical cells of exact verified variants and restores
**{crouch['scene_candidates']} CPU-valid crouch proposals** from the same immutable executions.

| branch | CPU funnel | calibrated/instantiated | physics | verified | terminal reason |
|---|---|---:|---:|---:|---|
{arms_row}
{crouch_row}

The arm branch's sole scene candidate remains refused after adapted-hard contacted its binding
panel before drift; accepted context evidence then placed the retry outside the registered
0.8–0.9 m bucket under the empirical engineering margin. The crouch branch delivered a
{crouch_window:.2f} mm raw executed window. Its prior terminal reason was an accounting defect,
not target infeasibility. The 20 restored trials span all three accepted CAL3 sources and now feed
the separately registered E6 critical-support pilot.

Combined funnel: **{cpu_trials} CPU trials/settings -> {selected} selections -> {physics_rollouts}
serial rollouts -> {crouch['scene_candidates']} reopened CPU proposals -> 0 verified families**.
Actual spend was **{gpu_hours:.3f} contended GPU-h**.
Generated artifacts remain isolated from claim 5.

Reproduce from `e3_lateral_cpu.json`, `e3_lateral_pilot.json`,
`crouch_strength_screen.json`, and `crouch_calibration.json` with:

```bash
python scripts/research/hallucination/render_e3_e4_closeout.py \\
  --lateral-cpu docs/hallucination/e3_lateral_cpu.json \\
  --lateral-pilot docs/hallucination/e3_lateral_pilot.json \\
  --crouch-screen docs/hallucination/crouch_strength_screen.json \\
  --crouch-calibration docs/hallucination/crouch_calibration.json \\
  --coverage-before docs/hallucination/coverage/summary.json \\
  --coverage-after docs/hallucination/e4_coverage/summary.json \\
  --e3-out docs/hallucination/REPORT_E3.md \\
  --e4-out docs/hallucination/REPORT_E4.md
```
"""
    args.e3_out.write_text(e3)

    def percent(value: float) -> str:
        return f"{100 * value:.2f}%"

    episode_row = (
        f"| indexed episodes | {before['episodes']} | {after['episodes']} | "
        f"{after['episodes'] - before['episodes']:+d} |"
    )
    graded_row = (
        f"| graded episodes | {before['graded_episodes']} | {after['graded_episodes']} | "
        f"{after['graded_episodes'] - before['graded_episodes']:+d} |"
    )
    variant_row = (
        f"| verified variants | {before['verified_variants']} | {after['verified_variants']} | "
        f"{after['verified_variants'] - before['verified_variants']:+d} |"
    )
    family_row = (
        f"| independent causal families | {before['independent_verified_families']} | "
        f"{after['independent_verified_families']} | "
        f"{after['independent_verified_families'] - before['independent_verified_families']:+d} |"
    )
    bin_row = (
        f"| occupied target bins | {before['occupied_target_bins']}/{before['target_bins']} | "
        f"{after['occupied_target_bins']}/{after['target_bins']} | "
        f"{after['occupied_target_bins'] - before['occupied_target_bins']:+d} |"
    )
    fraction_row = (
        f"| target-bin coverage | {percent(before['target_bin_fraction'])} | "
        f"{percent(after['target_bin_fraction'])} | "
        f"{percent(after['target_bin_fraction'] - before['target_bin_fraction'])} |"
    )
    e4 = f"""# E4 SweepCF-DCS Before/After Report

E4 adds the two E2-verified archetype variants to an isolated index view; E3 contributes no rows
because neither tail branch has yet verified a family. This final audit uses SweepCF-DCS v2:
probe/refused-variant rows cannot fill targets, and 48 impossible crouch/shoulder targets are absent.
The result is **more verified visual variants and one additional within-source DCS bin, but no new
causal source**. This distinction matters: archetype transfer can deepen observed support without
reducing source confounding.

| metric | Phase-1 baseline | after E2/E3 | change |
|---|---:|---:|---:|
{episode_row}
{graded_row}
{variant_row}
{family_row}
{bin_row}
{fraction_row}

Door-lintel and I-beam add eight physics-graded episodes while remaining variants of
`cf_005_056`; they cannot advance the independent-family denominator. Their canonical cells add the
previously probe-only `clear_25_50` bin, so verified-family target coverage changes from
{before['occupied_target_bins']}/{before['target_bins']} to
{after['occupied_target_bins']}/{after['target_bins']}. The E4 index and plots live under
`docs/hallucination/e4_index/` and
`docs/hallucination/e4_coverage/`; they remain separate from the release and claim-5 inputs.
"""
    args.e4_out.write_text(e4)
    print(
        f"PASS: E3 families=0 spend={gpu_hours:.3f}; "
        f"E4 bins={before['occupied_target_bins']}->{after['occupied_target_bins']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
