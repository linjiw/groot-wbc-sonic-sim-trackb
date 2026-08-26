# E3 Coverage-Targeted Tail-Synthesis Report

**Final-audit correction:** E3 produced **0 verified families**, but the earlier conclusion that
the crouch branch had zero scene candidates was wrong. SweepCF-DCS v1 let calibration `probe` rows
fill causal targets and included structurally impossible crouch/shoulder bins. V2 restricts target
occupancy to canonical cells of exact verified variants and restores
**20 CPU-valid crouch proposals** from the same immutable executions.

| branch | CPU funnel | calibrated/instantiated | physics | verified | terminal reason |
|---|---|---:|---:|---:|---|
| arms / lateral gap | 24 trials -> 1 candidate | 1 scene pair | 3/4 predicted outcomes | 0 | adapted-hard binding contact; no certified in-bin retry |
| crouch / overhead | 15 motions -> 60 settings -> 4 calibrations -> 36 face trials | 3 accepted pairs, 20 CPU candidates, 0 instantiated | 8 calibration rollouts | 0 | DCS-v1 suppression repaired; physics verification pending |

The arm branch's sole scene candidate remains refused after adapted-hard contacted its binding
panel before drift; accepted context evidence then placed the retry outside the registered
0.8–0.9 m bucket under the empirical engineering margin. The crouch branch delivered a
72.26 mm raw executed window. Its prior terminal reason was an accounting defect,
not target infeasibility. The 20 restored trials span all three accepted CAL3 sources and now feed
the separately registered E6 critical-support pilot.

Combined funnel: **84 CPU trials/settings -> 5 selections -> 12
serial rollouts -> 20 reopened CPU proposals -> 0 verified families**.
Actual spend was **0.102 contended GPU-h**.
Generated artifacts remain isolated from claim 5.

Reproduce from `e3_lateral_cpu.json`, `e3_lateral_pilot.json`,
`crouch_strength_screen.json`, and `crouch_calibration.json` with:

```bash
python scripts/research/hallucination/render_e3_e4_closeout.py \
  --lateral-cpu docs/hallucination/e3_lateral_cpu.json \
  --lateral-pilot docs/hallucination/e3_lateral_pilot.json \
  --crouch-screen docs/hallucination/crouch_strength_screen.json \
  --crouch-calibration docs/hallucination/crouch_calibration.json \
  --coverage-before docs/hallucination/coverage/summary.json \
  --coverage-after docs/hallucination/e4_coverage/summary.json \
  --e3-out docs/hallucination/REPORT_E3.md \
  --e4-out docs/hallucination/REPORT_E4.md
```
