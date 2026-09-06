# Next acquisition: freeze the inference methods and audit fresh motion sources

Historical acquisition design, 2026-09-05. The [registered protocol](FRESH_SOURCE_V1.md) now fixes the exact acquisition and inference contracts. The [completed acquisition result](FRESH_SOURCE_V1_RESULT.md) records the full funnel and stop-rule outcome. The design rationale below is preserved; it does not authorize another acquisition.

## Why this is the next discriminator

The [station-search study](STATION_SEARCH_V1_RESULT.md) removes all five remaining
original-gradient failures in its fresh-draw test panel. All three local alternatives
accept 192/192 test outputs and fix the known replay to 8/8. Pattern search is cheaper,
but all source motions were already observed during development. More tuning on these
sources would not establish fresh-source transfer. Preserve all frozen outcomes.

Also isolate whether the learned initialization helps **pattern search itself**. The
older uniform-gradient comparison cannot answer that question for a different search
algorithm. Keep uniformly initialized pattern search as a required control.

## Fixed acquisition target and lineage

Target eight new neutral walking source candidates, each with local_crouch derivatives
at route fractions 0.35 and 0.65, unchanged 0.055 m drop and 0.18 window. These are
16 derived cases, not 16 independent sources. Every derivative remains attached to
its source. All eight new sources are excluded from model training, normalization,
method design and checkpoint selection; current 410xx/420xx data remain development.

Before generation, assign eight unused source-generation identities and verify that
no matching model/prompt/seed artifacts or outcomes already exist. Record the exact
source-ID list in a hash-pinned acquisition manifest. Do not infer freshness from an
unused filename or from unseen random scene draws. Count every generation attempt.

The existing local setup inspected for this plan is:

- `scripts/research/generate_kimodo_motions.py` and `.venv_kimodo/bin/python`.
- The neutral straight-walking prompt/cache contract recorded in
  [the earlier route-retention design](evidence/E1_ROUTE_RETENTION_HELDOUT_V1.json).
- The prior Kimodo G1 settings: four seconds, 30 Hz, 100 denoising steps, cached text
  encoding and explicit model snapshot provenance. Verify current file/cache/model
  hashes before selecting them for the new manifest; this plan does not certify them.
- The existing local_crouch operator, persisted-reference FK, Q0/Q1 screening and
  reference route checks, with source/target hashes and unchanged endpoint/route gates.

Register candidate counts, stop rules, source roles, implementations, device/VRAM
preflight and a bounded generation budget before spend. Observe the standing GPU budget
and contention policy. Keep frozen SONIC and original corpora untouched. Retain failed
generations, route failures and reference-gate failures; do not replace them until eight
qualify. The final manifest must specify whether an acquisition failure stops the pilot
or permits analysis of the qualifying subset with the complete funnel reported.

## Inference comparison to freeze before inspecting new motions

Retain all six station-study methods and add uniformly initialized pattern search:
raw learned draws, ranked learned draws, original gradient, probe/refinement,
multiple starts, learned pattern, uniform pattern. Keep all three old all8/1200
checkpoints, fixed fresh scene-draw seeds, eight outputs per case/checkpoint, identical
original trust bounds, and 4,624 search queries per assisted job. Account for model
setup, inherited training, search, sampling and independent verification separately.
A uniform-pattern control must start from a fixed uniform eight-draw batch and use the
same local algorithm, domain, query budget and final checker as learned pattern.

Use source-level counts and paired output changes where correspondence exists. Report
both phases, all seeds, every source and both clearance-failure types. Repeat the
113-placement independent NumPy/Torch audit without tuning the methods, thresholds or
output quotas. Freeze directional comparisons and all denominators in the launch
protocol before source construction; do not choose them from the next observations.

## What follows and what does not

A fresh-source result would support transfer within this straight-walk / beam family.
It would not establish complex multi-object generation or a physical collision guarantee.
Next test continuous placement and between-frame clearance under explicit imported-body
geometry assumptions, then obstacle-present execution with the frozen controller.

For a more amortized learned generator, refine examples from TRAINING sources only and
study whether distillation reduces inference queries without collapsing valid support.
Keep the new audit pool excluded from those targets and all fitting. Neural supervision
may come from geometric search on training motion pairs without human obstacle labels;
that still does not identify a unique natural scene distribution. Broader 3D obstacle
families and joint scenes need their own motion alternatives and joint collision checks.
