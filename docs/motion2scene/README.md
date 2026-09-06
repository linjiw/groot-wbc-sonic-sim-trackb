# Motion2Scene visual research report

The [training-source distillation pilot](DISTILLATION_V1_RESULT.md) now completes a teacher bank, twelve student/control fits and 3,456 independently checked outputs. Hybrid raw accepts **334/384**, original raw **292/384**, and the stronger query-budget control **310/384**. Hybrid five-evaluation search accepts **377/384**, versus **384/384** for original seventeen-evaluation search. These are results on observed development sources; the 430xx pool remains excluded.

The recipe misses at least one registered source-wise raw-proposal or reduced-search criterion. Prioritize diagnosing teacher coverage, source-specific geometry and proposal concentration on development data before another fresh acquisition. Do not treat pooled gains or cheaper search alone as preserved performance. See the [research deep dive](RESEARCH_DEEP_DIVE.md) for the method, evidence and collision contract.

This static GitHub Pages report covers the fresh-machine shared-seed Kimodo motion qualification
study as of 2026-09-05. It does not pool outcomes with the earlier SweepCF or parallel learned
geometry-only hallucination track.

The [fresh-source audit](FRESH_SOURCE_V1_RESULT.md) now completes eight new source motions and 16 derivatives, with all seven methods frozen before acquisition. Learned pattern accepts **384/384**, uniform pattern **30/384**, original gradient **384/384** and the raw model **296/384**. Learned pattern improves over uniform on every source, but its prediction of higher acceptance than gradient fails because the counts tie. Probe/refinement retains one failure. The complete source-wise predicates and costs are retained. These are finite reference checks on CPU-generated straight walking, not physical execution or a collision guarantee.

The acquisition comparison is complete. Move the learning work toward [train-only search distillation](TRAIN_ONLY_DISTILLATION_DESIGN.md): test whether synthetic geometric targets improve raw proposals and reduce online query cost. Preserve all 43001–43008 sources and derivatives as excluded from training and selection. Another confirmatory transfer comparison requires a separately registered source pool. Develop imported-body and temporal clearance contracts alongside the learning work.

The earlier [fixed-budget station-search study](STATION_SEARCH_V1_RESULT.md) compares
three local alternatives on fresh draws. All reach 192/192 valid test outputs versus
187/192 for original gradient refinement, rescue five failures with zero paired losses,
and improve the separate known replay from 5/8 to 8/8. Gradient-free pattern search
uses 69.2% less measured search time. The [fresh-source acquisition](FRESH_SOURCE_V1_RESULT.md) follows this comparison; these earlier results remain
finite reference checks on observed development sources.

The [matched-query refinement study](REFINEMENT_V1_RESULT.md) adds a bounded
correction layer and a reusable checked sampler. Test acceptance rises from 123/192
(64.1%) raw to 189/192 (98.4%), versus 152/192 for ranking extra learned samples and
26/192 for uniformly initialized refinement. The hard case improves from 0/24 to
21/24, but a fixed fresh batch still rejects 3/8. The subsequent station study targets stalled station
proposals under a fixed query budget. These remain reference-only development results.

The [source diversity and unseen-phase study](SOURCE_PHASE_V1_RESULT.md) adds
80 targets from 16 sources and 12 fits. Doubling training sources raises unseen-location
yield from 49.5% to 62.0% at equal compute, or 64.1% with equal visits per motion.
One test source regresses, and the two six-source subsets differ sharply. The updated
plan prioritizes geometric feasibility/search and source composition before larger runs.
All 7,680 proposals receive independent reference checks; there is no execution guarantee.

The [paired timing follow-up](TIMING_DIAGNOSTIC_V1_RESULT.md) adds 12 separate development
executions and a complete held-out route measurement audit. It rejects the tested shared-clock
slowdown and identifies original-clock carrier 41002 for repeatability testing. No Q4 or learned
generator result is admitted. Its reproduction commands, evidence and hashes are separate from
the original visual-report bundle below.

The [repeatability and analytic-teacher follow-up](TEACHER_V1_RESULT.md) adds nine completed
physics runs, a new analytic capsule–box query, 27 finite-beam placement attempts, discrete jitter
checks and an intermediate-motion audit. The original-clock pair repeats at 3/3 seeds; the
intermediate level retains its required effect at 1/3. The beam remains a binary geometric
proposal and cannot establish ordinal minimality. No learning-eligible scenes are admitted.

The [learned-generator framework](LEARNED_GENERATOR_FRAMEWORK.md) reads the original LfH/LfLH
mechanisms against the humanoid problem and specifies a motion-only inverse objective, a first
model, distribution priors, collision-proof conditions, and discriminating experiments. It also
records a constructed counterexample to the archived sampled decoder's clearance claim. This is
a research design; it does not add trained-model or obstacle-present execution results.

The [first inverse-learning implementation and results](INVERSE_LEARNING_V1_RESULT.md) now add
12 CPU training runs with saved checkpoints, a motion-only mixture generator, loss ablations,
independent geometry checks, a sampling command, and distribution/validity plots. Preference
learning improves geometric yield on the selected development carrier; KL trades some yield
for broader placement coverage. Generalization and obstacle-present execution remain untested.

The [placement-uncertainty experiment](UNCERTAINTY_LEARNING_V1_RESULT.md) adds six CPU
training runs and checks all 768 comparator/new-model proposals at 113 placements.
Robustness improves in two of three seeds per KL family, but both regress on the third.
The rejection audit exposes a preference-loss/interference-margin mismatch. The
[updated research plan](NEXT_RESEARCH_PLAN.md) prioritizes an explicit interference
constraint, independent-carrier tests, and continuous geometry contracts.

The [explicit-margin experiment](MARGIN_LEARNING_V1_RESULT.md) adds twelve CPU runs.
Without KL, the preference model improves from 11/35/6 to 56/54/34 valid proposals out
of 64 across seeds; the KL setting still has a regression. A separate
[continuous-placement checker](PLACEMENT_CERTIFICATE_V1_RESULT.md) conditionally verifies
two selected scenes over their full translation/yaw domains. Numerical-error, imported
body geometry and between-frame assumptions still prevent physical certification.

The [grouped conditioning experiment](CARRIER_LEARNING_V1_RESULT.md) adds eight source
parents, 24 constructed early/middle/late crouch cases, nine generator runs and 36
per-motion searches. The conditioned model produces zero all-placement-valid proposals
on its two excluded test carriers and barely responds when the input event moves.
The [next model design](EVENT_CONDITIONED_GENERATOR_DESIGN.md) preserves event location
and specifies controls for the location prior introduced by that representation. This
is a reference-only development result; it does not qualify the original motion bank
or establish obstacle-present execution.

The [event and scale follow-up](EVENT_SCALING_V1_RESULT.md) adds 18 fits with two fixed
budget snapshots. Local features achieve 110/144 and 133/144 valid test proposals,
versus 2/144 and 4/144 for matched pooled features; swapping the event drops both to
zero. Doubling training parents helps in this comparison, while extra updates and
parameters do not consistently improve the small full-data model. The fixed fresh
sampler still accepts 0/8 on an early event, so rejection remains necessary. The
[data-scaling plan](DATA_SCALING_DESIGN.md) prioritizes source diversity and withheld
phases; [source notes](SCALING_SOURCE_NOTES.md) record the LfLH/LfH-CP recheck.

## Rebuild

Run from the repository root with the trusted local research bundle and standalone Motion2Scene
research repository available:

```bash
MUJOCO_GL=egl .venv_research/bin/python scripts/research/render_motion2scene_report.py \
  --data-root /path/to/research-data/groot-wbc \
  --research-repo /path/to/motion2scene
```

Dependencies: NumPy, MuJoCo, Matplotlib, Pillow, ImageIO/FFmpeg, and the repository's G1 meshes.
`--preview` renders only midpoint poster frames. Full generation produces three MP4 replays,
three posters, three SVG figures, portable evidence snapshots, and a hash manifest.

The first two videos replay recorded Isaac states. The third replays shared-clock reference CSVs
with a manually placed illustrative beam. No `mj_step` is called; none of these renders creates
a physics verdict. Existing recorder metadata supplies the verdicts for the first two videos.
The beam scene is not an admitted critical scene and must not be represented as a physical
preference-reversal result. Videos use a shared camera and elapsed time across panels.

`assets/manifest.json` records the selected source hashes and output hashes. Original absolute
machine paths are shortened in public JSON copies; source hashes and public snapshot hashes are
therefore separate. The full non-git motion bundle is needed to rebuild the media.

## Validation

```bash
.venv_research/bin/ruff check --select E,F,I scripts/research/render_motion2scene_report.py
.venv_research/bin/black --check scripts/research/render_motion2scene_report.py
git diff --check
.venv_research/bin/python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Browser validation: Chrome headless via Playwright, 1440×1000 desktop and 390×844 mobile;
no JavaScript errors or horizontal overflow. All three video sources decode and play; durations
are 3.967, 3.967 and 4.767 seconds, respectively. The synthetic beam slider is tested at both
endpoints (no feasible motion / neutral feasible). Public links, evidence hashes and media
metadata are checked before publication. No new controller experiment is performed by the original media renderer; the separate timing follow-up records its own physics runs.
