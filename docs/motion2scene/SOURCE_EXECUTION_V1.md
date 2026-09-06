# Frozen learned beams: eight-source execution pilot

Registered 2026-09-06 before new proposal draws, conversion or physics. The selected
41002 analytic beam establishes contact-free binary separation for one development
pair. The missing evidence is whether the learned reference pipeline produces useful
constraints across source motions with the release controller fixed.

This bounded pilot uses all eight previously observed development sources 41005–41008
and 42005–42008, with the existing event3 target (station 0.35, local_crouch 55 mm,
original clock). Event3 is the first declared unseen event and leaves downstream
stabilization room. These are existing reference-development sources, not freshly
acquired motions. Do not load 430xx or change earlier training/eligibility labels.

## Frozen procedure and information

Use original width-32 local-event checkpoint, fitting seed 8421, all8/1200. Draw eight
proposals per case with RNG seed 310000 + source ID, then the unchanged 17-evaluation
pattern search and independent 113-offset audit. The model receives target features;
search receives target and declared upright reference geometry. Neither receives achieved
trajectories, runtime contacts or event IDs. Before independent audit, retain the first
three numerically distinct output placements in proposal order (distance >1e-9 m in
station/height coordinates). Audit failures and missing distinct outputs become
scene refusals; never refill them. Retain all eight outputs and charge their full cost.
Freeze all scene coordinates and source references before any new execution outcome.

Convert both references with the existing Kimodo-to-SONIC adapter at 30 Hz, canonical
horizontal origin and zero scene yaw, without retiming or modifying poses. Reference
Q0/Q1 is an eligibility check for this diagnostic, not execution qualification.

## Execution and denominators

Run both motions for every source in the beam-absent scene at seed 7921: **16 runs**,
with no dependency skips. Use the unchanged beam sensor/recorder, release controller,
50 Hz first-episode passage scorer, 1 N force threshold, body-origin downstream margin
0.10 m, root height/alignment checks and 0.30 s stabilization. Retain the full tracker
verdict separately. The canonical station-0.35 plane is used for source qualification;
also score each frozen candidate plane in the same world frame.

A source pair qualifies when both first episodes pass the canonical passage and both
full-sequence tracker verdicts accept. No minimum semantic depth is added to binary
separation. d040 is not tested here, so no minimal-depth claim follows. No observed
empty-scene trajectory is used to move a beam or select a new candidate.

For each qualified pair, run both motions at every independently passing selected
placement using the same seed 7921: at most **48 present runs**. The subset is a
deterministic application of this rule to the complete absent batch; pin that result
when producing its manifest. Every source retains three requested scene slots in the
final 24-slot denominator, including source-qualification and geometric refusals.
Present-run separation requires target contact-free passage, upright prohibited beam
contact, and both absent motions passing at that candidate's plane. Report all four
rates, crossing/contact/stabilization/fall flags, and source-group outcomes.

This is a one-physics-seed source pilot preceding the proposed three-seed/192-run study.
Do not automatically expand repetitions or acquire replacement sources after a failure.
Existing positive/negative contact controls are hash-checked prerequisites; unchanged
instrumentation and collider settings are reused. Inventory every imported scene.

## Predictions and branch decision

- P1: at least 4/8 source pairs qualify in the absent scene.
- P2: at least half of qualified sources (and at least one) separate on all three
  requested beam placements; geometric refusals count against this predicate.
- P3: at least 12/24 requested source-scene slots achieve executed binary separation.

These are development decision thresholds, not power calculations. If P1 fails, treat
motion execution as the immediate bottleneck and defer additional scene-generator fitting
for execution yield. If P1 passes but P2/P3 fail, distinguish target beam contact from
missing upright interference, first-episode failure, reference refusal and native/proxy
mismatch before changing supervision. A successful pilot enables preregistration of
additional physics seeds; it does not establish arbitrary-motion generality.

## Cost, scope and intended contribution

Two CPU threads, no GPU fitting. Proposal preprocessing/search/audit capped at 600 s;
8 × 6,884 = 55,072 whole-motion geometry queries, including crosschecks. Charge model
loading, conversion and inherited offline fitting separately. Physics remains serial,
7500 MiB startup floor, 375 s per cell, stop on infrastructure failure, no scientific
retries. At most 64 runs give a 6.667 contended GPU-hour ceiling; charge actual usage
against the standing daily/weekly envelope before launching conditional runs. Yield
under contention and never stop another workload. Outputs use a fresh source-execution
lineage; all previous frozen artifacts remain unchanged.

The proposed evidence contribution is an eight-source execution funnel for a fixed
learned constraint procedure, with 24 scene requests and explicit refusals. This tests
the link from proxy geometry to fixed-controller behavior; it is not a new architecture
or a comparison of learned and analytic physical execution yield. CPU coverage-aware
supervision remains a coordinated separate experiment with its own cost-matched controls.
