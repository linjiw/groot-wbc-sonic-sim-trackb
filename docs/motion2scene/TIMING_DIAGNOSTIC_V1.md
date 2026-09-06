# Paired timing diagnostic v1

Registered 2026-09-05 before execution. Development-only performance experiment.

The held-out route result remains 5/7 retained tracker survivors, below its registered 80%
criterion. The original shared-clock Q3 pilot remains gated and unlaunched. This separate
experiment tests controller timing without admitting any motion to Q3/Q4 or learning data.
It does not revise route tolerances or validate the failed route instrument.

## Design and predictions

Use carriers 41001, 41002 and 41003 (first three IDs, development pool), with original and
shared-clock versions of neutral and d055: 12 registered cells, three independent units.
Use existing references, the same frozen SONIC controller, screen_empty and physics seed 7900.
The shared-clock map was derived from d085 activity with minimum pace 0.7; it is applied to
both tested levels. Retiming includes interpolation and changed reference velocities. This
tests that implemented timing intervention, not a pure playback-rate parameter.

The original and retimed d055 must share the source CSV hash. Each condition's crouch depends
on its own neutral's tracker acceptance. Alternate condition order by carrier parity; neutral
precedes crouch. No replacements or scientific retries. Report all 12 cells, including skips.

Predictions, adjudicated only after the entire batch finishes:

1. Shared-clock d055 reduces the tracker's endpoint-error diagnostic in at least two of three
   carriers. A missing paired execution supplies no evidence for improvement.
2. More shared-clock d055 cells pass tracker acceptance than original d055 cells.
3. At least two of three shared-clock neutral controls pass tracker acceptance.

Report paired endpoint-error differences, tracker reasons, schedule-error diagnostics, the
unchanged relative-route metrics, and achieved paired height/event retention. Use the existing
50 mm effect, 60% magnitude retention and 30% event-IoU criteria descriptively. Do not infer a
population improvement from three development carriers or pool with historical physics seeds.
Reduced endpoint error alone is insufficient if crouching or route accuracy worsens.

## Execution and provenance

Hash-pin the predictions, candidate sources, references, conversion provenance, checkpoint,
rollout and manifest drivers, diagnostic script and standalone motion-analysis modules before
launch. Fresh outputs live outside the frozen source bundles. Analysis verifies trajectory
hashes and refuses partial batches. Retain the original acceptance implementation and all
historical outcomes.

At most 12 serial, trajectory-only runs; require 9000 MiB free VRAM at each startup. Each run has
a 375-second timeout, bounding total runtime to 1.25 contended GPU-hours. Yield on contention;
stop on infrastructure failure. No other GPU workload is stopped. The standing research budget
is 8 contended GPU-hours/day and 24/week.

## Decision and connection to learned scenes

If the intervention improves tracking while preserving the motion effect, confirm on fresh
carriers and three physics seeds using a separately registered protocol. If it fails, diagnose
the measured tracking error and reference timing before scaling motion acquisition. Keep
route-estimator calibration separate from scene-clearance measurements.

The intended learning target is a motion-conditioned distribution over obstacle type, pose,
size and event position, initially an analytic LfH teacher followed by an LfLH model. First
establish executed target-versus-weaker preference reversal, removal controls, jitter robustness
and a qualified motion ladder. Begin with overhead beams; later add lateral gaps and step-over
obstacles once their motions and predicates qualify. Household/factory context is nuisance
variation around a verified critical obstacle, and must be rechecked for collisions.

Keep every derivative, obstacle variant and physics repeat of a carrier in one dataset split.
Record refusals, failures, exact-clearance margins, controller provenance and both alternatives'
outcomes. Compare learned placement against analytic and random placement on held-out carriers;
evaluate scene diversity and downstream scene-to-motion policy utility. This diagnostic produces
controller evidence, not positive training labels or a learned-generator result.
