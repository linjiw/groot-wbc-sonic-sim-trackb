# Selector breakpoint study v1: post hoc development diagnostics

The 120/600 original evaluation remains frozen. Its other 480 assigned runs are
operationally paused while this user-prioritized study diagnoses zero d040 requests.
No v1 input, checkpoint, scorer, layout, threshold or result is changed. These six
layout/seed conditions are now development evidence for any revised method.

## P0: recorded-input diagnosis (CPU only)

Before analysis, pin the 38 paired corpus records, the fixed eleven-example subsets,
twenty original checkpoints and 120 measured evaluation inputs. Report all 214
features' training standard deviations/ranges, stored scales, and evaluation
normalized displacements. The actual code clamps scales to 1e-3; the user’s 1e-6
example is illustrative, not a measured implementation value. Report hidden-layer
ranges, both logits, saturated probabilities and largest first-layer contributions
by feature group. Attribute no physical outcome to synthetic input interventions.

For each evaluation row and each of its eleven training examples, replace one group
at a time: rays [0:144], phase/skill/age [144:147], gravity [147:150], linear/angular
velocity [150:156], joint position [156:185], joint velocity [185:214], and all
non-ray values [144:214]. Preserve the remaining evaluation values. Report every
donor, request/refusal changes, logit changes and recovered d040 requests. Also test
the complementary all-state donor plus evaluation rays. No donor is selected by
its evaluation outcome. This is post hoc mechanism evidence, not a policy score.

Compute exact-observation groups on all 38 pairs and separately on each fitting
subset. Report outcome-label ambiguity, empirical observation-only action ceiling,
paired-action ceiling, and their difference. Compute the attainable masked BCE
infimum from each group's observed label frequencies; deterministic groups have
zero infimum. Do not interpret action-neutral outcome ambiguity as forced action
error, or a finite empirical ceiling as a population bound.

Fit four low-capacity development controls (one per original data arm), using only
the same fixed eleven examples, not the six diagnostic physics labels. Use a two-head
linear logistic predictor, fixed physical scales (ray/phase/gravity 1, velocities
1 m/s or rad/s, joint position 1 rad, joint velocity 5 rad/s), no mean subtraction,
zero initialization, full-batch Adam 0.01, 2000 updates and 0.01 mean squared weight
penalty. Keep the 0.5 threshold, walk preference and refusal semantics. This is a
shared learner control, not a generator result or a deployed policy. CPU limit 300 s;
no hyperparameter search, v1 refitting or inference-only normalization repair.

Hypothesis: narrow training-state variation makes evaluation state dominate original
logits. It is supported only to the extent measured group interventions restore
predictions; ray sensitivity and nonlinear interactions remain reported alternatives.

## P1: twelve new matched command executions

Source 41002; original first route station 0.35; heights 1.18/1.27/1.36 m; physics
seeds 8511/8512; both walk-commit and request-d040 = 12 new Isaac Lab executions.
Use the existing explicit command executor, frozen neutral/d040 bank, 0.30 s
decision, legal phase/jump guard, 3.3 s return request and unchanged passage scorer.
No learned model or scene-dependent rule selects these forced commands.

Freeze the manifest, code dependency closure, source block references and audit
registration before preflight/spend. Serial runs; inherited 7500 MiB free-memory
floor and 375 s cell timeout; reserve at most 1.25 contended GPU h. Check the standing
8 h/day and 24 h/week envelope using the latest-activity ledger before launch/resume.
No automatic retry of started/failed cells; no replacement of scientific failures.

Admission requires exact paired pre-decision state/action/token histories and captured
features, direct state/ray/bank audits, correct physical beam placement, synchronized
200 Hz contact, and legal intended command execution. Preserve and report any failed
predicate. Compare each new walking execution with its pinned historical walking
trace; exact prefix/features and the first-episode outcome are the registered
repeatability prediction. Full-trace differences are reported separately.

The task criterion remains measured force <=1 N, all recorded body origins crossing
by 0.1 m, and 0.3 s upright first-episode stability. Entry, return, reset and later
recovery remain separate; this is not a native-extent or continuous-time guarantee.
No old walking trace is counted among the twelve new executions.

Scientific prediction to test, not an admission requirement: at least one failed
walking condition is rescued by the legal d040 command. Report all four outcome
pairs, the six-condition paired-action ceiling, and whether seed 8512 at 1.27 m is
rescued. A failure to find a rescue is a useful negative result, not permission to
change the beam or command time.

## Decision after P0/P1

Use recorded empty-scene transitions and raw sensor visibility to design P2, giving
execution-aware analytic and learned construction the same information. Any shared
learner/sensor repair is a separately frozen version and applies to every arm.
Do not resume a large v1 allocation or start 24/48/96 acquisition before these
findings are reconciled. The proposed contribution remains a hypothesis: executable,
observable motion contrasts improve a fixed learner against execution-aware analytic
data. Final source candidates 9490001–9490008 still require acquisition and switching
qualification. No hardware or safe-stop claim is added.
