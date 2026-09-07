# Three development banks and the first transition-aware construction pilot

**September 7 result:** twelve new Isaac Lab executions qualify carriers 41001,
41002 and 41003 under the same 0.30 s command contract in two physics seeds. The
next construction pilot admits only **1/12 assigned scene slots**, from analytic
construction on 41001. Four obstacle-present commands on that scene establish
observable walk-fail/d040-pass contrasts in both seeds. No Motion2Scene slot survives.
The prediction that both methods yield a useful scene on every carrier fails.

This extends the declared switching bank beyond 41002 and physically validates one
achieved-transition analytic proposal on another development carrier. It does not
establish the learned generator's advantage, a complete training corpus or source-held-out
transfer. The original twenty learners and 120/600 evaluation remain unchanged;
480 original assignments are still paused.

## Qualification and complete requested funnel

| Source | Empty command pairs qualified | Analytic eligible slots | Motion2Scene eligible slots |
| --- | --- | --- | --- |
| 41001 | 2/2 | 1/2 | 0/2 |
| 41002 | 2/2 | 0/2 | 0/2 |
| 41003 | 2/2 | 0/2 | 0/2 |

All 12 empty-scene runs and six command pairs pass the direct-input/source-bank
measurement audit. All six d040 runs enter at 0.30 s and return legally; all twelve
have no recorded fall or reset. Qualification uses passage across a virtual plane
at source-relative station 0.50 and 0.3 s upright stability. It is not a claim that
every reference-tracking metric passes. The beam is physically disabled in these runs.

Each source's loaded bank is checked against its own CSV/PKL and repeated bitwise
across its four runs. Root XY tolerance is 1e-4 m; joint interpolation tolerance is
0.002 rad, explicitly accounting for the loader's float32 quaternion path. The
old 41002-only auditor is untouched. The new bank inputs are development data;
41001/41003 were selected before outcomes, and no reserved final ancestor was used.
[Bank protocol](DEVELOPMENT_TRANSITION_BANK_V1.md) ·
[full qualification record](evidence/transition-stage-20260907/bank-result.json).

Both construction arms receive the same full achieved seed-8721 walk/d040 trajectories.
The frozen learned initializer still sees its original reference representation;
only the search/audit evaluator changes to achieved geometry. Analytic initialization
uses achieved envelope heights and the existing global search. This is a deliberately
bounded evaluator-replacement experiment, not retraining the generator on transitions.

There are 96 retained outputs (16 per arm/carrier), but only the first two slots
per group are assigned physics eligibility. All six assigned learned slots fail the
achieved-geometry audit. Analytic 41001 slot 1 and both 41002 slots fail that audit;
both assigned 41003 slots intersect the reserved layout neighborhoods. Some slots
also fail the visibility proposal check. No refill, relaxed margin or moved beam is
used to create the one surviving placement. The unassigned outputs and every reason
remain in [the full proposal record](evidence/transition-stage-20260907/construction-proposals.json).

## All four obstacle-present executions

The surviving analytic scene on 41001 is station 0.68, underside 1.275515 m, with
0.1 × 1.2 × 0.1 m beam dimensions. It is generated from the achieved empty transition,
not selected after obstacle-present outcomes. Both seeds execute both commands.

| Physics seed | Walk task | Walk peak (N) | d040 task / peak (N) | d040 return |
| --- | --- | --- | --- | --- |
| 8721 | Fail | 1635.077 | Pass / 0.000 | True |
| 8722 | Fail | 1130.936 | Pass / 0.000 | True |

All four records pass paired-feature, state, ray-origin, source-bank, beam-placement
and synchronized 200 Hz contact audits. Physical packets show beam hits with free
space below in all four runs. Their pre-decision recorded state/action/token histories
and non-ray features exactly match the corresponding empty-scene records; the rays
change. These are matched forced commands, **not a new learned-selector evaluation**.

Both d040 executions pass the frozen contact-qualified body-origin crossing criterion,
remain upright without a reset or recorded fall, and log a return. Both nevertheless
receive the separate tracker's **reference_endpoint_tracking_error** rejection. Keep
that failure: contact-qualified passage is not exact reference completion. Walking
receives disallowed_robot_contact rejection in both seeds. Neither walker resets or
falls in the capture. The reported task criterion is <=1 N force through passage,
0.1 m body-origin crossing and 0.3 s first-episode stability; native-extent guarantees
and later recovery remain distinct.

Useful observable contrasts are 2/2 measured scene/seed pairs, but only **1/12 requested
scene slots** overall (analytic 1/6, Motion2Scene 0/6). Repeated physics seeds are not
new source ancestors or new geometrically distinct scenes. The generic batch scorer's
`source_qualified` field means both commands pass all its encounters; its false value
for this physical batch is expected from the failed walking comparator. Empty-bank
qualification is recorded separately above.
[Physical result](evidence/transition-stage-20260907/physics-result.json) ·
[admission](evidence/transition-stage-20260907/physics-admission.json) ·
[empty-comparator check](evidence/transition-stage-20260907/empty-comparator-check.json).

## Why the narrow proposal study mostly refuses

A separately registered post hoc map holds the achieved bank fixed and evaluates
1,420 station/height centres per source. These are sampled witnesses, not feasible
volume estimates or proof that other placements cannot work.

| Carrier | Nominal witnesses | 17-offset witnesses | 113-offset witnesses | Nominal / robust after visibility, reservation and pre-decision checks |
| --- | --- | --- | --- | --- |
| 41001 | 28 | 1 | 1 | 21 / 1 |
| 41002 | 19 | 0 | 0 | 14 / 0 |
| 41003 | 43 | 3 | 3 | 32 / 0 |

The old perturbation requirement sharply reduces the sampled support: 67 nominal
centres survive the basic/visibility/reservation checks, versus one robust centre.
For 41002 the best search-offset slack is −2.851 mm; this grid finds no robust witness.
All three robust centres on 41003 are excluded by the remaining checks. These facts
do not establish continuous infeasibility or justify altering the current assignments.

The original learned draws' unchanged ±0.05 station / ±0.03 m height trust boxes
cover only 17/90 nominal witness centres across the three sources (14/67 after the
other checks), and none of the four robust centres. Thus both geometric support and
initializer reach are implicated. More identical bounded searches cannot reach those
measured robust centres. This finding does not prove the optimizer would succeed
on every reachable centre.
[Diagnostic protocol](TRANSITION_SUPPORT_DIAGNOSTIC_V1.md) ·
[all maps and slacks](evidence/transition-stage-20260907/support-result.json).

## Cost, preserved failure and next decision

New physics: 12 empty-bank executions (0.105115 contended GPU h)
plus four physical scene executions (0.033405 h). Preparation
costs and every query are recorded per arm; analytic uses 4,624 search clearance
queries per carrier, learned pattern search 9,248. Each arm also incurs independent
verification, pre-decision and visibility work. This is not an equal-generator-compute
claim or the full acquisition-cost comparison. The finite-map diagnosis took
67.91 CPU wall seconds and produced no new physical labels.

The first bank preparation failed before a manifest or physics: 41001's root
reconstruction differed by 4.768e-7 m, above the initial 1e-7 tolerance. Its script,
protocol and failure record are preserved; the prelaunch revision declares 1e-6 m
for root conversion and retains 1e-7 for other fields. No motion or physical success
threshold changed. All prelaunch and scientific failures remain published.

**Next:** keep this robust pilot frozen. Register a separate nominal-task development
acquisition study, with perturbation performance reported as a separate measurement.
Use transition-conditioned teacher support to test a revised learned initializer,
against the equally informed analytic method, on the same three-carrier bank. Do not
relabel the 67 geometric witnesses as successful physical examples. Complete physical
pairs and related absent/raised controls before the next selector fit. A fixed common
learner and new source/layout split must precede the final five-arm 24/48/96 experiment.
If the learned proposal still provides no benefit against analytic construction, retain
that conclusion and test cost or a richer task family rather than weakening the baseline.

[All four execution replays](assets/transition-construction.mp4) use Isaac states and
forces rendered in MuJoCo with mj_forward only. The room is omitted and visual meshes
differ from Isaac colliders. No learned-policy or additional MuJoCo dynamics result
is implied. [Source archive](evidence/transition-stage-source-20260907/research-source.tar.gz) ·
[export hashes](evidence/transition-stage-20260907/exports.json).
