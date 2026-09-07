# M2S-ICRA-nominal-v1: a separately registered nominal placement contract

Registered September 7, 2026, after the M2S-ICRA-v1 acquisition funnel completed and
before any proposal, search or physics under this contract. M2S-ICRA-v1 is unchanged:
its assignments, refusals, labels, fits and evaluations stay exactly as recorded, and
this study neither replaces nor re-scores them.

## Why a second contract, and why it is not a relaxation

M2S-ICRA-v1 refused 47/48 analytic and 48/48 learned generated slots, nearly all for
`contrast_audit`. Two diagnostics locate the cause. The
[finite support map](TRANSITION_SUPPORT_DIAGNOSTIC_V1.md) found nominal witnesses on
all three carriers (28/19/43) but only 1/0/3 survivors under the inherited 113-offset
audit. The [envelope trade-off sweep](ENVELOPE_TRADEOFF_V1.md) measures how that
survival falls as the envelope grows.

The inherited offsets model uncertainty in **where the obstacle ends up**. That is the
right question for a placement certificate and for any hardware claim. It is not the
question this study asks: here the beam is authored into the simulator at an exact
pose, and the uncertainty that matters is in the **robot's execution**, which paired
physics measures directly and which two evaluation seeds already expose. Carrying a
placement-uncertainty envelope into a task whose placement is exact spends the entire
executed contrast window on a quantity that is identically zero.

So this is a different declared contract for a different question, registered before
its outcomes, not a knob turned until data appeared. The original contract keeps its
own results, and the 17-offset and 113-offset survivals are computed and stored for
every proposal here as reported diagnostics. They never gate acceptance. No result
from this study may be described as uncertainty-robust, and nothing here inherits the
earlier robust certificate.

## Frozen scope

Unchanged from M2S-ICRA-v1: SONIC, the 0.30 s decision, walk-commit versus request-d040,
the 3.3–3.5 s return, the 214-input two-head learner and its fixed-scale deterministic
regularized-linear form, the 214 features, the source banks, the twelve reserved
layouts, the physical scorer (≤1 N through body-origin crossing by 0.1 m with 0.3 s
upright stability), the frozen learned initializer, the frozen no-contrast checkpoint,
the analytic global distinct search, proposal seed 8841 and label physics seed 8722.

Changed, and only this: **acceptance geometry is evaluated at the nominal pose**, with
the same 10 mm target clearance and 10 mm walk interference.

Carriers 41001, 41002 and 41003, already inspected development ancestors. Sixteen
outputs per source and arm as before; **at most three eligible groups per source and
arm are assigned**, so the panel is bounded at 36 generated groups and 72 commands.
The six shared background groups from M2S-ICRA-v1 are **reused by reference with their
existing labels**; no background is re-executed and no reused trace counts as a new
example. Refusals are outcomes and are retained; there are no replacement proposals
and no refills.

## Predictions, before acquisition

1. Analytic and Motion2Scene each yield at least one eligible generated group per
   carrier under the nominal contract. Under the inherited contract they yielded one
   and zero in total.
2. At least one arm other than analytic acquires a walk-fail/d040-pass group. Only one
   such group exists in the whole of M2S-ICRA-v1.
3. Uniform's useful-contrast yield stays at most one across its assigned generated
   groups; its construction is unchanged and untargeted.
4. Target-only construction again yields predominantly both-pass groups.
5. Every arm whose fitted learner receives at least one useful contrast issues d040 at
   least once on the evaluation panel; every arm receiving none issues it zero times.
   In M2S-ICRA-v1 the three arms without a useful contrast were behaviourally identical
   and never adapted.
6. The majority of nominally accepted proposals fail the 113-offset diagnostic. If this
   fails, the two contracts are closer than the support map implies and the acquisition
   difference must not be attributed to the envelope.

Predictions 1–4 concern acquisition and are scored on this study's own funnel.
Predictions 5–6 concern the learner and diagnostics. A failed prediction is a result.

## Evaluation

The same twelve reserved layouts and three carriers at **physics seed 8511 only**:
36 traversal conditions for each of the four fitted arms, 144 executions. The scripted
upper/lower-ray rule and the privileged achieved-geometry predictor do not depend on
the training arm, so their existing M2S-ICRA-v1 measurements on exactly these
conditions are reused and reported as such, not re-executed. Comparisons against them
are therefore restricted to seed 8511.

Paired tables report both-pass, A-only, B-only and both-fail, paired by
(carrier, layout, seed). Passage is averaged within carrier before averaging carriers.
Conditions cluster inside three carriers: report the condition-level sign test and the
carrier-level agreement separately, and never present the condition count as an
independent sample size. Background adaptation and blocked-refusal suites remain the
separately reported M2S-ICRA-v1 suites and are not pooled into traversal success.

## Budget and stopping

Serial physics, unchanged 7500 MiB free-memory floor, 375 s cell timeout, unchanged
rolling 8 GPU-h/day and 24 GPU-h/week envelope. At most 72 label commands plus 144
evaluation executions: 216 cells, 22.5 reserved GPU-h at the standing ceiling and
about 1.9 h at the measured rate. This study launches only after M2S-ICRA-v1's own
evaluation completes; it must never compete with it for the same card. A failed cell,
changed reference hash or rejected measurement audit stops the study for review.

## What a result here can and cannot support

A larger accepted yield under this contract measures **geometric acceptance**, not
usefulness. Only the paired physical labels decide whether an accepted group is a
useful contrast, and only the evaluation decides whether it teaches. Three inspected
development carriers, one beam family, two commands and ideal rays remain the scope.
Nothing here is source-held-out transfer, a hardware claim, or evidence about the
learned generator's value at any other envelope.
