# Post-M8 decision: what the completed comparison changes

**2026-09-10.** The five-arm M8 common-set comparison is complete and the executed-envelope screen
has been calibrated against measured outcomes at zero physics cost. Together they answer the
[compass](RESEARCH_COMPASS.md)'s four questions unevenly, close one standing hypothesis, and
redirect the next experiment. This document records the decision. It runs nothing.

Sources: [M8 statistics](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-statistics-20260910-v1/result.json)
· [rendered panel](/home/linjiw/research-data/groot-wbc/m2s-M8-native-development-statistics-20260910-v1/report.md)
· [predictor calibration](/home/linjiw/research-data/groot-wbc/m2s-envelope-predictor-calibration-20260910-v1/result.json)

## 1. Where each method component now stands

| Compass question | Component | Status |
|---|---|---|
| Why condition construction on actual execution? | Executed envelopes + physical verification | Acquisition mechanism **holds**; downstream effect **null** |
| Why should the resulting data teach better decisions? | Contrast-targeted encounters | **Unsupported** |
| Why does commitment timing need special teaching? | Complete-continuation targets | **Untested** — common teacher across all five arms |
| Does prioritizing encounters add value? | Historical replay | **Null**, cleanly measured |

Executed contrast does not beat feasibility-screened uniform: passage difference 0 and paired time
difference 0.000 in all three corpora, same single failing context. Replay is identical to executed
contrast in all 18 cells, down to the selected schedule. Executed contrast beats only target-only,
the weakest arm.

The mechanism is response collapse: 13 of 15 learned policies emit one schedule on all six contexts,
and none passes anywhere the seven-schedule bank fails.

## 2. The correspondence, stated precisely

The two corpora that no single fixed schedule can cover are exactly the two whose policies use more
than one schedule; all thirteen single-schedule-coverable corpora trained constant policies. Fisher
exact **p ≈ 0.0095**.

Two qualifications belong with that number, and both matter.

The 15 corpora are 5 arms × 3 seeds over 6 shared contexts and one execution seed, so they are not 15
independent draws, and the hypothesis was formed after seeing the table.

**Corpus complementarity is necessary but not sufficient.** `seed93203_target_only` is one of the two
non-constant policies — it uses three schedules — yet on `complementary_late_development_0100`, the
one context that discriminates, it selects `short_e070_r265` and fails, where the bank's passing
schedules are the two sustained ones. It scores 5/6, the same as the collapsed policies. So
non-constancy is not the outcome. Correct selection is. Any success criterion phrased as "the policy
stops being constant" would have scored that corpus a win for no gain.

## 3. The negative-margin hypothesis is closed

The standing proposal was to relax the constructor's targeted interference threshold, on the evidence
of one acquired scene whose priors physically failed at only −2.65/−4.02 mm predicted interference.
It was recorded as *"unimplemented and untested; reconsider only after the M8 development gate."*

The gate has now been passed, and the calibration answers it. Over the 92 already-executed
encounters, against the measured property (every covering schedule fails, some other schedule
passes):

| Negative margin | Screen fires | Recall of the measured property |
|---|---:|---:|
| ≤ −20 mm | 1 | 1/7 |
| ≤ −10 mm (current) | 2 | **2/7** |
| ≤ −5 mm | 2 | 2/7 |
| ≤ −2 mm | 3 | 3/7 |
| ≤ 0 mm | 4 | **3/7** |

Precision is 2/2. Relaxing the threshold all the way to zero interference buys one additional
encounter. The binding constraint is **recall, not threshold**:

- `primary_candidate_93203_0148` (+15.80/+14.46 mm) and `primary_candidate_93203_0847`
  (+3.70/+2.70 mm) have covering schedules predicted **clear** that physically **failed**. No
  negative margin can ever reach them.
- `primary_candidate_93201_0417` is rejected by the **positive** robustness bar, not the negative
  one — and it is the round-8 encounter in `seed93201_reference_contrast`, the corpus that produced
  the only 6/6 learned policy in the study.

The two encounters the screen does select landed in `seed93201_target_only` and
`seed93202_target_only` — the two corpora that collapsed to a constant sustained policy at 4/6, the
worst learned scores on the panel.

So the geometric screen is a high-precision, low-recall proxy. That is exactly the property that
would produce the M8 null: it concentrates labeling on *predicted* contrast, and predicted contrast
is not the complementarity that produces a selective policy.

## 3a. What the replay null actually is

Replay is not a soft tie, and it is not a no-op on the fit. Comparing the M8 checkpoints of the
paired arms directly:

- `mean` and `std` are **bit-identical** across all three seeds, confirming the two arms saw the same
  supervised rows with the same feature values — the geometry and queues really are shared.
- The fitted parameters **do** differ: `weights` by up to 0.019 and `bias` by up to 0.26. Replay
  moved the ridge solution.
- Every decision is nevertheless unchanged: the per-phase argmax is identical for all three seeds,
  and the panel shows the identical selected schedule in all 18 executed cells.

So the honest statement is that the reweighting perturbed the fit by less than the margin separating
the options it was choosing between.

One tempting stronger explanation does **not** hold and should not be published. It is not the case
that the heads collapse to exact constants (`W = 0`) so that any positive weighting gives the same
solution: the largest linear coefficient is 0.027, not zero, and at ±3 standardized deviations the
linear term's swing (0.41–0.84 across phases) **exceeds** the bias gap between the top two qualified
options (0.01–0.44). The learner therefore has enough range to condition on features; on these six
contexts it simply does not travel far enough from the feature mean to flip an argmax. The observed
constancy is a property of what these corpora taught, not a structural impossibility of the ridge
head — which is consistent with, and not independent of, the corpus-complementarity account above.

## 4. What is registered next, and what is refused

**Registered direction: select on measured response, not predicted geometry.** The 92 executed
encounters already carry complete seven-branch outcome tables. A corpus-conditioned selector that
chooses the next encounter from encounters whose *measured* responses differ from those already
covering the corpus takes the screen's 29% recall off the critical path entirely. This is the
smallest change that addresses the measured cause, and it stays inside the compass's scope: same
tracker, same repertoire, same three decision phases, same teacher, same learner.

Before any physics is spent, three things must be settled, because adversarial review of the obvious
design refuted it on all three:

1. **The manipulation check is not the result.** Appending one encounter that defeats the covering
   set flips `minimum_cover_size` from 1 to 2 in all three corpora *by arithmetic*. That may be
   reported as confirmation the selector fired; it can never be the success criterion. The compass
   forbids an objective that becomes "maximizing the number of different labels", and this would be
   exactly that.
2. **The evaluation context must not be minted by the intervention's own predicate.**
   `complementary_late_development_0100` — the single context on this panel that discriminates — was
   generated by `motion2scene_complementary_late_screen.py` using the same clearance predicate and
   the same max-min-slack ranking a predicate-driven selector would use. Selecting training scenes by
   the rule that manufactured the only movable test item supports nothing. Confirmation must come
   from contexts minted by a different rule, or from the reserved lock.
3. **The control must vary one thing.** The obvious comparator — the frozen plan's next rounds — is
   `short_then_short`/`short_then_sustained` in all three corpora while every eligible targeted
   candidate is `short`/`sustained`, so the arms would share no scene family and would differ in
   positive schedule, negative schedule, stratum and ranking rule at once. That repeats precisely the
   confound the compass already flags for target-only versus executed contrast.

**Refused:** the "corpus-conditioned selection against the covering set" framing, as a headline. It
is a verified no-op on this pool — across all 3,840 proposals, every candidate whose
`prior_splice_e015_r265` clearance is ≤ −10 mm also has `prior_splice_e050_r265` ≤ −10 mm
(2070 of 2070), so a covering-set predicate selects the identical candidates as the single-schedule
one. Keep the edit as a correctness fix; do not sell it as the mechanism.

**Also refused:** continuing the declared M16/M32 acquisition trajectory as the response to this
result. It extends the same corpora under the same constructor, and 13 of 15 of those corpora train a
constant policy. The trajectory stays preserved and unchanged; it is simply not the experiment this
result calls for.

## 5. Consequences for the paper

The contribution that survives is narrower and more mechanistic than the one the manuscript currently
promises, and it is still a real one:

> Executed-motion construction reliably concentrates physical labeling on solvable
> adaptation-required encounters. That is not sufficient to teach perceptive selection: a corpus of
> per-encounter contrasts that one schedule can cover trains a constant policy. What distinguishes
> the corpora that trained selective policies is response complementarity across the corpus, and the
> executed-envelope screen predicts that property with high precision but low recall.

Replay should be reported as a tested component with a null result, per the compass's *"The paper
should survive a null replay result."* Complete-continuation teaching remains untested by this
experiment and must not be credited by it; a claim about it needs its own control.

The claim to avoid is the one the compass warns against — reporting the acquisition-stage yield
advantage as though it were the downstream benefit. M8 measured that benefit directly, and it is
zero against the uniform baseline.
