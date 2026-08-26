# E5 Learner Probe Design

**Status:** design only; run nothing. E5 is post-claim-5 and does not alter the frozen selector
analysis, training inputs, or 30-scene test set.

## Question

At equal physics and training budgets, does trajectory-conditioned critical sampling help a
behavior selector recover binding geometry better than uniform-feasible placement, DCS ranking,
or visual multiplication? E5 measures learning value; scene count and DCS occupancy are not
surrogate outcomes.

## Readiness Gate

Do not run E5 with the current evidence. E7/E7c give every source three verified archetypes, but
that count-only gate is insufficient: only `shelf_plank` and `ibeam` are verified for all four
sources. A held-out source × archetype comparison requires at least three **common** archetypes and
a complete crossed source–archetype matrix. E8 should transfer `hanging_panel` to `cf_005_056`,
089, and 090. Every included episode must pass keep-out, binding attribution, contact-before-drift,
and provenance gates. Generated data remains excluded from claim 5 unless the user makes a
separate explicit scope decision.

## Controlled Comparison

Starting from the completed claim-5 behavior-selector recipe, train equal-size corpora selected by:

1. **Critical support:** source-balanced `q_LFH` over route phase, finite extent, binding keypoint,
   paired margins, and normalized window position.
2. **Uniform feasible:** uniform sampling from the same deterministic geometry/keep-out support.
3. **DCS-ranked feasible:** current absolute-bin ranking, after the same feasibility gate.
4. **Visual-only:** additional archetypes at fixed canonical critical geometry.

Hold physics episodes, motion sources, optimizer, steps, rendering budget, candidate ordering, and
five training seeds fixed. Balance source and cell-role weights so variants do not duplicate a
source's label mass. Split jointly by causal source and archetype: no source or rendered room may
cross a held-out-source split, and the archetype under evaluation is absent from training.

## Measurements

Primary metrics are the existing counterfactual choice accuracy, unsafe-choice rate,
unnecessary-adaptation rate, and realized success minus adaptation cost on held-out archetypes.
Report per-source values and paired seed differences with intervals. Add diagnostic auxiliary heads
for binding-keypoint classification and signed-margin regression, plus scene-hidden and
wrong-scene controls. Auxiliary heads explain representation; they never replace physics outcomes.

Before training, register the minimum worthwhile effect and failure interpretation after a power
calculation on the attained source count. A null result is retained: it would show that the tested
critical proposal distribution did not improve geometry use at the available scale.
