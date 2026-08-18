# How many counterfactual families this corpus can actually supply

The plan asks for five verified families per geometry regime. Mining the corpus for pairs
says the overhead regime can currently supply **two**, and the reason is not the method.

## What the corpus offers

| | count |
|---|---|
| Accepted episodes in `g1_motionbank_v0.4` | 132 |
| — carrying no per-body geometry | 53 |
| — trajectory duplicates of another accepted episode | 7 |
| **Distinct episodes that can be mined** | **72** |
| Compatible overhead pairs among them | 55 |
| Pairs with a station-refined window above 0.05 m | 5 |
| **Distinct adapted motions across those pairs** | **2** |

The last row is the binding constraint. Four of the five viable pairs use the same adapted
motion, `clutter_061`; the fifth uses `clutter_056`. Five families built from this would not
be five independent families — they would be one duck motion measured against four different
walks, and a held-out split over them would leak immediately.

Two other numbers are worth stating plainly rather than leaving inside a script. **53 of the
132 accepted episodes carry no per-body geometry at all**, because they were recorded before
the recorder captured it. They remain perfectly good tracking episodes and nothing about
their acceptance changes, but they cannot enter any swept-volume computation, so the corpus
that can be mined is 72 rather than 132. And deduplication matters more than it looks: before
it was added, the ranking returned `density_dense`, `density_moderate`, `density_sparse` and
`density_tight` as four separate candidates with an identical 0.088 m screen and 0.127 m
refined window, because they are one trajectory replayed through four clutter scenes.

## Why the ceiling is two

Because the generator rarely produces a duck. The semantic predicates scored `duck_under` at
**3 of 7**, with the four failures dropping the torso 0.042–0.053 m — indistinguishable from
walking. Only motions that genuinely duck can serve as the adapted half of an overhead pair,
so the supply of adapted motions is the supply of semantically valid ducks.

That closes a chain across two workstreams that were running separately:

```
generator returns a walk for a duck prompt   (semantic validity: 3/7)
        v
few genuine duck motions in the corpus
        v
only 2 usable adapted motions for the overhead regime
        v
family count is generator-limited, not method-limited
```

The same chain predicts the floor regime is worse, and it is: `step_over` is **0 of 7**
semantically valid, with a trailing-foot apex statistically indistinguishable from walking
(Mann-Whitney p = 0.632). There is no adapted motion for a floor family anywhere in the
corpus, which is why that regime cannot start at all rather than merely starting small.

## What this changes in the plan

The prompt-taxonomy feedback loop was ranked P1 as a corpus-quality improvement. It is
actually a **prerequisite for the family scaling in P1**, because family count is bounded by
the number of semantically valid adapted motions and that number is currently 2 for overhead
and 0 for floor.

It is also the cheapest thing on the list. The predicates grade a *generated reference*
without any rollout, so testing whether a rephrased prompt produces a real duck costs seconds
rather than the six GPU rollouts a family costs. Measuring which phrasings work should come
before spending rollouts on families the corpus cannot yet support.

## Mining all three regimes: three disjoint families exist, against a target of fifteen

| Regime | Viable pairs | Distinct adapted | **Mutually disjoint families** |
|---|---|---|---|
| overhead | 5 | 2 | **2** |
| lateral | 3 | 3 | **1** |
| floor | 0 | 0 | **0** |

Disjoint is the number that counts. The overhead pairs reuse one adapted motion four times
and the lateral pairs reuse one nominal motion three times, so counting pairs overstates the
corpus by more than double. **Three independent families exist in total**, against a plan
target of five per regime.

## `side_step` does not serve the lateral regime, and that was a wrong assumption of mine

It was reasonable to expect lateral to be the healthy regime: `side_step` is the one
behaviour that scored 6/6 on semantic validity. It contributes nothing. Pairing each of the
six side-steps against each plain walk and counting why the pair is refused puts the reason
beyond doubt — the dominant rejection, 60 of them, is `min_half_width_m spread`.

Side-stepping is lateral *translation*. Passing a narrow gap requires width *reduction* —
tucking the arms, rotating the torso, turning a shoulder into the gap. A robot side-stepping
through a doorway is exactly as wide as one walking through it. The behaviour label and the
geometry regime look like they match and do not.

## The taxonomy has no mode that narrows the robot

Reading the 14 body modes against the three regimes explains every number above:

| Regime | Modes that target it | Disjoint families |
|---|---|---|
| overhead | `duck_under`, `crouch_walk`, `crouch_deep` | 2 |
| lateral | **none** | 1, and accidental |
| floor | `step_over` (0/7 semantically valid) | 0 |

Not one of the fourteen asks the robot to make itself narrower. `carry_walk` does the
opposite — it *widens* the silhouette, which is why `clutter_126`, a carry motion, is the
nominal in all three lateral pairs. The single lateral family that exists is an accident: a
duck and a pause-walk happen to be narrower than someone carrying a box, not because either
was asked to squeeze through anything.

So the taxonomy was built around behaviour names while the counterfactual method needs
motions that differ along a named geometric axis. Those are different design targets, and
the second one was never stated when the taxonomy was written.

## What this changes in the plan

The prompt-taxonomy feedback loop was ranked P1 as a corpus-quality improvement. It is
actually the **binding constraint on family scaling**, and it needs two distinct things:

1. **Fidelity** — make the existing modes produce their behaviour. `duck_under` is 3/7 and
   `step_over` is 0/7; the overhead ceiling of 2 and the floor ceiling of 0 are those two
   numbers.
2. **Coverage** — add modes that target a geometric axis the taxonomy does not reach. An
   arm-tuck or shoulder-turn mode is the entire lateral regime, and there is currently no
   prompt for it.

Both are cheap to test. The predicates and the envelope both grade a *generated reference*
with no rollout at all, so trying a rephrasing or a new mode costs seconds against the six
GPU rollouts a family costs. Measuring which phrasings produce a real duck, a real step-over
and a real arm-tuck should come before spending rollouts on families the corpus cannot yet
support.

The station refinement is what makes this measurable, and it is not a small correction. The
optimistic screen overestimated the one measured overhead window by 3.7x, and on one lateral
pair it reported +0.044 m where refinement returned **−0.020 m** — the wrong sign, not merely
the wrong magnitude.
