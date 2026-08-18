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

The lateral regime has not been mined yet and may be in better shape — `side_step` scored 6/6
on semantic validity, the only family that did. That makes lateral, not floor, the right
second regime to attempt.
