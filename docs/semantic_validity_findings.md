# Physical validity is not behavioural validity

Acceptance answers whether the robot tracked its reference safely. A benchmark asks whether
the robot did the thing its label claims. Those are different questions, and the corpus had
only ever asked the first.

Turning the review page's own captions into predicates answered the second for the first
time. The result splits three ways, and two of the three were errors in the predicates
rather than in the data — which is what a calibration pass is for.

## Where the corpus stands

Over the 132 accepted episodes in `g1_motionbank_v0.4`:

| Behaviour | Semantically valid | Note |
|---|---|---|
| `stand_to_walk` | **5/5** | was 0/5 under a threshold that was wrong |
| `side_step` | **6/6** | not one is a turn-and-walk in disguise |
| `turn_in_place` | **3/3** | feet pivot, root does not slide |
| `duck_under` | **3/7** | the four failures drop the torso only 0.042–0.053 m |
| `walk_pause` | **0/7** | genuinely mislabelled |
| `walk_to_stop` | **0/2** | genuinely mislabelled |
| 6 other families | — | **102 accepted episodes have no predicate at all** |

**17 of 30 semantically valid where a predicate exists**, against an 85% acceptance rate.

## The two families that are genuinely mislabelled

`walk_to_stop` never slows down. Minimum root speed across its accepted episodes is
0.879–0.909 m/s and they are still moving at 1.15–1.54 m/s in the final 0.4 s. The prompt
asked for a stop; the generator produced a walk; the physics accepted it because a walk is
perfectly safe.

`walk_pause` is the same story with more variance — minimum speed ranges 0.024 to 1.410 m/s
across its seven episodes, so some slow markedly and none holds still for the 0.30 s the
predicate asks for.

This is not a controller failure and not a gate failure. It is the generator not producing
the behaviour the prompt requested, which nothing downstream was positioned to notice.

## Two errors in the predicates, recorded because they nearly became findings

**`stand_to_walk` was judged against an absolute speed.** All five accepted episodes read as
failures at a 0.12 m/s threshold, while opening at 0.13–0.16 m/s and going on to reach
1.25–1.79 m/s. They are plainly starts from rest. The criterion is now the opening speed as a
*fraction of the episode's own peak*, which separates the cases by an order of magnitude and
passes all five. An absolute threshold cannot work across motions whose peak speed varies
threefold.

**`walk_look` was wired to the pause predicate.** It reported all four of its accepted
episodes as mislabelled. But "pauses and looks around" is a head and torso yaw excursion
while the root keeps traversing, and root speed cannot see that at all. The entry was
removed rather than replaced: a predicate measuring the wrong quantity manufactures findings,
and reporting "no predicate exists" is the honest answer until one does.

Both mistakes have the same shape as the corpus's earlier ones — a measurement that looks
principled, is applied to a population it was not calibrated on, and produces a confident
wrong answer. The defence is the same too: check the distribution before believing the
verdict.

## What this changes

- **Four validity axes, not one flag.** `physics_valid`, `tracking_valid`, `semantic_valid`,
  `scene_task_valid` answer different questions and an episode can pass one and fail another.
  Only the third is implemented.
- **"No predicate" is not a pass.** `check_behaviour` returns `None` for an unchecked
  behaviour, and callers must distinguish that from a verdict — the same distinction the
  accepted/unevaluable split makes elsewhere.
- **The prompt taxonomy needs a feedback loop.** Two of its body modes do not produce their
  behaviour. Whether that is fixable by rephrasing, by longer clips, or not at all is the
  next thing to measure, and it is cheap: the predicates now grade generated references
  without a rollout.
- **Per-family acceptance rates should be reported alongside semantic validity.** A family at
  100% acceptance and 0% semantic validity is worse than one at 60% and 100%, and the current
  review page shows only the first number.
