# Physical validity is not behavioural validity

Acceptance answers whether the robot tracked its reference safely. A benchmark asks whether
the robot did the thing its label claims. Those are different questions, and the corpus had
only ever asked the first.

Turning the review page's own captions into predicates answered the second for the first
time. The result splits three ways: behaviours that are present, behaviours the generator
never produced, and two cases where the *predicate* was wrong rather than the data — which is
what a calibration pass is for.

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
| `step_over` | **0/7** | genuinely mislabelled |
| 5 other families | — | **95 accepted episodes have no predicate at all** |

**17 of 37 semantically valid where a predicate exists**, against an 85% acceptance rate.

## The pattern: the generator returns a walk

Three families ask for a composite behaviour and receive plain walking. A fourth is partial.
This is upstream of the controller and of every gate — physics accepts them because a walk is
safe, and nothing downstream was positioned to ask whether it was the *right* walk.

| Prompt asked for | What came back |
|---|---|
| "stops and stands still for a moment, then walks on" | continuous walking, minimum speed 0.024–1.410 m/s |
| "comes to a stop" | never slows: 0.879–0.909 m/s minimum, 1.15–1.54 m/s at the end |
| "lifts a leg high to step over something low" | trailing foot apex indistinguishable from walking |
| "ducks down low to pass under an obstacle" | 3 of 7 duck; the rest drop 0.042–0.053 m |

The `step_over` result is the cleanest, because it is **threshold-free**. Over accepted
episodes the trailing foot's swing apex is 0.125–0.153 m for a plain walk (n = 14, median
0.136) and 0.131–0.145 m for episodes labelled `step_over` (n = 7, median 0.138).
Mann-Whitney **p = 0.632**. The verdict does not depend on where any threshold sits.

That control is the check I failed to run on `stand_to_walk`, where an uncalibrated absolute
threshold produced a confident wrong answer. Running it first is the difference between a
finding and a bug.

## The absence is in the reference, not the tracking

A natural objection to all of this: perhaps the generator *does* produce the behaviour and the
controller loses it. Forward kinematics on the reference clips answers it directly, with no
rollout, and for `step_over` the answer is no.

| trailing-foot swing apex | reference | executed |
|---|---|---|
| `step_over` (n = 7) | 0.193–0.226 m, median 0.213 | 0.166–0.180 m, median 0.173 |
| `walk` (n = 6) | 0.206–0.222 m, median 0.218 | 0.176–0.188 m, median 0.181 |
| step_over higher than walk? | **p = 0.693** | **p = 0.989** |

Indistinguishable in both domains. The behaviour is absent from what the generator produced,
so this is a generation failure and not a tracking one, and no amount of controller work
would recover it.

Tracking does lower the foot — by 0.034 m for a plain walk and 0.039 m for a `step_over`. That
the two are nearly equal is the point: the controller flattens *everything* by about 35 mm
rather than failing step-overs selectively.

**The screen is worth having, and it lies in a specific way.** Reference and executed grades
agree on 16 of 20 episodes, and silhouette peak transfers closely (reference minus executed is
−0.008 m mean, 0.034 m sd). Every disagreement was `step_over`, and every one was my own
threshold: `MIN_STEP_APEX_M = 0.18` was calibrated on rollouts and sits *between* the two
populations, so applied to references it passes nearly everything. For a while that looked
like evidence the generator had produced a step-over the controller then lost.

That is the same mistake as `stand_to_walk` and `walk_look`, for the third time: a threshold
applied to a population it was not calibrated on. It is now chosen from the payload's own
domain, and a reference clip declares itself as one.

## The families that are genuinely mislabelled

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

## The generator, not the controller: measured in both domains

Every verdict above was taken from an executed rollout, which cannot separate two very
different failures — the generator not producing the behaviour, and the controller not
tracking it. Forward kinematics on the reference clip separates them, and it costs about a
second per clip against minutes of contended GPU.

For the trailing foot's swing apex, over the same episodes:

| | n | reference (median) | executed (median) |
|---|---|---|---|
| `walk` | 6 | 0.206–0.222 m (0.218) | 0.176–0.188 m (0.181) |
| `step_over` | 7 | 0.193–0.226 m (0.213) | 0.166–0.180 m (0.173) |

**Reference p = 0.693. Executed p = 0.989.** The behaviour is absent in *both*, so
`step_over` is a generation failure and not a tracking one. The floor geometry regime is
blocked at the generator, and no amount of controller work reaches it.

Tracking does lower the foot, by a nearly uniform amount: 0.034 m for a plain walk and
0.039 m for a clip labelled `step_over`. The controller flattens everything; it does not
single out step-overs.

**That uniform shift nearly produced a false finding.** The executed threshold of 0.18 m
sits between the two populations, so applied to reference clips it passed almost all of
them — which read as the generator producing a step-over that the controller then lost.
Running the walk control in the reference domain is what showed otherwise. The predicate now
carries a separate, calibrated threshold for references, and the payload declares which
domain it came from.

It is the third time an absolute threshold applied to a population it was not calibrated on
has produced a confident wrong answer here, after `stand_to_walk` and the drift gate. The
defence has been the same every time: measure the control group before believing the verdict.

For the overhead regime the reference screen transfers well — silhouette peak differs from
executed by −0.008 m on average with a 0.034 m spread — so prompt iteration can be graded
without rollouts there.

## What this changes

- **Four validity axes, not one flag.** `physics_valid`, `tracking_valid`, `semantic_valid`,
  `scene_task_valid` answer different questions and an episode can pass one and fail another.
  Only the third is implemented.
- **"No predicate" is not a pass.** `check_behaviour` returns `None` for an unchecked
  behaviour, and callers must distinguish that from a verdict — the same distinction the
  accepted/unevaluable split makes elsewhere.
- **The prompt taxonomy needs a feedback loop.** Three of its body modes do not produce
  their behaviour and a fourth does so half the time. Whether that is fixable by rephrasing, by longer clips, or not at all is the
  next thing to measure, and it is cheap: the predicates now grade generated references
  without a rollout.
- **Per-family acceptance rates should be reported alongside semantic validity.** A family at
  100% acceptance and 0% semantic validity is worse than one at 60% and 100%, and the current
  review page shows only the first number.
