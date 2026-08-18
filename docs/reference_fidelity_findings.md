# What the generator actually produces, measured without a rollout

Behavioural fidelity used to be measurable only after a rollout, which meant it was measured
on whatever survived the acceptance gate — 37 episodes across seven modes — and cost GPU
minutes per clip. Forward kinematics on the reference answers the same question in about a
second, so the whole 150-prompt library can be graded, including clips that were rejected and
clips never rolled out at all.

That distinction matters, because the question is about the *generator* and the acceptance
gate is a filter on the *controller*. A mode whose clips are mostly rejected is invisible in
any rollout-based number, however badly it is phrased.

## Fidelity across the whole library

| body mode | graded | carries the behaviour | rate |
|---|---|---|---|
| `stand_to_walk` | 15 | 15 | **100%** |
| `side_step` | 3 | 3 | **100%** |
| `turn_in_place` | 3 | 3 | **100%** |
| `duck_under` | 15 | 3 | 20% |
| `walk_pause` | 15 | 0 | **0%** |
| `walk_to_stop` | 9 | 0 | **0%** |
| `step_over` | 15 | 0 | **0%** |

**24 of 75 references carry the behaviour their prompt asked for — 32%.** Seven of the
fifteen body modes still have no predicate, so this is a floor on what is wrong, not a
ceiling.

## Whole-clip styles work; mid-clip events do not

The split is clean and it is not about difficulty:

- **Present, 100%:** `side_step` and `turn_in_place` are postures held for the whole clip.
  `stand_to_walk` is a transition, but it is anchored at frame zero.
- **Absent, 0%:** `walk_pause` (stop in the middle, then resume), `walk_to_stop` (an event at
  the end), `step_over` (a discrete event mid-clip).
- **Intermittent, 20%:** `duck_under` — a posture change in the middle, which sometimes lands.

The generator produces a *style* reliably and an *event at a specified moment* unreliably.
Every failing mode asks for something to happen partway through a four-second clip.

That is a phrasing problem before it is a model problem, and it suggests the fix: ask for the
adapted behaviour as a whole-clip style rather than as an event. A clip that stays crouched
throughout is worth more to a counterfactual family than one that ducks briefly, because the
obstacle can then sit anywhere along the route instead of at one station.

## The crouch would be ideal, and the G1 cannot hold it

Measuring the lowest point each mode's silhouette reaches makes the case immediately:

| body mode | n | lowest silhouette peak | median |
|---|---|---|---|
| `crouch_walk` | 3 | 0.826–0.904 m | **0.849 m** |
| `walk` | 15 | 1.173–1.277 m | 1.244 m |

**0.395 m lower than a walk, three clips out of three, tightly clustered** — against the
0.178 m window the best available duck gives at a single station. No other mode comes within
0.05 m of a plain walk.

And none of them can be used. All three fail the joint-saturation prefilter before any GPU
time is spent, for one reason: `waist_pitch_joint` is clamped on **100% of frames**, while
the next worst joint sits at 0.05. Kimodo folds a human at the waist further than the G1's
range allows. Root height reaches 0.49–0.55 m against a walk's 0.75 m, so the pose is deep,
but it is deep in the one degree of freedom the robot does not have.

The failure being a single joint is what makes it actionable. The body does not need to fold;
it needs to lower. A knee-driven crouch with the torso upright reaches a similar height
through joints the G1 does have.

## The fix is already written down and has never been run

The prompt that produced those three clips is *"A person crouches down low and moves forward
{speed} while staying crouched"*. The current taxonomy says something different — *"A person
bends the knees and lowers **slightly**, then moves forward {speed} staying low"* — rephrased
after the saturation was diagnosed, and never generated, because generating needs the GPU.

So the next experiment is fully specified and cheap:

1. Generate the current `crouch_walk` phrasing, plus the new `arm_tuck` and `shoulder_turn`
   modes that the lateral regime has no prompt for at all.
2. Screen them for joint saturation — seconds, no GPU.
3. Grade the references for silhouette peak and half-width — seconds, no GPU.
4. Only then spend rollouts.

A knee-driven crouch landing anywhere between 1.00 and 1.10 m would give a 0.15–0.25 m
overhead window along the *entire route*, from a mode that is 3/3 consistent rather than
3/15. That would lift the overhead regime's ceiling of two disjoint families, which is
currently set by how rarely the generator produces a real duck.
