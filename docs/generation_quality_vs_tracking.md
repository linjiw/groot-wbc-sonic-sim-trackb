# Does the generator produce motion the controller can execute?

A dataset built on generated motion has two failure surfaces, and reporting one acceptance rate
hides which is which. The generator can emit a clip the robot's **body** cannot hold — joints past
their limits, a root height no leg length reaches — which is caught cheaply on CPU. It can also emit
a kinematically fine clip the **controller** cannot track, which costs a rollout to discover.

Joining the generator's own screen to SONIC's verdicts on the same clips, matching on clip index:

| | clips | share |
|---|---|---|
| generated | 150 | |
| kinematically feasible | 144 | 96% |
| rolled out and joined | 47 | |
| of those, SONIC tracked | 45 | 96% |

Read alone, that looks like a healthy pipeline. Broken down by behaviour it is not.

## The failure is concentrated in exactly the behaviour the dataset needs

| behaviour | generated | kinematically feasible | rolled out | tracked |
|---|---|---|---|---|
| **crouch** | 9 | **4 (44%)** | 2 | 2 |
| side / narrow | 3 | 2 (67%) | 2 | 2 |
| arms | 6 | 6 (100%) | 1 | 1 |
| turn | 67 | 67 (100%) | 33 | 33 |
| stop / start | 23 | 23 (100%) | 5 | 5 |
| walk | 42 | 42 (100%) | 4 | 2 |

Two things stand out.

**Crouch is the only behaviour with a kinematic failure rate at all** — 44% feasible against 100%
for everything else. The rejections are all `joint_saturation`, and the earlier prompt audit found
the mechanism: every phrasing that lowers the body pins `waist_pitch_joint` at its limit on 100% of
frames. The generator lowers the torso by folding the waist, and the G1 cannot fold that far.

**The generator over-produces the category the corpus already has.** 67 of 150 prompts are turn
variants and 42 are plain walks — 73% of the corpus is unconstrained locomotion, which is precisely
the diversity that does not help. The behaviours a counterfactual needs, crouch and lateral
narrowing, are 12 of 150 and are the ones that fail.

## Why this matters for the method

This is the quantitative case for constructing adapted motions with a **local operator** rather than
requesting them from the generator. It is not that prompting is impossible in principle; it is that
under this checkpoint and these templates, the constrained behaviours are both rare and
disproportionately infeasible, while the unconstrained ones are abundant and nearly always fine.

An operator applied to an already-accepted walk inherits that walk's feasibility and changes only
what one obstacle requires. That is why `local_crouch` exists, and why its output is matched to its
nominal by construction rather than paired after the fact.

## What this is not

**Not a verdict on the generator.** One checkpoint, one set of prompt templates, 4-second clips.
A different phrasing strategy or a fine-tune could move these numbers, and this measures neither.

**Not a tracking benchmark.** 47 clips joined out of 150 generated, because only part of the corpus
has been rolled out and 165 rollouts belong to batches with no screen record. The 96% tracked figure
is over the joined subset and is not a corpus-wide rate.

**Not a semantic check.** These rows say a clip was *executable*, not that it performed the
behaviour its prompt named. Of 75 prompts admitting a semantic predicate, only 24 contained their
target behaviour — that is a separate and much worse number, tracked separately on purpose.
