# The first counterfactual family, demonstrated in physics

Same task, same start, same goal, same corridor. One number changes — the underside of a
shelf — and the whole-body behaviour that succeeds changes with it.

| | easy scene (1.391 m) | hard scene (1.212 m) |
|---|---|---|
| **nominal walk** | accepted, 0.0 N | **rejected, 137.2 N** |
| **adapted duck** | accepted, 0.0 N | **accepted, 0.0 N** |

The failing contact is `torso_link` at frame **113**, against a swept-volume prediction of
frame **112** — one frame, 20 ms. Window **0.178 m**. Reference drift begins at frame 120,
*after* the contact, so it is the collision's consequence rather than its cause.

## Two families, and what moving the obstacle bought

`duck_003` supersedes `duck_002`. Same motion pair, same task, same room; the only difference
is where the shelf sits.

| | `duck_002` (path midpoint) | `duck_003` (station) |
|---|---|---|
| shelf position | x = 2.23 m | x = 2.67 m |
| window | 0.053 m | **0.178 m** |
| penetration at the hard shelf | 27 mm | **89 mm** |
| force on the nominal motion | 3.0 N | **137.2 N** |
| observed vs predicted contact frame | 94 vs 92 | **113 vs 112** |

The midpoint is a guess and it was the wrong one: it caught the leading edge of the duck
rather than the duck. Placing the obstacle where the two motions actually differ tripled the
window and turned a 3 N graze into a firm collision. The "negative is marginal by
construction" caveat recorded against the first family was a property of that placement, not
of the method.

**Penetration depth, not clearance, is the number to tune against.** Clearance saturates at
−0.068 m — the torso capsule's radius — the moment the capsule is engulfed, so it cannot say
how deep a collision is. The height the shelf would have to rise to stop touching does not
saturate, and it tracks force: 27 mm gave 3.0 N, 89 mm gave 137.2 N. A 3.3× penetration
bought an 18× force, so the boundary margin is a real knob and it trades a negative that is
too weak to be convincing against one violent enough to be a crash rather than a failure.
89 mm sits in a good place — decisive, and the robot does not fall.

## What is and is not being claimed

Three claims sit behind this result and only the first two are established. Keeping them
apart is the difference between a defensible contribution and an overreach a reviewer will
find immediately.

| Level | Claim | Status |
|---|---|---|
| 1 | The swept volumes of two executed motions separate, so a geometric window exists | **established** |
| 2 | The same controller, in the same place, succeeds or fails as that geometry changes | **established, twice** |
| 3 | The result survives a perturbed start pose rather than one deterministic replay | **established** |

## It is not one deterministic replay

The obvious objection to a family discovered from an executed trajectory is that an obstacle
was placed on a recorded path and that path then replayed. SONIC rollouts *are* highly
deterministic under a fixed configuration, which is what makes scene-around-motion generation
work at all, so the objection deserves an answer rather than a denial.

The 2×2 was re-run from three jittered start poses — 8 to 18 mm of translation and 0.5° to
1.0° of yaw, small enough that the task is unchanged and large enough that bit-identical
replay is impossible. **All twelve cells came out as predicted, and all three perturbations
hold.**

| | jitter | outcome |
|---|---|---|
| p1 | +15 mm, +10 mm, +0.5° | holds |
| p2 | −12 mm, +18 mm, −0.8° | holds |
| p3 | +8 mm, −15 mm, +1.0° | holds |

One number moves a great deal and is worth reporting for that reason: the contact force on
the failing cell reads 126.8, 658.3 and 95.5 N across the three jitters, against 137.2 N
unperturbed. A centimetre of placement decides *how hard* the torso meets the shelf. It does
not decide *whether* it does — the verdict is identical in every case. That is what makes the
family robust and also why the force should never be quoted as a property of the family.

Beyond all three sits a claim this work does **not** make: that the robot *perceives* the
scene and *chooses* to duck. Both motions here are prescribed references. What changes is
which prescribed behaviour physics permits — that is the supervision signal a scene-conditioned
policy would need, not evidence that one exists.

The honest one-line statement of the method today: *we mine discriminative geometry intervals
between physics-executable humanoid motions and use them to generate controlled counterfactual
scenes in which the same task requires different whole-body behaviours.*

In both families the walking robot's torso meets the shelf, is knocked off its reference, and
is rejected for `disallowed_robot_contact` and `unstable_reference_drift`. Every other cell
records **no lateral contact at all** — not a small force, no contacting bodies.

## The rejection is attributable to the shelf

"Rejected" alone would not support the claim. A rejection arriving by another route — drift
into a wall, a fall, tangled legs — produces the same table and means something else, which
is exactly what happened on an earlier attempt. So the failure is attributed, not just
counted:

| Question | `duck_002` | `duck_003` |
|---|---|---|
| First lateral contact | `torso_link`, frame 94, 3.0 N | `torso_link`, frame 113, 55.4 N |
| Frame geometry predicted the interference | 92 | 112 |
| Is the body in the overhead regime's group | yes | yes |
| Drift onset in the **easy** scene | frame 175 | frame 175 |
| Drift onset in the **hard** scene | frame 105 | frame 120 |

Two facts carry the attribution, and they hold in both families. The observed contact lands
**two frames** and then **one frame** — 40 ms, then 20 ms — from where the swept-volume
geometry said it would, which is a real validation of the predictor against physics rather
than against itself. And the reference drift moves later or earlier with the shelf, always
arriving *after* the contact: the drift is the collision's downstream consequence, not a
tracking failure that the shelf happened to coincide with.

Getting this right needed two corrections. Reading the raw contact array flagged frame 0 of
every cell — 237.4 N on `right_hip_roll_link`, in episodes with no collision at all — because
the robot is dropped into the scene and its hips carry the settling load. The acceptance gate
has always decomposed contact by Newton's third law, and a collision is the lateral part;
reaching past that re-answers a question that was already answered correctly. Separately,
comparing an observed *first* contact against a predicted *deepest* frame charged the
predictor for the 0.5 m depth of the shelf, turning a 2-frame agreement into an apparent
8-frame error.

This is the supervision no scene-around-motion episode can provide. Those rooms are built so
the motion fits, which produces positives efficiently and cannot show that geometry
determines behaviour: when the furniture never touches the robot, two layouts give two
different images and a bit-identical trajectory. The middle cell here is a **matched
negative** — the same motion, in the same place, failing because the room changed.

## How the pair was found

The window could not have been guessed. Each motion is rolled out on a bare plane, and its
executed swept volume is binary-searched against a shelf lowered toward it until the first
interference. At the station where the duck is deepest:

- the walk clears a shelf down to **1.302 m**
- the duck clears one down to **1.123 m**

A **178 mm window**. The hard scene sits at 1.212 m, the easy scene clear of both at 1.391 m.

Only part of the duck's torso drop converts into head clearance, because the torso pitches
forward during a duck and the limiting capsule is the head at its top: a motion whose torso
descends 0.214 m buys 178 mm of shelf at the right station, and only 53 mm at the midpoint.

## Five attempts, and what each one taught

**The clearance metric saturates.** The first run reported both motions at exactly
−0.0680 m in the hard scene — the torso capsule's radius. A capsule wholly inside the
obstacle has point-to-box distance zero, so the clearance stops distinguishing depth. Fine
for a boundary search, useless for ranking two motions at a fixed obstacle position, which
is what the construction was doing. `build_paired_family` compares each motion's own
boundary instead.

**The adapted motion was not adapted.** The second run used the reference labelled *"ducks
down low to pass under an obstacle"*. Executed, its torso drops 0.061 m — *less* than the
plain walk's 0.063 m — giving a 0.006 m window and no family. The semantic predicate had
independently scored `duck_under` at 3/7. Using that predicate to **select** the adapted
motion is what produced the pair above: it picks the reference whose torso actually descends
0.214 m.

**The room was sized from one motion.** The third run gave half a counterfactual and one
puzzle: the adapted motion failed both scenes at an identical 710.9 N. Identical is what gave
it away, since a shelf-related failure would differ between shelf heights. The contact was on
the legs at frame 187 with the root at x = 4.85 m, against a wall at 5.0 m — the room had
been sized from the nominal path, and the adapted motion travels further. Rooms are now
sized from every motion in the family.

**The shelf position was not single-sourced.** Moving the obstacle to the station where the
motions differ was applied to the boundary search and not to the renderer, which recomputed
the position internally as the path midpoint. Physics loaded a shelf 0.44 m from the one the
search had optimised, both motions hit it, and a 0.178 m window produced no family. The
attribution report caught it by flagging `adapted_hard` as impure, rather than letting
"counterfactual established: false" stand as a finding about the motions. The builder now
reads the shelf box back out of the USDA that physics will load and checks all four
clearances have their intended sign before spending a rollout — recomputing it from the
builder's own variables would only confirm the builder agrees with itself, which was true
throughout the bug.

**Two runs over the same pair overwrote each other's scenes.** Same motion indices give the
same `family_id`, so the second run's USDA replaced the first's silently. Re-running the
earlier family's report then read the later family's geometry and reported clearances for
scenes that had never been rolled out. Scene ids now carry the run directory.

Four of the five failures were in the harness, not the geometry. That is worth stating
plainly next to any yield number: counting them as method failures understates the method,
and excluding them silently overstates it, so `report_family_yield` prints both rates and
names every exclusion.

## What to be careful about when citing this

**The negative's severity is a choice, not a property of the method.** The boundary margin
decides how far below the nominal motion's clearance the hard shelf sits, and that fixes the
collision: 27 mm of penetration produced 3.0 N, a graze that barely clears the gate's 1.0 N
threshold, while 89 mm produced 137.2 N without the robot falling. Report which was used.
Anyone scaling this should tune the margin deliberately rather than inheriting it, because
both ends are bad — a graze is unconvincing, and a crash stops being a behaviour failure.

**One family is not a result.** This demonstrates the machinery end to end and gives a
protocol that costs six rollouts. The claim that the corpus teaches scene-conditioned
behaviour needs the family count, the geometry regimes beyond overhead, and a model that
learns from them.

**The scenes are training material, never evaluation material.** The geometry is fitted to
executed swept volumes, so it encodes one policy's trajectory. `eval_scene_gate` refuses such
scenes for evaluation splits, and that refusal must survive the temptation to reuse a family
that took six rollouts to build.
