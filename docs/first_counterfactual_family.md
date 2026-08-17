# The first counterfactual family, demonstrated in physics

Same task, same start, same goal, same corridor. One number changes — the underside of a
shelf, by 53 mm — and the whole-body behaviour that succeeds changes with it.

| | easy scene (1.325 m) | hard scene (1.272 m) |
|---|---|---|
| **nominal walk** | accepted, 0.0 N | **rejected, 3.0 N** |
| **adapted duck** | accepted, 0.0 N | **accepted, 0.0 N** |

The failing contact body is `torso_link`: the walking robot's torso meets the shelf, is
knocked off its reference, and the episode is rejected for `disallowed_robot_contact` and
`unstable_reference_drift`. Every other cell records **no external contact at all** — not a
small force, no contacting bodies.

This is the supervision no scene-around-motion episode can provide. Those rooms are built so
the motion fits, which produces positives efficiently and cannot show that geometry
determines behaviour: when the furniture never touches the robot, two layouts give two
different images and a bit-identical trajectory. The middle cell here is a **matched
negative** — the same motion, in the same place, failing because the room changed.

## How the pair was found

The window is narrow and could not have been guessed. Each motion is rolled out on a bare
plane, and its executed swept volume is binary-searched against a shelf lowered toward it
until the first interference:

- the walk clears a shelf down to **1.298 m**
- the duck clears one down to **1.245 m**

A **53 mm window**. The hard scene sits in the middle of it, at 1.272 m; the easy scene sits
clear of both, at 1.325 m.

Only about 37% of the duck's torso drop converts into head clearance, because the torso
pitches forward during a duck and the limiting capsule is the head at its top. That is why
a motion whose torso descends 0.214 m buys only 53 mm of shelf.

## Three attempts, and what each one taught

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

## What to be careful about when citing this

**The negative is marginal by construction.** 3.0 N against a 1.0 N threshold is a graze, not
a collision — the hard scene sits 26 mm below the nominal motion's boundary because the window
is only 53 mm wide. That is arguably the most informative place for a negative to sit, since
it is the discriminating case, but it is not a dramatic failure and should not be described
as one.

**One family is not a result.** This demonstrates the machinery end to end and gives a
protocol that costs six rollouts. The claim that the corpus teaches scene-conditioned
behaviour needs the family count, the geometry regimes beyond overhead, and a model that
learns from them.

**The scenes are training material, never evaluation material.** The geometry is fitted to
executed swept volumes, so it encodes one policy's trajectory. `eval_scene_gate` refuses such
scenes for evaluation splits, and that refusal must survive the temptation to reuse a family
that took six rollouts to build.
