# The first matched counterfactual family

One journey, two behaviours. The nominal walk and the local crouch share their root path,
duration, gait phase, start and goal exactly; the crouch differs only by a knee-driven squat
centred on the obstacle station. A shelf 98 mm below one and above the other then decides which
succeeds.

| | easy scene (1.355 m) | hard scene (1.257 m) |
|---|---|---|
| **nominal walk** | accepted, 0.0 N | **rejected, 3253.5 N** |
| **local crouch** | accepted, 0.0 N | **accepted, 0.0 N** |

Family `mf_005_c08`, station x = 2.229 m, room 10.634 × 5.000 m, predicted window 1.2074 →
1.3051 m (97.7 mm).

## Why this is stronger than the earlier families

`cf_005_056` and `duck_003` put a shelf between *two different motions*. That establishes that
geometry decides outcomes, but it invites a fair objection: the shelf separated two different
journeys, and the difference in outcome could ride on any of the ways those journeys differ.

Here the two clips are the same clip, with one local operator applied. Measured on the executed
files rather than asserted:

- root XY identical, frame for frame
- 120 frames in both
- exactly six joints move: knees 0.994 rad, hip pitch and ankle pitch 0.497 rad each, in the
  coupled squat ratio
- `waist_pitch` change **0.000 rad** — the crouch is knee-driven, not a torso fold

So the shelf separates two *behaviours* on one route. That is the supervision a scene-conditioned
policy needs.

## The rejection is attributable to the shelf

Contact lands on `torso_link` at frame 90. The root-height deficit appears at frame 97 — seven
frames, 140 ms *later*. The drift is the collision's consequence, not a tracking failure the shelf
coincided with. Drift rate is 0.040 m/s in the easy scene against 0.344 m/s in the hard one, and
the crouch reads 0.100 m/s in **both** scenes: the shelf never touches it.

## Two cautions

**The negative is violent.** 3253.5 N, against the 137.2 N that an earlier family recorded as a
good operating point. Penetration here is only 48 mm where that family used 89 mm, so force does
not follow penetration across motions and must not be modelled as if it does. The hard shelf sits
at the window's centre, which is the most robust placement and also the least informative one — it
cannot say where the real boundary is. Rungs 8 and 25 mm inside each predicted boundary are
written and awaiting rollout for exactly that reason.

**One family is still not a result.** This is one motion pair in one regime. The paper-level claim
needs 24–30 verified families across overhead and lateral, and the lateral operator is not there
yet: the arm tuck is accepted on 1 of 3 valid nominals, and [two attempts to predict which
one](tuck_trackability_is_not_predictable.md) both failed.

## A reproducibility failure this exposed

The clip physics verified moves its knees 0.994 rad. Two features added to the operator *after*
that clip was generated — a 0.40 rad excursion cap and waist-headroom spending — meant the
repository could no longer produce it: capped at 0.400 rad the same call yielded 31.7 mm of drop
instead of 80.0 mm, 41% of it from a waist fold the verified clip does not contain.

The cap was the deeper mistake. It came from an **arm tuck** failure at 1.300 rad and was applied
to both operators, while the crouch's own strength sweep had put its trackability boundary between
1.05 and 1.31 rad. One number for two operators silently forbade a family that works.

Both are fixed: the caps are per operator (tuck 0.40, crouch 1.00) and the waist is off by
default, matching both the guidance and the verified clip. The operator now reproduces it to
within 0.0002 rad. Three regression tests pin this, because the failure was silent — a test
asserting the waist stays still had been red since the commit that moved it, and the suite was not
run.
