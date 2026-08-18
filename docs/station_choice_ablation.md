# Where the obstacle goes is the algorithm, not the scene

An obstacle's position has been treated as something a room generator picks. It behaves like
an algorithm parameter, and on the overhead regime it is not an improvement over the
alternatives — it is the only thing that produces a family at all.

Running all three placement policies over every compatible pair in
`g1_motionbank_v0.4`, geometrically, with no GPU:

## Overhead — 55 compatible pairs

| placement policy | median window | best | pairs ≥ 50 mm |
|---|---|---|---|
| random station | 3 mm | 40 mm | **0 / 55** |
| route midpoint | 3 mm | 43 mm | **0 / 55** |
| maximal separation | 15 mm | **186 mm** | **5 / 55** |

Neither baseline yields a single usable overhead family. Every one that exists does so
because the station was chosen.

## Lateral — 88 compatible pairs

| placement policy | median window | best | pairs ≥ 50 mm |
|---|---|---|---|
| random station | 4 mm | 136 mm | 4 / 88 |
| route midpoint | 10 mm | 142 mm | 6 / 88 |
| maximal separation | **31 mm** | **172 mm** | **25 / 88** |

Four times the yield, and **19 pairs become usable that the midpoint would have discarded**.

## Why the midpoint fails

It is not that the midpoint is unlucky. It is that a whole-body adaptation occupies part of a
route rather than all of it, and the middle of the path is not where it happens. The chosen
station sits a median **0.50 m** from the midpoint in the overhead regime and **0.40 m** in
the lateral one — far enough that a 0.5 m-deep shelf placed at one does not overlap the
other at all.

The first family measured this directly. At the midpoint the walk and the duck were 53 mm
apart; at the station where the duck is deepest they were 178 mm apart. Physics agreed: 27 mm
of penetration produced a 3.0 N graze that barely cleared the acceptance gate's 1.0 N
threshold, and 89 mm produced a 137.2 N peak with the robot still on its feet.

## What this measurement is and is not

**It is geometric yield, not verified yield.** A window of 50 mm is necessary for a family,
not sufficient: the pair still has to survive four rollouts and an attribution check, and of
the three overhead attempts made so far one failed on room sizing and one on a shelf-placement
bug of mine. Read the table as *how many pairs are worth spending six rollouts on*, which is
exactly the decision it is meant to inform.

**The 50 mm bar is calibrated on one family, not on a distribution.** It sits between the
53 mm build that produced a graze and the 178 mm build that produced a decisive collision.
It should be recalibrated once a handful of families have been physics-verified at known
window widths, and the number should be expected to move.

**Random is averaged over 24 draws per pair**, so its median is the expected window from an
uninformed placement rather than a single unlucky one.
