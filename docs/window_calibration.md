# How well does the predicted window match the real one?

A family is built by binary-searching each motion's executed swept volume against a descending
shelf, taking the two boundaries, and putting one shelf between them. Until now that prediction
had only ever been checked at a single height — the window's centre — which is the placement most
likely to work and the least able to reveal an error. This measures both boundaries directly.

Family `mf_005_c08`. Predicted window **1.2074 → 1.3051 m (97.7 mm)**.

| shelf | motion | predicted | observed | external force |
|---|---|---|---|---|
| 1.3551 m | nominal | clears | accepted | 0.0 N |
| 1.2971 m | nominal | **fails** | **accepted** | 0.0 N |
| 1.2801 m | nominal | fails | rejected | 85.0 N overhead, 0.0 lateral |
| 1.2574 m | nominal | fails | rejected | 2758.2 N overhead, 1725.6 lateral |
| 1.3551 m | crouch | clears | accepted | 0.0 N |
| 1.2574 m | crouch | clears | accepted | 0.0 N |
| 1.2154 m | crouch | **clears** | **rejected** | 1017.4 N overhead |

## The predictor is optimistic on both sides

- the nominal's real boundary lies in **(1.2801, 1.2971)** — a 17 mm bracket; it survives 8 mm
  below where the prediction says it fails, and the prediction is high by 8 to 25 mm
- the crouch's real boundary lies in **(1.2154, 1.2574)** — it fails 8 mm above where the
  prediction says it clears

Both errors push the same way: they *shrink* the usable window. Combining the two brackets, the
real window is between **22.7 mm and 81.7 mm** wide, against a predicted 97.7 mm. In the worst case
the prediction over-states it more than fourfold. It does contain 1.2574 m, verified from both
sides, which is the family's placement.

The direction is not symmetric in cause. Swept capsules are conservative outer approximations, so
they should make each motion look *taller* than it is and predict interference too early — which
is what happened to the nominal. The crouch failing *higher* than predicted is the opposite, and
the reason is not geometric: at 1.2154 m the shelf pressed `torso_link` down at 1017.4 N and the
robot lost its reference. A swept volume knows where the robot went; it does not know that a
controller squeezed into a gap stops being able to track.

**So the practical rule is to place against the centre, not against an edge.** The `mf_005_c08`
hard shelf sits at the window's midpoint and is verified from both sides; a family placed within
about 20 mm of either predicted boundary would have inverted. Because the real window may be a
quarter of the predicted one, the margin cannot be replaced by a predicted-boundary offset — which
is what task G3 set out to do, and this measurement rules out.

## This measurement found a gate defect

Both rejections here are **overhead** pushes — the shelf pressing down — and
`disallowed_robot_contact` tested only the horizontal component. The crouch at 1.2154 m carried
1017.4 N of downward force on `torso_link` with a horizontal component of exactly 0.0, so the
gate stayed silent and the episode was caught only by the drift that followed. A milder jam would
have been **accepted**.

The fix is in `contact_decomposition.py`: sign separates an overhead collision from the settling
load the lateral-only test was written to exclude. Auditing all 230 evaluable episodes found 10
carrying an overhead push above the lateral one, all already rejected on other grounds, so the
corpus held no false accepts from this.

**That audit result was luck, not safety, and the next rollout proved it.** The 1.2801 m probe run
after the fix is rejected for `disallowed_robot_contact` *alone* — its drift of 0.125 m/s sits below
threshold, so nothing else disqualifies it. Regraded with the lateral-only gate restored, the same
episode comes out **accepted**: a walk whose torso is pressed down 85 N by a shelf, recorded as a
clean traversal. The bug was one boundary probe away from putting a collision into the corpus as a
positive. It did, however, correct three claims — see the
correction block in [the first family's record](first_counterfactual_family.md).
