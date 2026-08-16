# Design plan: dense, semantic, multi-view indoor humanoid dataset

Status: draft 2026-08-16. Successor plan to
[`design_kimodo_sonic_isaaclab_g1_dataset.md`](design_kimodo_sonic_isaaclab_g1_dataset.md),
which took the pipeline from nothing to 15 validated episodes. This plan covers the
richness axes that dataset does not yet have.

## 0. Where we actually are

Proven and in the repository:

- Kimodo reference -> SONIC physics tracking -> validated LeRobot `synthetic_g1` export.
- 15 episodes x 248 rows, passing schema, latent-parity and semantic evaluation.
- Scene-aware placement, and clutter generated *around* a known-good path
  (21-30% floor occupancy, measured lateral scene contact 0.00 N).
- Acceptance gates that compare against the commanded motion, not absolute thresholds.
- Visual validation that caught debug-marker goal leakage no numeric gate saw.

Honest gaps this plan addresses:

| Gap | Today |
|---|---|
| Clearance model | a single 0.45 m cylinder around the **root**, plus a 0.30 m margin |
| Clutter geometry | 2D footprints only; no vertical structure |
| Motion variety | 2 usable bundled demos: one walk, one bow. No household activity |
| Semantics | furniture proxies have a `kind` string; nothing is exported as a label |
| Views | ego RGB only in the dataset; third-person exists for review only |
| Scene styles | one procedural style, one density setting |
| Motion generation | **blocked**: no `sm_120` PyTorch build, so no new prompts |

## 1. What "great" means for this dataset

The dataset should let a model learn, and let a reviewer verify, that a humanoid:

1. **navigates tight, realistic indoor clutter** without contact,
2. **does human household activity** (reach, wipe, pick up, place, crouch to a low shelf),
3. in scenes whose objects **mean something** (a mop is near a bucket; a plate is on a
   counter, not on the floor),
4. observed from **the views a real robot has** plus the views a human needs to review it,
5. across a **controlled density sweep**, so robustness to clutter can be measured rather
   than assumed.

Every episode must remain physics-validated and provenance-bound. Richness must not cost
the guarantees already established.

## 2. Item A -- true swept-volume clearance (unlocks narrow passages)

**Why.** The current model is a 0.45 m cylinder about the root. That is simultaneously too
conservative and too permissive: too conservative near the feet, where the robot is narrow
and could pass a 0.6 m gap; too permissive around swinging arms and the head, which the
root cylinder does not represent at all. Narrow corridors are unreachable with it.

**What exists.** `motion_cmd.robot_body_pos_w` is `(num_envs, num_bodies, 3)` for all 30
tracked bodies, and the recorder already reads `[:, 0]` for the root. **Per-body positions
are not recorded today** -- that is step one.

**Design.**

1. Extend `TrajectoryRecorderTerm` with `body_pos_w (T, 30, 3)` and `body_names`, matching
   the existing contact-body ordering so contact and geometry index identically.
2. Add `swept_volume.py`: from `body_pos_w`, build a per-frame set of capsules along the
   kinematic links (a sphere per body plus a capsule per parent-child pair), with a
   per-link radius table derived from the G1 collision geometry.
3. Clearance becomes `min over (frame, link, obstacle)` of capsule-to-box distance in 3D,
   replacing the 2D root-cylinder test in both the planner and the acceptance gate.
4. Keep the tracking margin, but apply it to the swept volume rather than a nominal radius.

**Payoff -- and the prediction above was wrong, corrected by measurement.** Implemented
and measured on a real 249-frame walking rollout (see section 11):

| quantity | value |
|---|---|
| swept half-width, arms tucked | **0.273 m** |
| swept half-width, median | **0.377 m** |
| swept half-width, peak arm swing | **0.664 m** |
| old uniform assumption | 0.450 m |

So a *uniform* narrow corridor of 0.35-0.45 m, as this plan originally claimed, would be
**unsafe**: the robot exceeds it on 22 of 249 frames. The old 0.45 m root cylinder was
optimistic by 0.056 m on that episode, and 0.45 + 0.30 = 0.75 m was under-specified at
peak swing, where the true requirement is 0.964 m.

The real unlock is therefore a **per-frame, spatially varying corridor**: 0.573 m where
the arms are tucked (24% tighter than the old uniform value) and 0.964 m where they swing
(21% wider than it). Tighter *and* safer, in different places along the same path.

**Risk.** Under-approximating link radii turns a false-accept into a real collision.
Mitigation: derive radii from the G1 USD collision meshes, not by hand, and cross-check
that the swept volume predicts the contact frames already recorded -- we have ground truth
from `robot_contact_force_w`, so the model can be validated against observed contacts
before it is trusted to license narrower gaps.

## 3. Item B -- vertical clutter, not just footprints

**Why.** Everything today is a floor-standing box, so the only navigation skill exercised
is planar avoidance. Real indoor space is layered.

**Design.** Three occupancy bands, checked in 3D against the swept volume:

| Band | Height | Examples | Constraint |
|---|---|---|---|
| Floor | 0 - 0.25 m | rug edge, cable tray, low box, pet bowl | must not intersect foot/ankle capsules |
| Body | 0.25 - 1.4 m | table, counter, chair, open drawer, mop bucket | must not intersect torso/arm capsules |
| Overhead | 1.4 - 2.2 m | wall cabinet, hanging lamp, shelf, doorway header | must not intersect head/shoulder capsules |

Two additions make this interesting rather than merely 3D:

- **Cantilevered geometry** -- a table top or wall shelf whose footprint overlaps the
  corridor while its *free space underneath* does not. The robot walks under a shelf its
  2D footprint would have blocked. This is the case the current 2D model cannot express at
  all, and it is common in homes.
- **Per-band density knobs**, so a scene can be floor-cluttered but overhead-clear, or the
  reverse.

**Validation.** The static preflight must gain a 3D route check; it currently rejects any
solid whose vertical extent overlaps `[support_z, support_z + robot_height)` regardless of
whether the robot's *body at that point* is actually there.

## 4. Item C -- semantic objects and labels

**Why.** "Cluttered" is not the same as "meaningful". A model cannot learn a household task
from anonymous boxes, and an evaluator cannot check semantic grounding that was never
recorded.

**Design.**

1. **Object catalog** with `category`, `affordance`, `typical_support` (floor / counter /
   wall / ceiling), size distribution, and a `semantic_id`.
2. **Placement grammar** rather than uniform sampling: objects are placed relative to
   *anchors*. A kitchen anchor spawns counter + sink + fridge; small items (plate, cup,
   bottle) go **on** counters, not on the floor; a cleaning anchor spawns mop, bucket, bin.
   This is what makes a scene read as a room rather than a warehouse of cubes.
3. **Export semantics.** Add to the episode record: per-object `semantic_id`, category,
   pose, and support relation; plus, when the renderer supports it, an instance-segmentation
   sidecar aligned to the ego frames. The existing `synthetic_g1` profile stays unchanged
   for training features -- semantics are added as metadata/non-training features, per the
   original contract's rule.

**Note on assets.** Real meshes remain future work (see section 7). The catalog and grammar
are asset-independent: proxies today, textured assets later, same semantic ids. Building the
grammar first means swapping assets does not invalidate the labels.

## 5. Item D -- household-activity motions

**Why.** The dataset is currently *walking*. The stated goal is home cleaning and human-style
household work.

**Blocked on:** Kimodo generation. The local environment has no `sm_120` PyTorch build for
the RTX 5090, so no new prompt has ever been sampled; everything to date re-exports bundled
demos. **This is the single highest-value unblock in this plan** -- without it, motion
variety is capped at two usable clips.

**Design once unblocked.**

1. Prompt families: *sweeping the floor*, *wiping a table*, *picking up an object and
   carrying it*, *crouching to a low shelf*, *opening a cabinet*, *pushing a chair in*.
2. Kimodo end-effector constraints to place hands at task-relevant poses, which the design
   doc's section 10 already anticipates.
3. **Motion-conditioned scene generation**: the clutter builder already places furniture
   around a path; extend it to place the *task object* at the end-effector target. If the
   motion reaches to a point, put a table there. This is the same inversion that made
   clutter work, applied to manipulation.
4. Keep contact-rich manipulation out of scope until section 10 of the parent doc is
   addressed; the first step is reach-and-gesture near objects, not grasping.

**Interim, unblocked:** the two curved motions (`05_root_path`, `06_root_waypoints`) are
already richer navigation than what is in the dataset and fail only on a provisional
tracking threshold. Calibrating that threshold adds behaviours today at zero generation cost.

## 6. Item E -- multi-view capture

**Why.** One ego camera is what the robot has; it is not enough to review, and not enough to
train a model that reasons about a scene.

**Design.** Per-episode view set, all sharing one clock:

| View | Purpose | Status |
|---|---|---|
| ego RGB 640x480 @ 50 Hz | training | done |
| wrist RGB (L/R) | manipulation phase | env supports it; not wired |
| third-person chase 1280x720 | human review | works; review-only, not exported |
| overhead orthographic | layout/coverage review | `overview_camera` exists |
| ego depth + instance segmentation | semantic supervision | sidecar, per parent doc 5.4 |

Rendering every view for every episode is wasteful. Follow the parent doc's two-pass design:
physics first, then **replay accepted trajectories** through the render pass with whatever
view set that episode needs.

## 7. Item F -- density sweep and scene styles

**Why.** "Cluttered" should be a measured axis, not an adjective.

**Design.** Parameterise the generator by a `density` level with recorded metrics
(occupancy, min corridor width, object count per band):

| Level | Floor occupancy | Min corridor | Character |
|---|---|---|---|
| `sparse` | 5-10% | 1.2 m | open room, few landmarks |
| `moderate` | 15-25% | 0.9 m | furnished living space (today's output) |
| `dense` | 30-40% | 0.7 m | busy room, frequent detours |
| `tight` | 40%+ | 0.45-0.55 m | requires the swept-volume model from item A |

Plus **style presets** -- living room, kitchen, bedroom, hallway, office -- each an anchor
set and object mix in the item-C grammar. Split by style and layout so held-out evaluation
means something.

## 8. Item G -- photorealistic assets

Deferred deliberately, not forgotten. The NVIDIA Omniverse asset server is reachable from
this machine (verified HTTP 200). What adopting it costs:

- the dependency-free preflight validates only `Plane` + axis-aligned `Cube`; real meshes
  need a USD-native geometry gate to replace or supplement it;
- collision approximation (convex decomposition) per asset, and a check that the collision
  proxy matches the visual mesh closely enough that clearance guarantees still hold;
- a licence and redistribution record per asset, per the parent doc's section 5.1 rule;
- render cost, which the two-pass design already anticipates.

Sequencing: geometry and semantics determine whether the *navigation and task* data is any
good. Textures determine what the ego camera sees, which matters for sim-to-real transfer.
Layout first is deliberate; this is the natural next milestone after items A-C.

## 9. Proposed order

Each step is independently useful and leaves the dataset valid.

1. **A1 -- DONE.** `body_pos_w` and `body_quat_w` are recorded (orientation is required
   to place body-local capsules; positions alone only support bounding spheres).
2. **A2 -- DONE.** Swept-volume model built from the G1's 29 collision capsules and
   validated against independently recorded physics before being used to narrow anything.
3. **B** -- vertical bands and cantilevered geometry, 3D preflight.
4. **F** -- density sweep, now that narrow corridors are safe.
5. **C** -- semantic catalog, placement grammar, semantic export.
6. **E** -- multi-view render pass over accepted trajectories.
7. **D** -- household motions, the moment Kimodo generation is restored.
8. **G** -- photorealistic assets.

Items 1-6 are unblocked today. Item 7 is gated on an environment fix. Item 8 is gated on the
scene-contract work in section 8.

## 10. What must not regress

The guarantees already paid for, which every item above must preserve:

- physics-validated acceptance with reason codes, compared against the commanded motion;
- `action.motion_token` encoder-equivalence to the deployment `token_state`;
- typed `EpisodeRequest -> GenerationResult -> ConversionResult` provenance bound to the
  runtime capture;
- no simulator overlay in any exported observation;
- split-safe scene/layout grouping;
- honest reporting: quarantined rejects keep their reason codes, and provisional thresholds
  stay labelled provisional.


## 11. Implementation log

### 11.1 A1 -- per-body pose recording (done)

`TrajectoryRecorderTerm` now stores `body_pos_w (T, 30, 3)`, `body_quat_w (T, 30, 4)` and
`body_names`, in the same scene-local frame as `root_pos_w` (verified: the pelvis column
matches `root_pos_w` to 0.000000 m).

Orientation is recorded as well as position because the collision geometry is defined in
each link's local frame; positions alone would only support conservative bounding spheres,
which are too coarse to narrow anything.

**A landmine worth stating.** The articulation body order and the contact-sensor body order
**differ at 27 of 30 positions**, though they cover the same set. Indexing `body_pos_w`
against `robot_contact_force_w` positionally would silently pair the wrong link with the
wrong force. Everything downstream aligns by name.

### 11.2 A2 -- swept-volume model (done)

[`swept_volume.py`](../gear_sonic/dataset_generation/swept_volume.py) places the G1's 29
primitive collision capsules -- covering the 14 links that have collision geometry -- in
world frame from the recorded pose.

Geometry source: the Kimodo G1 MJCF, whose link structure was already verified against this
repository's G1 (same 29 actuated joints, order, axes, limits, parent links, transforms).
The repository's own `g1_29dof.xml` defines collisions as meshes with no usable radii. The
numbers are transcribed into the module so it does not depend on an external checkout.

A satisfying cross-check fell out: **exactly those 14 links carry collision geometry, and
exactly those links were ever observed to register contact force.** The other 16
articulation bodies reported identically zero force in every rollout because they have no
collision shape -- which also explains the "24 of 30 bodies are all-zero" observation from
the contact-decomposition work.

**Validation against independent ground truth**, done before the model was allowed to
license narrower gaps:

1. *Floor reconstruction.* Foot capsule surfaces reach **z = -0.0008 m**. Nothing in the
   capsule table knows where the floor is; the model recovers the ground plane to
   sub-millimetre from body pose alone, which confirms the rotation and offset placement.
2. *Agreement with recorded contact.* On an episode with **0.000 N** of recorded lateral
   scene contact, the model predicts **1.070 m** of clearance -- positive, consistent. The
   nearest link is `right_wrist_yaw_link`, precisely the body the root cylinder ignored.
3. *Disagreement with the old model, in the direction the measurements predict.* The root
   cylinder assumed 1.576 - 0.45 = **1.126 m**; the true surface clearance is **1.070 m**.
   The old model was optimistic by 0.056 m, matching the measured 0.514 m wrist reach
   against its 0.45 m assumption.

### 11.3 Per-frame corridors in the clutter builder (done)

`build_clutter_scene` accepts `per_point_clearance_m`, so the corridor follows the robot
instead of being a tube sized for its worst moment. Measured against the uniform model on
the same path and seed:

| | uniform 0.75 m | per-frame swept |
|---|---|---|
| nearest obstacle | 0.762 m | **0.643 m** |
| median obstacle distance | 1.426 m | **1.258 m** |
| pieces placed | 27 | 26 |
| floor occupancy | 27.1% | 24.9% |

Furniture comes 12 cm closer at the tightest point and 17 cm closer on average. Piece count
and occupancy are slightly *lower*, and that is the model working correctly rather than a
regression: this motion swings its arms hard, so the per-frame requirement rises to 0.964 m
at peak swing, and the space the corridor gives back where the robot is slim does not fully
pay for the space it correctly reclaims where the robot is wide. A motion with tucked arms
would gain more.

### 11.4 Next

Item B (vertical bands and cantilevered geometry) is the natural follow-on: the swept volume
is already 3D, and `swept_volume_clearance` takes boxes with `z` extents, so overhead and
under-table geometry can now be checked properly rather than collapsed to a footprint.
