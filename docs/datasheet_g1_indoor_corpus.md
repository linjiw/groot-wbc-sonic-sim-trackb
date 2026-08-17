# Datasheet: G1 Indoor Corpus

Status 2026-08-16. Every number here is measured from a recorded rollout or a checked-in
scene manifest, produced by
[`build_dataset_gallery.py`](../scripts/research/build_dataset_gallery.py) in one pass with
the footage it describes. Where something is an assumption rather than a measurement it is
labelled as such.

Companion documents:
[design](design_kimodo_sonic_isaaclab_g1_dataset.md) ·
[forward plan](design_plan_dense_indoor_humanoid_dataset.md) ·
[action contract](dataset_contract_sonic_vla.md)

---

## 1. What this corpus is

A humanoid (Unitree G1, 29 actuated joints) walking through indoor scenes under full physics,
recorded as **paired observation and action** in the form a vision-language-action model would
consume: an ego RGB stream, robot state, a language instruction, and the 64-dimensional SONIC
latent action that actually drove the decoder.

The distinguishing property is not scale — it is small — but that **nothing in it is asserted
without evidence**. Every episode is executed by a real controller in a real simulator and must
clear eleven reason-coded gates. Rejected episodes keep their reason codes instead of
disappearing.

## 2. What exists today

### 2.1 Episodes

| | count |
|---|---|
| Rollouts recorded and analysed | **24** |
| Accepted | **20** (83%) |
| Rejected, reason-coded | 4 |
| Accepted footage | **100 s** at 50 Hz |
| Exported to LeRobot `synthetic_g1` | **15** episodes, 3 720 rows |

Exported episodes carry 248 rows each: 249 recorded frames minus one dropped to the causal
shift that pairs each observation with the action taken *after* it.

### 2.2 Where the episodes come from

| Shelf | Accepted | Total | What it exercises |
|---|---|---|---|
| `01_placed_scenes` | 12 | 12 | Hand-authored household room and factory aisle, with search-planned placement |
| `02_generated_clutter` | 3 | 7 | Furniture generated around the motion path, 2D footprints |
| `03_clutter_3d` | 1 | 1 | Vertical bands and cantilevered geometry |
| `04_density_ladder` | 4 | 4 | Sparse → tight density sweep over one motion |

### 2.3 Scenes

18 scenes in 5 packages, all passing the same dependency-free preflight (content hashes, stage
metadata, collision coverage, floor support, route clearance — no USD or Isaac import needed).

| Package | Scenes | Pieces | Floor occupancy |
|---|---|---|---|
| `g1_dataset` | 2 | hand-authored | — |
| `g1_clutter` | 6 | 23–26 | 22–28% |
| `g1_clutter_curved` | 4 | 28–32 | 20–28% |
| `g1_clutter3d` | 2 | 33–34 | 31–38% |
| `g1_density` | 4 | 12–33 | 13–34% |

Across the 16 generated scenes: **169 furniture solids**, median 27 per scene, occupancy 13–38%
(median 27%), **523 collision prims** in total.

**By vertical band** — 101 body, 45 overhead, 23 floor. **24 pieces are cantilevered directly
over the walking corridor**: their footprint covers the path while the robot passes underneath.
A 2D footprint model cannot express those at all.

### 2.4 Per-episode record

Every accepted episode carries, at 50 Hz:

| Field | Shape | Note |
|---|---|---|
| `observation.images.ego_view` | 640×480 RGB | head-mounted, the trained visual observation |
| `observation.state` | 43 | whole-body configuration, MuJoCo joint order |
| `observation.projected_gravity` | 3 | |
| `action.motion_token` | 64 | the SONIC decoder input — see §4.1 |
| `reference.g1_qpos` | 36 | the commanded motion, which the gates compare against |
| `teleop.left/right_hand_joints` | 7 each | neutral; Kimodo carries no hand trajectory |

The raw trajectory additionally holds per-body pose (`body_pos_w`, `body_quat_w`, 30 bodies),
per-body contact force vectors, filtered foot-to-floor sensors, and tracking metrics. That raw
record is authoritative; the LeRobot export is a derived training view.

---

## 3. Distribution — including where it is weak

### 3.1 The scene axis is genuinely varied

Occupancy spans **13% → 38%**, corridor clearance **0.986 m → 0.287 m**, and layouts range from
an open room to one where furniture presses in from both sides. Four density levels were rolled
out and all four were accepted.

### 3.2 The motion axis is nearly degenerate

This is the corpus's real limitation and it should be read before anyone plans to train on it.

| Quantity | min | median | max | spread |
|---|---|---|---|---|
| Path length | 3.241 m | 3.697 m | 3.706 m | **14%** |
| Net displacement | 3.044 m | 3.495 m | 3.509 m | 15% |
| Mean speed | 0.653 m/s | 0.745 m/s | 0.747 m/s | **14%** |
| Action effective rank | 3.28 | 4.70 | 4.71 | of 64 dimensions |

Every accepted episode is a **~3.5 m forward walk at ~0.7 m/s**. The 64-dimensional latent has a
participation ratio of roughly **4** — the action manifold the corpus actually covers is about
four-dimensional, not sixty-four.

The cause is direct: **only two source motions are usable**, because no motion has ever been
generated from a new prompt (§6.1). The scene generator multiplies contexts, not behaviours.
Twelve of the fifteen exported episodes are the same two motions under different placements and
layouts.

**What that means in practice.** The corpus supports questions of the form *"does visual context
change what the policy should do?"* It does **not** yet support *"can the policy produce a
different behaviour?"*, because it has barely seen one.

### 3.3 Execution quality, where accepted

| Quantity | min | median | max |
|---|---|---|---|
| Endpoint error | 0.130 m | 0.203 m | 0.221 m |
| p95 path error | 0.136 m | 0.228 m | 0.234 m |
| Foot support fraction | 0.984 | 0.988 | 1.000 |
| Self-contact peak | 36.0 N | 282.8 N | 316.8 N |
| **Lateral scene contact** | **0.000 N** | **0.000 N** | **0.000 N** |

Zero scene contact on every accepted episode, at every density level — the clutter generator's
guarantee holds in physics, not just in the planner.

Self-contact is wrist-against-hip during arm swing. It is a gait property, not a collision: on a
bare-plane control with no obstacles at all it peaked *higher* (309 N) than in either furnished
scene.

### 3.4 Rejections

4 of 24, all from `02_generated_clutter`, all for `reference_path_tracking_error` at p95
0.271–0.272 m against a provisional 0.25 m threshold — **8% over, with zero collisions**, and one
of them landing its endpoint to 0.017 m. These are the two curved motions (`05_root_path`,
`06_root_waypoints`). They are a controller-tracking limit and a threshold-calibration question,
not scene failures. Earlier, in fixed scenes, the same motions drove the robot into racks at
**1 576–1 850 N**; scene generation removed the collision entirely and left only the tracking
gap.

---

## 4. Design criteria

These are the rules the pipeline is built to satisfy. Several were learned by violating them.

### 4.1 The action must mean what the deployment runtime means

`action.motion_token` is the one field whose semantics are owned by the C++ deploy runtime, not
by this repository. Both sides must occupy the same slot, in the same representation, with the
same memory layout.

**Verified:** recorded tokens lie exactly on the FSQ codebook lattice — 25 distinct values, all
`k/16` — on **26 of 26 rollouts**, residual RMS 0. That makes them byte-comparable with the
`token_state` a local encoder produces on the robot. The width is bound to the deploy
observation config by a test, so drift on either side fails CI.

### 4.2 Gate against the command, not an absolute threshold

An absolute threshold cannot tell "the robot collapsed" from "the reference asked for a low
pose". Three separate false rejections proved it:

| Gate | Rejected | Reference commanded | Tracking error |
|---|---|---|---|
| `fall_root_height` @ 0.50 m | pelvis 0.408 m | **0.323 m** — a crouch | 0.039 m |
| `disallowed_robot_contact` @ 1 N | knee 105–549 N | crouch puts the knee on the floor | force purely +Z |
| `fall_root_tilt` @ 0.60 rad | 0.708 rad | **0.732 rad** — a bow | 0.124 rad |

Each gate is now a pair: a reference-relative limit, plus an absolute limit reserved for states
no command can legitimately request (pelvis below 0.25 m, torso beyond 1.40 rad).

### 4.3 Measure the robot, don't approximate it

Clearance was a 0.45 m cylinder about the root. The G1's 29 collision capsules, placed in world
frame from recorded pose, show that is wrong in both directions: half-width runs **0.273 m**
(arms tucked) to **0.664 m** (peak swing), median 0.377 m. A uniform narrow corridor would be
unsafe; a uniform wide one wastes space. The corridor varies per frame.

Validation before the model was trusted to narrow anything: foot capsule surfaces land at
**z = −0.0008 m**, recovering the floor plane from body pose alone.

### 4.4 Generate the scene around the motion

Fitting motions into fixed scenes is placement-limited and yields episodes crossing empty floor.
Building the room around a known corridor makes the episode traversable **by construction** and
lets furniture hug the path.

This is also why `tight` works at a 0.12 m margin — below the measured p95 path error. The
clutter is generated against the **executed** swept volume, so tracking error is already baked
into the geometry; the margin only absorbs simulation nondeterminism.

### 4.5 The observation must not contain the answer

Debug markers rendering the tracking goal into the ego camera — 243 of 249 frames, peaking at
15.1% of the image — passed every numeric gate. Only looking at frames caught it. Contamination
is now detected automatically per episode, and it was **pose-dependent** (15.1% on one motion,
0.01% on another), so it would never have shown up uniformly.

### 4.6 Say what cannot be checked

The one-frame causal alignment is *not* verifiable by regression here, and the evaluator says so
rather than reporting a passing check. Reversing the shift moved R² by 0.015 — the wrong way —
and a deliberate 25-frame offset only moved it 0.717 → 0.556, because the latent has
autocorrelation **0.981 at lag 1**. The contract is guaranteed structurally by the exporter and
its sentinel tests.

### 4.7 Quarantine, don't delete

Rejected rollouts keep their reason codes. They are the raw material for calibrating thresholds
and for future failure-recovery work.

---

## 5. How each part could serve humanoid navigation and locomotion research

Stated as what the data can support, with the caveat that follows from §3.2.

**Ego-vision navigation policies.** Paired ego RGB, robot state, language, and a physically
executable action, with the visual modality verified free of simulator overlay. The density
ladder gives a controlled axis for asking how clutter degrades a policy — the same motion at
four occupancies with everything else fixed. That is a clean ablation, and rare.

**Traversability and clearance learning.** Per-body pose and per-body contact forces are
recorded, so the swept volume can be reconstructed exactly. A model can be asked to predict
whether a corridor is passable given the robot's *actual* geometry rather than a cylinder
approximation, with ground truth available at 0.273–0.664 m resolution.

**Contact-aware safety.** The contact decomposition separates self-contact, floor support and
lateral collision from a signal that natively conflates them. That distinction is a prerequisite
for any learned safety critic — a model trained on undecomposed force would learn that walking
is dangerous, since gait self-contact reaches 300 N routinely.

**Sim-to-real latent transfer.** Because the recorded action is byte-comparable with deployment
`token_state`, a policy trained here emits tokens the C++ runtime accepts without translation.
This is the piece most datasets get wrong silently.

**Scene generation research in its own right.** The generator is a reusable result: given any
executed trajectory it produces arbitrarily dense scenes guaranteed traversable, at a measured
density, with cantilevered geometry. It is not tied to this controller or this robot.

**Threshold and gate calibration.** The quarantine set, with reason codes and full metrics, is a
labelled corpus for deciding where acceptance thresholds actually belong.

**What it cannot serve yet.** Behaviour diversity, manipulation, long-horizon navigation, or
anything requiring more than a ~3.5 m forward walk. Those wait on §6.1.

---

## 6. Known limitations

### 6.1 No motion has ever been generated from a prompt — the binding constraint

Every episode re-exports one of Kimodo's bundled demo clips. The local Kimodo environment has no
`sm_120` PyTorch build for the RTX 5090. This is why the motion axis is degenerate (§3.2), why
there is no household activity, and why 12 of 15 exported episodes are two motions in different
clothes. **It is the single highest-value unblock and it is an environment fix, not a research
problem.**

### 6.2 Scale

15 exported episodes, 3 720 rows, 100 s of accepted footage. This is a proven pipeline and a
seed corpus, not a training set.

### 6.3 Not photorealistic

Scenes are untextured primitive proxies with plausible furniture dimensions. The NVIDIA asset
server is reachable; adopting real meshes needs collision approximation, per-asset licence
records, and a USD-native geometry gate to replace the primitive-subset preflight.

### 6.4 No semantics exported

Furniture proxies carry a category internally but nothing is exported as a label, and there is
no placement grammar — a plate should sit on a counter, not the floor.

### 6.5 Review tooling gaps

The overhead render produces blank frames (`overview_camera`, cause not found; the speculative
fix was reverted rather than shipped unverified). The chase camera clips inside furniture in
dense rooms. Wrist cameras are wired into the environment but their render pass is unverified.

### 6.6 Provisional thresholds

Self-contact 343 N is a body-weight anchor; the command-deviation limits (0.15 m, 0.35 rad) and
absolute limits (0.25 m, 1.40 rad) are physical anchors. All are labelled provisional in code,
pending a reviewed pilot.

### 6.7 Human review is partial

3 of 15 exported episodes have been inspected frame by frame. The rest pass the automated marker
detector, but nobody has looked at them.

### 6.8 One episode pair is a visual variant, not a behaviour

Two clutter seeds of one motion have **bit-identical** state trajectories — the furniture never
touches the robot, so the physics is unchanged — while their imagery differs by 21.2/255 per
pixel. Useful as visual-robustness augmentation; not behavioural diversity. The evaluator reports
this: 15 episodes cover **14 distinct behaviours**.

---

## 7. Where things are

| What | Path |
|---|---|
| Exported dataset | `<work>/dataset_v2/` — 15 episodes, LeRobot v2.1 |
| Video gallery | `<work>/gallery/` — 32 videos on 6 shelves |
| Distribution report | `<work>/distribution.json` |
| Figures | `<work>/figures/` |
| Scene packages | `gear_sonic/data/assets/scenes/` |
| Raw rollouts | `<work>/{clean,clutter_rollouts,clutter3d_rollouts,density_rollouts}/` |

`<work>` is `/data/robotixx/groot-wbc-kimodo-m0` on this machine.

Gallery shelves: `01_placed_scenes`, `02_generated_clutter`, `03_clutter_3d`,
`04_density_ladder` hold ego videos named `<config>__<accepted|rejected>__ego.mp4`;
`05_third_person` and `06_multiview` hold review renders. Videos are copied, not moved — the
working directories stay intact.

## 8. Reproducing

```bash
# scenes
python scripts/research/build_clutter_scenes.py --csv-dir <ref> --out <pkg> --motions ... --seeds ...
python scripts/research/check_g1_dataset_scenes.py <pkg>

# rollout, gate, export
scripts/research/run_kimodo_sonic_rollout.sh --scene <id> --motion <pkl> --out <dir> --task "..."
python scripts/research/analyze_kimodo_rollout_batch.py <rollouts>
scripts/research/export_kimodo_dataset_batch.sh --work-dir <dir> ...

# validate
python scripts/research/check_sonic_vla_dataset.py <ds> --profile synthetic_g1 --groot-loader-ready
python scripts/research/check_latent_parity.py --dataset <ds> --require-encoder-equivalent
python scripts/research/evaluate_synthetic_g1_dataset.py <ds>

# gallery + distribution
python scripts/research/build_dataset_gallery.py --work <work> --out <work>/gallery --json <work>/distribution.json
```

238 offline tests run without Isaac Sim: `pytest tests/dataset_generation`.
