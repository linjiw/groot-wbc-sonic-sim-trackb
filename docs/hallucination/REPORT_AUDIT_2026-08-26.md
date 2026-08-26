# LFH Code and Validity Audit — 2026-08-26

**Scope:** the geometry, scene-authoring, and model layers of `gear_sonic/dataset_generation/hallucination/`,
their tests, and the claims the docs draw from them. Every finding below was reproduced by running
code; nothing is reported from reading alone.

**Headline:** the overhead critical-support path is correct and reproduces bit-exact, so every
published overhead result stands. Eleven defects were found elsewhere, most of which **fail open** —
they return a plausible number for a question the instrument did not actually answer. One of them
had blocked the lateral constraint axis outright. Separately, the conditional `q_LFH` result does
not survive an honest baseline.

---

## 0. What was verified as correct

- `capsule_height_over_rectangle` against 20,000-case brute force: worst deviation 2.7e-6 m, all
  attributable to brute-force discretisation.
- **Golden re-derivation, after all fixes:** recomputing `overhead_face_reach` on the stored
  `lfh_089_crouch` nominal/adapted trajectory pickles at station (1.8716927, 0.1385739), 0.10 m
  along, 3.0 m across, reproduces `critical_support_cal3.json` exactly — nominal reach
  1.2961514882246337, adapted 1.2265280042653741, support 1.2445720042653741–1.2781074882246337 m,
  binding `head_torso`, critical frames 109/113.
- Binding faces land at the requested coordinate for all five overhead and three lateral archetypes
  (offset 0.0000 mm). Keep-out dilates all 29 capsules against **both** payloads. All non-binding
  archetype parts carry `constraint_context` and are checked. Committed E7 keep-out reports
  reproduce exactly.
- Full offline suite after all fixes: **839 passed**.

## 1. Confirmed defects and fixes

| # | site | defect | consequence | status |
|---|---|---|---|---|
| 1 | `reach.py` `lateral_face_reach` | reach was the max over *whole-capsule endpoints* after a mere AABB overlap test | up to **8× overestimate**; a capsule whose in-slab part lies entirely outside the height band still returned a finite reach | **fixed** — conditioned on the exact face rectangle via `capsule_height_over_rectangle` |
| 2 | `window.py` `solve_window` | groups that never occupy the face carried `-inf` reach into the margin subtraction | `+inf` margins serialise as the non-JSON token `Infinity` and are rejected downstream, so **no lateral window could ever populate a spec** | **fixed** — non-occupying groups are omitted; missing keys tolerated |
| 3 | `reach.py` `overhead_reach_profile` | `overhead_face_reach` raises when any group misses the footprint, and the profile calls it past the route ends | a face shorter than the body's along-route spread killed the **entire** profile — exactly the short-face case §2.1 introduces the profile for | **fixed** — `require_all_groups=False` in profile mode |
| 4 | `keypoints.py` `__post_init__` | `root_pos_w`/`root_quat_w` never validated finite | NaN propagates through every `argmin` station selector, so one bad frame is silently selected as *every* station | **fixed** — root tracks validated |
| 5 | `stage_geometry.py` `_records` | `direct_body` was sliced at the first nested `def` | xform ops authored after a prim's first child were dropped, and a sibling block's attributes were folded into the preceding prim — a **0.5 m** binding-coordinate error passed the 0.5 mm gate | **fixed** — all nested brace blocks excised |
| 6 | `validate_keepout.py` | `_capsule_bounds` raised uncaught when the crossing interval overran the payload | reachable on reset-spanning captures (evaluated on one shorter segment); produced **no `KeepoutReport` at all** instead of a refusal | **fixed** — converted to a `crossing_out_of_range` violation |
| 7 | `delivery.py` `DeliveryModel.predict` | `np.interp` clamps outside the calibrated range | a 120 mm strong command silently received the 60 mm response **with an unchanged residual band** — a screening estimate wearing confidence it does not have | **fixed** — raises `DeliveryOutOfSupport` |

Regressions for all seven live in `tests/dataset_generation/test_hallucination_audit_regressions.py`;
each fails on the pre-audit code.

### Why defect 1 matters beyond the code

`docs/lfh/lfh.md` records lateral critical support as empty and attributes it to physics: E3's
binding face changed the executed path before drift. That physics observation stands, but it was
never the only obstacle. Defects 1 and 2 mean the lateral instrument **could not have produced a
usable window even where one existed** — the reach was wrong, and any window containing a
non-occupying group failed to serialise. The claim "lateral is unsupported" is therefore
confounded between a physics finding and a broken measurement, and must be re-tested on the fixed
code before it is written into a paper as a negative result.

## 1b. Closed later the same day

| # | site | defect | status |
|---|---|---|---|
| 8 | `run_approved_manifest.py` | `subprocess.run(timeout=...)` kills its direct child; the driver is a shell script that execs Isaac in a further process, so a timeout killed the wrapper and **orphaned the rollout, which kept its VRAM** | **fixed** — the driver runs in its own session and the timeout signals the whole process group (SIGTERM, then SIGKILL) |
| 9 | `materialize_context_rich_scene.py:79` | `route_frame_offset_m` hardcoded `[0, lateral, z]` | **fixed** — records the true offset from the binding station; the across component measured from the route median is kept under its own name |
| 10 | `archetypes.py` / `instantiate.py` | the seed drawing non-binding **collision** dimensions was published as `visual_seed` | **fixed** — emitted as `geometry_seed`, with `visual_seed` retained as a deprecated alias |
| 11 | `trajectory_segments.best_evaluable_payload` | "best" means *first*, not best-scoring, and nothing said so | **documented** — the first pass is the one the reference commanded and the only one a stored spec's frame indices refer to |

Defect 8 was not hypothetical: it was observed live during LFH-E16. Another user's job took 23.8 GB
of the shared card, a rollout stalled in Isaac startup, the runner's 1800 s timeout fired and
recorded the cell as an infrastructure failure — and the Isaac process survived, orphaned, holding
**4.65 GB for 35 minutes** after the runner had exited. The fix is verified by a test that spawns a
wrapper with a long-lived grandchild and confirms the grandchild is reclaimed on timeout.

## 2. Findings recorded but not code-fixed

- ~~**`route_frame_offset_m` is wrong in the context-scene manifest.**~~ *(fixed, see 1b)*
  `materialize_context_rich_scene.py:79` hardcodes `[0.0, lateral, z]`. Against the committed
  `e10b_context_rich_cpu.json`, ContextLeftPillar records `[0.0, 1.05, 0.65]` while its true offset
  from the binding station is `[-1.173, +1.100]` — the along component is route-progress dependent
  (−1.17 m to +1.07 m across the four obstacles) and the across component is measured from the
  route's median, not the station. **The authored USDA geometry is correct**; only the record is
  wrong, so E10b's physics result is unaffected. Anyone reconstructing world position from the
  record is off by more than a metre.
- **`visual_seed` controls collision geometry, and the register's E10 explanation does not hold.**
  *(the naming is fixed, see 1b; the register correction stands)*
  `archetypes.py:172` draws the hanging panel's binding-cube thickness and `:97` the I-beam web
  height from the seed the manifest calls `visual_seed`; both feed `PhysicsCollisionAPI`. The
  coupling is real and the field is misnamed. But measured, seeds 33101 and 32301 give panel
  thicknesses of 0.41799 m and 0.41784 m — a **0.15 mm** difference. The E10/E10b register entry
  offers the seed-controlled thickness as the mechanism for E10's failure; that mechanism is
  numerically negligible. E10's failure is a *simulator* seed effect, not a geometry difference,
  and the register should say so.
- ~~`trajectory_segments.best_evaluable_payload` returns the **first** split segment~~ *(documented,
  see 1b; first-pass selection is the correct behaviour, it simply was not stated)*.

## 3. The conditional `q_LFH` result does not survive an honest baseline

Reproduced exactly (`evaluate_critical_distribution.py`, exit 0): 1.51373 nats, top-3 11/12,
400/400. The evaluation is mechanically leak-free — held-out atoms excluded, bandwidths constant,
global archetype vocabulary.

The baseline was the problem. Replacing the kernel with a constant, holding source balancing and
the 0.10 exploration floor fixed:

| sampler | LOSO mean log loss |
|---|---:|
| uniform over 5 archetypes | 1.60944 |
| **conditional kernel (`q_LFH_conditional_v1`)** | **1.51373** |
| **feature-blind marginal counting** | **1.46416** |
| Laplace count baseline (alpha = 0.25) | 1.43543 |

Counting beats the fitted model at every exploration setting tested. The kernel differs from
counting on exactly the three `door_lintel` labels and is worse on all three, because it upweights
source 086 — the only source with `hanging_panel` — in every fold, moving mass toward the wrong
unique archetype. Shuffling the trajectory features improves log loss in 3 of 5 shuffles.

Two supporting numbers are true by construction:

- **Top-3 recall 11/12.** `ibeam` and `shelf_plank` are verified for every source and sit at
  exactly 0.32 in every fold, so 8 of 12 labels are automatic. Counting also gets 11/12; shuffled
  features get 10–11/12.
- **400/400 in-support.** `sample()` bounds the quantile to [0, 1]; `coordinate_at()` maps [0, 1]
  affinely onto the same interval the check tests against. It cannot fail for any seed, or with the
  exploration width forced to 5.0 (confirmed 1000/1000).

Also: `hard_xi` is 0.5 for all four atoms by construction (hard is defined as the window midpoint),
so the empirical quantile spread is ~3.7e-15 and coordinate sampling is `N(0.5, 0.05)` regardless of
input. And the "novelty" ranking that puts `hanging_panel` at 0.95 for the three missing crosses is
pure counting — it is verified once and `hvac_duct` never, for any query whatsoever.

**Diagnosis: the learning target is wrong, not the fit.** The executed pair determines *where* the
face must be, and that is closed-form (§2.4) and deliberately unlearned. Archetype identity is
appearance — several archetypes realise the same face — and the trajectory does not constrain it.
The model is being asked to predict the one part of the scene its inputs do not determine.

## 4. What the evidence says the trainable target is

The pipeline's actual gate is not archetype choice; it is **delivery**. E9a stopped because a
freshly generated motion's crouch twin did not track in an *empty* scene. That is where a learned
predictor would pay for itself, and it is what `docs/lfh/lfh.md` §4.1 already names `D_phi` and
marks unvalidated (E-D1 leave-one-motion-out CV "still required").

`docs/hallucination/delivery_corpus.csv` (new, built by
`scripts/research/hallucination/build_delivery_corpus.py`) indexes every executed LFH cell — **118
cells, 89 accepted, 35 distinct reference clips** — joining the commanded edit, measured
geometrically from the stored clip pair rather than trusted from a manifest, to the recorded
verdict and tracking diagnostics. Outcomes are copied verbatim from the immutable run records;
nothing is re-graded.

The empty-scene subset is the clean delivery signal, and it is small but already directional:

| operator | cells | accepted |
|---|---:|---:|
| `local_arm_tuck` | 15 | **15** |
| `local_crouch` | 6 | 3 |

The tuck delivers at every amplitude tested, including a "strong" 85.5 mm lateral reduction at
0.400 rad of joint change. The crouch is the fragile operator, which is consistent with the
operator-survival split already recorded (arm departures 88–105%, leg departures 46–67%).

Six crouch points cannot fit a model, and the obvious single predictor fails on them: motion 089
was accepted at a nominal endpoint error of 0.139 m while 095 was rejected at 0.113 m. Two
confounds are visible instead, and both are now measured by the ladder screen:

- **095's "crouch" is not a pure crouch** — it also narrowed the arms by 32.4 mm. `lfh.md` §2.2
  anticipates exactly this coupling; it had never been measured per clip.
- **The E9a fresh motion had route straightness 0.747**, against the `MIN_ROUTE_STRAIGHTNESS = 0.95`
  that the CAL3 selection itself requires (verified sources are 0.985–0.991). It was carried to
  physics outside the calibrated envelope on straightness *and* at the top of the amplitude ladder.

**E9a's registered conclusion is therefore confounded.** It is recorded as evidence that a fresh
generated motion fails on controller delivery. What it actually shows is that *this* fresh motion,
at 80 mm on a curve of straightness 0.747, failed. Whether fresh motions can source families is
open, and the ladder screen is what tests it.

## 5. Source supply is not 4

The CAL3 screen swept four target drops and then discarded three, recommending only "the
historically verified 80 mm command", and it screened only the 15 `stand_to_walk` clips. The pool
holds **150 clips, 94 gated worth-a-rollout, of which 58 have straightness ≥ 0.95** across ten body
modes — walk, walk_look, carry_walk, reach_walk, backward, squat_pick, duck_under, side_step,
turn_in_place. Four of those 58 have ever been carried to a crouch calibration.

`scripts/research/hallucination/screen_crouch_ladder.py` screens the whole eligible pool at
40/55/70/85 mm and records, per rung, the delivered reference drop, joint excursion, cap status,
and the lateral coupling that 095 exhibited. Its output is reference-side support only: a usable
rung licenses an empty-scene physics cell and nothing more.

## 6. Consequences for the paper

1. Overhead results stand unchanged; the golden re-derivation is bit-exact after seven fixes.
2. The "learned proposal distribution" cannot be claimed. Report source-balanced counting as the
   proposal distribution, and report the refuted conditioning as a negative result with its
   baseline — it is a better contribution than an unearned gain of 0.096 nats.
3. The lateral negative result must be re-derived on fixed code before it is written down.
4. E9a must be re-run inside the calibrated envelope before "fresh motions fail on delivery" is
   stated as a finding.
5. The delivery corpus and the ladder are the route to both more sources and the first
   validated `D_phi`.
