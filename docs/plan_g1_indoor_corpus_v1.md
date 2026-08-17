# G1 Indoor Corpus: Plan to a Benchmark-Grade, Real-Anchored Dataset

Scope: one RTX 5090 (32 GB, sm_120) · target ICRA 2027 (deadline Sept 15, 2026; conference Seoul, May 2027) · fallbacks RA-L or RSS 2027
Baseline: dataset_v2 as of 2026-08-16 — 15 episodes / 3,720 rows, 18 scenes in 5 packages, 26/26 latent parity, 238 offline tests, 2 usable motions.

---

## Progress against this plan (updated 2026-08-16)

**Phase A closed except the throughput benchmark.**

- **A1 Kimodo unblocked.** Both existing envs carried `torch 2.5.1+cu124`, arch list ending at
  `sm_90`. New cu128 env at `/data/robotixx/envs/kimodo_sm120`
  (`install_scripts/install_kimodo_sm120.sh`). Generation costs **13.5 s per 5 s motion at
  ~2 GB**. The 15 GB LLM2Vec encoder is now a separate CPU stage writing a prompt cache
  (3.8 s/prompt), so it never competes with the GPU.
- **A2 First new-prompt motions rolled out.** 2 of 3 accepted; the third was correctly
  rejected (see below). One walks at **1.278 m/s** against the corpus's entire 0.653–0.747
  band.
- **A3 GR00T loader smoke passes for real** at `ab88b50c`, against the official
  `ShardedSingleStepDataset` rather than the fake. Env `/data/robotixx/envs/groot_loader`.
- **A5 Licensing settled — we can release the motions.** NVIDIA Open Model License; an output
  is explicitly not a Derivative Model. Details and the one open item (Llama-3 gating for
  reproducers) in `docs/kimodo_output_licensing.md`.

**Phase B underway.** Taxonomy of 150 prompts across 14 body modes × 3 speeds × 5 turn styles
(B1); scene-first route sampler (B2); goal-object and task-language derivation (B3); diversity
dashboard (B4). Measured over 114 generated references, the library already spans **0.301–1.739
m/s** and heading changes to **6.721 rad**, against a corpus that turns essentially not at all.

### Four findings that change the plan

1. **Effective rank was being measured the wrong way.** The datasheet's "≈4" was the
   *within-episode* rank. Corrected: pooled **5.22**, between-episode **1.16**, within-episode
   4.10. The between-episode figure is the damning one — the accepted episodes' mean actions
   span essentially one direction.
2. **Not every generated motion is trackable, and it is predictable.** A deep crouch produced
   9686 N of hip-into-pelvis force from an unreachable reference. Joint-limit saturation
   predicts peak self-contact at **r = 0.933** over 11 motions and screens it out before any
   GPU time (`docs/motion_prefilter_validation.md`).
3. **Only 2 of 21 scenes can grade a policy.** Every generated scene encodes the tracking error
   of the motion it was built around. `density_dense` and `density_tight` admit *no* independent
   route at all. This makes eval-scene generation (C3) blocking for the Phase D baselines,
   earlier than this plan assumed.
4. **No acceptance gate is marginal — but nine have never fired.** Every threshold that fires
   sits in an empty gap wider than the passing range, so its exact value changes no outcome
   today. The calibration pilot (C1) should therefore run *after* Phase B diversity populates
   the gaps, and should target episodes near a threshold rather than a random 100.

### Round two: what running 40 varied motions broke

Full write-up in `docs/round2_diversity_findings.md`. Five problems, all invisible with two
motions, and one hypothesis that was wrong.

5. **Every capture was truncated at 59% of its motion**, cutting the second clause off every
   composite behaviour the taxonomy exists to produce. Fixed: the step count is read from the
   motion library.
6. **"Unevaluable" was indistinguishable from "rejected".** One episode reported
   `accepted=False` with an *empty* reason tuple while carrying 1776 N of scene contact no
   gate had seen. Fixed: reset-spanning captures are split and recovered, and the outcome is
   now three-way with the acceptance rate divided by *evaluated*.
7. **The body radius described the pelvis, not the robot** — 0.45 m against a measured
   0.434–0.664 m swept half-width, exceeded on 98% of episodes. Corrected, along with the
   route clearance.
8. **Episode length is a hidden parameter of every acceptance rate.** Grading the same
   episodes at increasing horizons gives 83% at 1.2 s falling to 64% at 4.8 s, while p95 path
   error grows 0.098 → 0.248 m. **The usable horizon of SONIC on novel motions is about 4–5 s
   at the current threshold** — a controller property worth reporting on its own. Generation
   now defaults to 4.0 s, chosen off that curve.
9. **The clearance budget did not explain the collisions.** The two episodes with scene
   contact had the *lowest* clearance requirement; the three that exceeded the budget had
   none. Recorded so it is not retried.

**C3 delivered:** eval-eligible scenes went from 2 to 14. Rooms are cleared against the
*commanded* path plus a measured drift allowance, and each is certified to admit routes other
than the reference before it ships. The cost is stated: eval rooms run 9.5–21.1% occupancy
against training rooms' 20–38%.

**Throughput is dominated by contention, not by our code.** The identical 150-motion sweep
took 1984 s with six other GPU processes and 290 s with two — a 7× swing. No throughput
number from this machine means anything without the contention it was measured under.

---

## 1. The story: replicable by construction, verified by measurement

Your strongest asset is one nobody else in synthetic humanoid data can claim, and it is currently framed as a weakness. The scenes are primitive Plane-and-Cube geometry — which means **every scene manifest is a build sheet**. Axis-aligned boxes with recorded dimensions, positions, and clearances can be reproduced on a lab floor with foam blocks, tables, and tape in an afternoon. Combine that with two facts you have already measured: the exported `action.motion_token` is byte-comparable with the `token_state` the C++ deployment runtime consumes (26/26 rollouts, residual RMS 0), and you own a real G1 plus a motion-tracking pipeline.

That closes a loop no photorealism-first dataset can close:

> Text prompt → Kimodo motion → scene generated around the executed corridor → SONIC physics validation → deployment-parity action labels → **the same scene rebuilt physically from its manifest → the same motion executed on the real G1 through the same deployment runtime → motion-tracked and compared against the simulated trajectory.**

The paper's one-sentence claim becomes: *every episode in this dataset is physically realizable, and we prove it by realizing a sample of them.* Photorealistic corpora argue their pixels transfer; you demonstrate your **behaviors and geometry** transfer, with the action labels already in the robot's native contract. Do not chase photorealistic assets for v1 — the primitive constraint is now load-bearing for the story. Textured realism becomes a later test-split axis, exactly as section 22.5 already sequenced it.

Keep the epistemic voice of the status page ("every claim is a measurement, including the ones that came out worse than expected"). If the real-world replication shows a larger gap than hoped, report it — the measurement is the contribution, and that honesty is what will distinguish this from marketing-grade dataset papers.

## 2. What the paper harvests from work already done

Freeze and cite, do not rebuild. Four of your engineering episodes are publishable methodology findings in their own right, and together they form a "lessons for synthetic robot data" section that reviewers will remember:

1. **Goal leakage caught only by eyes** (section 21): debug markers rendered the tracking target into the observation on 243/249 frames while every numeric gate passed. This is a warning every synthetic-data group needs, plus the automated detector you built afterward.
2. **Contact decomposition by Newton's third law** (section 18.3): splitting self-contact from scene contact via paired equal-and-opposite forces, which raised usable data 3.8× by measuring the right quantity rather than relaxing a threshold.
3. **Gates relative to the command** (section 20): the tracked-crouch/kneel/bow false rejections and the principle that acceptance must compare against what was commanded, with absolute limits reserved for states no command can request.
4. **The robot is not a cylinder**: per-frame swept volume (0.273–0.664 m half-width, 1.317 m ceiling) licensing cantilevered clutter that a 2D footprint model cannot express.

The provenance chain, quarantined-failure corpus, and honest negative results (the causal-alignment check that cannot work and says so; the broken overhead camera that was reverted rather than shipped) round out that section.

## 3. The three gaps that decide acceptance

**Gap 1 — Behavior diversity.** Two usable motions, latent effective rank ~4, every accepted episode a ~3.5 m walk at ~0.7 m/s. Your status page already names this the honest headline; a reviewer will name it the fatal flaw of a "navigation dataset." This is entirely blocked on the Kimodo generation environment, and Phase A removes the block.

**Gap 2 — Scene→action causality.** In scene-around-motion data, the geometry is generated *from* the action, so actions are statistically independent of the scene given the motion — obstacles never cause behavior. With only two motions, a policy can fit the data while ignoring vision entirely. Two fixes compound: motion diversity makes vision informative (with 50+ motions, the corridor shape in the ego view becomes the only signal disambiguating which path is being executed — which is precisely the visual-navigation correlation you want, produced for free by the inverted generator); and mixing in **scene-first episodes** via the section 6.3 route sampler restores true geometry→route causality, with the generation direction recorded per episode in provenance. Ship the diagnostic as a headline ablation: a sighted baseline versus a blind (no-image) baseline on held-out scenes. If sighted wins decisively, the dataset demonstrably teaches visual navigation; report the gap as a standing dataset-health metric.

**Gap 3 — No learning or deployment evidence.** Without a trained policy and a real-robot anchor, this reads as an excellent engineering report. Phase D delivers sim closed-loop results; Phase E delivers the minimum viable real anchor; and there is a cheap third leg — because the token conventions coincide (measured), a sim-trained VLA can be evaluated **offline on your existing real teleop recordings** by action-prediction error against the recorded `token_state`, giving a real-data evaluation with zero robot risk.

## 4. Operating a single RTX 5090

**Unblock Kimodo first.** The block is the environment, not the hardware: current stable PyTorch cu12.8+ wheels support Blackwell/sm_120. Recover disk (the doc flags this as the reinstall risk), then build Kimodo a fresh venv or container with a modern torch, leaving the Isaac Lab 2.3.2 / Isaac Sim 5.1 environment untouched — the design's file-based stage separation exists exactly so these never share dependencies. Fallback if Kimodo rejects modern torch: a few dollars of rented A100 time, shipping qpos CSVs back through the same artifact contract. Verify with one genuinely new prompt→CSV→rollout — the first newly sampled motion in the project's history.

**Time-multiplex by day and night.** Daytime GPU: interactive Isaac rollouts, renders, and debugging. Nighttime GPU: unattended batch rollouts or, later, training. Everything CPU-bound — scene generation, preflight, adapters, export, validation, analysis, all 238 offline tests — runs in parallel while the GPU is occupied. Your resumable batch runners and explicit success markers are exactly the right shape for a machine that gets preempted daily; keep enforcing marker-plus-artifact success, never exit codes.

**Skip building render-replay for v1.** The two-pass design (section 8.3) exists to avoid rendering rejected rollouts, but scene-around-motion acceptance is now high (20/24), so rendering with physics wastes little. Deleting one engineering task from the critical path matters more on one GPU than saving ~20% of render time. Revisit at M1 scale.

**Measure throughput before promising scale.** Run a one-hour benchmark night: rollouts completed, wall time each, disk per episode. Working estimate: if a 249-frame rollout with ego camera costs ~2–4 minutes, then ~1,500–2,500 accepted episodes fit in roughly 10–14 unattended nights at your current acceptance rates — but plan from the measurement, not this guess. Storage discipline: H.264 ego video dominates; keep physics trajectories compressed and prune quarantine videos aggressively (keep their trajectories and reason codes).

**Training fits with care.** GR00T fine-tuning on 32 GB should be approached as LoRA or action-head tuning with bf16 and gradient checkpointing, small batch with accumulation. Freeze the dataset **before** training starts so the GPU calendar has no contention between generation and learning. Measure a 30-minute VRAM probe before committing to a multi-day run.

## 5. Phase plan

| Phase | Dates (2026) | GPU load | Deliverable | Exit gate |
|---|---|---|---|---|
| A — Unblock & close M0 | Aug 16–21 | light | Kimodo generating on the 5090; GR00T loader smoke at pinned rev; 20 reviewed episodes | first new-prompt motion rolled out and exported; M0 checklist fully green |
| B — Behavior library & scale | Aug 20–30 | heavy nights | ≥50 distinct accepted motions; both generation directions; 1–3k accepted episodes; goal labels + language | diversity dashboard flips: effective rank, path/speed spread, behavior clusters |
| C — Benchmark freeze | Aug 28–Sep 4 | light | calibrated thresholds (100-episode reviewed pilot); frozen splits + leakage test; closed-loop eval harness | dataset v1.0 tagged; no PROVISIONAL labels remain |
| D — Baselines | Sep 1–9 | heavy days+nights | GR00T LoRA fine-tune; blind ablation; density- and scene-generalization evals; offline real-episode action prediction | sighted-vs-blind gap and closed-loop success table complete |
| E — Real-world anchor | Aug 25–Sep 10 (robot-parallel, GPU-light) | none | 2–3 sim episodes physically replicated, mocap-compared | paired sim/real trajectory figure with quantified gap |
| F — Paper & release | Sep 5–15 | light | paper, dataset card, HF release, eval code, quarantine corpus | submitted |

**Phase A details.** Disk cleanup; fresh Kimodo environment; one new prompt end-to-end. Install the pinned Isaac-GR00T revision (`ab88b50c`) and run the real `ShardedSingleStepDataset` smoke — the last unchecked M0 box. Finish the 34-config placement batch and human-review to 20 accepted episodes. Also run the Phase E go/no-go: confirm the C++ deployment runtime tracks a bundled motion on the real G1 on open floor. If that works in week one, the real anchor is low-risk; if not, descope Phase E to the follow-up paper without touching the rest of the plan.

**Phase B details.** Build a prompt taxonomy (~100–200 prompts): gait styles × speeds × turn radii × stops-and-holds × crouch/duck-under × side-steps × backward segments × pause-and-look. Kinematic prefilters (section 7) keep simulation time on plausible candidates. Generate scene-around-motion episodes for new motions *and* scene-first episodes by sampling routes in the existing 18 scenes plus generated layouts, feeding routes to Kimodo as root-path constraints. Add the minimal semantics slice: export the furniture category strings already carried internally, annotate the goal object nearest each endpoint, and template task language ("walk between the shelves and stop at the crate") — this upgrades style prompts into task-conditioned instructions, which is what the VLA benchmark needs. Track behavior diversity nightly with the metrics you already compute.

**Phase C details.** The 100-episode reviewed calibration pilot has been deferred three times in the docs; it is now blocking, because a released benchmark with provisional gates is a review liability. Calibrate the self-contact threshold (measured peaks 8–911 N against the 343 N anchor), the 0.15 m / 0.35 rad command-deviation limits, and the 0.25 m p95 path threshold that currently rejects clean-but-8%-over curved tracking. Freeze `split_group` partitions with the automated leakage test. Critical benchmark rule: **evaluation scenes must never use executed-swept-volume geometry** — the `tight` preset bakes one policy's exact trajectory into the room and is only collision-free for that policy; eval scenes use reference-path clearance plus a margin covering policy variation, or scene-first routes. State this distinction explicitly in the paper; it is the kind of self-aware precision that builds trust.

**Phase D details.** Closed-loop evaluation is the money result: the trained policy emits 64D tokens, the SONIC decoder executes them in held-out scenes, and you report success rate, lateral scene-contact force, endpoint error, path efficiency, and goal-hold time — the same gates, now grading a policy instead of a rollout. Ablations: sighted vs blind; train sparse/moderate → test dense/tight; held-out scene families; and the offline action-prediction evaluation on real teleop episodes.

**Phase E details.** Pick one scene (the household room, or a `moderate` clutter layout). Write a small script that turns a scene manifest into a printable build sheet: floor-plan with tape coordinates, box footprints and heights, origin and heading marks. Build it with foam/cardboard boxes. Run 2–3 accepted motions through the deployment runtime, mocap markers on pelvis and feet, two repetitions each. Align via the taped scene origin, then produce the signature figure: real executed path overlaid on sim executed path over the obstacle footprints, with endpoint error, minimum clearance, and contact-free confirmation. Safety: start open-floor, add boxes incrementally, respect the sim-measured clearances as the expected envelope, e-stop briefed. Three paired episodes are enough — this is an anchor, not a study.

**Phase F details.** Paper assembly (section 6 below), dataset card, licenses, HuggingFace hosting in the LeRobot layout, eval harness release, and the quarantine corpus with reason codes as a secondary artifact — releasing labeled failures is rare and genuinely useful to controller researchers. Emphasize regenerability: byte-identical reconversion from manifests means others can extend the corpus, which honestly answers "only a few hours of motion."

## 6. Claims → evidence map

| Paper claim | Evidence | Produced in |
|---|---|---|
| Action labels match the deployment contract | FSQ-lattice parity, 26/26 + all new rollouts, gated at export | done; re-verified per batch |
| Scenes are physically replicable | manifest→build-sheet + mocap-compared real runs | Phase E |
| The dataset teaches *visual* navigation | sighted vs blind gap on held-out scenes | Phase D |
| Behavior axis is no longer degenerate | ≥50 motions; rank/spread/cluster dashboard | Phase B |
| Geometry→action causality exists | scene-first subset + direction recorded in provenance | Phase B |
| Gates are calibrated, not provisional | 100-episode reviewed pilot; frozen thresholds | Phase C |
| Policies generalize across density and scenes | closed-loop success on held-out axes | Phase D |
| Synthetic transfers toward real data | offline action prediction on real teleop episodes | Phase D |
| Methodology findings | goal leakage, contact decomposition, command-relative gates, swept volume | done; write up |

## 7. Risks and honest outs

**Kimodo output licensing.** Before any public release, confirm the license terms for redistributing Kimodo-generated motions and derived episodes; NVIDIA research licenses sometimes restrict this. Check now, not in Phase F — it shapes whether you release episodes, manifests-plus-regeneration-scripts, or both.

**VRAM for fine-tuning.** If even LoRA does not fit comfortably, tune the action head only, or accept longer wall-clock with aggressive checkpointing. The blind ablation is cheap either way and must not be cut.

**Real-robot gap larger than expected.** Report it with the same voice as the causal-check demotion. A quantified sim-real gap on paired episodes is a result; a suppressed one is a scandal waiting for a reproducer.

**Timeline slip.** The most likely slip is Phase D training. If baselines are not solid by Sept 10, submit to RA-L (dataset-friendly, with an ICRA presentation option) or hold for RSS 2027 (~late January) — which would also buy time for photorealistic test splits and longer stitched routes. Do not compromise the calibration pilot or the leakage test to make the date; those are correctness, not polish.

**Single-GPU contention.** The calendar only works if the dataset freezes before training begins and nights are treated as batch capacity. Any "quick daytime training run" during Phase B will silently cost a generation night.

## 8. Beyond v1 (the research program)

The ICRA paper is the locomotion benchmark plus the sim-real anchor. The arc after it, in order of leverage: photorealistic and textured scenes as a *test-split* generalization axis (with the USD-native geometry gate and per-asset licensing the design already specifies); a semantic placement grammar so goals become object-centric household tasks; longer stitched multi-segment routes; dynamic obstacles as a time-aware task family; and the M2/M3 manipulation extension riding on the same contracts. Your motion-tracking pipeline also opens a second data direction later — captured human motion as an additional reference source alongside Kimodo — but keep it out of v1 scope.

## 9. Definition of done — dataset v1.0

- [ ] Kimodo generates new motions locally on the 5090; ≥50 distinct accepted motions in the library.
- [ ] ≥1,000 accepted episodes spanning both generation directions, with direction, goal object, and task language in every episode.
- [ ] All thresholds calibrated by the reviewed pilot; no PROVISIONAL labels in code or docs.
- [ ] Splits frozen by `split_group`; leakage test green; eval scenes contain no executed-swept-volume geometry.
- [ ] Official GR00T loader smoke passing at the pinned revision.
- [ ] Sighted vs blind and closed-loop generalization tables complete.
- [ ] 2–3 mocap-compared real replications, or an explicit descope note moving them to the follow-up.
- [ ] Licenses confirmed; dataset card, eval harness, quarantine corpus, and regeneration manifests released.
- [ ] Stratified human review with documented false-accept/false-reject findings (the M1 gate), not just automated detectors.
