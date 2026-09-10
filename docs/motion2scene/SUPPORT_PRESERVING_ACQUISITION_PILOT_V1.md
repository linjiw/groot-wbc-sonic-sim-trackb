# Support-preserving acquisition pilot (v1)

**Registered 2026-09-10.** One bounded intervention, chosen after the five-arm M8 comparison
returned a null downstream effect and the envelope-screen calibration located the cause in
**recall, not threshold** ([POST_M8_DECISION_V1.md](POST_M8_DECISION_V1.md) §3).

This document registers the design and records what has actually executed. It is not a summary of
the M8 result and it does not restate that null as a success.

## 1. The proposed intervention

The current hard executed-envelope acceptance gate may suppress useful physical teaching
encounters. Keeping its geometry as a **proposal bias** while allowing bounded exploration
**outside** it may produce better teaching data and better executed decisions.

This is a proposed intervention, not a demonstrated improvement. It changes proposal support. It
does **not** change the robot controller, the tracker, the schedule repertoire, the teacher, the
learner, or the task's success rule. Exploration happens in simulation only; nothing here supports
a claim about safe physical-robot exploration or robust deployment.

### Why this and not the threshold sweep

The −10/−2/0 mm negative-margin sweep is closed. It is not repeated. Over the 92 already-executed
encounters the screen's precision was 2/2 and its recall 2/7 to 3/7, and two of the encounters it
misses have covering schedules predicted **clear** that physically **failed** — no negative margin
can ever reach them. A third is rejected by the **positive** robustness bar. Requiring *less
negative interference* cannot recover any of them. The 2/2 precision figure is retrospective and
selection-conditioned; it is not an established population precision and the sweep is not held-out
calibration.

## 2. The three nested proposal controls

One fresh common candidate reservoir, same declared task domain, matching strata, no physical label
used for selection.

| Arm | Rule |
|---|---|
| **A — `support_strict`** | The unchanged executed-contrast rule and max-min-slack ranking, delegated verbatim to the frozen `acquisition_queues`. |
| **B — `support_broad`** | Every validly drawn candidate in original draw order. **Neither** the positive 81-offset robustness bar **nor** the negative nominal-interference test is applied. Environment validity, interior station, finite locked bounds and the same seven-branch action interface are preserved. |
| **C — `support_mixture`** | A committed finite alternation of the two channels, ε = 1/2, fixed before any new outcome: slots 0 and 2 guided, slots 1 and 3 exploratory. |

Conceptually μ_C(q) = (1 − ε)·μ_A(q) + ε·μ_B(q) with ε = 1/2, realised as an exact finite sampling
schedule rather than a probability that has to be estimated afterwards. Every selected encounter
records which channel proposed it.

**What the mixture gives up:** the statement that every proposed scene satisfies the old geometric
screen. It does **not** relax the physical passage/contact/recovery contract.

### Measured support of the exploration channel

The exploration channel must have support outside **both** original screens, or it does not test the
reported omission mechanism. Measured on the fresh reservoir (3,840 candidates, 1,280 per corpus
seed), at the unchanged `margin_m = 0.01`:

| Seed | Reservoir | A-eligible | Outside positive 81-offset bar | Outside negative test | Outside both |
|---|---:|---:|---:|---:|---:|
| 93201 | 1,280 | 82 | 751 | 464 | 17 |
| 93202 | 1,280 | 83 | 771 | 450 | 24 |
| 93203 | 1,280 | 81 | 797 | 425 | 23 |

The strict gate admits **6.4%** of the declared domain; **60.4%** of it lies outside the positive
robustness bar alone. The restored support is real and was not manufactured by moving a margin.

## 3. What is held fixed, and what varies

Common and unchanged across all three arms: the frozen SONIC tracker, the seven qualified
schedules, the causal 114-D student input, the complete-continuation teacher, the phase ridge
learner at `l2 = 10.0`, uniform per-phase weighting, the three decision phases, one adaptation
followed by recovery, the physics seed, and the scorer.

**Exactly one thing varies:** which channel proposes each of the four new encounters per corpus.

All three arms import the **identical** four-encounter executed-contrast prefix and the identical
M4 checkpoint, so they share a byte-identical starting point:

* shared prefix: **39 episodes, 46,488 measured physics steps** per corpus (bootstrap's 7 teacher
  branches plus 4 encounters × 8 episodes), inherited, not new, and reported once per arm rather
  than attributed to any one of them;
* new addition: **4 encounters = 32 episodes = 38,144 steps** per corpus, over a stratum schedule
  computed once per seed from the shared prefix and applied identically to all three arms.

No information-consistent teacher, feasibility classifier, new sensor, new skill or replay rule is
introduced. No learned acquisition surrogate is built.

Bounded addition budget, committed before any new outcome: **9 corpora × 32 = 288 new episodes,
343,296 physics steps.** The budget is not expanded to reach a desired result, scenes are never
replaced after an outcome, and an acquisition shortfall is retained rather than refilled from the
other channel.

## 4. Evaluation independence

The original six-context panel is development/regression evidence. One of its six contexts,
`complementary_late_development_0100`, was minted by the same clearance predicate and max-min-slack
ranking a predicate-driven selector would use, so it cannot confirm this intervention. The pilot is
therefore read out on a **separate development-validation sample**, declared before any of its
physics ran:

* drawn from the locked task domain only — its beam families and its common ranges;
* stratified jointly over **(beam family × underside band)**, 5 × 2 = **10 contexts**, both axes
  task geometry, neither an acceptance predicate;
* **zero clearance queries, zero outcome queries, no rejection sampling**, no reference to any
  arm's acceptance predicate, predicted useful contrast, or known policy failure;
* exact-identity exclusion against the 18 reserved layouts and their stress variants, the six M8
  development contexts, and every candidate in both training reservoirs — the only use made of
  reserved geometry;
* every drawn context retained with its measured outcome, **including bank-unsolved and unknown
  cases**. Because feasibility is never consulted, the sample will contain contexts no schedule
  solves. It is not redrawn because it turns out easy, hard or unfavourable. An uninformative panel
  is an explicitly inconclusive pilot.
* each layout is an independent draw and therefore its own base layout; the ancestry grouping is
  emitted so a later perturbation variant can never be reported as an independent validation item.

The eighteen reserved layouts stay inaccessible to method selection. They are not rerendered, no
response forecast is computed on them, and their geometry does not tune the pilot. They currently
have **no physics results at all** and no adoption, and both locks verify byte-for-byte against
their independently recorded `.sha256` companions.

### Panel

20 rows × 10 contexts = **200 episodes**, at one physics seed, same bank, same scorer:

* the 9 pilot policies (3 arms × 3 corpus seeds) at M8;
* the 3 frozen M8 executed-contrast checkpoints, **reused** as a located comparator and never
  counted as independently trained pilot policies;
* the unchanged strong sensor script;
* all 7 fixed schedules, which establish measured bank solvability per context.

## 5. What the comparison tests

* **C vs A** — whether preserving exploratory support improves the strict-gate method.
* **C vs B** — whether the geometry-guided portion adds value beyond broad coverage.
* **B vs A** — whether broadening support alone changes anything.

If B and C improve equally, the supported conclusion is that **broadening support** helps, not that
the mixture is superior. If screen recall or action diversity improves without physical policy
improvement, the central method hypothesis remains unsupported. If useful measured distinctions
exist but the policy cannot use them, the next step is one bounded teacher/information/fitting
intervention — not a capacity verdict read off coefficient magnitudes.

Primary: actual passage, per corpus, every assigned context, plus matched passage-time differences
over mutually successful assignments only. Passage time is crossing plus stabilization; it is kept
distinct from whole-episode adaptation and recovery completion, and no energy cost is invented.

Secondary mechanism readouts — bank solvability, passing-set patterns, added response coverage,
presence of usable continuation targets, and the physical outcomes of off-gate proposals — explain a
result. They are not the success criterion. A manipulation check that fires is not the result.

Nine corpora across three seeds are not nine independent replications, and seven teacher branches
within an encounter are not replications at all.

## 6. Preserved work

The declared **M16/M32 trajectory is preserved and unchanged**. It has zero progress past encounter
index 8 and nothing was in flight when this pilot was registered; the pilot writes a separate
execution root and shares the common acquisition lock, so neither can silently overwrite or
contend with the other. The five-arm M8 corpora, their checkpoints and the completed M8 panel are
untouched. Replay remains a documented, tested component with a measured null, not a headline.

## 7. Implementation

New, all under `scripts/research/` — nothing under `gear_sonic/`, whose contents are a hash-bound
editable runtime source (`m2s-native-runtime-freeze-v6` pins it to one commit plus exactly 44 dirty
files, and any added file there fails the launch preflight):

| Script | Role |
|---|---|
| `motion2scene_support_pool.py` | the three proposal controls, the shared stratum schedule, the committed mixture schedule |
| `motion2scene_support_validation.py` | the task-geometry-only validation draw |
| `motion2scene_prepare_support_pools.py` | fresh reservoir, unchanged clearance screen, queue construction, scene materialization |
| `motion2scene_support_plan.py` | the pilot plan: shared prefix plus four new rounds per arm |
| `motion2scene_support_acquisition.py` | execution, reusing the frozen controller, backend and fit |
| `motion2scene_support_validation_sample.py` | mints and registers the validation contexts |
| `motion2scene_support_validation_panel.py` | declares, prepares and runs the 200-episode panel |
| `motion2scene_support_pilot_readout.py` | the readout described in §5 |

Tests: `decoupled_wbc/tests/test_motion2scene_support_pool.py`,
`decoupled_wbc/tests/test_motion2scene_support_validation_panel.py`.

The panel's resource gate replaces the inherited `pgrep -f` pattern with a `/proc` scan that
matches only a Python process whose own arguments name the driver. The old pattern also matched any
shell command that merely mentioned the driver path — including an operator's own status check —
which made a live runner believe the GPU was busy and stall in `waiting_for_native_capacity`.

## 8. Execution status

Recorded honestly; this section is the only part of this document that changes as physics runs.

**Completed, zero physics:**

* Fresh reservoir registered and screened: 3,840 fresh candidates over 3 corpus seeds,
  `--rounds-per-stratum 512` with the M8 pool's first 256 rounds reserved and never reselected.
* Draw-stream audit: all 3,840 previously screened candidates reproduce **exactly** (0 geometry
  mismatches across all three seeds), so the declared domain has not drifted.
* Support measured (§2): the exploration channel has support outside both screens.
* Pilot plan built: 9 corpora, ε realised at exactly 0.5 in every seed, **no acquisition
  shortfall**.
* Validation sample minted: 10 contexts, 0 clearance queries, 0 physics steps, 0 exact-identity
  collisions against reserved layouts, development contexts or training reservoirs.
* Runtime readiness verified once: the frozen runtime context, all four source closures and the
  inherited M4 ledger load and hash-verify for a pilot arm.
* 29 tests pass; Ruff and Black (the gate `lint.sh` enforces) pass on all new files.

**Measured cost of one episode**, from the M8 panel's own 138 `attempt.json` files: median
**56.5 s**, p95 60.3 s of driver wall time. The 32 s/cell figure in earlier notes describes the
retired 199-frame/4-second generation and does not apply to this 299-frame path. The full pilot is
therefore 488 episodes ≈ **7.6 GPU-hours**, of which acquisition is 288 episodes ≈ 4.5 h.

At the measured 56.5 s that fits inside 8 actual GPU-hours, but **not** inside the rolling gate's
conservative 375 s/cell *reserve*, which would admit about 76 cells today. That reserve is a ~6.6×
overestimate of measured cost and is **not enforced on this code path** (acquisition and panel
episodes write `attempt.json`, not the `run_record.json` the ledger reads). The reserve constant was
left unchanged; whether to raise it is the principal investigator's call, and this pilot does not
raise it.

**Physics executed:** see the receipts under the pilot execution root. Acquisition advances all nine
corpora in lockstep by encounter boundary, so partial progress stays matched across arms rather than
completing one arm first.

### First matched layer: all nine corpora at slot 0

Measured, not predicted. Each corpus's first new encounter, with the gate membership recorded in the
plan before any physics ran:

| Channel | Encounters | Gate membership | Passing branches | With a usable continuation target |
|---|---:|---|---|---:|
| strict | 6 | inside both screens | 4/7 in every case | **6 of 6** |
| broad | 1 | inside positive, outside negative | 5/7 | 1 of 1 |
| broad | 2 | **outside positive**, inside negative | 0/7 both | **0 of 2** |

Three observations, all provisional at nine encounters:

* The strict channel is perfectly consistent — 6 of 6 solvable, all at 4/7. The acquisition
  mechanism was never what the M8 null impugned.
* The exploration channel splits entirely on the **positive 81-offset bar**: inside it, the richest
  table measured so far; outside it, nothing passes at all. The positive bar certifies that some
  schedule can physically clear. The negative test only certifies that some schedule collides, and
  its recall was the measured problem.
* The one useful off-gate encounter passed `neutral` and `short`, which strict encounters exclude
  by construction — requiring an interfering alternative selects scenes where walking through fails.
  If off-gate-but-positive-clearing encounters systematically teach *when not to adapt*, that is a
  lesson the negative requirement structurally cannot propose.

**How this fails.** If that corpus's policy still collapses to a constant schedule on the validation
panel, the richer teacher table bought nothing and the mechanism fails at the learning step, not the
collection step. Three of nine corpora are drawing from a region that produced nothing, so the broad
arm may be *worse* than strict. The mixture is the arm positioned to benefit, and ε was fixed first.

Two earlier readings of this data were wrong and are corrected here: the all-fail encounters are not
explained by the command-tick guard (one failed across the full horizon with no guard violation, so
the property is simply that no schedule clears), and encounters that teach nothing are not reliably
cheap (one cost 4,564 steps, the other the full 8,344). Only the narrow claim stands — cheapness can
correlate with uselessness, which is why measured steps must not normalise improvement.

### Two defects found before physics, both fixed outside frozen files

1. **Rejected split.** The sample originally stamped `split="development_validation"`, but the
   frozen collector's `validate_scene` accepts only `("development", "reserved_evaluation_v3")`.
   The panel's very first `prepare` would have aborted — *after* all 288 acquisition episodes were
   spent. The collector is hash-pinned in three closure lists of the adopted runtime, so widening it
   would break every acquisition arm. Fixed by stamping `split="development"` and carrying the role
   as a separate `sample_role` field; all 10 definitions now pass the frozen validator, and a
   regression test pins the contract.
2. **Latent `artifact(str)` in the frozen runner.**
   `ExpandedController.acquire_expanded` calls `artifact(slot["inherited_model"])`, which arrives
   from JSON as `str` while `artifact()` requires a `Path`. Both files match the sha256 values
   recorded in the M8 plan and the adopted runtime, so the defect is latent and pre-existing rather
   than introduced here — and **a resume of the declared M16/M32 trajectory would hit it too**, since
   that path also imports an inherited prefix. It is repaired by overriding the method in the
   pilot's own `SupportController`, leaving the frozen file byte-identical. This is flagged, not
   fixed in place: changing that file would invalidate the M8 plan's `implementation` hashes.
