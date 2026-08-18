# Guidance, 2026-08-18 (C): stop expanding the apparatus

Saved verbatim in substance; the operative instruction is at the end.

## The judgement

SweepCF is now a credible counterfactual-data **construction method**. What remains is to show
that the data it produces actually teaches a learner to select a different whole-body adaptation
from geometry, and that the ability transfers to scenes never fitted to a trajectory. The board
itself marks claims 5 and 6 as not started. The research bottleneck has moved from the data
pipeline to **learning evidence, dataset scale and generalisation**.

## Positioning: this is not another large humanoid dataset

The neighbourhood is crowded. MTC has 348 clutter-aware trajectories over 145 3D scenes and already
argues geometry-induced whole-body adaptation. VLK has 48,000 vision-language-kinematics
trajectories tested on a real Unitree G1. HumanoidPF has studied crouching, hurdling and squeezing
with real-robot clutter traversal. On raw scale, photorealism and real-robot demonstration this work
will not win.

What is unique:

> Most datasets add **scene diversity**. SweepCF adds **decision diversity**.

Fully:

> SweepCF constructs matched, physics-verified motion–scene counterfactuals in which the task,
> route, start, goal, controller and motion phase remain fixed, while one geometric constraint
> changes the minimum feasible whole-body adaptation.

Title, converging: *SweepCF: Physics-Verified Counterfactual Data for Scene-Conditioned Humanoid
Traversal* (safer), or *When Must a Humanoid Adapt? Counterfactual Supervision for Whole-Body
Traversal*.

## What is strongest today

**A real causal unit.** The scene-around-motion corpus produced different images with a bit-identical
state trajectory — visual augmentation, not behavioural diversity. The matched family changes that:
root XY identical frame by frame, duration and gait phase and start and goal identical, waist change
0, six leg joints altered, both succeed in easy, only the nominal fails in hard. A reviewer can no
longer say the shelf separated two different journeys.

**Geometry proposes; physics certifies.** The predicted 97.7 mm window is really 22.7–81.7 mm. The
nominal capsule is conservative; the adapted crouch under a lower shelf fails not by geometric
penetration but because downward pressure costs the tracker its reference. This is not a method
failure — it is a stronger method design. Capsule envelopes screen pairs, body groups, stations and
parameter ranges cheaply; physics rollouts find the real feasible window; only families whose real
window is non-empty are released.

**The validation system finds its own errors.** `disallowed_robot_contact` saw only horizontal force
and so missed pure −z pressure on the torso. It was fixed, 230 episodes re-audited, and several
earlier force/penetration conclusions corrected. Valuable for a dataset paper — but the paper
carries only the final definition, not the debugging chronology:

    support contact:            +z
    overhead external contact:  -z
    lateral external contact:   xy
    self contact:               paired internal forces

## Where a reviewer would reject this today

**Four families is not a dataset.** Enough to show the method exists; not enough to show generator
yield is stable, that a selector generalises across motions, that SweepCF beats random sampling,
that scene-first generalisation is not luck, or that the learner is not memorising motions. "This
question needs contrast, not corpus size" was a fair early engineering judgement and is not a
dataset-paper strategy. More precisely: *method existence needs contrast; a dataset paper needs
independent contrast at sufficient scale.*

| dimension | minimum to submit | stronger |
|---|---|---|
| distinct nominal motions | 8 | 12–16 |
| geometry regimes | 2 | 2 (overhead + lateral) |
| verified counterfactual families | 24–30 | 40–60 |
| matched-operator families | ≥12 | ≥24 |
| frozen scene-first scenes | 30 | 30 is enough |
| human-reviewed rollouts | 100 | 150 |
| two-rater independent review | 30–50 | 50 |
| learner seeds | 5 | 5–8 |

Report simultaneously: family count, distinct nominal-motion count, distinct adapted-motion count,
distinct route count, scene-instance count. Five shelf heights on one nominal are five scene
instances, never five independent behaviour families.

**The selector's 0.963 is not a paper result.** It shows the loader, metric and controls work and
that ordering is not being read — benchmark unit tests. It is not evidence for claim 5: the families
are synthetic and built to validate the harness, and the privileged model was handed the limiting
margin. The main result must come from physics-verified families, held-out nominal motions, frozen
scene-first scenes, a deployable scene representation such as depth-derived geometry, and the
decorated/random/SweepCF comparison.

**Behaviour labels cannot trust the prompt.** Of 156 evaluable episodes 132 were accepted on
tracking, but of 37 with a semantic predicate only 17 performed the labelled behaviour; of 150
prompts, 75 admitted a predicate and only 24 contained the target behaviour. Whole-clip style is
reliable; events at a specified moment essentially fail. So the release must separate
`prompt_intent`, `reference_semantic_valid`, `executed_semantic_valid`, `operator_type`,
`physics_outcome`, `scene_compatibility`. SweepCF's advantage is exactly that a local operator's
semantics are guaranteed by construction rather than by trusting a prompt.

**The canonical hard negative is too violent.** 1.2574 m gives a 3253.5 N nominal collision — a
crash, not a clean near-boundary matched negative. The already-measured **1.2801 m** is better:
85.0 N overhead, 0.0 lateral, drift below threshold, rejected solely by the corrected contact gate.
Re-verify nominal × 1.2801, crouch × 1.2801, and the three registered jitters; if all hold, make it
canonical. Define the hard scene as the *minimum reliable intervention that produces the intended
failure across registered perturbations*. Do not target peak force — force is highly sensitive to
contact detail. Gate on outcome, first-contact body, contact direction, contact timing, impulse, and
robot-remains-upright.

## The method, as six steps

1. **Qualification** — start from an already-accepted nominal motion.
2. **Matched local adaptation** — `m_c = T_crouch(m_0)`, `m_a = T_arm(m_0)`, preserving root XY/yaw,
   duration, gait phase, start/goal, and everything outside the operator window.
3. **Station selection** — `s* = argmax_s [E(m_0,s) − E(T(m_0),s)]`. Random station and route
   midpoint find zero ≥50 mm families across 55 overhead pairs; maximal separation finds 5. That is
   already a strong method ablation.
4. **Geometric candidate window** — `W_geom`, a *proposal* only, no longer called the feasible window.
5. **Physics refinement** — bracket the nominal's failure boundary and the adapted motion's success
   boundary by rollout, giving `W_phys ⊆ W_geom`. Report predicted window, observed lower and upper
   bounds, window-realisation ratio, probes consumed, failure body/time accuracy.
6. **Family certification** — release only when `y(S_e,m_0)=1`, `y(S_e,m_a)=1`, `y(S_h,m_0)=0`,
   `y(S_h,m_a)=1`, with the nominal's hard failure occurring first at the intended obstacle, on the
   intended body group, before unrelated drift; positive cells free of scene contact; and the
   operator's semantics still present in execution.

## Claim 4 can be marked established now

Do not claim "more natural". Define the **minimum-edit feasible behaviour** under a lexicographic
rule: `m*(S) = argmin_m D(m, m_0) s.t. y(S,m)=1`, with `D(m_0,m_0)=0` and `D(T(m_0),m_0)>0`. In the
easy scene both succeed and the nominal is chosen at zero edit; in the hard scene the nominal fails
and the adapted motion becomes the minimum feasible one. That is a strict reversal and needs no
weights over knee angle, energy or joint travel. So claim 4 becomes *the scene reverses the
minimum-edit feasible behaviour*, established on the matched overhead family. Keep the continuous
cost for comparing two different adaptations later; do not let it block the paper.

## The central experiment is the 3×3, not many separate 2×2s

Bank `M = {m_nominal, m_crouch, m_arm-tuck}` against easy / overhead / lateral:

| candidate | easy | overhead | lateral |
|---|---|---|---|
| nominal | success, preferred | fail | fail |
| local crouch | success | success, preferred | fail or non-preferred |
| local arm tuck | success | fail or non-preferred | success, preferred |

This shows the robot knows not merely *whether* to adapt but *which part of the body to change*.
Better Figure 1 than two unrelated 2×2s. Floor is 0 — do not chase high-knee to fill a third regime;
overhead and lateral already act on lower and upper body, which is orthogonal enough.

## Arm tuck: stop hunting a universal predictor

The honest conclusion is subtler than "predictable / not predictable": crouch excursion shows a
monotone dose–response with drift *within* a motion, but the threshold is **motion-specific**;
x000's trackable range yields only a 43.1 mm window while the excursions giving 92–96 mm are
untrackable. So the correct screen computes on CPU the excursion a target window needs, then spends
**one** rollout testing trackability at exactly that strength — not a global cap, not a clip-only
predictor. Board wording should become *crouch feasibility is motion-specific but cheaply screenable
with one targeted rollout*.

For the arm tuck, two predictors are refuted; accept it as a measured-yield process and do not hunt a
sixth proxy from three samples. One bounded, physically motivated change is still worth making:
**bilateral tuck → critical-side one-arm tuck**, since lateral obstacles are usually one-sided. It
reduces joint intervention, lowers self/ground collision risk, keeps the other arm's natural swing,
and matches the signed envelopes: compute `w_L(s) = max_b y_b(s)` and `w_R(s) = −min_b y_b(s)`
separately rather than `max_b |y_b(s)|`. If yield stays low, record 1-in-3 as a measured generation
cost and stop modifying the retargeter.

## Dataset target: two views

**Motion qualification view** — prompt, reference, semantic validation, embodiment feasibility, SONIC
tracking outcome, rejection reason.

**Counterfactual family view** — family/nominal/adapted ids, operator type, route, scene template,
obstacle station, easy and hard parameters; reference and executed envelopes; predicted and
physics-verified windows; per cell: success, contact body, contact direction, contact frame,
force/impulse/duration, drift onset, semantic validity, robustness outcomes; observations: ego RGB,
ego depth or local point cloud, proprioception, SONIC 64D token, reference qpos.

Raw episodes stay authoritative, LeRobot is a derived training view, rejected rollouts are retained.
The dataset card must state: prompt_intent ≠ verified behaviour; fitted scenes are train/validation
only; scene-first scenes are the formal test; family variants always share a split; geometry-predicted
and physics-verified labels stay separate; supported regimes are overhead and lateral; floor is
unsupported OOD and must not be quietly dropped from headline metrics.

## The three experiments that decide the paper

**A — data construction efficiency.** Random station vs route midpoint vs maximal separation vs
maximal separation + physics refinement. Metrics: verified counterfactual yield per 100 rollouts,
intended-failure purity, window-realisation ratio, contact-body accuracy, contact-time error,
unrelated-failure rate, physics cost per useful family. Start with 24–36 stratified pairs
(near-threshold / medium / large window × overhead / lateral), not 120 rollouts.

**B — decorated vs random vs SweepCF**, twice: **budget-matched** (same physics rollout budget —
does SweepCF produce more useful counterfactual supervision per unit cost?) and **contrast-matched**
(same number of verified preference-reversal families — are near-boundary, station-targeted examples
more valuable to learn from?). Without the second, a reviewer says SweepCF only won on balance.

**C — scene-conditioned selector.** `f_θ(S,m) → P(survive | S,m)`; drop predicted-unsafe candidates,
choose minimum edit among the rest, allow abstention when none is feasible. Baselines: always
nominal; always crouch/tuck; no-scene; scene-shuffle; fixed cylinder; analytic capsule oracle;
depth-derived geometry selector; raw ego-depth model. Headline metric
`TripletChoiceAccuracy = 1[easy→nominal ∧ overhead→crouch ∧ lateral→tuck]`, plus unsafe-choice rate,
unnecessary-adaptation rate, abstention/missed-feasible rate, cost regret, scene-shuffle degradation,
calibration error. **Statistical unit is the family or nominal-motion identity, never the frame**;
family-level bootstrap CIs; ≥5 learner seeds.

## Claim 6: scene-first generalisation

| motion | fitted scenes | scene-first scenes |
|---|---|---|
| seen nominal | standard validation | scene generalisation |
| held-out nominal | motion generalisation | **joint motion + scene generalisation** |

The bottom-right cell is the valuable one. On the frozen test, do not keep only the pretty preference
reversals — run the full candidate bank and report the real distribution: all safe, nominal only,
crouch only, tuck only, multiple safe, none safe, unsupported floor. All-safe means nominal is
correct; none-safe means the selector should abstain. Also enforce the observability gate
`t_visible < t_decision < t_adaptation_onset < t_bottleneck`, or a clip crouching from frame 0
supports only map-conditioned selection.

## C1 human review cannot be deferred further

Already observed: accepted rate recomputed after gate semantics changed; a horizontal-only contact
gate missing overhead collisions; prompt labels widely inconsistent with reference semantics; visual
marker leakage passing every numeric gate. So the 100-episode reviewed pilot is a necessary part of
credibility, not optional polish. Stratify across accepted-easy, accepted adapted-hard, nominal-hard
negatives, near-boundary cases, tracking-drift failures, overhead/lateral contact, high self-contact,
semantic predicate failures, and operator recovery transitions. At least 30–50 episodes reviewed
independently by two people, reporting automatic false accept, automatic false reject, semantic false
label, inter-rater agreement, adjudication count.

## Paper shape

ICRA 2027 closes **2026-09-15, 11:59 PM PST**: 8 pages including everything, no supplementary PDF;
video (≤180 s) uploadable 5 Aug – 9 Sep or 17–22 Sep. The chronological "we were wrong, then fixed
it, then withdrew it" narrative is an excellent research log and must not enter the paper.

| section | pages |
|---|---|
| Introduction + contributions | 0.8 |
| Related work | 0.6 |
| Problem formulation | 0.5 |
| SweepCF method | 1.8 |
| Dataset and validation | 1.0 |
| Experiments | 2.3 |
| Limitations/conclusion | 0.4 |

Figures: (1) decorated scene variation vs SweepCF decision variation; (2) matched local operator +
station mining + physics refinement; (3) the 3×3 specificity matrix; (4) main A/B/C learner result on
fitted and scene-first test. Tables: dataset comparison by *supervision type* not raw scale; main
selector results; a small station-selection + physics-refinement ablation. Commit counts, test counts,
gate bugs and failed predictors belong in the repository and dataset card, not the abstract.

## Four-week plan

**18–22 Aug — close the method definition.** Grade the 8 queued x002/x003 cells; restate claim 4 as
minimum-edit reversal; restate crouch wording as motion-specific one-rollout screening; verify the
cleaner overhead negative at 1.2801 m; build the first one-sided lateral matched family; fix the
artifact's internal snapshot drift (header says 4 families while the draft section still says one);
freeze `sweepcf_method_v0.1`; start C1 human review.

**23–30 Aug — reach trainable scale.** 8–12 distinct nominal motions; ≥8 matched families in each of
overhead and lateral; 24–30 verified families total; every nominal with easy + at least one hard
regime; physics labels on all frozen test scenes; ego depth / route-aligned clearance profile
readable; dataset manifest and family loader frozen. No floor, photorealism, VLA or manipulation.

**31 Aug – 6 Sep — the decisive learning experiment.** Train analytic oracle, privileged geometry
selector, depth-derived structured selector, ego-depth model; run decorated/random/SweepCF,
budget-matched and contrast-matched, no-scene, scene-shuffle, held-out nominal, scene-first test,
5 seeds, bootstrap CIs. Claims 5 and 6 must be answerable by the end of this week.

**7–10 Sep — ablation and paper freeze.** Fill only the most informative gaps. **11–15 Sep** —
writing and audit: every number generated from one frozen manifest; anonymous paper and repo;
one-command figure reproduction; no further gate changes.

## The operative instruction

> Stop expanding the apparatus. The next milestone is a paper-level dataset, not another validator.
>
> First, finish and grade the two queued overhead families. Rebuild the canonical matched overhead
> negative at the mildest physics-verified shelf height — test 1.2801 m for both motions and
> registered jitters — rather than using the 3253 N crash as the representative cell.
>
> Correct the claim language: the matched family establishes a minimum-edit feasible-behavior
> reversal without requiring a calibrated scalar cost. Crouch trackability is motion-specific and
> screenable with one targeted rollout at the excursion required for a usable window; it is not
> globally predictable from motion identity or a universal cap.
>
> Build one critical-side arm-tuck operator and a one-sided lateral obstacle using signed left/right
> capsule envelopes. If its yield remains low, record the empirical 1-in-3 cost and stop modifying
> the operator.
>
> Scale to at least 24–30 verified families over 8 or more distinct nominal motions before treating
> the learning comparison as scientific evidence. Keep family count separate from nominal-motion
> count.
>
> Run two dataset comparisons: fixed physics budget and fixed number of verified contrasts. Train a
> scene–motion compatibility selector, evaluate easy/overhead/lateral triplet choice, and report
> unsafe and unnecessary-adaptation rates on held-out motions and the sealed scene-first test set.
>
> Begin the 100-episode human review now. Freeze label semantics before scaling, because physics
> acceptance, behavior semantics and scene compatibility are distinct labels.
>
> Do not pursue floor adaptation, photorealism, manipulation or an end-to-end VLA until claims 5 and
> 6 have been answered.

## The closing judgement

The real progress is not 638 tests or 4 families. It is that a vague ambition — "generate lots of
humanoid motion and clutter scenes" — has been compressed into a precisely checkable question:

> Within the same task and the same journey, can scene geometry change the minimum feasible
> whole-body edit; and does data built around that reversal teach a learner to choose the right
> adaptation in a new scene?

The obstacle now is not technical difficulty but research inertia: the better the apparatus gets,
the more tempting it is to keep improving the apparatus, because that is familiar and controllable,
while the learning experiment might fail. But claims 5 and 6 *are* the paper.

> By September 6, SweepCF must either demonstrate that counterfactual supervision improves
> minimum-adaptation selection on scene-first environments, or reveal precisely why it does not
> under a sufficiently powered experiment.

Equality across three or four families explains nothing; a negative result is only valuable if
family count, controls, oracle and test split are all solid.
