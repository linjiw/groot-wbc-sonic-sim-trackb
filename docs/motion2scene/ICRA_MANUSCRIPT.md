# Motion2Scene: Constructing Training Scenes from Executed Humanoid Motion Contrasts

Working manuscript, September 7, 2026. Abstract intentionally deferred until the
registered result decision. This draft separates completed measurements from the
unfinished M2S-ICRA-v1 comparison. It is not a submission-ready paper.

## I. Introduction

A humanoid that can walk and crouch still needs to decide when crouching helps.
Training this decision requires encounters whose observations distinguish the outcomes
of the available commands. A randomly placed obstacle can be irrelevant to both
commands, or defeat both. A beam placed above a crouching reference can appear useful
while colliding with the robot during its actual transition into the crouch.
The resulting label concerns an intended posture rather than the command the robot
can execute from its current state.

We study scene construction from a pair of supported humanoid commands: commit to
walking, or request a crouching reference at a fixed decision time. Both commands run
through the same frozen controller. Their achieved trajectories provide a proposal
model for placing overhead constraints, and paired obstacle-present simulations
provide outcome labels. A fixed perceptive selector then predicts the success of
each command. This separates three questions: whether a scene has a geometric
contrast, whether that contrast survives execution, and whether the resulting data
teaches a useful decision on other layouts.

Learning environments from motion is not itself our novelty claim. Learning from
Learned Hallucination (LfLH) already learns obstacle configurations from open-space
motion and uses hallucinated environments to train navigation policies [1]. Our
controlled instance addresses a different supervision problem: full-body humanoid
commands must enter and leave an adaptation through a legal transition, and an
obstacle may change contact outcomes without making progress physically impossible.

The study makes three contributions at distinct evidence levels. First, it defines
a command-bound construction and labeling protocol that preserves all four paired
outcomes, including encounters that neither command solves. Second, it measures the
mismatch between complete-reference screening and achieved-command screening, and
validates source-specific switching banks and physical contrasts in Isaac Lab.
Third, it specifies a four-arm comparison—uniform, analytic, target-only and learned
construction—with one shared outcome learner. The third contribution remains an
experimental question, and the two completed panels answer it: contrast construction
is what teaches the selector, and learning the proposal neither helps nor hurts once
the acceptance geometry lets it acquire data at all. The learned proposal is a tested
factor, not an assumed source of superiority.

The experiment's most transferable outcome is about scarcity rather than about which
generator wins. A training scene helps this decision only if it carries a contrast
that survives execution and is visible when the command must be issued, and such
scenes are rare in a way that is measurable before any simulation: the acceptance
geometry alone can remove almost all of them, and we report the price of that choice
as a curve rather than a single threshold. On the other side, scarcity cuts both ways.
A single verified contrast is enough to change what the fixed learner does, which is
what the removal diagnostic in Section VI shows.

## II. Related work

LfLH learns hallucinated environments in which open-space plans are useful and trains
reactive navigation from those environments [1]. This establishes the motion-to-scene-
to-policy loop that motivates our experiment. We investigate how paired executed
humanoid commands constrain the labels needed by that loop, rather than claiming
that inverse scene generation is new.

Recent humanoid systems address perception-conditioned traversal and skill composition
at broader scales. HumanoidPF represents humanoid–obstacle relationships for cluttered
traversal [2]. Perceptive Humanoid Parkour composes atomic skills with motion matching
and studies long-horizon vision-based execution [3]. Our experiment retains a narrow
beam family and command bank to isolate training-scene construction. It does not
compare against those systems' full task breadth or sensing pipelines.

SONIC supplies the inherited motion-tracking foundation and command interface [4].
We freeze its parameters and do not attribute its locomotion capacity to scene
construction. The learning component is a supervised two-head outcome predictor.
Because its command is issued once from a shared approach, the present task is not
an end-to-end reinforcement-learning study. Sequential extensions would need to
address policy-induced state distributions, as emphasized by DAgger [5]. Evaluation
reports paired outcomes and the small number of carrier groups, consistent with the
need to expose uncertainty rather than infer robustness from point estimates [6].

## III. Problem and command-bound supervision

Let S denote a static scene, x_t the measured robot state, phi_t the reference phase,
h_t the relevant approach history, and a a supported command. Define the binary task
outcome as Y_a = Y(S, x_t, phi_t, h_t, a). A0 commits to walking through the encounter.
A1 requests d040 at t=0.30 s and uses the shared legal return interface. A0 is not a
wait-and-reconsider action. A refusal from the predictor continues A0; it is not a
physical stop or successful avoidance.

The contact-qualified task requires at most 1 N measured beam force through passage,
body-origin crossing 0.1 m beyond the beam, and 0.3 s upright stability in the first
episode. We record resets, falls, legal entry/return and endpoint tracking separately.
This definition admits contact-qualified passage with an endpoint-tracking rejection;
it does not imply exact reference completion or an all-time collision guarantee.

Each encounter receives the pair (Y_0,Y_1). Both-pass examples permit a preference for
walking. Walk-fail/crouch-pass examples establish useful adaptation. Walk-pass/crouch-
fail examples warn against the transition. Both-fail examples describe the limits of
the current command bank. Missing or mismatched executions are masked rather than
converted to negative labels. Paired executions must agree on recorded pre-decision
state, sensing, action and token histories. This is measured prefix agreement, not
a claim to restore an unobserved complete controller state.

## IV. Construction and fixed learner

For each qualified carrier, empty-scene executions produce achieved walk and d040
body trajectories. A capsule representation supplies inexpensive geometric queries.
The station coordinate follows the source reference route; beam dimensions are
0.10 by 1.20 by 0.10 m, with station in [0.1,0.9] and underside in [1.1,1.45] m.
The achieved forecast includes entry, passage and exit. Physics remains the labeling
oracle because obstacle interaction and tracking deviations can invalidate that
forecast.

Uniform construction samples the common domain without a future-trajectory contrast
filter. Analytic construction uses achieved envelopes and global distinct search.
Motion2Scene retains the frozen local-event initializer and 17-query bounded pattern
search, replacing its geometric evaluator with achieved trajectories. The target-only
ablation uses its previously fitted no-contrast model and removes walking interference
from search and acceptance. All arms share finite-domain, duplicate, reserved-layout
and pre-decision validity checks. Target and contrast screens retain the declared
perturbation offsets; every rejected attempt remains charged.

The observation contains 144 values from twelve upper rays and their conditional
lower-ray queries, plus 70 state/phase/history values. Neither source identity nor
true beam coordinates enters a learned selector. One deterministic logistic model
predicts two probabilities from the 214 inputs. All inputs have physical scale one
except joint velocity, scaled by five. Zero initialization, 2000 Adam updates at
0.01 and a 0.01 mean-squared-weight penalty are common across arms. The >=0.5 rule
prefers walking when both heads are positive and requests d040 only when walking
is predicted infeasible and d040 feasible. The transition manager enforces legality;
it does not make a hidden scene-dependent choice beneath the model.

## V. Completed physical and mechanism evidence

The earlier generated-source pilot admits six source pairs from eight requested and
obtains eighteen contact-avoidance contrasts from twenty-four requested scene slots.
Walking still crosses in those cases but violates the contact criterion. Thus the
result distinguishes task outcomes, not physical impossibility. d040 also passes the
selected 41002 beam in three runs, so d055 is not established as minimally necessary.
[Internal evidence: SOURCE_EXECUTION_V1_RESULT, BEAM_D040_EXECUTION_V1_RESULT.]

The selector-breakpoint study identifies one d040 rescue among six matched layout/seed
conditions on carrier 41002. A shared analytic-trained linear control realizes this
rescue: at underside 1.27 m and seed 8512, d040 records 0 N through qualifying passage,
where matched walking records 57.947 N. Its 4/6 passage contrasts with 3/6 for the other
three data arms. This is development evidence on one inspected carrier. Changing
scaling, capacity and regularization together does not isolate normalization as the
sole causal explanation of the original MLPs' failure.

The achieved-transition forecast catches thirteen command contacts that the complete-
reference screen falsely predicts clear, but misses two other contact-qualified
failures. It improves a measured diagnostic and remains imperfect. The next empty-
command bank qualifies 41001, 41002 and 41003 in both seeds: twelve executions, six
paired approaches and legal d040 entry/return. Qualification does not require every
reference-tracking metric to pass.

The first bounded construction pilot accepts only one of twelve assigned slots,
from analytic construction on 41001; no learned slot survives. Both seeds then show
walking contact (1635.077 and 1130.936 N) and d040 contact-qualified passage (0 N).
Both d040 runs also receive reference_endpoint_tracking_error rejections. The full
funnel therefore demonstrates one unique analytic scene, not broad learned-scene yield.

### Why almost nothing is acceptable: the placement envelope consumes the window

The acceptance geometry, not the proposal model, is the dominant cause. Sweeping
11,421 station/height centres per carrier over the recorded achieved transitions and
scaling the inherited audit envelope gives the number of 10 mm two-sided contrast
witnesses below. The nominal-pose column is the achievable support; the last row is
the envelope the study inherited.

![Contrast witnesses against placement-envelope half-width, and eligible generated proposals per arm under the inherited and nominal contracts.](assets/envelope-tradeoff.png)

| Envelope (xy / z) | 41001 | 41002 | 41003 |
| --- | ---: | ---: | ---: |
| nominal pose | 200 | 168 | 337 |
| +-5 mm / +-2.5 mm | 120 | 86 | 238 |
| +-10 mm / +-5 mm | 50 | 27 | 150 |
| +-20 mm / +-10 mm (inherited) | 2 | 0 | 32 |

The best nominal joint margin, the smaller of target clearance and walk interference
at a centre's own pose, is 24.42, 20.48 and 26.99 mm. The inherited envelope asks a
centre to keep a 10 mm two-sided contrast while the beam moves +-20 mm horizontally,
which is the size of the entire executed window. It is the correct instrument for a
placement certificate and for any hardware claim, where the obstacle's realized pose
is uncertain, and the wrong one for a simulator study that authors the beam at an
exact pose. Reported this way the table also prices a future hardware claim directly.

A second and separable cause belongs to the proposals. On 41003 the inherited envelope
still leaves 32 sampled witnesses and the frozen learned draws, whose +-0.05 station
and +-0.03 m trust boxes are unchanged, reach none of them. Reachability and acceptance
therefore fail independently, and only the first is addressed by a different contract.

A third property is measured and not gated by either contract. Among proposals that
are critical at the nominal pose, 44 of 47 analytic scenes but only 24 of 34 learned
scenes are visible to the decision-time sensor, with 8 of 12 invisible on 41003, where
the learned initializer concentrates late-station draws. A contrast the robot cannot
observe cannot inform the selector whatever its geometry. No visibility gate is added
after the fact; the quantity is reported as a covariate.

These are sampled witnesses on one finite grid at one physics seed per carrier. They
do not prove infeasibility, do not estimate feasible volume, and do not revise any
earlier outcome.

## VI. Registered learning-utility experiment

M2S-ICRA-v1 assigns 24 encounter groups per arm across the three qualified development
carriers: eighteen generated and six identical shared backgrounds. All four methods
retain their refused assignments. At most 156 unique command executions label the
four arms and shared backgrounds. If acquisition produces fewer complete groups,
we report actual counts and treat the result as an equal-requested-slot comparison;
we cannot claim to have completed an equal-24-label experiment.

Four fixed learners and two declared comparators face twelve reserved station/height
layouts, two physics seeds and three carriers, yielding 432 traversal assignments.
The comparators are the upper/lower-ray decision rule and a privileged achieved-
geometry predictor, not an optimal transition-time oracle. Passage is averaged within
carrier before averaging carriers. Paired tables report both-pass, method-A-only,
method-B-only and both-fail. Repeats share carriers and do not create independent
source evidence. Background adaptation/refusal outcomes form separate suites.

All 82 assigned labeling executions and 41 unique pairs pass measurement admission.
The resulting training sets are the study's central observation.

| Arm | Groups | Generated | Both pass | d040 only | Walk only | Both fail |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 24 | 18 | 10 | 0 | 0 | 14 |
| Analytic | 7 | 1 | 4 | **1** | 0 | 2 |
| Target-only | 22 | 16 | 20 | 0 | 0 | 2 |
| Motion2Scene | 6 | 0 | 4 | 0 | 0 | 2 |

Every arm contains the same six shared background groups. Across 192 proposals, 72
assigned generated slots and 82 executed label commands, **exactly one** acquired group
is a walk-fail/d040-pass contrast, from analytic construction on 41001. Consequently
the analytic and Motion2Scene training sets differ by exactly that one group and are
otherwise identical, and the equal-24-complete-label target fails in three arms. This
is an equal-requested-slot comparison with unequal acquired data, and the acquisition
shortfall must be disclosed wherever the downstream numbers are quoted.

The comparison this creates is unusually clean and unusually narrow at the same time.
Adding eighteen untargeted groups (uniform) or sixteen target-only groups changes the
fitted policy not at all; adding one contrast changes it completely. Both statements
rest on a single acquired example, so the effect size is not estimable from this study.

Four primary fits and 24 registered leave-four-assignment-out refits are complete.
All 540 policy evaluations are admitted: six methods on all 72 traversal conditions,
plus the 108 background control runs. Passage is uniform 28/72, analytic 39/72,
target-only 28/72, background-only Motion2Scene 28/72, scripted rays 46/72 and
privileged geometry 39/72. Nothing remains pending.

Analytic versus uniform yields 28 both-pass, eleven analytic-only, zero uniform-only
and 33 both-fail conditions. The carrier-averaged difference is +15.278 percentage
points, with six additional passes on 41001, three on 41003 and two on 41002. All
eleven additional passages request d040 and measure 0 N beam force; matched walking
records 385.1–1685.5 N. This supports useful learned adaptation on the inspected
layouts. It does not establish a learned-generator advantage or source-held-out
generalization.

The comparison is one contrast against three copies of the same policy, and must be
reported as such. The uniform, target-only and background-only Motion2Scene learners
issue walking on all 72 conditions and are behaviourally identical; each therefore
gives the same 28/72 and the same eleven-to-zero paired table against analytic. The
condition-level discordance test over eleven one-directional pairs gives p = 0.00098,
but the 72 conditions are twelve layouts and two seeds inside only three carriers, so
that figure overstates independence. The carrier-level statement is the honest one:
the direction is the same on all three carriers, with six, three and two extra passes
and no reverse case, which is n = 3.

What the single contrast taught is legible, and it is narrow. That group's beam sits
at underside 1.2755 m. The evaluation panel has three underside bands, and the fitted
learner's behaviour separates along them.

| Beam underside | Conditions | Analytic requests d040 | Analytic passes | Scripted requests | Walking-only passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.18 m | 24 | 11 | 0 | 11 | 0 |
| 1.27 m | 24 | 14 | 15 | 23 | 4 |
| 1.36 m | 24 | 4 | 24 | 12 | 24 |

The learner almost never requests the adaptation at 1.36 m, where walking already
passes 24 of 24, and all eleven of its additional passages fall in the 1.27 m band, the
band its one training contrast came from, spread across all three carriers with six on
41001. At 1.18 m neither command ever succeeds, so its eleven requests there are neither
rewarded nor punished by this suite. Within the 1.27 m band it requests the adaptation
14 times in 24 conditions against the scripted rule's 23, which is where its remaining
gap to the observed ceiling lies.

So one example transferred across carriers within its own geometry band and not beyond
it. That is a real but strictly band-local generalization, and it is the most this
study can claim about what an acquired contrast teaches.

Adaptation is close to free on this panel and every learner under-uses it. Of the 47
conditions in which both commands were actually executed, 19 are d040-only successes,
16 both-pass, 12 both-fail, and none is a walking-only success. The best outcome
available from the observed commands is 47 of 72, which the scripted rule nearly
attains at 46 by requesting d040 46 times; analytic requests it 29 times and reaches
39. Under-adaptation, not misfiring adaptation, is what separates the learners from the
ceiling here.

The background suites settle the cost side. Across the absent and raised controls no
policy requests the adaptation even once, so no arm pays an unnecessary-adaptation
penalty on this panel and the learned request is not indiscriminate. The blocked suite
reverses the ranking: every learner and the privileged forecast classify the scene as
infeasible and refuse on 6 of 6, while the scripted rule refuses on none. No blocked
scene is passable, so refusal there is correct classification rather than successful
avoidance, and it is reported apart from passage. The stronger traversal baseline is
therefore the weaker infeasibility detector, and neither dominates.

The first analytic refit removes the sole useful generated label and three refused
assignments. Its remaining training IDs, weights, biases and scales exactly equal
the background-only Motion2Scene primary model. On the 61 recorded inputs, its d040
requests fall from 22 to zero. The five other analytic folds retain 22 requests;
two remove no available labels. This controlled removal links one acquired contrast
to the fixed learner's decisions. It is an offline refit diagnostic, not additional
physics evaluation or a population claim about single-example learning.

Acquisition spends 0.824108 contended GPU h and admitted policy evaluations spend
3.146765 h. These are physical execution costs, not a complete end-to-end cost
comparison: generator fitting, proposals, rejected searches and shared bank costs
remain separately recorded. Remaining evaluation and the final grouped uncertainty
analysis must precede the registered result decision. [Internal evidence:
M2S_ICRA_540_RESULT; all assigned outcomes and refit checkpoints are released.]

### A second contract, and the equal-label comparison it makes possible

Because the envelope and not the proposal model dominates the funnel, a second contract
is registered separately, with its predictions filed before any proposal was drawn. It
changes exactly one thing: criticality is judged at the nominal pose, with the 17-offset
and 113-offset survivals still computed and stored for every proposal as reported
diagnostics that never gate acceptance. Everything else is unchanged, and the six shared
background groups are reused by reference rather than re-executed, so no reused trace
counts as a new example. A unit test asserts that both contracts reach identical
decisions whenever their geometry verdicts agree.

Under this contract eligible generated proposals move from 1 of 48 to 41 of 48 for
analytic and from 0 of 48 to 33 of 48 for Motion2Scene. Three groups per carrier and arm
are assigned, giving 36 groups and 72 label commands, all of which completed. Every arm
holds exactly fifteen groups, so this is the equal-label comparison the first contract
could not produce.

| Arm | Groups | Generated | Both pass | Useful contrast | Both fail | Training BCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 15 | 9 | 7 | 0 | 8 | 0.001762 |
| Analytic | 15 | 9 | 4 | **9** | 2 | 0.000542 |
| Target-only | 15 | 9 | 13 | 0 | 2 | 0.000175 |
| Motion2Scene | 15 | 9 | 4 | **9** | 2 | 0.139004 |

Both contrast constructions acquire nine useful contrasts from nine generated groups.
Both non-contrast constructions acquire nine groups and no useful contrast at all. All
four registered acquisition predictions hold, including the two that failed under the
inherited envelope.

All 144 evaluation executions on the twelve reserved layouts at seed 8511 are admitted.

| Policy | Passage | d040 requests | Refusals | Per-carrier passage |
| --- | ---: | ---: | ---: | --- |
| Uniform | 14/36 | 0 | 17 | 0.333 / 0.417 / 0.417 |
| Analytic | 24/36 | 24 | 0 | 0.667 / 0.667 / 0.667 |
| Target-only | 14/36 | 0 | 0 | 0.333 / 0.417 / 0.417 |
| Motion2Scene | 22/36 | 22 | 0 | 0.583 / 0.583 / 0.667 |

Motion2Scene beats uniform and target-only by eight discordant conditions to zero, and
analytic beats them by ten to zero, with no reverse case on any carrier. Motion2Scene
and analytic are indistinguishable: two discordant conditions of 36, both favouring
analytic, p = 0.50. So contrast construction is what teaches the selector, and learning
the proposal neither helps nor hurts at this label count. As before, 36 conditions are
twelve layouts on three carriers, so the carrier-level statement is the honest one:
contrast arms win on three of three carriers, and the learned-analytic difference is
-0.083, -0.083 and 0.000 by carrier.

One deficiency is specifically the learned generator's. Its training loss is 0.139
against analytic's 0.000542 on identical architecture, optimizer and label count. The
construction record explains it: among proposals critical at the nominal pose, 44 of 47
analytic scenes but only 24 of 34 learned scenes are visible to the decision-time sensor,
and 8 of 12 are invisible on 41003, where the learned initializer concentrates
late-station draws. Four of the nine assigned Motion2Scene groups record zero sensor ray
hits. A contrast the robot cannot observe is a label the learner cannot fit. Visibility
is gated by neither contract and none was added after the fact.

The whole second study cost 1.442 contended GPU-hours.
[Internal evidence: M2S_ICRA_NOMINAL_V1_RESULT.]

## VII. Limitations and result decision

This is a controlled instance with one beam family, two commands, ideal simulator rays,
three inspected development carriers and one frozen controller. It has no hardware or
source-held-out transfer claim. The original 120/600 MLP evaluation remains a retained
partial failure, with 480 assignments paused. Existing layout diagnostics have
informed this revision, so their reuse is development evaluation.

Geometric proposal acceptance does not guarantee a useful physical contrast: under the
inherited envelope the learned initializer produced no accepted scenes at all, and the
24-label quota failed in three arms. Contact-qualified passage does not eliminate
endpoint tracking error, and neither a predicted refusal nor continued neutral walking
is a qualified protective action.

Two results are supported and one is not. Contrast construction improves the fixed
selector over untargeted and target-only construction, at equal labels, with no reverse
case on any carrier. Learning the proposal distribution neither helps nor hurts relative
to the execution-aware analytic solver at this label count; the two are indistinguishable
on 36 paired conditions, which is a null at n = 3 carriers and not an equivalence claim.
What the learned generator does buy is measured elsewhere and is modest: a 68.7 per cent
reduction in search time and acceptance on fresh sources where uniform sampling fails.
What it costs is also measured: its critical scenes are less often visible at decision
time, and its training loss is two orders of magnitude higher for that reason.

Every conclusion here is a simulation result on three development carriers under one
frozen controller. The evaluation varies layout and carrier, not source ancestry, so it
is unseen-layout transfer rather than source-held-out transfer. No claim extends to
hardware, and the placement robustness a hardware claim would require is exactly what the
envelope sweep prices at 2, 0 and 32 surviving centres.

## References (verified primary records, September 8)

[1] Z. Wang et al., “From Agile Ground to Aerial Navigation: Learning from Learned
Hallucination,” IROS, 2021. https://arxiv.org/abs/2108.09793

[2] H. Xue et al., “Collision-Free Humanoid Traversal in Cluttered Indoor Scenes,”
arXiv:2601.16035, 2026. https://arxiv.org/abs/2601.16035

[3] Z. Wu et al., “Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via
Motion Matching,” arXiv:2602.15827, 2026. https://arxiv.org/abs/2602.15827

[4] Z. Luo et al., “SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body
Control,” Science Robotics, vol. 11, no. 117, eaed4592, 2026.
https://arxiv.org/abs/2511.07820

[5] S. Ross, G. Gordon, and D. Bagnell, “A Reduction of Imitation Learning and Structured
Prediction to No-Regret Online Learning,” AISTATS, 2011.
https://proceedings.mlr.press/v15/ross11a.html

[6] R. Agarwal et al., “Deep Reinforcement Learning at the Edge of the Statistical
Precipice,” 2021. https://arxiv.org/abs/2108.13264
