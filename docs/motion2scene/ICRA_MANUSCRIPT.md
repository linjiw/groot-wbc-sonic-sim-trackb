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
experimental question: acquisition is complete, but policy evaluation is partial.
The learned proposal is a tested factor, not an assumed source of superiority.

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
A finite support map finds 67 nominal candidate centres after other checks, but only
one under the inherited perturbation audit. None of its four pre-filter robust
centres lies inside the frozen learned draws' trust boxes. These finite samples do
not prove infeasibility or justify revising the earlier outcomes.

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
Complete groups are 24 uniform, seven analytic, 22 target-only and six Motion2Scene,
including the same six shared backgrounds per arm. Only one generated group, from
analytic construction on 41001, is walk-fail/d040-pass. Uniform supplies six generated
both-pass and twelve both-fail groups; all sixteen target-only generated groups are
both-pass. Motion2Scene supplies no accepted generated groups. Its fitted learner is
therefore background-only. The equal-24-complete-label target fails in three arms.

Four primary fits and 24 registered leave-four-assignment-out refits are complete.
At the September 7 snapshot, 366 of 540 policy evaluations are admitted: all six methods
on 61 of 72 traversal conditions. Passage counts are uniform 21/61, analytic 30/61,
target-only 21/61, background-only Motion2Scene 21/61, scripted rays 36/61 and privileged
geometry 29/61. These are ordered partial results; 66 traversal and all 108 separate
background runs remain pending under unchanged rolling compute and memory gates.

Analytic versus uniform yields 21 both-pass, nine analytic-only, zero uniform-only
and 31 both-fail conditions. The carrier-averaged difference is +14.603 percentage
points in this slice, with five additional passes on 41001 and two each on 41002 and
41003. All nine additional passages request d040 and measure 0 N beam force; matched
walking records 385.1–1685.6 N. The scripted rule passes six further conditions with
no reverse difference. This supports useful learned adaptation on the inspected
layouts, while the script remains the stronger observed baseline. It does not
establish a learned-generator advantage or source-held-out generalization.

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
M2S_ICRA_366_RESULT; all assigned outcomes and refit checkpoints are released.]

## VII. Limitations and result decision

This is a controlled instance with one beam family, two commands, ideal simulator rays,
three inspected development carriers and one frozen controller. It has no hardware or
source-held-out transfer claim. The original 120/600 MLP evaluation remains a retained
partial failure, with 480 assignments paused. Existing layout diagnostics have
informed this revision, so their reuse is development evaluation.

Geometric proposal acceptance does not guarantee a useful physical contrast. The
current learned initializer has produced no accepted scenes in the achieved-transition
study, and the matched 24-label quota fails in three arms. Contact-qualified passage does not
eliminate endpoint tracking error. Neither a predicted refusal nor continued neutral
walking is a qualified protective action. The final conclusion must follow the
registered comparison: report contrast-construction utility if supported, learned
proposal benefit only if measured, and a construction/validation result if learning
utility remains unresolved.

## References (verified primary records, September 7)

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
