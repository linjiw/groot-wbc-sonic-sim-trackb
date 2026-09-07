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
experimental question until the registered acquisition and evaluation complete.
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

[Results pending: complete acquisition funnel, paired labels, fitted controls,
432-cell traversal panel, comparator characterization and requested-slot cost table.
Do not replace these entries with predictions or offline command lookups.]

## VII. Limitations and result decision

This is a controlled instance with one beam family, two commands, ideal simulator rays,
three inspected development carriers and one frozen controller. It has no hardware or
source-held-out transfer claim. The original 120/600 MLP evaluation remains a retained
partial failure, with 480 assignments paused. Existing layout diagnostics have
informed this revision, so their reuse is development evaluation.

Geometric proposal acceptance does not guarantee a useful physical contrast. The
current learned initializer has produced no accepted scenes in the achieved-transition
pilot, and the matched 24-label quota may fail. Contact-qualified passage does not
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
