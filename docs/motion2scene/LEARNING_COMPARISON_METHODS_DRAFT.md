# Learning comparison: methods draft

This draft describes the implemented development study. It supplies the experiment
section, not an abstract claiming a downstream advantage. Independent-layout outcomes
belong in the [measured result](INDEPENDENT_LAYOUT_V1_RESULT.md); pending trials are
not filled from training losses or geometric screening.

## Question and controlled comparison

We test whether the environment-generation method changes the usefulness of training
data for a fixed humanoid behavior selector. The robot, low-level SONIC tracker,
neutral/d040 reference bank, observations, transition guard, learner architecture,
optimizer and checkpoint rule are shared. The primary comparator is a strong
motion-conditioned analytic generator. Uniform placement tests the value of targeted
scenes, while a model without the contrast objective tests whether interference with
the alternative motion provides useful supervision.

The current development study uses one observed source, 41002. A beam is parameterized
by normalized route station and underside height; its depth, width and thickness are
0.1, 1.2 and 0.1 m, respectively. Uniform proposals cover the declared station/height
domain. Analytic construction uses local motion envelopes, event ranking and bounded
global search. Motion2Scene uses its frozen local model and pattern search. The
no-contrast model retains the architecture and fitting budget but omits alternative
interference in training, search and rejection. Basic domain, initial-penetration,
duplicate and held-out-neighborhood checks apply throughout. Full contrast screening
is retained for the analytic and Motion2Scene pipelines; uniform noncritical scenes
are not removed merely because they teach no adaptation.

## Transition-conditioned supervision

For a scene S, the action label is the measured outcome of a command issued at a
particular pre-decision state and controller history:

\[
Y_a = Y(S,x_t,\phi_t,h_t,a),\qquad a\in\{\text{walk-commit},\text{request-d040}\}.
\]

Walking commits to the encounter; it does not mean waiting until another decision.
The alternative requests d040 once at reference time 0.30 s. The shared transition
manager checks phase and reference continuity, then requests return at 3.3 s. It
does not use scene geometry to override the learned choice. Paired training runs use
matched physical seeds and exact pre-command state/action/token histories. Direct
callback capture binds observations and robot state before either command acts.
Whole-reference execution labels and earlier 0.20 s labels are excluded.

Each encounter retains both binary outcomes: both succeed, only walking succeeds,
only d040 succeeds, or both fail. Missing execution is unknown and masked rather than
converted into failure. The complete development corpus has 38 scene pairs from 76
Isaac Lab executions. The equal-count fitting subset uses the first eight eligible
assigned generated scenes in fixed slot order and the same absent, raised and blocked
controls in each arm: eleven complete examples per learner. Its 3/11 background share
is disclosed separately from the original 3/12 assignment quota. The three unused
ninth generated pairs and the analytic geometric refusal remain charged and recorded.

## Learned decision and information boundary

A two-head network predicts command success from 214 values: 144 collision-ray
features, reference phase, current skill, observation age, gravity, linear/angular
velocity, and 29 joint positions and velocities. The ray features encode observed
hits, distances, hit heights and conditional lower-ray observations. Obstacle identity,
true beam parameters, source ancestry and future outcome labels are not inputs.
The sensor is an ideal simulator scene-query observer, not a measured depth camera.

The shared network has two ReLU hidden layers of widths 64 and 32. Each arm uses
train-only normalization, binary cross-entropy, Adam at 0.001, 1,000 full-batch updates,
and the final checkpoint for five optimizer seeds. The twenty fits use one dataset
size; they are not a data-efficiency curve. At deployment, predicted-feasible walking
is preferred, otherwise predicted-feasible d040 is requested. If neither head reaches
0.5, the system logs refusal and continues with neutral commitment. This fallback
does not stop the robot and does not solve a blocked passage.

## Independent-layout execution and endpoints

The prespecified test grid crosses four route stations with three underside heights;
its twelve layouts were reserved before fitting. Two physical seeds evaluate each
of the twenty fixed models, giving 480 traversal assignments. Absent, raised and
blocked suites add 120 assignments. The first wave evaluates all heights and both
physical seeds at the first station, retaining the other assignments as pending.
This holds out layouts on the shared bank, not source ancestors.

The frozen scorer requires every recorded body origin to cross the beam by 0.1 m
and remain upright for 0.3 s in the first episode. A reset before this qualifying
window prevents passage. The peak measured beam force through that window must not
exceed 1 N. Full-capture resets, return commands and measured forces are also reported.
The endpoint does not guarantee subsequent neutral recovery or native-collider
extent clearance. Physics-step contact is captured at 200 Hz and checked against
50 Hz trajectory records. Loaded-bank, direct-state, ray-origin and model-readout
checks precede admission of each completed evaluation block.

We report each optimizer seed's passage count, paired Motion2Scene-minus-analytic
differences on matched layout/physics encounters, and complete assigned denominators.
Controls retain separate adaptation/refusal measures. Optimizer seeds are repeat fits;
they do not increase the number of source ancestors. This development study does not
support source-held-out intervals, significance, noninferiority, or hardware claims.
Generation, rejected proposals, labeling and evaluation costs remain separate. The
larger acquisition-budget comparison requires complete upstream cost accounting.

## Evidence map for manuscript assembly

| Proposed statement | Evidence location | Present boundary |
| --- | --- | --- |
| Motion contrasts can propose reference-separating constraints | Fresh-source geometric and generator audit records | Reference geometry; separate from command success |
| Generated scenes can distinguish executed behaviors | SOURCE_EXECUTION_V1_RESULT.md | 18/24 requested development slots; upright can still cross with contact |
| Learning labels must bind the issued transition | ACTION_LABEL_COMPLETION_V1_RESULT.md and COMPARATIVE_CORPUS_STAGE_RESULT.md | 0.30 s labels cannot be inferred from whole-motion or 0.20 s success |
| The fitted predictor can issue the measured commands | evidence/learned-command-check-result.json | Twelve observed-scene integration executions; not independent performance |
| Motion2Scene data outperform analytic data | INDEPENDENT_LAYOUT_V1_RESULT.md | Hypothesis; requires the completed comparison and final source transfer |

The existing development corpus is unfavorable to the proposed generator: all nine
Motion2Scene scenes have both commands fail, while four of eight analytic scenes
make d040 useful. Thirteen scenes also share identical deployed feature vectors with
differing labels. These findings limit transition support and observable information;
they are not removed to improve the comparison. If the analytic arm matches or wins,
the paper must separate the usefulness of motion-conditioned constraints from an
unestablished advantage of learning their distribution.
