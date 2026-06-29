# CG-WBC Curriculum Algorithm v0

## Research Question

CG-WBC tests whether competence-gated sampling improves humanoid fetch/place
reliability versus uniform GR00T fine-tuning while preserving the fixed
SONIC/VLA interface:

```text
GR00T action = 64D SONIC motion token + 7D left hand + 7D right hand
```

The algorithm does not modify SONIC deployment, ZMQ protocol, observation
ordering, or low-level G1 command generation.

## Questions

- RQ1: Does competence-gated sampling improve closed-loop success versus
  uniform fine-tuning?
- RQ2: Does curriculum reduce fall rate and object drop rate, or only improve
  task success?
- RQ3: Which stage is the bottleneck: reach, grasp, place, standing
  stabilization, carry, or generalization?
- RQ4: Does anchor replay prevent forgetting of earlier skills?
- RQ5: Are open-loop GR00T errors predictive of closed-loop SONIC/VLA success?
- RQ6: Does stage-wise evaluation expose failures hidden by aggregate success?

## Algorithm

```text
Input:
  D: validated SONIC/VLA demonstration dataset
  G: curriculum stage graph
  M: stage-wise evaluation metrics
  theta: GR00T policy parameters
  pi_sonic: fixed SONIC whole-body controller
  A: fixed SONIC/VLA action interface

Initialize:
  unlocked_stages = {S0}
  difficulty[S] = 0.0 for all stages
  competence[S] = 0.0 for all stages
  replay_buffer = validated episodes grouped by stage

Loop:
  1. Validate dataset and G1 SONIC schema.
  2. Build curriculum manifest from episodes.
  3. Select training batch using unlocked stage mask, learning progress,
     anchor replay, and difficulty weighting.
  4. Fine-tune GR00T using the standard action prediction loss.
  5. Run open-loop evaluation on held-out episodes.
  6. Run closed-loop MVP evaluation for each unlocked stage.
  7. Update competence score for each stage.
  8. If stage gates pass, unlock successor stages.
  9. If success is high and safety metrics pass, increase difficulty.
 10. If fall/drop/jerk exceeds threshold, decrease difficulty.
 11. Save curriculum state, metrics, and model checkpoint.

Output:
  Fine-tuned GR00T policy
  Stage-wise evaluation report
  Curriculum state trajectory
  Baseline comparison against uniform fine-tuning
```

## Stage Graph

The MVP graph is defined in
`configs/research/curriculum_graph_fetch_place.yaml`:

```text
S0 schema_validity
S1 static_reach
S2 grasp_lift
S3 place_nearby
S4 stand_stabilized_pick_place
S5 short_carry_place
S6 perturbation_carry_place
S7 language_visual_generalization
```

Each stage has prerequisites, task difficulty parameters, and gates. Fall rate
and drop rate are gates, not just reported metrics.

## Competence Score

For v0:

```text
competence =
  0.50 * success credit
+ 0.20 * fall safety credit
+ 0.15 * drop safety credit
+ 0.10 * action smoothness credit
+ 0.05 * schema validity credit
```

The score is normalized over the terms active for a stage. That keeps
schema-only and safety-heavy stages comparable to stages with success, fall,
drop, and jerk gates. A stage unlocks successors only when hard gates pass,
competence is at least the configured threshold, and the evaluation sample count
is large enough.

## Difficulty Controller

Each stage owns `difficulty in [0, 1]`.

```text
unsafe fall/drop: decrease by 0.10
success > 0.90:  increase by 0.05
success < 0.50:  decrease by 0.05
otherwise:       unchanged
```

This keeps the task near the learning zone instead of blindly increasing
difficulty every epoch.

## Curriculum Sampler

Training samples should be drawn stage-wise:

```text
20% anchor replay from solved earlier stages
60% active frontier stages
20% hardest unlocked/generalization stage
```

The current v0 implementation exposes generic sampling weights:

```text
weight(stage) =
  anchor_replay(stage)
  + unlocked(stage) * softmax(beta * learning_progress(stage))
```

Where:

```text
learning_progress = abs(EMA_success_now - EMA_success_previous)
```

The sampler output is a stage distribution. The training integration can use it
to construct balanced dataset manifests, per-stage shards, or a custom GR00T
data sampler in a later increment.

## Golden-Path Proof

Before any large training run, the required proof is:

```text
schema check passes
dataset validation passes
curriculum manifest builds
open-loop eval command is known
MVP eval report is produced
stage-wise summary is produced
```

No model-performance target is required for this proof. The target is pipeline
validity and measurable stage-wise reporting.
