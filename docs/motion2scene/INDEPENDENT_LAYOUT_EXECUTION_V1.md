# Independent-layout execution v1: frozen development learners

Registered before any reserved-layout rollout. Implements the existing
[600-cell evaluation plan](INDEPENDENT_LAYOUT_EVALUATION_V1_PLAN.md), with unchanged
four arms, twenty final checkpoints, twelve independently reserved layouts,
physics seeds 8511/8512, observed source 41002 and the one-shot 0.30 s decision.
No fitting, checkpoint selection, scene replacement or transition tuning occurs.

The first bounded wave is the first six blocks in the original layout order:
station 0.35, underside heights 1.18/1.27/1.36 m, both physics seeds, all twenty
learners per block = **120 actual executions**. This is an explicitly partial
first wave of the 600-cell assignment, not a representative or complete test score.
Finish and audit each twenty-cell block before launching the next. An operational
stop leaves all remaining assigned trials pending; it does not remove them.

## Binding and execution

Materialize the twelve beams from the original normalized source route and yaw;
retain the existing 0.1 m depth/thickness and 1.2 m width. Absent, raised and blocked
controls reuse their declared geometry and are separate suites. Hash-bind all
checkpoints, scene files, controller/source inputs, the existing learned runtime,
scorer, input auditor, this protocol and the derived master/block assignments.
The source library, learned runtime, legal phase/jump guard and SONIC stay frozen.
The learner alone requests walk or d040 from its directly captured causal inputs.
A neither-feasible prediction records refusal and executes neutral fallback; it is
not a stop and cannot turn collision into successful avoidance.

Serial Isaac Lab with the inherited 7500 MiB free-memory floor and 375 s per-cell
wall timeout. Each twenty-cell block reserves at most 2.083334 contended GPU h;
check actual cumulative 8 h/day and 24 h/week use before each launch or resume,
including every already spent block. Chunk records live under discoverable top-level
m2s-* directories so existing budget accounting sees them. No concurrent physics
cells, no automatic retry of infrastructure failures and no scientific failure refill.
The 62.5 h worst-case master is an assignment only and is explicitly nonexecutable.

## Before-run predictions and admission

For each completed block, check:

1. All twenty learners receive exactly the same pre-decision state/action/token
   history and 214-value observation at the matched layout/physics seed.
2. Direct features recompute exactly, ray origins agree within 1e-6 m, the same-index
   recorder state agrees within 1e-7, and the loaded neutral/d040 bank is bitwise
   equal to the already qualified bank.
3. CPU re-evaluation of each pinned model agrees with recorded probabilities within
   1e-6 and with its actual request/refusal. The captured checkpoint hash matches.
4. Every executed switch satisfies the existing phase/jump and no-state-write guard.
   Report denied requests and missing returns separately; a reset is not recovery.
5. Imported beam placement/scale matches the assigned geometry within 1e-6 and
   200 Hz contact capture synchronizes to the 50 Hz recorder using the existing audit.

An input/runtime/measurement failure stops promotion and later launches until it is
resolved with a versioned analysis or runtime; preserve the original failure. A
contact or passage failure is a scientific outcome and remains in the denominator.
No prediction requires a successful scene or an advantage for Motion2Scene.

## Analysis fixed before outcomes

Use the existing score_passage function without modification: all recorded body
origins must cross the beam by 0.1 m and remain upright for 0.3 s during the first
episode. It measures body-origin crossing, not native collider extent clearance.
Beam contact is flagged when the measured force exceeds 1 N, so the implementation
accepts <=1 N. Earlier prose said <1 N; disclose this exact boundary, report any
exactly-1 N maxima and do not silently change the frozen scorer. Report peak forces,
resets, falls and return logs separately from the binary outcome.

The primary completed-panel effect is Motion2Scene minus analytic contact-qualified
passage at eleven complete training examples per arm. Report each optimizer seed
separately, paired across the same layouts and physics seeds. Five fits are not five
independent datasets, and one source does not support source-held-out inference.
For partial waves show block-level counts and a completed/assigned denominator;
do not call their mean the full test success rate. No significance/noninferiority
claim is registered at this development scale.

Absent/raised unnecessary adaptations and blocked refusal rates use their own
assigned denominators. Refusal correctness on blocked scenes is not physical
avoidance. Keep the planned scripted comparators, perceptual shortcut diagnostics,
larger 24/48/96 fitting budgets and final-source transfer as outstanding experiments.
