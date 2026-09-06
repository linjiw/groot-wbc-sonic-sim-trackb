# Comparative acquisition v1: exact target binding and first paired data slice

Registered September 6, before this acquisition's proposals and physics. This advances
[learning contract v2](LEARNING_CONTRACT_V2.md). SONIC, generator checkpoints, learner,
214 features, one 0.30 s decision, return at 3.3 s and the shared phase/jump guard stay
fixed. No new geometry or timing campaign is a prerequisite.

The frozen generator was fitted on 55 mm contrasts. This acquisition supplies the
actual **neutral/d040 CSV pair used by the deployed motion bank**. Recompute both
SONIC entries with the pinned converter and compare every entry array and FPS to the
existing files (absolute array tolerance 1e-7, exact FPS and field set). A preflight
reconstruction exposed a 1.49e-8 m serialization difference, so this is a bounded
numerical binding, not bitwise conversion identity. Apply the shared horizontal origin; require the same route and root
orientation. Build all-frame capsule states by MuJoCo forward kinematics. This is
reference geometry, not MuJoCo dynamics. The legacy encoder and analytic helper have
hardcoded d055 slot names; the explicit compatibility view supplies d040 arrays to
those slots, while all provenance and public case labels remain d040. No d055 file is
read and no old d055 scene is reused. No refitting is allowed.

Four arms each assign the first nine of sixteen outputs, followed by the same absent,
raised and blocked background encounters: twelve assigned groups per arm, one quarter
background. This is a **development corpus smaller than the proposed primary 24/48/96
budgets**, not the primary comparison. All sixteen generated outputs and their costs
are retained even though seven are outside this first assignment. Uniform uses seed
8601 and the common parameter domain. Both learned arms sample sixteen proposals
with seed 8601 and apply the same 17-query pattern budget; no-contrast passes only
target geometry into search and rejection. Analytic uses the existing event-aware
20-station envelope, 136-candidate global search and distinct selection, requesting
sixteen distinct outputs. Its geometry-query cost is lower than the sixteen-output
learned searches; report the difference, not a matched-query claim.

All arms enforce finite/domain, initial nonpenetration, duplicate-within-arm and the
already reserved station/height test exclusion. Only Motion2Scene and analytic require
the declared robust neutral-interference/target-clearance audit; no-contrast requires
target clearance only; uniform has no criticality filter. Rejection never triggers
refill or resampling. Every attempt remains in the acquisition denominator. Independent
NumPy clearance and search-implementation agreement are checked for every arm.
Generator fitting costs remain inherited with their known timing gaps; new setup,
synthesis, audit and physics times are measured separately. Shared backgrounds are
acquired once under identical commands/seed and attributed equally to all arms; they
are not independent repeated encounters. Same-seed beam-removed observations can use
the identical shared absent encounter; sharing is disclosed rather than counted anew.

## Bounded first slice

Freeze the full proposal assignment before physics. This manifest executes the first
two assigned generated slots per arm if geometrically eligible, plus shared absent
and blocked: **at most ten scene pairs / twenty executions**. Missing or rejected slots
remain missing; no later output replaces them. Raised and the remaining assigned slots
are pending a subsequent spend manifest. Physics seed 8602 is shared across arms and
both commands. The twelve reserved layout tests and final transfer sources are not
executed or used for fitting. No selector fit is authorized by this incomplete slice.

Capture delivered rays, exact robot state, joint order, current skill, clocks and
214 computed features directly at the first decision callback, before issuing the
command. The shared execution code has no scene-dependent selection rule. Require
exact pre-command capture and recorded state/action/token/reference-prefix agreement
between paired commands; this is replay matching, not a hidden-state snapshot. Validate
reset-spanning captures segment by segment without discarding failed episodes. Missing
labels remain masked; both-fail is a valid label. Qualify passage using the frozen
body-origin crossing plus 0.3 s recovery and >1 N per-body, 200 Hz contact criterion;
report peak forces, resets, falls, command refusals and incomplete crossing separately.

Predictions, adjudicated on all assigned first-slice pairs:
1. Every actually completed pair has a matching direct decision capture and prehistory.
2. Every admitted d040 request executes legally at 0.30 s and returns legally.
3. Shared absent has both outcomes pass; shared blocked has both outcomes fail.
4. At least one assigned Motion2Scene or analytic pair yields walk-fail/d040-pass.
A rejection or missing run cannot count as support for prediction 4. No arm-ranking or
learning-utility prediction is tested by this small acquisition slice.

Serial Isaac Lab execution; inherited 7500 MiB resource floor and 375 s timeout;
maximum 20 × 375 / 3600 = 2.083334 contended GPU h. Stop infrastructure failures and
hash changes, retain science failures, no retries. Check the standing 8 h/day and
24 h/week envelopes before launch. CPU proposal ceiling 600 s. The full corpus and
policy fits require subsequent manifests, not informal promotion of this slice.
