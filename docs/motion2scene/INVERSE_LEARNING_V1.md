# First motion-only inverse-learning diagnostic

Registered development design, 2026-09-05. Implements the first learning experiment from
[the framework](LEARNED_GENERATOR_FRAMEWORK.md). This file is hash-pinned before the run.

## Question and scope

Can a temporal motion encoder and correlated four-component beam distribution learn a useful
station/height distribution from fixed geometric feedback, without receiving obstacle labels,
analytic intervals, or the previous teacher's selected beam?

Use the already selected carrier 41002, upright versus d055. References and executions 7901/7902
provide gradients; 7903 is excluded from gradients. That seed was examined in prior development
studies. It is a sensitivity check, not a fresh confirmatory holdout. All derivatives belong to one
carrier. No generalization or ordinal-minimality claim is possible from this experiment.

This first implementation optimizes two beam coordinates, with depth 0.10 m, width 1.20 m,
thickness 0.10 m, and route-chord yaw fixed. Station domain [0.35,0.65] and underside-height domain
[1.10,1.45] m are shared by every arm. Heights are not constrained to analytic feasible intervals.
The beam is abstract unsupported geometry. Imported robot geometry equivalence, continuous-time
clearance, and obstacle-present execution remain unverified. All training-eligibility flags remain
false for the downstream execution-qualified dataset.

## Fixed experiment

- Four arms: encoder with preference/clearance/KL; encoder without KL; encoder with clearance/KL
  only; direct per-motion mixture optimization with preference/clearance/KL and no encoder.
- Three optimizer seeds: 8121, 8122, 8123. CPU float64, two threads; no GPU or physics launches.
- 300 Adam updates at learning rate 0.002; four mixture components and two samples per component
  per update. Components are enumerated and weighted, with pathwise continuous gradients.
- All recorded frames and complete retained capsule axes. A domain-wide vertical bounding test
  removes only capsules unable to affect a distance minimum capped at 0.10 m.
- Frozen motion costs: upright=0, crouch=1, a declared adaptation proxy. Collision coefficient
  200 per metre, softplus distance temperature 0.005 m, choice temperature 0.25; target barrier
  normalized by 0.01 m with weight 5. KL weight 0.02 where used, measured against a standard normal
  prior in logit coordinates. The equivalent transformed-space KL has cancelling Jacobians.
- Final checkpoint, without best-step selection. 256 actual mixture proposals per arm/seed.
  Save checkpoint and proposals before evaluating the excluded recording.
- Independent NumPy piecewise segment/box evaluation, all eight source recordings, 10 mm target
  clearance and 10 mm upright primitive-overlap criterion. The latter is not penetration depth
  or proof of actual-robot interference.
- Compare with 256 transformed-prior proposals per seed and an independent 30-by-70 midpoint grid.
  A grid oracle samples from centers satisfying gradient-source constraints, without selecting on
  7903. Its 2100-query search cost is disclosed separately; it is not a matched-compute learner.
- Report raw target-clear yield, joint-valid yield, gradient-source yield, valid grid-center bins
  reached, per-seed outcomes, density parameters, loss history and training time.
- Choose one representative per arm/seed using the maximum minimum margin over gradient sources
  only, then apply the existing 81-point discrete scene-jitter audit to all sources. No jitter
  result is used to choose a replacement proposal.

## Predictions and falsifiers

1. The PyTorch spatial query agrees with the independent NumPy query within 1e-8 m on evaluated
   proposals and has finite gradients. An agreement failure invalidates learning conclusions.
2. Preference-supervised arms should improve joint geometric validity over random placement and
   clearance-only learning. Report each seed and misses; no significance claim from this carrier.
3. Adding KL may improve valid-bin coverage relative to no KL, at some cost in raw validity. A
   failure or reversed tradeoff is an outcome, not permission to retune this run.
4. A learned encoder need not beat direct optimization on one carrier. This experiment can show
   optimization works; amortization/generalization must be tested on separate source carriers.
5. Excluding a physics recording may reduce validity. Discrete jitter success is not guaranteed;
   failures remain visible and do not change the 10 mm thresholds.

Stop on source-hash mismatch, nonfinite loss/gradient, or 1800 seconds total run wall time.
Preserve partial logs and failure records. A short separately labelled engineering smoke run may
precede the registered 300-step experiment; disclose its scope and do not pool its measurements.

## Validation before launch

Analytic cases, missed-axis-sample regression, independent NumPy agreement, finite-difference
gradients for endpoints and box pose, finite overlap/parallel gradients, mixture-density agreement,
component-weight gradients, fixed-decoder permutation invariance, meaningful height-gradient
directions, and conservative-domain-culling agreement. No test is a physics or mesh certificate.
