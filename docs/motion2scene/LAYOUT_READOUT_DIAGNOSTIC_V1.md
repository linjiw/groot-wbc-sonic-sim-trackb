# Readout diagnostics for the independent-layout first wave

This is a post hoc development diagnostic, specified after inspecting the first
completed independent-layout block (layout_00, seed 8511). All twenty policies
requested neutral walking, including refusal fallback, and all contacted the beam.
The full primary assignment, fitted policies and execution contract remain unchanged.
No diagnostic outcome is a substitute for an actual baseline rollout or a prespecified
main comparison. CPU ceiling: 180 s; no new physics or fitting.

For every admitted first-wave row, recompute its pinned model readout. Replace only
the 144 ray features with the previously measured shared_absent capture at the same
0.30 s phase, retaining that row's seventy state/phase/skill/age values. This tests
sensitivity to an observation intervention. It is not a physical beam-removal rollout,
and does not establish that the altered observation is sufficient for avoidance.
Report request changes, refusal changes and probability changes by data arm.

A phase-only predictor is constant in this experiment because every decision occurs
at 0.30 s. Use the empirical two-action outcome frequencies in each arm's same eleven
training examples as the maximum-likelihood constant Bernoulli predictor, with the
same 0.5 threshold, walk preference and refusal fallback. Do not call this another
neural fit or a matched optimization experiment. Compare its requested actions and
refusals with the learned readouts; no physical baseline success rate is inferred.

Finally check whether non-ray inputs and pre-decision histories are identical across
the first three layouts at each physics seed. This is an observed input audit, not a
causal explanation of a passage failure. Missing/unadmitted blocks stay missing, and
no diagnostic changes training, sensor encoding, command timing or test assignments.
