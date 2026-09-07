# Learned command interface check v1

Registered before fitting results or learned-policy executions. After the full corpus
passes admission and all twenty development fits complete, use optimizer seed 8501
from each arm on three already observed encounters: analytic_01, shared_absent and
shared_blocked. These twelve executions test the new learned command implementation;
they are not independent-layout evaluation or comparative evidence of data utility.
The analytic encounter was already observed to have walk-fail/d040-pass outcomes.
Its use here must not be presented as an unseen test or fair arm-ranking benchmark.

The outcome predictor receives the same directly captured 214 values at 0.30 s. Its
checkpoint and train-only normalization are hash-bound. CPU model construction and
inference preserve the controller RNG. The model makes the scene-dependent decision;
the existing phase/jump guard handles legality and the common 3.3 s return. If both
predicted outcomes exceed 0.5, prefer walking; otherwise request predicted-feasible
d040. Neither positive means a recorded refusal with neutral commitment as fallback.
The robot still moves. Refusal correctness does not make a collision an avoidance
success. No extra rule overrides the model and no parameters are tuned on these runs.

Physics seed remains 8602. Before running, compute the expected model probabilities
and request from the existing pre-decision capture. Predictions for all twelve cells:
(1) the in-engine feature vector matches the paired capture exactly; (2) the recorded
model probabilities agree within 1e-6 and the selected/fallback action matches; (3)
the complete state/action/token/reference trace matches the corresponding already
measured explicit-command rollout; (4) passage/contact outcome agrees with that
rollout. Any mismatch is an implementation finding, retained and resolved before an
independent test panel is consumed. No simulated outcome is invented from the lookup:
all twelve new cells must actually execute to count as this integration result.

Serial Isaac Lab, fixed 7500 MiB floor and 375 s per-cell timeout; at most twelve
executions / 1.25 contended GPU h. Launch only after the completion batch, within the
standing daily/weekly budget. Preserve all failures, no retries. The twelve reserved
independent layouts and the final-transfer candidates remain untouched by this check.
