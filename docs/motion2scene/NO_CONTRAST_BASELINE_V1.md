# No-contrast baseline construction v1

Registered 2026-09-06 before CPU fitting. Construct the fourth generator arm needed
for the learning-data comparison; do not improve or refit the frozen Motion2Scene arm.
Same EventMixture(width=32), initialization seed 8421, 24 all8 training derivatives,
training-only normalization, Adam rate, 1200 updates, source permutation and proposal/
pose random streams as the selected all8_8421/1200 Motion2Scene model. Retain the
five-placement sampled target penalty with coefficient five. Remove the preference
and neutral-interference penalties completely; no KL term is added.

Target-only pattern17 correction and target-only independent rejection must be used
when sampling this arm. Neutral geometry cannot rank, repair or reject its outputs.
All fitting sources are the original 41001–41004 and 42001–42004 development ancestors;
430xx and reserved downstream test layouts are excluded from fitting/selection. The
historical target references are 55 mm local edits. Applying either model to the
common deployed d040 bank requires explicit input binding, not relabeling old targets.

This is one baseline-generator fit, not a robot-data selector fit, comparison result,
or new source acquisition. Record finite losses/gradients, all source visits, full
CPU wall time and the output checkpoint. No outcome-based checkpoint selection;
fixed update 1200. CPU only, two threads, maximum 300 seconds of training; retain any
failure without restart. Exact fit metadata and original checkpoint are hashed before
optimization. Prediction: finite fixed-budget completion with all loss terms except
target clearance structurally absent. Learning utility is tested later in physics.
