# No-contrast generator baseline: construction result

The required comparison arm completed **1,200 CPU updates** in **53.800 s**,
with 50 visits to each of the same 24 all8 training derivatives and initialization
seed 8421. The matched local model architecture, normalization, Adam configuration,
source-order and proposal/pose streams were retained. Preference and neutral-interference
loss terms are absent; only the target-clearance penalty remains.

The fixed-endpoint checkpoint is hash-recorded. Its sampled training loss reaches zero;
this does not establish useful scene diversity, criticality or downstream performance.
The original Motion2Scene checkpoint is unchanged. This is one baseline-generator fit,
**zero robot-data selector fits**, zero new sources and zero GPU spend.
The timer covers optimization only; initial source loading and model setup were not
separately timed. The protocol requested full wall-cost accounting, so this is an
instrumentation gap to preserve, not a complete generation-cost comparison.

The ablation's reusable search accepts only target clearances, retains the same 17-query
budget and uses target-only rejection. Its synthetic unit checks cover both the query
budget and acceptance of target-clear scenes without a neutral witness. A complete
sampling/acquisition manifest still needs to bind the deployed d040 target inputs and
charge all rejected proposals and physics labels before a learning comparison.

[Protocol](NO_CONTRAST_BASELINE_V1.md) · [Result and checkpoint hash](evidence/no-contrast-baseline.json) · [Registration](evidence/no-contrast-baseline-registration.json) · [Main comparison design](LEARNING_UTILITY_PLAN_V1.md)
