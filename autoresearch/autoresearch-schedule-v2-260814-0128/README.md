# LACE schedule schema v2 autoresearch loop

Goal: make new rollout schedules derive clip lengths only from a validated,
split-bound reference-length inventory while retaining read-only validation of
legacy schema-v1 artifacts.

Metric: all schedule, atlas, and storage-lock CPU tests pass; the frozen pilot
artifact is byte/hash locked and live storage readiness succeeds without Isaac
or GPU use.

Iterations are recorded in `results.tsv`.
