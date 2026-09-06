# Source execution: recover saved artifacts and resume unstarted cells

2026-09-06. The original eight-source pilot stopped with `driver exited -13` on
`source_41005_absent_d055`. Its log ends with a saved 199-frame trajectory and
`SONIC_EVAL_SUCCESS`. Signal 13 is consistent with a broken output pipe; its cause
is not established. No simulator process from the pilot remains running.

Before any further physics, preserve the original manifest, run record and all
outputs. In a separate recovery directory, admit existing data only if the unchanged
manifest driver's success-marker, unique-trajectory, 50 Hz and evaluability checks
pass, and the unchanged source scorer validates synchronized filtered contacts and
the native inventory. Admit an evaluable rejection as well as an acceptance. Do not
rerun either completed cell or replace a scientific result. Missing or invalid
artifacts stop this recovery. Keep the original infrastructure-failure record linked
by hash, and retain its already charged execution cost exactly once.

Copy the scientific manifests/proposals unchanged into the recovery directory.
Continue the remaining 14 absent cells and the originally conditional present cells
with the original seeds, scenes, source IDs, controller, thresholds, resource limits
and refusal denominators. Existing output paths remain those in the original frozen
manifest; only derived run records/results live in the recovery directory. Redirect
the coordinator's stdout/stderr to a persistent file so terminal lifetime cannot
close its output pipe. Stop on another infrastructure failure. Do not alter the
registered source-execution predictions or launch replacement sources.

This is artifact recovery, not another rollout or evidence of successful passage.
The unchanged first-episode passage scorer and separate full-sequence tracker
determine the scientific results. No recovered outcome was inspected before this
recovery rule was written. Remaining maximum physics cost is 62 × 375 / 3600 =
6.4584 contended GPU-hours; the original 0.044968 hours stays charged. The recorded
prior weekly use plus this maximum remains below the standing 24-hour envelope.
