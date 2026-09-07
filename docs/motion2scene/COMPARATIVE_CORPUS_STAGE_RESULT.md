# Comparative command corpus: measured outcomes and learning readiness

The first twenty Isaac Lab runs are complete, with 10/10 matched command-label pairs. The remaining fixed batch has completed 56/56 runs. The combined corpus contains 38 measured scene pairs; input admission is True. All 76 direct feature/state/ray/bank checks pass; 34/38 d040 requests log a return. Missing returns remain failures, not completed recovery.

All 20 matched development fits are complete: four data arms × five optimizer seeds, eleven complete encounters per arm, 1000 updates each.

**Unseen-layout policy executions and learning-utility comparisons remain incomplete.** Training losses are fit diagnostics, not traversal success or evidence that one generator teaches a better policy.

| Arm | Pairs | Both pass | Walk fail / d040 pass | Walk pass / d040 fail | Both fail | Missing |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 9 | 3 | 0 | 1 | 5 | 0 |
| analytic | 8 | 0 | 4 | 0 | 4 | 0 |
| no_contrast | 9 | 8 | 0 | 1 | 0 | 0 |
| motion2scene | 9 | 0 | 0 | 0 | 9 | 0 |
| shared | 3 | 2 | 0 | 0 | 1 | 0 |

**Main observed limitation:** all nine Motion2Scene generated encounters are both-fail under this fixed transition contract; four of eight analytic encounters admit useful d040 adaptation. This is a development data-yield result, not an independently evaluated policy ranking. The generator remains frozen and no failed assignment is replaced.

Post-fit input diagnostic: 1 exactly identical feature group(s) contain conflicting measured outcomes, covering 13 scenes. A deterministic predictor given these same 214 values cannot match every outcome in the group. Keep this observation-interface limitation separate from generator quality; no sensor or label is changed here. [Exact feature aliases](evidence/corpus-feature-aliases.json).

Shared controls are listed once. Repeating the same controls in four fitting datasets does not create four independent scene samples. The single source remains 41002; optimizer seeds are not source replication. All generated assignments and the analytic test-neighborhood rejection are retained.

First-slice capture audit: `{"features_exact": true, "ray_origin_bound": true, "recorder_state_bound": true, "reference_bank_exact": true}`. These checks compare direct pre-command features, ray origins, recorder state and the actual loaded neutral/d040 banks. Prior 0.20 s development labels are excluded.

The automatic first-slice legal-switch predicate is `True`, but the protocol's stronger entry-and-return prediction is `False`. 8/10 d040 executions log a return. A reset before the return phase is not evidence that recovery occurred. [Return review](evidence/first-slice-return-review.json).

Measured new physics cost across both batches so far: **0.673639 contended GPU h**. The union record is an analysis-only derivation of the two real batches and is explicitly nonexecutable; it is never counted as a third run.

The eleven-example fitting subset uses the first eight geometrically eligible assigned slots, in fixed slot order, plus the same three backgrounds. Its IDs were frozen before admission or fitting and do not depend on passage outcomes. Three unused ninth generated pairs remain in the corpus. The assigned background quota is 3/12; the disclosed fitting subset is 3/11 backgrounds in every arm. This development size does not replace the planned larger data-budget experiment.

Eight final-transfer candidate IDs (9490001–9490008) are reserved against ten named acquisition-provenance files, with canonical fingerprints for 212 prior reference files. This revised scope does not establish absence from all JSON metadata; the four broad inventory attempts remain failed/refused history ([v4 timeout](evidence/transfer-reservation-v4-failure.txt)). No final source is generated or qualified. Post-generation duplicate checks and the same switching qualification remain mandatory. The twelve independent layout tests remain untouched. [Reservation scope](FINAL_SOURCE_PROVENANCE_RESERVATION_V1.md) · [Ledger](evidence/final-source-provenance-ledger.json) · [Reservation](evidence/final-source-provenance-reservation.json).

The first combined CPU analysis stopped because the assembly omitted a local copy of the frozen proposals. A versioned adapter copies the identical hash-bound proposal bytes and reruns the unchanged analyzer; the original failure and scripts remain preserved. No physics was repeated and no labels or fitting IDs were changed. [Assembly repair](evidence/corpus-completion-analysis_assembly_repair.json).

## Complete paired labels

| Scene | Arm | Walk | d040 request | Matched inputs |
| --- | --- | --- | --- | --- |
| uniform_00 | uniform | fail | fail | True |
| uniform_01 | uniform | fail | fail | True |
| analytic_00 | analytic | fail | fail | True |
| analytic_01 | analytic | fail | pass | True |
| no_contrast_00 | no_contrast | pass | pass | True |
| no_contrast_01 | no_contrast | pass | pass | True |
| motion2scene_00 | motion2scene | fail | fail | True |
| motion2scene_01 | motion2scene | fail | fail | True |
| shared_absent | shared | pass | pass | True |
| shared_blocked | shared | fail | fail | True |
| uniform_02 | uniform | pass | pass | True |
| uniform_03 | uniform | pass | pass | True |
| uniform_04 | uniform | fail | fail | True |
| uniform_05 | uniform | fail | fail | True |
| uniform_06 | uniform | fail | fail | True |
| uniform_07 | uniform | pass | pass | True |
| uniform_08 | uniform | pass | fail | True |
| analytic_02 | analytic | fail | fail | True |
| analytic_03 | analytic | fail | pass | True |
| analytic_04 | analytic | fail | pass | True |
| analytic_06 | analytic | fail | fail | True |
| analytic_07 | analytic | fail | pass | True |
| analytic_08 | analytic | fail | fail | True |
| no_contrast_02 | no_contrast | pass | pass | True |
| no_contrast_03 | no_contrast | pass | pass | True |
| no_contrast_04 | no_contrast | pass | pass | True |
| no_contrast_05 | no_contrast | pass | pass | True |
| no_contrast_06 | no_contrast | pass | pass | True |
| no_contrast_07 | no_contrast | pass | pass | True |
| no_contrast_08 | no_contrast | pass | fail | True |
| motion2scene_02 | motion2scene | fail | fail | True |
| motion2scene_03 | motion2scene | fail | fail | True |
| motion2scene_04 | motion2scene | fail | fail | True |
| motion2scene_05 | motion2scene | fail | fail | True |
| motion2scene_06 | motion2scene | fail | fail | True |
| motion2scene_07 | motion2scene | fail | fail | True |
| motion2scene_08 | motion2scene | fail | fail | True |
| shared_raised | shared | pass | pass | True |

[Before-run completion and fitting protocol](COMPARATIVE_CORPUS_COMPLETION_V1.md) · [First capture audit](evidence/comparative-acquisition-capture_audit.json) · [First measured rows](evidence/comparative-acquisition-result.json) · [Completion record](evidence/corpus-completion-run_record.json) · [Updated source snapshot](evidence/corpus-stage-source-20260906/research-source.tar.gz)

## Training diagnostics only

| Arm | Optimizer seed | Training BCE | CPU seconds |
| --- | --- | --- | --- |
| uniform | 8501 | 0.347277 | 0.292 |
| uniform | 8502 | 0.347311 | 0.261 |
| uniform | 8503 | 0.347306 | 0.258 |
| uniform | 8504 | 0.347306 | 0.259 |
| uniform | 8505 | 0.347298 | 0.257 |
| analytic | 8501 | 0.00017986 | 0.257 |
| analytic | 8502 | 0.000154542 | 0.256 |
| analytic | 8503 | 0.000240872 | 0.256 |
| analytic | 8504 | 0.000195606 | 0.255 |
| analytic | 8505 | 0.000132183 | 0.256 |
| no_contrast | 8501 | 8.23682e-06 | 0.256 |
| no_contrast | 8502 | 7.86062e-06 | 0.256 |
| no_contrast | 8503 | 4.38522e-06 | 0.257 |
| no_contrast | 8504 | 1.44398e-05 | 0.258 |
| no_contrast | 8505 | 4.21159e-05 | 0.259 |
| motion2scene | 8501 | 5.21515e-05 | 0.263 |
| motion2scene | 8502 | 2.14977e-05 | 0.261 |
| motion2scene | 8503 | 5.0427e-05 | 0.262 |
| motion2scene | 8504 | 4.28818e-05 | 0.260 |
| motion2scene | 8505 | 4.59034e-05 | 0.258 |

[Fit manifest](evidence/corpus-fits-manifest.json) · [All fit diagnostics](evidence/corpus-fits-result.json)

## Learned command integration on observed scenes

Twelve actual learned-policy Isaac Lab runs completed. Predicates: `{"feature_exact": true, "model_request_matches": true, "outcome_matches": true, "trace_exact": true}`.

This is an implementation check on analytic_01, absent and blocked scenes already observed during development. It is not an independent test or a fair arm-ranking benchmark. A refusal is logged separately from neutral commitment; it does not stop the robot.

Additional integration cost: 0.110135 contended GPU h.

[Before-run protocol](LEARNED_COMMAND_CHECK_V1.md) · [Complete integration result](evidence/learned-command-check-result.json)

[All four learned-command replays](assets/learned-integration.mp4) · [Replay source receipt](evidence/learned-command-demo.json). Observed analytic_01 was in the analytic training set; this is an integration illustration, not a fair test comparison. Measured Isaac poses are rendered with MuJoCo mj_forward; no new dynamics are simulated.
