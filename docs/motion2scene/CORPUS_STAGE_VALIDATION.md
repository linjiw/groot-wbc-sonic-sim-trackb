# Corpus stage validation, September 6

- 76/76 corpus physics cells completed; 38/38 paired inputs admitted. Every direct
  state/feature/ray/bank audit passes. All physical failures remain labeled.
- 20/20 CPU development fits completed under the frozen eleven-example design.
- 12/12 actual learned-command physics integrations pass feature, request, full
  trace and outcome equivalence. These scenes were already observed.
- 1,020 current artifact references independently hash-checked: all match.
  [Hash audit](evidence/corpus-completion-final_hash_audit.json).
- Focused impacted tests: **194 passed, 1 skipped**. Command:

```bash
PYTHONPATH=. .venv_research/bin/pytest -q \
  tests/dataset_generation/test_motion2scene*.py \
  tests/dataset_generation/test_capsule_box*.py \
  tests/dataset_generation/test_placement_certificate.py \
  tests/dataset_generation/test_trajectory_validation.py \
  tests/dataset_generation/test_trajectory_segments.py \
  tests/research/test_run_approved_manifest.py
```

Black checks pass on the new Python files. Ruff passes with two precise E501
exemptions for prose lines in the already hash-frozen completion and v4 reservation
scripts. Their source bytes are preserved rather than rewritten after registration.
New implementation files pass Ruff without exemptions. The combined analysis's
missing proposal-copy failure and the bounded v4 inventory timeout remain published.

The learned replay uses the actual twelve-cell result and shows all four seed-8501
learners on analytic_01, including failures. That scene was in the analytic training
set. It illustrates integration, not a fair generalization comparison. MuJoCo uses
mj_forward for rendering recorded Isaac poses; no MuJoCo dynamics are integrated.
External motion banks, controller checkpoints and environment installations remain
outside the downloadable source archive, as declared in its manifest.
