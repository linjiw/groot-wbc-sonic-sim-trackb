# Locked traversal layout and comparison recipe

The exact geometry and comparison protocol are frozen in
[TRAVERSAL_EVALUATION_LOCK_V2.json](TRAVERSAL_EVALUATION_LOCK_V2.json), with
[SHA-256 companion](TRAVERSAL_EVALUATION_LOCK_V2.sha256):

```text
7898862a0a02ab7d65a5e21f245c15a77b82a74c86245e88a611ad48ffd58dfc
```

This is a prospective evaluation recipe, not completed evaluation or evidence
that every course is feasible. It fixes 24 single-beam layouts and 12 two-beam
compositions, balanced across development carriers 41001/41002/41003, with physics
seeds 93101 and 93102: **72 nominal episodes per policy**. The eight nonzero axis
perturbations form a separately reported 576-episode stress panel; they are not
silently added to the nominal denominator.

Geometry was sampled once with Python `random.Random(202609081601)` and rounded
to nine decimal places. The file records every draw, range, and source assignment.
Single passages cover short, sustained, and early constraints. Both beams of a
course must be instantiated together and executed without reset. No clearance or
physical outcome selected the geometries. A geometry-only audit against 267
manifest/registration/data snapshots found zero exact matches among 92 historical
`(route progress, underside, length)` triples. This is a bounded duplicate audit;
it does not establish dissimilarity to all scratch scenes or new motion ancestry.

These layouts are reserved from generator fitting, imitation, dataset aggregation,
threshold selection, and teacher-based tuning. Perturbations, branches, frames,
and derivatives inherit the layout's evaluation role. Ground-truth geometry,
object identity, source/layout IDs, and outcomes are excluded from student inputs.
Motion-source transfer is explicitly deferred: all three carriers have prior
development exposure, and no new ancestry is qualified here.

The acquisition checkpoints are 50,000 / 150,000 / 450,000 actual physics integration
steps, with acquisition seeds 93201/93202/93203. All evaluated physical branches
count, including failures and teacher/student branches. Geometry search, proposal
fit/inference, and learner computation are reported separately. Uniform,
target-only, analytic contrast, and the proposed analytic observation-aware
curriculum share the same qualified controller/options, sensor, and learner.
Conditional learned and analytic local-resampling proposals are separate ablations.
Always-walk, constant-adaptation, and scripted sensor policies remain comparators.

The initial shared integration uses a 35-ray body-mounted fan, causal surface and
occupancy history, 100 common sensor/state features, and a guarded binary linear
readout. Its actual calibration, qualification, timing and learner settings still
require a **separate immutable implementation lock** linked to this file's hash.
The multi-reference policy and course-continuation interface remain unqualified.
Do not run or rank the course panel until those execution requirements are met;
unsupported executions are `not_run`, not assigned outcomes.

Success requires first-episode course completion within 8 seconds for single
passages or 12 seconds for courses, with no fall, reset, illegal command, or beam
force above 1 N. The final crossing uses all recorded body origins at least 0.1 m
beyond the last beam and 0.3 seconds of upright hold; it is not an assertion about
every native collider extent. Complete thresholds and precedence are in the JSON.
Course/layout and acquisition-corpus replication determine uncertainty.

Verify the frozen bytes from the repository root:

```bash
cd docs/motion2scene
sha256sum --check TRAVERSAL_EVALUATION_LOCK_V2.sha256
```

Do not rewrite the frozen JSON or regenerate it with a different seed. A
substantive protocol change creates a new version that cites this hash and states
whether any evaluation outcomes were inspected. Completing the separate runtime
implementation lock must not alter these geometries or use their outcomes to
tune the implementation.
