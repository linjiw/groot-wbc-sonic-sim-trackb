# LACE runtime-provenance autoresearch

Goal: replace the atlas recorder's declared-seed placeholder with a CPU-testable,
fail-closed post-reset realization contract before any GPU smoke run.

Acceptance metric: canonical realization records must be deterministic, sensitive
to realized state, identical across probe policies for a common-random-number cell,
rejected when active interval events or required readbacks are missing, and accepted
by the broader CPU LACE suite.

Outcome: keep. The recorder now hashes resolved ordered event configuration,
per-environment physical/randomized parameters, reset state, and policy-neutral
scheduled-reference state. A separate invariant robot-contract digest retains ordered
body/joint names and hard/soft position and velocity limits. Scientific atlas ingest
recomputes both realization/config hashes and rejects policy-dependent realizations.

No simulator or GPU run was performed. Independent multi-hot termination predicates
remain an explicit scientific-runtime blocker; cached Isaac termination winners are
not misrepresented as multi-hot observations.
