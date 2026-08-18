# LACE SONIC-Lite throughput gate

Scope: implement a plan-first, fail-closed throughput benchmark for
`N_env in {128,256,512,1024}` without launching Isaac Sim or allocating GPU
memory during implementation.

Metric: focused CPU tests and lint pass while the frozen protocol enforces at
least 20 timed PPO iterations after warm-up, exact cross-cell invariants,
fresh `/data` outputs, global GPU-memory preflight/safety margin, and
deterministic selection by sustained end-to-end control transitions/second
among passing cells. Iterations/hour remains a diagnostic because iteration
work scales with `N_env`. Every cell keeps the same 233 lexicographically
ordered `D_curriculum` motions resident on plane terrain and must attest that
runtime identity on every recorded iteration.

Verification commands and keep/discard decisions are recorded in
`results.tsv`.
