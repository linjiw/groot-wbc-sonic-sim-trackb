# LACE probe milestone autoresearch

- Started: 2026-08-13 23:49 America/New_York
- Scope: advance the frozen CPU-only LACE foundation to a bounded, deterministic
  SONIC failure-probe/atlas smoke path.
- Primary metric: all new LACE probe contract tests pass, with no silent-zero or
  missing mechanism channels and byte-identical native sampling when LACE is disabled.
- Secondary metric: a two-motion smoke artifact validates against the versioned atlas
  schema and is written under `/data/robotixx/groot-wbc-sonic-research/lace`.
- Verify: focused Pytest, Ruff, artifact self-digest validation, deterministic rerun
  hash comparison, and existing compatible SONIC research tests.
- Keep rule: retain only changes that improve the metric without touching unrelated
  Kimodo/dataset-generation work or enabling LACE in the release config.
- Stop rule: do not launch a full atlas, PPO sweep, or kill the existing GPU process.

