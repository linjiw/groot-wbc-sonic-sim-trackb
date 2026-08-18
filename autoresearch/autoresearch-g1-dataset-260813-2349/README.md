# Kimodo + SONIC + Isaac Lab G1 dataset autoresearch

- Started: 2026-08-13 23:49 America/New_York
- Iteration bound: 25
- Scope: advance the M0 synthetic G1 dataset pipeline from offline contracts to
  physics-executed, latent-recorded, loader-validated household and factory spikes.
- Primary metric: number of M0 gates proven end-to-end without regressing the
  offline dataset-generation suite.
- Baseline: contracts, qpos adapter, guarded `scene_usd`, and reduced
  `synthetic_g1` LeRobot profile pass offline validation; physics and GR00T loader
  gates are still open.
- Verify: focused Pytest, Ruff, `git diff --check`, real Kimodo-to-SONIC FK error,
  one-environment Isaac rollouts, latent shape/finite/non-constant checks, and
  LeRobot/GR00T dataset loading.
- Keep rule: retain only changes that close an M0 gate or expose a reason-coded
  blocker without weakening existing live-VR validation.
- Stop rule: avoid downloads while free disk is below 10 GB; do not interrupt the
  unrelated GPU process or modify unrelated LACE/research files.
- Outcome: the seven implementation-spike gates are proven with fresh household
  and factory Kimodo-derived, SONIC-executed, Isaac-rendered rollouts that each yield
  an accepted subclip. Each final export has 65 causal rows and includes the audited
  joint-order conversion, pair-resolved support evidence, typed lineage, runtime
  artifact binding, strict file validation, and a passing pinned official GR00T
  loader smoke. This does not waive the design's formal 20-episode visual-review or
  C++ latent-parity gates.
  Live Kimodo model inference was not restored because the existing environment is
  incompatible with the RTX 5090 and free disk remains constrained.
