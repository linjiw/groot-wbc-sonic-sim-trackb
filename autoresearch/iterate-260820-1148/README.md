# LFH Phase 1.1 / Phase 2 CPU Iteration Log

- Objective: close the accepted Phase-1 review tasks, then implement and verify the authorized
  Phase-2 CPU machinery without running physics or touching frozen evaluation artifacts.
- Primary metrics: canonical index = 2 independent sources / 3 verified variants; `duck_003`
  shelf faces reproduced within 0.5 mm; every generated face passes station and measure-back
  checks within 0.5 mm; injected keep-out violations are refused; at least five overhead and two
  lateral-gap archetypes are available.
- Verification: explicit pytest file list, legacy release consumers, canonical coverage render,
  deterministic golden instantiation, and property tests.
- Safety boundary: no GPU rollouts, prediction-register edits, frozen-split writes, pushes, or
  deployments.

See `results.tsv` for each keep/discard decision.
