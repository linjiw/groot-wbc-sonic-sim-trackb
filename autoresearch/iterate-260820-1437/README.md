# LFH Phase-2 Critical-Set Continuation

- Objective: close the LFH method-module gap after the accepted Phase-2 review and refuse any
  CPU artifact whose generated binding footprint no longer preserves the golden 2x2 geometry.
- Primary metrics: all 29 collision capsules map to one semantic group; the exact duck_003
  footprint recovers the historical reach window within the source search tolerance; generated
  easy/hard scenes retain the intended capsule-sign pattern; focused tests pass.
- Safety boundary: CPU-only. No GPU rollout, manifest authorization, prediction-register edit,
  frozen-split write, paper edit, push, or deployment.

See `results.tsv` for keep/discard decisions.
