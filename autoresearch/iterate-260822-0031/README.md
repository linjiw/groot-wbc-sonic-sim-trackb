# Autoresearch iteration 2026-08-22 00:31

Objective: make LFH infer a proposal distribution over critical scenes from an executed motion
pair, verify the small-data mechanism, and produce an auditable simulator render.

## Iterations

1. **Conditional kernel v1 — keep.** Added source balancing, trajectory-local kernel weighting,
   archetype exploration, bounded critical-coordinate sampling, and leave-one-source-out tests.
   It improves log loss over the uniform-compatible prior and retains all sampled coordinates in
   exact support.
2. **Default 768 px MuJoCo render — discard/fix.** The upstream G1 MJCF exposed a 640 px offscreen
   framebuffer. No result was inferred from the failed render.
3. **Explicit offscreen framebuffer and compiled-geometry audit — keep.** Both hard trajectories
   render at 640x360/25 fps. The generated MJCF compiles all selected USDA boxes with at most
   `2.22e-16 m` centre error and zero size error.

The NVIDIA driver is unavailable (`nvidia-smi` cannot communicate with it), so no new Isaac rollout
was attempted. Existing validated E10b Isaac trajectories provide the authoritative physics labels;
MuJoCo is used only for kinematic visual cross-checking.
