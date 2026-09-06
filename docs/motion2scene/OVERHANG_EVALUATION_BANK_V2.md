# Overhang interface v2: evaluation-only alternate reference bank

Registered 2026-09-06 after completing all 14 v1 cells, before v2 physics.
V1 P1/P2 fail while guard and contact-measurement predictions pass. The first
seed's switched reference becomes constant after frame 23: 175 subsequent
adjacent frame differences are exactly zero, and the robot stops near X=0.54 m.
The second seed's reference keeps moving. All failed cells and costs remain.

The inherited `ReactiveCommand` calls `load_motions_for_training` for its alternate
library. `MotionLibBase.load_motion_with_skeleton` can freeze the remaining frames
when `is_evaluation=False`. This code path and the recorded flat reference tail
support training augmentation as the cause; v1 did not log the sampled augmentation
flag. Do not call the zero-force stalled trial successful traversal.

Repeat the same 14 condition/seed requests with the same observer, guard, geometry,
thresholds and controller, but reload the alternate library using
`load_motions_for_evaluation(start_idx=0)`. Save/restore Python, NumPy and Torch
CPU/CUDA RNG states around that deterministic reload so subsequent randomization
is not shifted by the repair. Do not edit the motion files or the release controller.

Before the first command update, verify each loaded 199-frame root XY route
against interpolation of its hashed 30 Hz source at 50 Hz (maximum error <=1e-4 m).
Reject a frozen/shifted route instead of silently executing it. Save the loaded root
and DOF banks, source-route errors and loader identity. This route check detects
freezing/heading changes; it is not a full asset/pose collision certificate.

P1–P4 and the complete denominators remain those in [v1](OVERHANG_INTERFACE_V1.md),
scored separately. Additional P5: all 14 runtime bank audits pass and the loaded
root/DOF arrays are identical across all condition/seed cells. Do not pool v1/v2
as independent transfer observations. Newly saved bank artifacts support independent
post-run source-route checks and cross-cell array comparison.

Hash-pin the completed v1 manifest, run record and result, the source implementation
of the training/evaluation loaders, the new subclass and root-route checker, and
this protocol. Budget: 14 serial cells, 375 s/cell, <=1.45834 contended GPU-hours,
7500 MiB free-memory startup floor; no cameras. Stop on infrastructure failure.
Retain all scientific failures without rerunning tuned parameters. Check for leaked
rollout descendants after completion; charge reserved GPU time explicitly.
