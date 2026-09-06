# Registered overhang variation panel v1

Registered 2026-09-06 before manifest creation and simulator spend. This implements
the preceding variation design, retaining the repaired evaluation banks, SONIC
checkpoint, observer, and phase/joint/anchor switch thresholds. The changed component
is causal packet delivery; scene changes are explicit experimental factors.

The intended contribution remains motion-conditioned generation of useful training
constraints, measured closed-loop execution, and a downstream learning comparison.
This panel addresses only the interface's finite perturbation and delay limits.

## Complete denominator and controls

Carrier 41002; new physics seeds 8041/8042. Five beam poses: nominal underside
1.2652671813964844 m; underside ±0.01 m; longitudinal center ±0.15 m along the
registered beam normal. All five poses run reactive, blind and oracle selectors:
30 executions. Oracle uses the same legal onset/return switches; it does not receive
an unrestricted motion controller.

Three additional nominal-scene reactive delays, 0.10/0.25/0.50 s, add six executions.
They share the nominal zero-delay blind/oracle controls; these controls are not counted
again as independent repeats. Packets are sampled at 50 Hz and delivered only after
the requested age: effective delays are 0.10/0.26/0.50 s. Preserve complete hit geometry,
original command timestamp and monotonic capture index. No packets precede capture;
missing startup packets mean no detection. A final motion-clock wrap does not rewind
the packet queue. Reset/fall outcomes remain in the denominator.

Absent, raised (underside 2 m), and blocked-underpass reactive controls add six
executions. The blocked control extends the same instrumented rigid cuboid from the
floor to the nominal beam top, preserving its XY footprint. Thus lower rays see solid
occupancy and the contact sensor measures the blocking cuboid. This is a wall-like
negative classification control, not a navigable scene. Refusing to crouch does not
make its traversal successful. Total: **42**, serial, no scientific replacements.

## Predictions fixed before execution

- **P1 nominal:** reactive and oracle both complete contact-free passage in both seeds;
  blind has measured obstacle contact in both seeds.
- **P2 finite pose separation:** the same separation holds for all five registered
  poses in both seeds. A failure is informative; do not adjust heights or omit cells.
- **P3 negative specificity:** all six absent/raised/blocked cells have zero raw
  overhang-positive frames and zero switches over the complete capture. Report empty/
  raised passage and blocked contact separately from this classification prediction.
- **P4 delay boundary:** both 500 ms reactive cells issue denied requests, never switch,
  and fail contact-free passage. The 100/250 ms outcomes are descriptive boundary
  measurements, not pass criteria chosen after observation.
- **P5 measurement contract:** all 42 cells preserve state/clock and legal reference
  jumps; exact delayed packet reconstruction passes; loaded reference banks agree
  across cells; native scene transforms agree with each cell's declared cuboid; direct
  200 Hz contact and the last-substep 50 Hz record agree within 1e-5 N.

Use the frozen first-episode passage score: every recorded body origin beyond the
beam plane by 0.10 m, upright stabilization for 0.30 s, and no per-body normal force
above 1 N through that horizon. Also report complete capture length, reset/fall,
incomplete crossing, force duration/impulse, raw/delivered observations, denied reasons,
and reference fidelity. Crossing remains a body-origin criterion, not full colliders.
The force score is simulator-specific and is not a hardware safety certificate.

## Resource and failure rules

Hash-pin the manifest, script, protocol, scenes, motion/asset/controller dependencies
before launch. Retain the established trajectory-only 7500 MiB floor, serial execution,
375 s cell timeout and success-marker/50 Hz capture checks. Ceiling 42×375 s = 4.375
contended GPU h. Prior M2S run records conservatively sum below 2 h including duplicated
recovery accounting and the separate orphan reservation supplement; this ceiling fits
the standing 8 h/day and 24 h/week envelope. Actual cost must be recorded separately.
On infrastructure failure stop, report the failed cell, inspect scoped process cleanup,
and register any recovery. Never modify pinned dependencies or the 430xx audit sources.

Publish the complete table and deterministic seed-8041 replays of nominal reactive,
nominal blind, and 500 ms delay regardless of outcome. A replay of recorded body origins
must be labeled as such; it is not another simulator execution or a mesh visualization.
No downstream learning, source transfer, noisy depth, full-collider crossing, or hardware
claim is licensed by this panel.
