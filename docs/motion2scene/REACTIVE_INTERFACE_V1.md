# Phase-aligned reactive interface pilot v1

Registered 2026-09-06 before physics. Development carrier 41002 only; no new
source-generalization or learned-policy claim. SONIC checkpoint remains frozen.

## Fixed design

12 serial Isaac Lab runs: seeds 8021, 8022 crossed with absent/reactive,
present/reactive, raised/reactive, present/blind, absent/oracle, present/oracle.
Use the existing analytic beam_021, underside 1.2652671813964844 m; raised
underside 2.0 m. Absent disables collision and visibility. Both alternatives are
the existing neutral and d040 references, with equal clocks and duration. d040
is permitted because it already cleared this beam; no d055 necessity claim.

A separate command subclass loads both clips. Start neutral. After each command
clock update, cast twelve PhysX scene-collision rays from pelvis XY + 0.2 m
forward and pelvis Z + 0.4 m. Gravity-aligned heading; azimuth {-10,0,10}
and elevation {-5,0,5,10} degrees; range 3 m. Ignore the robot itself; retain the
nearest other collision on each ray. Occupied means a nearest hit has world
height in [1.15,1.50] m (known flat-floor task assumption). Rule does not access
beam identity or authored pose. Sensor is ideal sparse collision lidar, not depth
rendering, a ground height map, or a learned representation.

At motion time >=0.2 s, occupied triggers neutral→d040 and latches; oracle
triggers regardless of sensing. At >=3.3 s, return neutral. Blind always stays
neutral while recording the same sensor. This is a constrained phase-aligned
interface, not arbitrary gait switching or late-obstacle recovery. No blending,
root/joint writes, environment reset, clock reset, controller fitting or tuning.
Record reference discontinuities and exact state/clock equality at every switch.

Capture actual filtered beam normal forces directly from the PhysX contact view
after every 0.005 s physics step, bypassing cached sensor timestamps. Preserve
50 Hz forces too. Require consecutive physics step IDs, four substeps per
control frame, and last substep agreement with 50 Hz contact samples (1e-5 N
absolute tolerance). Aggregate each body's largest substep vector for the existing
first-episode passage rule (>1 N rejects); report true 200 Hz peak, duration >1 N
and sum of body normal-force magnitudes integrated over the passage horizon.
This impulse statistic is not net vector impulse. Crossing still uses body origins.

## Predictions, scored across both seeds without replacement

P1: absent/raised reactive make no switches; present reactive makes exactly two,
neutral→d040 before 1.0 s and d040→neutral at 3.3 s, without robot state/clock writes.
P2: all reactive and oracle cells pass contact-free first-episode passage, with
zero observed falls/resets; both blind present cells exceed 1 N.
P3: physics-step synchronization passes in all cells, and blind present supplies
positive contact while absent supplies negative controls.
These are falsifiable development predictions, not guarantees. Preserve every miss.

## Resource and stop contract

Hash-pin this protocol, addon code, original runtime, checkpoint, both references,
scenes, inputs and installed stepping/contact implementations before launch.
Use the standing 9000 MiB free-memory floor (GPU was ~15 GiB free during planning),
375 seconds per cell, at most 12 cells / 1.25 contended GPU hours. No cameras.
Stop on infrastructure failure; no scientific retries. A necessary infrastructure
repair gets a separately registered revision retaining original failure and hashes.
No training promotion and no access to 430xx fresh-source audit for fitting.
