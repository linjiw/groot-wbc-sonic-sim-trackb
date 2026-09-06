# Action-label completion and legal timing v1

Registered 2026-09-06 before the following 12 Isaac Lab runs. This is development
interface qualification, not generator improvement or learned-policy evaluation.
Frozen generator, SONIC, old 42-cell panel, assets and contact criterion are retained.

## Commands and paired histories

A0 commits to neutral for the encounter; it does not mean waiting and reconsidering.
A1 requests the phase-aligned d040 reference once at the declared 50 Hz decision tick,
then requests the common return at 3.30 s. Entry permission remains 0.20–0.40 s,
maximum joint-reference jump 0.05 rad and root-reference jump 0.01 m. Refusal is
recorded as failure to execute the command, never physical stopping or successful
avoidance. No new edit operator or skill is introduced.

Six controls: absent/raised/blocked × seeds 8041/8042 execute the existing scripted
oracle (d040 request at 0.20 s) on exactly the previous scenes. Their comparator is
the previous no-switch reactive trace in the same scene/seed. Every pair must match
recorded pre-decision root/joint positions and velocities, applied actions, motion
tokens, references and clocks exactly. Sensor packets at the decision must match.
This uses deterministic replay from initialization, not a simulator/hidden-state
snapshot. Failure to match invalidates pairing; it does not become an outcome label.

Six timing cells: station −15 cm × seeds 8041/8042 × decision times 0.20/0.30/0.40 s.
A new opt-in command executor accepts an explicit A1 and enforces the same geometry-
independent guard. The overhang observations are logged but cannot select the action.
The two new 0.20 s cells audit reproduction of the old scripted timing rule; they are
not new independent source evidence. 0.20 s was already the earliest legal baseline,
so this sweep cannot test an earlier command outside the existing legal window.
Each timing cell is compared with the old blind pre-decision history in that scene.

## Predictions and endpoints

P1: all six control pairs and all ten prior critical pairs have matching pre-decision
recorded history and decision sensor packet, yielding 16/16 complete binary outcome
pairs. Preserve [pass,pass], [fail,pass], [pass,fail] and [fail,fail]; missing/invalid
comparators remain masked. This small one-ancestor panel is training-ineligible.
P2: absent/raised d040 passes 4/4; blocked d040 fails 2/2. No extra stop action.
P3: explicit 0.20 s execution reproduces both old oracle outcomes and full recorded
root/joint/action histories exactly. If it fails, the new command interface is not
qualified as a drop-in replacement and fitting dependent on it must wait.
P4 (mechanism hypothesis): at least one tested legal time permits contact-qualified
passage in the failed seed 8042. A negative result means no usable time among these
three tested ticks, not impossibility over all continuous times or all motions.
P5: every new cell retains causal packet identity, 200/50 Hz contact synchronization,
native beam transform, identical evaluation reference arrays, source-route binding,
and unchanged robot state/clock across legal reference changes.

Primary passage is the existing body-origin crossing plus 0.30 s stabilization,
no reset/fall and no per-body beam normal-force sample above 1 N. Report force peak,
duration and summed-body magnitude impulse; these are sampled simulator measurements.
Do not mix this with the older 14-cell native-envelope audit or rename thresholded
passage as literal zero contact. Legal command acceptance is separate from passage.

## Resources and stopping

12 serial, trajectory-only cells; 375 s per-cell timeout; 7500 MiB free-memory gate
inherited from the registered resource revision. Ceiling 1.25 contended GPU h,
within standing 8/day and 24/week. Yield on contention, resume only the same manifest;
stop on infrastructure failure or hash mismatch. No scientific rerolls or replacement.
All outcomes and costs remain. Analyze only after all 12 cells complete. Inputs,
source result, protocol, new executor and analysis are hashed before launch.
