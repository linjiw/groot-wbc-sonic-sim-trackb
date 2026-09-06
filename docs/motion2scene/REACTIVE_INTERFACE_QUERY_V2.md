# Reactive interface v2: enable PhysX scene queries

Registered 2026-09-06 after inspecting the first absent and critical-beam v1
recordings, before v2 simulation. V1's first critical-beam cell recorded zero
scene-ray hits despite 261.463 N sampled contact. Installed Isaac Lab declares
`SimulationCfg.enable_scene_query_support=False`; v1 did not override it.
This is a missing sensor-runtime setting, not evidence that the physical beam
is outside the sensor field of view.

Preserve the entire original 12-cell batch and all its failures. Repeat the
same 12 condition/seed requests as a separately identified development revision,
with **one runtime change**: `sim.enable_scene_query_support=True` in an opt-in
environment subclass. Reuse the immutable original sensing, switching and contact
recorder code. No threshold, phase, reference, checkpoint, scene or score tuning.
Hash-pin the completed v1 manifest, run record and result before v2 launch.
The prelaunch-v1 manifest had no simulator spend; prelaunch-v2 fixed only the
analysis reader's handling of a raised collision-enabled beam before any run.

The design, P1/P2/P3 predictions and stop conditions are those in
[REACTIVE_INTERFACE_V1.md](REACTIVE_INTERFACE_V1.md), scored separately in v2.
Resource ceiling: another 12 serial cells, 375 s/cell, at most 1.25 contended
GPU-hours and the 7500 MiB trajectory-only floor documented in
[the resource continuation](REACTIVE_INTERFACE_RESOURCE_V1.md). Compare against
the consolidated v1 under that same floor; only scene-query support changes the
simulation configuration. Report both actual costs.
Do not pool repeats into 24 independent observations or hide the v1 sensor failure.
This repairs an interface prerequisite; it is not a held-out confirmatory study.
