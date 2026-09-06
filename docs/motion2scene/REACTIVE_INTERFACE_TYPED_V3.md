# Reactive interface v3: typed PhysX hits and explicit callback failure

Registered 2026-09-06 before v3 physics. V2 enabled scene queries but still saved
zero valid ray hits. Its logs contain `AttributeError` because `raycast_all`
passes `omni.physx.bindings._physx.RaycastHit` objects, not dictionaries. The
installed binding declares `.collision`, `.distance` and `.position` attributes.
PhysX swallowed the Python callback exceptions, so the ordinary rollout marker
was insufficient evidence of valid sensing. V2 was stopped during its sixth
cell after five cells completed; retain those raw artifacts, the interrupted
cell, all unstarted requests and its charged GPU time. No V2 sensing success is
admitted from a normal rollout exit.

V3 uses the same 12 condition/seed requests and P1/P2/P3 rules as
[the original protocol](REACTIVE_INTERFACE_V1.md). Only repair hit normalization
and failure propagation: support typed objects and mapping objects; collect a
callback exception and raise it immediately after the query returns; require
scene queries enabled at runtime. Preserve nearest-hit occlusion, robot exclusion,
ray geometry, height band, latch, switch phases and d040 reference. Do not use
beam identity to select a hit. Unit tests cover the installed typed-hit shape,
occlusion, robot exclusion, absent/high negatives and swallowed-callback failure.

A separate subclass overrides only the observer; immutable original switching,
reference loading and 200 Hz recording code are reused. Hash-pin v2 manifest,
its final failure record, its rollout logs, binding declaration, new code and
this protocol. All repeats are separately labelled development repairs.

Budget: 12 serial cells, 375 seconds/cell, 1.25 contended GPU-hours ceiling,
7500 MiB free-memory floor, no cameras. Stop on any new infrastructure failure;
never retry a scientific failure. Report actual costs of v1, interrupted v2 and
v3 separately, including the interrupted cell. No training promotion or new
source-generalization claim. A successful v3 cannot erase the v1/v2 failures.
