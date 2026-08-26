# E9a Fresh Motion-to-Scene Report

The fresh Kimodo transaction is **refused before scene construction**.

Kimodo generated a 120-frame curved-left walking motion from the registered prompt at seed 45001.
The CPU motion gate passed, and `local_crouch` produced a reference twin with a 79.999 mm local
head/torso drop while preserving the root path. Isaac empty-scene physics then produced:

| motion | outcome | endpoint error | p95 path error | external contact |
|---|---|---:|---:|---:|
| nominal | accepted | 0.08377 m | 0.11147 m | 0 N |
| crouch twin | rejected | 0.37573 m | 0.41021 m | 0 N |

The adapted rejection reasons are `reference_endpoint_tracking_error` and
`reference_path_tracking_error`. Because both executions did not accept, no trajectory-conditioned
obstacle support or USD scene was authored. This is the intended fail-closed behavior: CPU-valid
reference editing does not imply controller-deliverable motion.

The stored `motion_pair.json` already carries the route-relative basis and candidate overhead,
lateral, floor, and oblique normals. Those normals are representational only. None becomes feasible
support without an accepted paired execution and measured finite-face separation.

Evidence SHA-256: motion pair
`f93d3f3561b88d1b05765e1e12bd9dd87ef642edb6f502e689c891daab2c0cc1`; approved manifest
`ebcb944b33cd01edce171485f5a20b6426fa59b922badf516378f0679789ee0a`; physics run
`068c04c3df67a7584f84a037357362d274aaaaf55b98bdf5808d808eebedc193`.
