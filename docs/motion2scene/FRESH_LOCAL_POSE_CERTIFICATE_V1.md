# Nonzero local six-dimensional pose domains for all 384 frozen outputs

Registered 2026-09-06 before measuring nominal slack or constructing these domains.
Use all 48 learned-pattern jobs and all eight requests per job in the original
fresh-source audit: three fitting seeds, eight sources, two events. Keep all 384
requests; do not select only high-slack outputs or tune a model. The 430xx sources
remain excluded from training and method selection. This is a contract audit of
already published frozen outputs, not new-source confirmation.

The question differs from certifying the entire original ±2 cm/±0.02 rad yaw domain:
does each accepted request have a nonzero local feasible region in R^3 × SO(3)?
Use the original static 29-capsule reference model and exact segment/box query.
Read nominal target and upright clearances from the saved zero-offset audit entry,
and independently reproduce them before constructing a domain. Define nominal slack
s = min(c_target - 0.01, -0.01 - c_upright) - 1e-8 m. Nonpositive s is unresolved.

Let R be the norm of the box half-extents. For positive s, set translation component
halfwidth a = min(0.02 m, s/(4 sqrt(3))) and world rotation-vector component halfwidth
b = min(0.02 rad, s/(4 sqrt(3) R)). Pose is t0 + delta_t and Exp(delta_rotvec) R0.
Every box point moves by at most sqrt(3)a + 2 R sin(sqrt(3)b/2) <= s/2. Subtract
this displacement and the numerical allowance from the nominal margin slack.
If the remaining bound is positive, the entire six-dimensional cube passes under
these assumptions. The exp chart is injective inside the pi-angle ball used here,
so positive chart volume implies positive local product/Haar measure. Report chart
volume (m^3 rad^3), not a computed Haar volume or the original uncertainty-set volume.

For every positive domain, additionally check all 64 six-dimensional corners and
eight deterministic interior points (RNG seed 6941). Use full-axis NumPy box queries
at the actual three-dimensional rotations, with the existing safe 20 mm clearance
cap/broad phase. These finite probes test implementation agreement; the displacement
inequality, not their finite count, supplies the conditional domain claim. Save all
scores and witnesses. Do not shrink a failed domain after observing these probes.

P1: all 384 nominal scores reproduce the saved audit within 1 micrometre. P2: all
384 have positive certified local chart volume. P3: all diagnostic probes retain
both 10 mm constraints. Report every failure and source-wise radius/volume ranges.
No numerical interval-arithmetic proof, native mesh/body result, temporal clearance,
tracking guarantee or complete original-domain certificate follows.

Limit this CPU experiment to two threads, 600 seconds, zero GPU use and at most
384 × 73 × 2 = 56,064 whole-motion queries. Preserve all outcomes and registrations;
do not modify the 430xx files, checkpoints, outputs or historical acceptance labels.
