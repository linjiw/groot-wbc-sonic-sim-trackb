# Native temporal audit on the fixed 41002 beam

Registered 2026-09-06 before evaluating the new interpolants. Use all six original
empty-scene repeatability recordings and all three d040 extension recordings. Keep
the fixed midpoint beam and all 81 original placement offsets. No beam selection,
physics, fitting or outcome-dependent sampling occurs in this audit.

Compare the 29-capsule proxy, cached native primitive subset (27 shapes), and native
outer union (45 shapes, including enclosing hand/wrist mesh spheres), each at 50 Hz
and 200 Hz. Read the hash-pinned body-local geometry from the prior native audit;
this is not a new claim about official URDF convex decomposition or cooked PhysX
geometry. Time each full 81-offset panel on the same CPU, retaining preprocessing
and query time separately. These are single-panel timings under shared-machine load.

Interpolate each recorded rigid body with linear translation and shortest-arc SLERP
of normalized wxyz quaternions. Split each original 20 ms interval into four 5 ms
intervals. Normalize only near-unit quaternions (norm error <=1e-3). Evaluate only
the first recorded episode; never interpolate across resets. The interpolant is a
declared body-motion model, not a reconstruction of unobserved joint dynamics.

For each interval and capsule axis, bound its distance to the temporally nearest
endpoint by `||p1-p0||/2 + 2 R sin(theta/4)`, where R is the largest body-local axis
endpoint norm and theta is the body's full geodesic rotation angle. Subtract this
bound and an assumed 1e-8 m numerical allowance from the smaller endpoint clearance.
A positive lower bound excludes tunneling for this interpolant. A negative lower
bound is unresolved, not proof of collision. Interference uses a primitive witness,
never an outer mesh-sphere intersection. Placement offsets remain a finite set.

P1: all three d055 trajectories retain >=10 mm native-outer clearance at all 81
placements at 200 Hz. P2: their conservative interval lower bounds also retain the
10 mm margin at all placements. P3: each upright recording retains a <=-10 mm
native-primitive witness at all placements. Report each predicate and every d040
margin without requiring a favorable ordinal result. The 50 Hz comparison is checked
against the prior native audit to 1 micrometre (normalization may change last bits).

Bound the CPU audit at 300 seconds and zero GPU use. Store all per-offset minima and
all interval lower bounds in compressed arrays with hashes. This does not certify
continuous placement uncertainty, native cooked contacts, actual between-step motion,
the complete 384-placement batch, or full SO(3) obstacle-pose volume.
