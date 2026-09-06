# Cached native geometry audit of the frozen development beam

Registered 2026-09-06 before clearance evaluation. GPU contention prevents the registered
intervention from launching. This CPU audit addresses collision-model agreement using
the cached imported robot asset and the six already recorded empty-scene 41002 rollouts;
it cannot substitute for obstacle-present execution.

Freeze beam_021 at the already selected median height, including centre, yaw, dimensions
and the same 81 offsets used by the original finite-beam teacher: dx/dy ±0.02 m,
dz ±0.01 m, dyaw ±0.02 rad, each including zero. Use the first motion episode in each
of the six original-clock seed 7901–7903 recordings. Preserve world coordinates.

Extract enabled collision capsules and spheres from the cached imported USD, including
instance proxies. Transform each primitive into its owning rigid body's local frame;
reject nonuniform primitive scale or unsupported shape types. For each mesh below a
collision-marked transform, construct a body-local sphere containing every transformed
vertex. That sphere also encloses the mesh triangles and their convex hull. Keep its
role explicitly **outer approximation**, never an interference witness. Record layer
hashes, local primitive parameters, mesh vertex counts/digests and enclosing radii.

For target d055, independently compute capsule–beam distances for all native primitives
and enclosing mesh spheres at every recorded frame/offset. For upright, use only native
capsule/sphere primitives as an inner subset of the cached collision union. Clearing the
outer union supports cached-geometry target clearance; intersecting its native primitive
subset supplies a cached-geometry interference witness. The finite thresholds remain
target >=10 mm and upright <=-10 mm. No beam adjustment or failure replacement.

Prediction: all three targets and all three upright recordings retain their respective
margin across all 81 offsets. A failure names the recording, offset and shape, and does
not trigger margin relaxation. Cross-reference the result with the old proxy evidence.
Use CPU only; cap the complete audit at 180 seconds, record wall time and distance-query
counts. A query is one whole-motion clearance for one geometry role and beam placement.

Limits: cached USD is not proof of the currently cooked PhysX shapes, collision filters,
contact/rest offsets or contact forces. Mesh cooking inflation is not enclosed by this
construction without an additional bound. Recorded time samples do not establish
continuous-time clearance. Source-level independence remains one selected carrier.
This audit supplies no new physical rollout and does not change the pending intervention.
