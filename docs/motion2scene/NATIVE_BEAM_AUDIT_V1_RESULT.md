# Cached native beam audit: all six recorded margins retained

2026-09-06. Under the [registered audit](NATIVE_BEAM_AUDIT_V1.md), the frozen beam
retains target clearance and upright interference on all six original-clock 41002
recordings at all 81 tested placements. This is a CPU evaluation of existing
empty-scene trajectories, not an obstacle-present execution result.

| Recorded physics seed | Target outer-union minimum clearance | Upright primitive-subset worst clearance | Passing offsets, target / upright |
| --- | ---: | ---: | ---: |
| 7901 | +20.285 mm | -20.013 mm | 81/81 / 81/81 |
| 7902 | +21.771 mm | -12.175 mm | 81/81 / 81/81 |
| 7903 | +13.535 mm | -22.729 mm | 81/81 / 81/81 |

The frozen requirements are target >=+10 mm and upright <=-10 mm. Their smallest
remaining slacks are therefore **3.535 mm for target clearance** and **2.175 mm for
upright interference**. Every worst witness belongs to `torso_link`. The prediction
that all six cached-geometry margins survive passes; the narrow remaining slack
still motivates the registered native-contact experiment without moving the beam.

The cached USD supplies 26 capsules and one sphere. These native primitives form
the interference subset. The target-clearance union additionally includes 18
enclosing spheres for the wrist and hand meshes. Each sphere encloses all transformed
mesh vertices, and hence their triangles and convex hull. Those spheres are used only
for clearance; no interference claim relies on hitting an outer mesh approximation.

All geometry is expressed relative to its owning rigid-body frame, then transformed
using the recorded body poses. No achieved trajectory is recentered. Only the first
recorded episode is used. Capsule/sphere intersections and outer-union clearances use
the existing exact static capsule–box evaluator at recorded frames. There is one
selected source carrier and three repeated physics seeds, not six independent motions.

The audit spent **486 whole-motion queries**, **2.775 s wall time**, and **zero GPU
time**. Each query evaluates one geometry role for one recording and beam placement.
The geometry has 27 primitive shapes for the upright subset and 45 shapes for the
target outer union. No new controller rollout, teacher fitting, scene adjustment or
failure replacement occurred.

Evidence: [full per-offset results](evidence/native-beam-audit.json) and
[body-local geometry plus source-layer hashes](evidence/native-beam-geometry.json).
Original data and registration are under
`/home/linjiw/research-data/groot-wbc/m2s-native-beam-audit-v1/`.

Four focused tests pass: body-local transforms despite world translation; meshes under
collision-marked transforms retained only in the outer union; refusal of nonuniformly
scaled primitives; and enclosure of mesh vertices/convex combinations. Ruff and Black
checks pass on the new helper, driver and tests. Without the optional USD library path,
the native test module skips cleanly (the ordinary check ran four passage tests and
skipped this native module). They use the installed USD libraries
without starting Isaac Sim:

```bash
M2S_USD_LIB_DIR=/home/linjiw/isaaclab-install/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
M2S_PY_LIB_DIR=/home/linjiw/.local/share/uv/python/cpython-3.11.16-linux-x86_64-gnu/lib
PYTHONPATH=.:scripts/research:$M2S_USD_LIB_DIR \
LD_LIBRARY_PATH=$M2S_USD_LIB_DIR/bin:$M2S_PY_LIB_DIR \
.venv_research/bin/python -m pytest -q tests/dataset_generation/test_motion2scene_native_geometry.py
```

The cached asset is not a running-stage audit. Cooked PhysX shapes, active collision
filters, cooking inflation, contact/rest offsets and contact forces remain unverified.
Recorded frames and the finite placement set provide no continuous-time/uncertainty
certificate. Do not promote this result to physical passage or learned-generator yield.

The intervention was reattempted through the existing registered runner and yielded
at **5,288 MiB free GPU memory**, below its **9,000 MiB** gate. It still has **zero new
physics runs**. The beam, physics seeds, controller and execution criteria remain
unchanged. The next required action is the two native-contact controls, followed by
the 12 paired cells if the controls pass; no background process is waiting to launch.
