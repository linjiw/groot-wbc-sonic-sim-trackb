# E1 Report — Core Machinery and CPU Golden

**Status:** Phase-2 CPU gate complete on 2026-08-20. No physics rollout was
launched, no prediction-register entry was changed, and no frozen evaluation
artifact was modified or used for generation. E1a and the physics half of E1b
remain proposed, not authorized.

**Post-review correction:** LFH critical-set recomputation found that the
reviewed 1.0811 m plank reversed the adapted-hard capsule sign. D2-005 repairs
the temporal footprint and the regenerated CPU golden below supersedes that
artifact. See `REPORT_LFH_KCS.md`; the original E1b and probe manifests remain
as non-authorizing records and are superseded by unapproved V2 proposals.

## Result

The `duck_003` source now regenerates as a content-addressed `ConstraintSpec`,
and all five overhead archetypes pass the mandatory face, station, route,
extent, and single-cause CPU preflight. The library also contains three lateral
archetypes. The fixed `shelf_plank` golden realizes easy/hard undersides at
**1.390625 / 1.212500 m**; maximum face and station error are below
`5e-5 mm`, comfortably inside the 0.5 mm gate.

Fresh extraction exactly matches the committed spec fingerprint
`sha256:cf761303c1461e49c681e72de95a614fe741529561c3b067cee6ee28ced38d75`.
It records distinct executed crossing frames (nominal 125, edit 95), the
measured station `(2.6739, -0.2632)`, and hashed `scene=plane` trajectories.
The finite footprint is the historical 0.5 m along-route × 3.0 m across-route.
Across-route coverage prevents a lateral skirt; along-route depth defines the
local crouch's active interval and may not be enlarged independently (D2-005).

| CPU check | Result |
|---|---:|
| Archetype inventory | 5 overhead / 3 lateral-gap |
| Overhead easy/hard pairs certified | 5 / 5 |
| `shelf_plank` face/station maximum error | <0.00005 mm |
| Closest non-binding context | `ibeam`, 53.42 mm |
| Generated 2×2 capsule signs (easy orig/edit; hard orig/edit) | +89.58 / +202.30 / −68.00 / +66.66 mm |
| Six-file KCS + Phase-2 suite | 36 passed |
| Physics verdicts generated | 0 |

The first thin-flange `ibeam` attempt was correctly refused: its web entered
the hard-cell corridor by 6.6 mm. Moving the web behind a 140 mm binding flange
produced 53.42 mm minimum context clearance and was retained. Geometry was
changed to satisfy the certificate; no verdict was invented.

The machine-readable evidence is in `e1_cpu/summary.json`; generated scenes,
pair manifests, and keep-out reports live in the separate
`gear_sonic/data/assets/scenes/hallucinated_variants_v1_e1/` package. Unique
package-level filenames make every scene addressable by the existing rollout
driver without modifying any frozen package.

## Refusals and Open Evidence

`mf_005_c08` remains a valid historical family but is **not** an extractable LFH
source. The extractor now reports all gaps instead of failing on the first one:
no explicit axis, no nominal/adapted motion identities, no matching nominal or
adapted plane trajectory/log, and no contact-attribution artifact. An exhaustive
exact-motion audit found that the apparent modebank pair ran in
`cf_005_056_easy`; all local executions used shelf scenes. This corrects the
earlier claim that a matching nominal plane rollout existed (D2-002).

Plane probes alone will not fabricate the missing metadata or attribution. If
they pass, a later CPU repair must preserve and cite the historical evidence
before an `mf_005_c08` spec or golden can be admitted. Consequently, the CPU
half of E1b is green for `duck_003` only; full E1b is not green.

The `door_lintel` side jambs pass the duck keep-out check (minimum 924.18 mm)
but remain optional and are excluded from every proposed GPU cell. Whether
keep-out-passing jambs are acceptable is a user decision at this gate.

## Proposed Physics Requests

Every manifest is hash-pinned, defaults to no earlier than Aug 25, requires
explicit user approval, runs serially behind 9000 MiB free-GPU capacity, uses an
1800 s timeout, and requires `SONIC_EVAL_SUCCESS`, `success_manifest.json`, a
trajectory, and 50 Hz recording. Infrastructure failure is never a physics
rejection. The blank overhead camera is forbidden.

| Manifest | Cells | Contended ceiling | Purpose |
|---|---:|---:|---|
| `manifests/E1A_REPEATABILITY_PROPOSED.json` | 16 | 1.667 GPU-h | 2 families × 4 cells × 2 fresh runtime seeds; stop all LFH work on any source outcome flip |
| `manifests/E1B_DUCK003_PHYSICS_PROPOSED_V2.json` | 4 | 0.417 GPU-h | Corrected finite-footprint golden; supersedes the stale reviewed manifest and requires new review |
| `manifests/MINIMAL_PLANE_PROBES_PROPOSED_V2.json` | ≤6 | ≤0.625 GPU-h | Adds the required `keypoint_response.csv` postflight contract; requires new review |

The minimal probe cohort is target-driven, not a broad screen. Motion 089-left
has the strongest screened one-sided window (60.004 mm); motion 092-right has a
59.452 mm window plus two accepted nonempty-scene executions. Their conservative
0.70 delivery prior leaves 42.003 and 41.616 mm respectively, above the 30 mm
CPU spend gate. Both nominal and edited artifacts are exact and content-addressed
under `/data/robotixx/groot-wbc-kimodo-m0/lfh_probe_candidates/`. Nonempty-scene
success is only a prioritization signal; it is never promoted to plane evidence.
An adapted probe is skipped if its paired nominal is rejected, so six is a cost
ceiling rather than a quota.

The physics postflight is specified to keep contact attribution explanatory:
physics supplies the verdict; the capsule proxy must use the documented 10 µm
resolution floor, report its timing disagreement, refuse ambiguity as
`unattributed`, and treat contact with any non-binding authored primitive as
`secondary_contact`.

## Reproduction

```bash
./.venv_sim/bin/python scripts/research/hallucination/run_e1_cpu.py
./.venv_sim/bin/python scripts/research/hallucination/prepare_probe_candidates.py
./.venv_sim/bin/python scripts/research/hallucination/build_phase2_manifests.py \
  --only e1b --only probes

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv_sim/bin/python -m pytest \
  tests/dataset_generation/test_hallucination_keypoint_window.py \
  tests/dataset_generation/test_hallucination_delivery.py \
  tests/dataset_generation/test_hallucination_propose.py \
  tests/dataset_generation/test_hallucination_spec_and_stage.py \
  tests/dataset_generation/test_hallucination_archetypes.py \
  tests/dataset_generation/test_hallucination_keepout.py -q
```

The tests additionally cover the 29-capsule semantic partition, analytic finite-
face support, exact golden binding attribution, extended-plank refusal, monotone
delivery fitting/refusal, and the all-keypoint proposer screen. Phase 2 stops
here for renewed review; no Phase-3 work is started.
