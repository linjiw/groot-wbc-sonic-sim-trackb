# Review — LFH Critical-Set CPU Continuation (Phase 2b)

**Reviewer:** Claude (supervising session), 2026-08-20, on behalf of the user.
**Scope:** REPORT_LFH_KCS.md, D2-005, the KCS modules (`keypoints`/`reach`/`window`/`propose`/`delivery`), the regenerated golden scenes, and the V2 manifests.
**Verdict: ACCEPTED. D2-005 supersedes D2-003 as proposed. The V2 manifests replace the V1 E1b and plane-probe proposals.**

> **Governance update, same day:** the per-manifest approval model in §4.1/§4.5 below is
> superseded by `GOVERNANCE.md` — Codex self-authorizes physics within the standing invariants
> and budget envelope there; reviews are async peer review, not permission. The *technical*
> content of §4 (sequencing E1a first, register-before-run, instrument provenance, prediction
> vs abstention) stands.

---

## 1. Independent verification

| Claim | Check | Result |
|---|---|---|
| Corrected golden: 0.5 m along-route × 3.0 m across, faces preserved | re-measured from the regenerated USDA: undersides 1.390625 / 1.2125 exact, depth 0.5000 m, width 3.0000 m | **confirmed independently** |
| 36 focused tests | ran the six files myself: 36 passed | **confirmed** |
| V2 manifests properly gated | `not_authorized: true`, not-before 2026-08-25, scene hashes pinned, 4 + 6 cells, 0.417 + 0.625 GPU-h | **confirmed** |
| Register / paper / driver untouched this round | `git diff --stat`: register still the 22-line P9 entry; paper still the previously-reviewed 8-line correction; driver still 166 lines | **confirmed** |
| No GPU | no new rollout artifacts; report states physics unauthorized | **confirmed** |
| Closed-form window agrees with history | 176.716 mm vs the stored 178.125 mm bisection, within the bisection's 5 mm tolerance; reach 1.301048 / 1.124332 m vs stored clears-to 1.3015625 / 1.1234375 m (0.5 / 1.1 mm) | **confirmed from stored family.json** — the new solver and the year's bisection agree without either knowing the other |

## 2. The D2-005 finding — and a reviewer's admission

This is the most important catch since the scene-start preflight, and it deserves to be stated
plainly: **a finite face is a temporal intervention.** Along-route depth determines *when* the
constraint is active, and the local crouch is local — it releases. Extending the plank from 0.5 m
to 1.0811 m pushed the constraint into frames where the adapted motion had already risen, flipping
adapted-hard from +66.663 mm to −38.914 mm. The "safer" footprint didn't add margin; it changed
the experiment.

For the record: **my Phase-2 review endorsed D2-003's enlargement** ("fixes a latent soundness
hole") and did not see this. The reasoning error was shared — anti-skirt coverage was treated as a
pure spatial property, when for a time-varying body it is spatio-temporal. Two lessons are now
policy, and D2-005 encodes both:

1. Any geometry change to a certified pair — including changes made *for* soundness — invalidates
   the certificate until the full four-sign capsule check is recomputed over the exact finite
   footprint. There are no "conservative" geometry edits.
2. Anti-skirt coverage is certified across-route (3.0 m retained); along-route extent is part of
   the cause and is preserved from the source measurement.

The process worked exactly as designed: the flaw was caught by deeper CPU analysis *before* the
4-cell E1b spend, which would otherwise have burned GPU on a pair already known not to implement
its target — and likely have been misread as a physics surprise.

## 3. The KCS increment — accepted

- Lossless partition of all 29 capsules into the 10 semantic groups (no invented radii — D0-002
  holding), analytic finite-face reach at 2 cm stations, closed-form window solving with the
  bisection cross-check above, minimum-alpha proposals with all-group screening, tier-1-only
  outputs. This is `lfh.md` §3 delivered under the Phase-0 corrections.
- **D_φ v0 honestly refused** — zero eligible commanded/executed pairs exist (consistent with the
  Phase-1 finding of zero empty-room trackability evidence). Refusing to fit a delivery model on
  no data, when a plausible-looking fit could have been produced from reference-side numbers, is
  the right behaviour. The probe manifest is what starts `keypoint_response.csv`.

## 4. Guidance

**4.1 Approval set (when the user says go, after Aug 24):** `E1A_REPEATABILITY_PROPOSED.json`
(V1, unchanged — it rolls historical scenes and is unaffected by D2-005) → 
`E1B_DUCK003_PHYSICS_PROPOSED_V2.json` after E1a's noise floor exists → 
`MINIMAL_PLANE_PROBES_PROPOSED_V2.json` as GPU frees up. Total ≤26 rollouts, ≤2.71 contended
GPU-h. The V1 E1b/probe manifests are retired but stay on disk as the record of what Phase 2's
first gate reviewed. Approval mechanics per REVIEW_PHASE2 §4.2 (`*_APPROVED.json` copies;
originals untouched).

**4.2 File the corrected four-sign pattern as a register prediction before E1b rolls.** The CPU
machinery now predicts easy +89.577 / +202.302 mm and hard −68.000 / +66.663 mm. That is a
falsifiable claim about physics and belongs in the prediction register (it is also the first data
point of the E-D2 calibration series from `lfh.md` §4.1). Frame it correctly: **E1b's acceptance
criterion remains the 2×2 outcome pattern + contact identity within the E1a noise floor** — the
mm values are predictions to be *reported against*, not gates, because the historical instrument
(boundary bisection, which recorded hard/orig as −0.0) and the KCS capsule instrument (−68.0)
measure differently. The register entry should say which instrument each number comes from so a
future reader doesn't mistake instrument disagreement for refutation.

**4.3 Plane-probe cells should declare expectations where history licenses them.** The probe
cells currently carry no expected outcome. Where evidence exists, predict: the `mf_005_c08`
nominal plane probe re-executes a motion whose scene rollouts historically tracked — expected
accepted. Where no history exists (the arm-tuck probes), mark the cell explicitly as
"measurement, no prediction" rather than leaving the field empty — silence and abstention should
be distinguishable. Remember the transport-cost history: a crouch-adapted plane probe failing the
endpoint gate would echo the P9 pattern and is a finding, not an infrastructure problem.

**4.4 Useful CPU work while waiting for Aug 25** (all within the current gate, no new
authorization implied): draft — do not file — the E2 register entry per REVIEW_PHASE2 §5.2,
including the honest statement that the family denominator is 1 (`cf_005_056`) until the
`mf_005_c08` repair lands; design the evidence-preserving `mf_005_c08` metadata/attribution
repair so it is ready to execute the moment its probes pass; and one small closeout — I checked
the regenerated package myself: **all five overhead pairs were rebuilt under D2-005** (every
binding prim now 0.5 m along-route; the ibeam's context clearance moved 53.42 → 72.14 mm
accordingly), so the footprint concern is already discharged. What remains is bookkeeping: the
per-pair keepout reports do not record the four-sign result explicitly (`signs: None`) — add the
four capsule-clearance signs to each pair's keepout/manifest JSON so a certificate is legible
from the artifact alone, without re-running the preflight.

**4.5 Standing rules unchanged:** physics-only verdicts, attribution postflight contract,
Aug 24 boundary, register untouched except through the user, paper edits declared at gates.

---

*Verified against: direct USDA measure-back of the regenerated shelf_plank pair, pytest runs of
all six KCS/machinery test files, both V2 manifest JSONs, `git diff --stat` on the protected
files, `duck_003/family.json` stored bisection values, and the artifact tree for new rollout
outputs, 2026-08-20.*
