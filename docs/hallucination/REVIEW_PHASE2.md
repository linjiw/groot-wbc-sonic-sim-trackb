# Phase-2 Review — CPU Machinery, Golden Reproduction, and Proposed Physics

**Reviewer:** Claude (supervising session), 2026-08-20, on behalf of the user.
**Scope:** REPORT_E1.md, `e1_cpu/summary.json`, the `hallucinated_variants_v1_e1` scene package, the three proposed manifests, the D2 deltas, probe candidates, and the Phase-1.1 closeout, at the gate defined in `docs/lfh/implementation.md`.
**Verdict: ACCEPTED. The CPU half of E1 is green for `duck_003`. Two decisions are teed up for the user (§4); my recommendations are given. No physics is authorized by this review.**

---

## 1. Independent verification

| Claim | Check | Result |
|---|---|---|
| Golden faces 1.390625 / 1.212500 m within 0.5 mm | **I re-measured the undersides directly from the generated USDA** (Cube size × scale, translate): easy = 1.390625, hard = 1.2125, exact | **confirmed independently** |
| Keep-out numbers | `ibeam` keepout.json: hard-cell min context clearance 53.4205 mm; binding-face offsets ≤ 2.2e-13 mm; station offsets ≤ 1.1e-13 mm | **confirmed** |
| Fresh extraction matches committed fingerprint | `summary.json`: `fresh_extraction_matches: true`, sha256:0daaf895… | confirmed (as reported) |
| 23 Phase-2 tests + 10 coverage tests pass | ran them myself with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` | **confirmed** |
| No GPU spent | no `success_manifest.json` in the artifact tree newer than 2026-08-20 00:00; `physics_executed: false` | **confirmed** |
| Register/prereg untouched; pre-existing diffs preserved | `prediction_register.md` still the 22-line P9 entry; `run_deployable_family.py` still 166 lines; `analysis_prereg.md` unmodified | **confirmed** |
| Phase-1.1 closeouts | canonical `sweepcf_release/index` regenerated (mf_005_c08 present, additive columns, dated backup `index_pre_lfh_2026-08-20/`); progress board now says 2 sources / 3 variants, zero stale claims; COVERAGE.md reconciles the 21 tests (10+5+6) and documents the pandas/numpy environment breakage | **all three discharged** |
| Manifests are proposals, not authorizations | `not_authorized: true`, `default_not_before: 2026-08-25`, every cell hash-pinned with `scene_start_xyz_expected`, serial behind the 9000 MiB gate, 1800 s timeout, infrastructure-vs-science split, ego-only cameras, overhead pass forbidden, 1e-5 m attribution floor | **confirmed** — Phase-0 C3/C4/C5 are all institutionalized in the schema, not just prose |

## 2. Judgement calls that were made well

- **The keep-out validator earned its keep before any GPU.** The first `ibeam` design was refused
  for a 6.6 mm web intrusion into the hard corridor, and the fix moved geometry rather than the
  gate. That is the validator doing precisely what Phase 0 said it must — refusing candidates
  cheaply so physics only sees configurations that can work.
- **D2-002 is an honest self-correction of Phase 0.** RECON's R1 claimed `mf_005_c08` had a
  nominal plane trajectory; an exhaustive log-matching audit found **no** `scene=plane` execution
  at all — every apparent probe ran in a shelf scene. Codex corrected its own accepted recon
  rather than quietly building on it, widened the refusal (`extract_spec` now reports every
  missing prerequisite), and downgraded "one reroll completes the evidence" to "two probes plus a
  later evidence-preserving metadata repair." This is the register culture applied to the
  project's own documents. Full E1b is correctly *not* claimed green.
- **D2-003 fixes a latent soundness hole in the golden itself.** The historical 0.5 m shelf depth
  never certified anti-skirt coverage; the regenerated golden derives 1.0811 m from the union of
  both executed capsule spans while preserving the face coordinates exactly. The right call:
  reproduce the *cause*, not a known-insufficient artifact.
- **D2-001 and D2-004** (per-motion crossing intervals; driver-addressable package-level scene
  names) are the kind of reality-over-plan deltas the protocol exists for.
- The probe cohort is target-driven and hash-pinned end to end, with the 0.70 delivery prior
  applied conservatively (42.0 / 41.6 mm predicted windows above the 30 mm spend gate), and
  nonempty-scene success used for prioritization only — never promoted to plane evidence.

## 3. Notes and one boundary flag

- **`docs/paper/sweepcf_draft.md` was edited without being listed at the gate.** The edit itself
  is a factual repair I endorse — the claim table and regime table now state 2 independent
  causal families / 3 verified variants (lateral 1 → 0), matching the audited count, and the
  pre-registered analysis file is untouched. But the paper draft is user-owned surface: any
  future edit to `docs/paper/*` must be named in the gate summary, not left to be discovered in
  `git status`. Same for `docs/progress/` — the board refresh was authorized (Phase 1.1), so
  that one is fine.
- The `autoresearch/iterate-260820-1148/results.tsv` keep/discard trail (11 rows) is a good
  practice; keep producing it.
- Reproduction commands reference `./.venv_sim/bin/python`; the tests also pass under system
  `python3`. Fine as-is.

## 4. Decisions for the user (with recommendations)

**4.1 `door_lintel` side jambs — recommend ALLOW.** The jambs pass keep-out with 924.18 mm
minimum clearance — 17× the corridor margin. The entire point of the C \ B certificate is to
license exactly this kind of context geometry on evidence rather than taboo. Condition: jambs
remain excluded from E1b's physics cells (as proposed) until one jambed variant earns its own
manifest line in E2, where `secondary_contact` refusal protects us if the certificate missed
something.

**4.2 The three physics manifests — recommend APPROVE ALL THREE for after Aug 24,** total ≤26
rollouts, ≤2.71 contended GPU-hours, in this order:
1. **E1a repeatability (16 cells)** first — everything downstream is judged against its noise
   floor, and its stop condition (any source outcome flip halts all LFH work) is the cheapest
   possible insurance on the whole corpus. Note this doubles as a corpus-repeatability check the
   claim-5 paper benefits from regardless.
2. **E1b duck_003 physics (4 cells)** after the noise floor exists.
3. **Minimal plane probes (≤6 cells)** — independent of 1–2 in content; runs whenever GPU is
   free after E1a starts reporting. The skip rule (adapted probe skipped if its nominal is
   rejected) makes 6 a ceiling, not a quota.

Approval mechanics when the user says go: flip each manifest's authorization in a copy named
`*_APPROVED.json` (leave the `_PROPOSED` originals untouched as the record of what was reviewed),
and file the E1a prediction-register entry — expected outcomes are already declared per cell
(`expected_source_outcome`) — **before** the first rollout, per Invariant 10.

## 5. Guidance for what follows the physics

1. **If E1a flips any source outcome: full stop, escalate.** That is a corpus-level finding that
   outranks LFH; it goes to the user before any interpretation.
2. **E1b physics green + E1a floor** ⇒ the golden loop is closed and Phase 3 (E2 variant
   transfer) may be *drafted*: register entry with thresholds first, then the 24-cell manifest
   for approval. Default K=3 archetypes × the one extractable family — note that with
   `mf_005_c08` refused as a source, E2's "2 families" default from the design plan shrinks to
   `cf_005_056` until the metadata repair lands; say so in the register entry rather than
   silently halving the denominator.
3. **Plane probes green for `mf_005_c08`** ⇒ the evidence-preserving metadata/attribution repair
   (CPU) is the next gate item, and only after it may an `mf_005_c08` spec be extracted.
4. **Probe results feed `keypoint_response.csv`** (the D_φ data contract from `lfh.md` §4.1) —
   record commanded-vs-executed per keypoint from the first probe onward, so the delivery model's
   active-learning loop starts with these very rollouts.
5. The attribution postflight in the manifests (1e-5 m floor, `unattributed` refusal,
   `secondary_contact` on non-binding contact) is now the standing contract — future phases cite
   it, they don't renegotiate it.

---

*Verified against: direct USDA measure-back of `cs_duck_003__shelf_plank__s00000017__{easy,hard}.usda`,
`cs_duck_003__ibeam__s00000017.keepout.json`, `e1_cpu/summary.json`, the three manifest JSONs,
`lfh_probe_candidates/candidates.json`, pytest runs, `git diff` on register/driver/paper/prereg,
the regenerated `sweepcf_release/index`, the refreshed progress board, and a timestamp sweep for
new success manifests, 2026-08-20.*
