# Phase-1 Review — Coverage Baseline and Index-Contract Repair

**Reviewer:** Claude (supervising session), 2026-08-20, on behalf of the user.
**Scope:** COVERAGE.md, `coverage/` artifacts, `sweepcf_coverage.py`, the index-builder repairs, and the Phase-1 additions to DESIGN_DELTAS.md/RECON.md, at the gate defined in `docs/lfh/implementation.md`.
**Verdict: ACCEPTED — with three closeout tasks (§3) to finish before Phase 2 work starts. Phase 2 CPU machinery is authorized under §4.**

---

## 1. Independent verification

Re-measured, not taken on faith:

| Claim | Check | Result |
|---|---|---|
| C1 acceptance test: exactly 2 independent families | `summary.json` `independent_verified_families: 2`; families view maps `duck_002`/`duck_003` → source `cf_005_056`, `mf_005_c08` standalone | **confirmed** — the fixed acceptance test passes |
| 68 episodes / 60 graded; 3/168 bins; 165 targets | `summary.json` | **confirmed, exact** |
| 2,000 mm station mismatch on two variants | `summary.json` `binding_station_mismatch` = `mf_x002_c08`, `mf_x003_c08`; D1-001 | **confirmed** — see §2, this is a finding, not just bookkeeping |
| C2 (sealed split named) | RECON now identifies `scene_first_v1.json` as primary **with SHA fingerprint**, `scene_first_v2.json` as the pre-registered contingency vintage (own fingerprint, activated only by the ambiguous-zone rule), `review_cohort_v1.json` as the human-review cohort, corroborated by the analysis prereg | **discharged, thoroughly** |
| Pre-existing dirty files preserved | `git diff --stat` on `prediction_register.md` (22 lines) and `run_deployable_family.py` (166 lines) — byte-identical to their pre-Phase-1 state | **confirmed** — register untouched (C6 respected) |
| New tests pass | `test_sweepcf_coverage.py`: 10 passed | **confirmed** (but see §3.3 on the "21" count) |
| `report_dataset_counts.py` repaired | diff shows `verdict()` path now imports `check_route_meets_obstacle` **and** `binding_station_offset_mm` | **confirmed** — C1(a) discharged, and strengthened beyond what C1 asked |
| Trackability not inflated | 45 nonempty-scene successes left unpromoted; fully-gated LFH candidates = 0 | **confirmed and correct** — the conservative reading |

## 2. What is genuinely good here

- **D1-001 is the best finding of the phase.** C1 asked for the route check; Codex discovered the
  route check is *insufficient* — the 3 m shelf footprint still intersects a route displaced 2.0 m
  from its manifest station — and replaced it with route-intersection ∧ station-offset ≤ 0.5 mm.
  Note the convergence: this is now the **third independent detection of the same defect class**
  (the P9 void attempt's `--scene-start` mismatch was −2.0 m along the route; the gate audit read
  the same cells as "stopping short"; Phase 1 measured the displacement as exactly 2000.0 mm).
  Three instruments, one mechanism. The 0.5 mm station-offset check must become a standing
  preflight for every generated scene from Phase 2 onward.
- **Refusing to promote nonempty-scene successes to empty-room trackability** is exactly Invariant
  6 applied where it costs something: it leaves the pipeline with **zero** fully-gated candidates
  today rather than a comfortable, wrong number.
- **Unknown bins reported as unknowns** (35–47% on the constraint fields) instead of being
  reconstructed from directory names. The wall families' lateral coordinates staying "unknown"
  until measured is the right kind of empty.
- The report regenerates from one CLI with `--expect-independent-families 2` — the acceptance test
  is wired into the tool, not the prose.

## 3. Closeout tasks (Phase 1.1 — CPU-only, finish before Phase 2 code)

**3.1 Repair the canonical release index.** The repaired index exists only under
`docs/hallucination/coverage/index/`. The canonical artifact at
`/data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/index/families.csv` **still omits
`mf_005_c08` and still carries the pre-repair labels** — the known-wrong numbers remain at the
address every existing consumer reads. Not overwriting a release without approval was the right
instinct; here is the approval: regenerate the release index in place with the repaired builders
(additive columns, same location), keep a dated backup of the old index beside it
(`index_pre_lfh_2026-08-20/`), and rerun the legacy-consumer smoke tests against the regenerated
artifact. This discharges C1(b) at the canonical location.

**3.2 Refresh the progress board.** `docs/progress/sweepcf_board.html` (stale, 2026-08-19) still
says **"Three independent families."** Rerun `docs/progress/refresh_board.py` (or correct the
claim at its source if the board is hand-fed) so the board states 2 independent families / 3
verified variants. This discharges C1(c).

**3.3 Reconcile the test count.** The report claims "21 focused tests"; I can confirm 10 in
`test_sweepcf_coverage.py`. List in COVERAGE.md (or a delta) the exact files and invocation that
constitute the 21, so the number is reproducible. Environmental note, not Codex's defect: a
one-shot `pytest tests/dataset_generation/` is broken on this machine by a pre-existing
pandas/numpy incompatibility (numpy 1.21.5 vs pandas needing ≥1.22.4) that kills collection for
the whole directory — run test files in explicit lists until the environment is fixed, and say so
wherever a "full suite" claim would otherwise be implied.

Minor, no action required: `duck_000` maps to source `unknown` while `duck_001` maps to
`cf_005_056` — if that asymmetry has a reason (missing manifest field, different provenance),
one sentence in COVERAGE.md's unknown-bin section would pre-empt the question.

## 4. Phase 2 authorization and guidance

Phase 2 (core machinery + golden reproduction) is authorized **for its CPU portion** once §3 is
closed: ConstraintSpec, `extract_spec`, the archetype library (≥5 overhead, ≥2 lateral-gap),
`instantiate`, the keep-out validator, and their unit tests. Constraints:

1. **Golden targets are fixed and now doubly verified:** `shelf_plank` re-instantiation of
   `duck_003` must reproduce underside 1.3906 / 1.2125 m to ≤0.5 mm, measured back from the
   composed stage. E1b's CPU half (authoring + measure-back) can be built and gated green now;
   its physics half waits.
2. **The D1-001 station-offset check (≤0.5 mm) joins the keep-out validator** as a mandatory
   preflight on every generated scene — manifest station vs loaded constraint, measured, never
   assumed from authoring parameters.
3. **All physics stays gated.** E1a needs its redesigned manifest (RECON flagged the `mf_005_c08`
   pair as incomplete); the `mf_005_c08` empty-room adapted reroll (Phase-0 C4) and any E1a
   rollouts are **post-Aug 24 by default** (C5) and each requires a written manifest — cells,
   scene, motion artifact, GPU cost, stop conditions — for explicit user approval. Reuse the
   hardened-driver conventions (success markers, scene-start preflight, contention margin, hang
   timeout, infrastructure-vs-rejection split).
4. **The looming bottleneck, named now so it surprises nobody:** fully-gated LFH candidates = 0
   because *no* motion has strict empty-room trackability evidence. Workstream B cannot start
   without empty-room probe rollouts, which are GPU. Phase 2 should therefore end by proposing a
   **minimal empty-room probe manifest** (likely: the `mf_005_c08` adapted reroll + the handful of
   motions the 165-bin target ranking actually needs first), sized in rollouts, for the user to
   approve after Aug 24 — rather than discovering this dependency mid-Phase-4.
5. Contact attribution per Phase-0 C3 (physics wins, stated resolution floor, `unattributed` is a
   refusal code); material provenance stays `runtime_default` (D0-007); the blank overhead camera
   is never scheduled (D0-010).

Stop at the Phase-2 gate with REPORT_E1.md's CPU half (authoring golden test + validator
property tests) and the proposed physics manifests, per `implementation.md`.

---

*Verified against: `docs/hallucination/coverage/summary.json` and `coverage/index/*.csv`,
`/data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/index/families.csv`, `git diff` on the
pre-existing dirty files and `report_dataset_counts.py`, `pytest` runs of the new tests,
`docs/progress/sweepcf_board.html`, and RECON.md's split-fingerprint section, 2026-08-20.*
