# Phase-0 Review — LFH Reconnaissance and Design Deltas

**Reviewer:** Claude (supervising session), 2026-08-20, on behalf of the user.
**Scope:** `RECON.md` and `DESIGN_DELTAS.md` at the Phase-0 gate defined in `docs/lfh/implementation.md`.
**Verdict: ACCEPTED — all 14 deltas adopted, with the conditions in §3. Phase 1 is authorized under §4.**

---

## 1. Independent verification of the recon's factual claims

I did not take RECON.md on faith; I re-measured its load-bearing claims against the artifacts.
Every one checked out:

| Claim | Check | Result |
|---|---|---|
| `duck_002`/`duck_003` share one causal source (`005`/`056`) | both `family.json` files name `005_…_s0.csv` nominal + `056_…ducks…_s0.csv` adapted | **confirmed** |
| `cf_005_056` window 0.178125 m | `duck_003/family.json` `window_m` | **confirmed, exact** |
| `mf_005_c08` peaks 1.30515 → 1.20743 m, hard shelf 1.25743 m | `matched/mf_005_c08.json` | **confirmed, exact** |
| Design plan's E1b golden numbers (duck_003: 1.3906 / 1.2125 m) | `duck_003/family.json` | **confirmed** — valid golden targets |
| Release views disagree | `episodes.csv` = 42 rows; `families.csv` marks both ducks `verified` and **omits `mf_005_c08`** | **confirmed** |
| Frozen split exists as parameters only | `gear_sonic/data/splits/scene_first_v1.json` present; no materialized scenes | **confirmed** |
| No code / register / scene changes in Phase 0 | `git status` — only new files under `docs/hallucination/` and `docs/lfh/` | **confirmed** — protocol respected |

Independently significant: RECON's family-identity conclusion (**2 independent causal families;
duck heights are variants**) converges exactly with the prior file-anchored audit
`docs/gate_audit_2026-08-19.md`, which RECON does not cite and apparently did not use. Two
independent derivations landing on the same count is the strongest confirmation either could get.

## 2. What is genuinely good here

- **D0-002 (semantic capsule groups over invented keypoint radii)** is the right call and better
  than the spec it corrects. `lfh.md` §1.1's ten fixed-radius keypoints were never going to be
  sound against a 29-capsule model with no head link; grouping existing capsules keeps tier 1
  honest (it can never certify what tier 2 would reject) and inherits tested geometry. Accepted
  as written.
- **D0-004 (refusing `mf_005_c08` as a delivery seed until a plane reroll exists)** is exactly the
  measurement discipline this corpus is built on: a shelf-scene rollout is not an empty-room
  probe, and substituting it would poison D_φ's training data at the root.
- **D0-006 (face extents derived from both swept bodies, not historical constants)** closes a real
  hole — `scene_route_check` proves the route meets the footprint, not that the constraint is
  non-skirtable. The plan assumed this; the repo never guaranteed it.
- **The Executive Verdict's honesty.** "Implementable, but not literally from the current plan"
  with four named corrections is what a Phase 0 is for.

## 3. Conditions attached to acceptance

**C1 — Cite and reconcile with the existing audit (RECON.md amendment, cheap).** Add references to
`docs/gate_audit_2026-08-19.md` and `docs/constraint_distance_findings.md`. The Phase-1 identity
repair (D0-005) must also cover the three counting defects already on record there: (a)
`report_dataset_counts.py` re-admits `mf_x003_c08` because `verdict()` never calls
`scene_route_check`; (b) `families.csv` omits `mf_005_c08`; (c) the progress board says 3.
**Acceptance test for the repair: the independent-family count that falls out is 2.** If your
repaired contract produces any other number, stop and escalate rather than shipping it.

**C2 — Name the sealed split (RECON.md amendment, required before Phase 1 claims).**
`gear_sonic/data/splits/` contains `scene_first_v1.json`, `scene_first_v2.json`, and
`review_cohort_v1.json`; RECON names only v1. Determine which file the pre-registration text
actually seals (check the register and `docs/g0_real_dataset_runbook.md`), record the answer and
its evidence in RECON.md, and treat only that file as the frozen artifact. "Read-only" applied to
the wrong version is not read-only.

**C3 — D0-008 contact attribution needs a stated resolution floor.** The capsule proxy has already
been measured running out of resolution at ~10 µm separations, and its conservative envelope leads
recorded contact by ~14 frames (`docs/constraint_distance_findings.md`). Adopt that document's
protocol wholesale: physics is the verdict, the proxy explains, disagreements are *reported* with
a stated ambiguity threshold — never reconciled by tuning, and `unattributed` is a refusal code,
not a judgement call.

**C4 — D0-004's plane reroll is approved in principle, gated in practice.** When you request it,
bring a manifest (cells, scene, motion artifact, expected cost — this is ~1–2 rollouts) and run it
through the hardened driver conventions in `scripts/research/run_deployable_family.py`: explicit
success markers (exit status is not authoritative), `scene_start_xyz` preflight against the source
family, 9000 MiB free-GPU margin under contention, 1800 s hang timeout, and infrastructure
failures recorded as unevaluable, never as rejected cells. This repo paid for each of those rules
separately; reuse them, do not rediscover them.

**C5 — The Aug 24 boundary is absolute.** No GPU work, however small (including C4's reroll and
E1a), runs before the claim-5 pre-registered analysis without the user's explicit line-item
approval. Default: all rollouts wait until after Aug 24.

**C6 — Do not touch the pending P9 adjudication.** The prediction register carries an
unadjudicated result (deployable-retiming rerun of 2026-08-19, see
`docs/handoff_2026-08-20_codex.md` §4.2). That is a separate workstream with its own owner. RECON
already says the register is off-limits; this makes explicit that *adjudicating* is also off-limits
from within LFH work.

**C7 — Expect your files to be swept into someone else's commits.** A parallel session commits in
this repo every few minutes with `git add -A`; `docs/hallucination/` is untracked and will be
absorbed under an unrelated message. Nothing is lost when that happens. Do not rewrite history to
fix attribution, and check `git log --oneline -3` before reasoning from "the tree is clean".

**Minor (no action blocking):** RECON's `157 USD-family files` vs the plan's `117 scenes` — the
artifact-computed number wins per D0-013; publish the reconciliation, don't average them. The
`drwx------`/`600` permissions on `docs/hallucination/` differ from repo norms; loosen to match
(`664`/`775`) so review tooling and Pages builds can read them.

## 4. Per-delta disposition

| Delta | Disposition |
|---|---|
| D0-001 module layout | Accept. Matches repo convention (library in `gear_sonic/dataset_generation/`, thin entry points in `scripts/research/`). |
| D0-002 semantic capsule groups | Accept as written — improvement over spec. |
| D0-003 family vs variant identity | Accept. Variants never advance the 24–30 gate. |
| D0-004 refuse `mf_005_c08` seed pending plane reroll | Accept, with C4 + C5. |
| D0-005 additive index-contract repair first | Accept, with C1 (must also fix the three recorded counting defects; must reproduce count = 2). |
| D0-006 swept-derived face extents + C \ B validator | Accept. |
| D0-007 material provenance `runtime_default` | Accept — never invent friction numbers. |
| D0-008 offline contact attribution | Accept, with C3. |
| D0-009 v1 operators = crouch + arm-tuck only | Accept. Step/stance edits are a user-level scope decision, exactly as `implementation.md` says. |
| D0-010 ego + CPU-render evidence sheets | Accept. Never schedule the blank overhead camera. |
| D0-011 generic binding-face handle + measured-back ≤0.5 mm | Accept. |
| D0-012 50 mm as configurable default, C \ B measured explicitly | Accept. |
| D0-013 artifact-computed inventory + reconciliation warnings | Accept. |
| D0-014 local tests now, CI as separate scope | Accept. |

## 5. Phase 1 authorization

Proceed with Phase 1 as scoped in RECON.md's recommendation, under these bounds:

1. **CPU-only, additive-only.** No scene authoring, no GPU, no register edits, no frozen-split
   reads beyond what C2 requires to identify the sealed file.
2. Deliverables: the identity-contract repair (with C1's acceptance test), joined
   trackability in the motions view, DCS with explicit `unknown` bins and separated
   independent-family / variant / motion counts, `targets.json`, and `COVERAGE.md`.
3. Run the existing render/report scripts afterward as the no-consumer-broken smoke test.
4. Stop at the Phase-1 gate with `COVERAGE.md` for review, per `implementation.md`.

---

*Review artifacts verified against: `/data/robotixx/groot-wbc-kimodo-m0/counterfactual/duck_00{2,3}/family.json`,
`/data/robotixx/groot-wbc-kimodo-m0/matched/mf_005_c08.json`,
`/data/robotixx/groot-wbc-kimodo-m0/sweepcf_release/index/{episodes,families}.csv`,
`gear_sonic/data/splits/`, and `git status` on the working tree, 2026-08-20.*
