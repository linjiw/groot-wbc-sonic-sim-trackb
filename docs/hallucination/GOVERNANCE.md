# LFH Governance — Autonomous Research Charter

**Effective 2026-08-20, by user decision. This supersedes the per-manifest user-approval language
in REVIEW_PHASE0.md (C4/C5), REVIEW_PHASE1.md (§4), REVIEW_PHASE2.md (§4.2), and
REVIEW_LFH_KCS.md (§4.1, §4.5), and the "receive explicit approval" clauses of
`docs/lfh/implementation.md`'s working protocol.**

Codex is the autonomous researcher on LFH — not a supervised contractor. Claude's reviews are
**peer review after the fact**, not permission before it. Work never blocks waiting for a
reviewer. What follows is the difference between the two kinds of rules this project has: the
ones that made the corpus credible (which stay, because breaking them voids results), and the
ones that were workflow ceremony (which are now Codex's own judgement calls).

## 1. Codex decides alone — no approval needed

- **Whether and when to run physics**, including E1a, E1b-V2, the plane probes, and every future
  batch Codex authors. The manifest discipline stays (§2.3) but is self-approved: write it, record
  it, run it.
- **Timing relative to Aug 24/25.** The date was never sacred; the *claim-5 isolation* behind it
  is (§2.1). If a batch cannot touch claim-5 inputs and yields the GPU under contention, Codex may
  run it earlier at its own judgement, stating the reasoning in the run record.
- Batch sizing, ordering, retries, and re-rolls within the standing budget envelope (§2.4).
- Archetype and validator design, CPU analysis of any depth, screening and proposal work.
- **Filing its own prediction-register entries** for its own experiments (in a clearly marked LFH
  section), and adjudicating *those* entries against results — before/after discipline per §2.2.
- Revising its own earlier decisions when evidence demands it (the D2-005 pattern), recorded as
  design deltas.

## 2. Standing invariants — these are what "valid" means, not manager gates

1. **Isolation of pre-registered work.** Frozen SONIC untouched; `scene_first_v1/v2` read-only;
   claim-5 inputs, the pre-registered analysis, and user-owned register entries (e.g. P9)
   untouched. Physics-only verdicts; geometry proposes, physics disposes.
2. **Register-before-run.** Predictions filed before each physics batch; outcomes reconciled
   after; misses reported, never buried. This is what makes autonomous results trustworthy
   without a supervisor.
3. **Manifest-before-spend.** A hash-pinned manifest with stop conditions exists before launch —
   as a record and a preflight, not as a request. The hardened-driver rules ride along (success
   markers, scene-start check, 9000 MiB gate, serial execution, timeout, infrastructure-vs-science
   split).
4. **Budget envelope: 8 contended GPU-hours per day, 24 per week,** self-tracked in the run
   records. Within it, spend freely; beyond it, check in with the user (a resource question, not
   a science question). The user can resize this line at any time.
5. **Honest reporting continues unchanged**: reports, funnels, refusals-as-results, the
   `autoresearch/` keep/discard trail. Reviews happen asynchronously against these artifacts.

## 3. Stop-the-line — the short list that still escalates to the user

- An **E1a source-family outcome flip** (a corpus-level finding, bigger than LFH).
- Any evidence a frozen or pre-registered artifact was disturbed, by anyone.
- Work requiring a **new edit operator** (side-step/stance/stride) or changing the paper's
  claims — scope, not execution.
- Budget beyond the §2.4 envelope.

Everything else: decide, record, proceed, and let the review catch up.

## 4. Context Codex should carry (so autonomy is informed, not blind)

The full background lives in: `docs/lfh/` (the plan and method), `docs/hallucination/RECON.md` +
`DESIGN_DELTAS.md` (what reality corrected), the four review files (what peer review verified and
where it erred — note REVIEW_LFH_KCS §2's shared D2-003 mistake), `docs/gate_audit_2026-08-19.md`
and `docs/constraint_distance_findings.md` (the corpus's measurement discipline), and
`docs/prediction_register.md` (read-only: the register culture to emulate, including recorded
void attempts). The recurring lesson across all of them: the expensive failures were never bad
judgement about *whether* to run — they were unverified assumptions about *what* was being run.
Autonomy changes who decides; it does not change what must be checked.
