# What the first reviewer pass found

A full pass over the 98-card pack produced three defects in the pack itself and three disagreements
with the automatic gates. The defects are fixed; the disagreements are the point of the exercise.

## Defects in the pack

**Nineteen cards rendered no image.** Cards from roughly the eightieth onward showed only questions.
The cause was `loading="lazy"` on the contact sheets: images below the fold never load when a
reviewer prints to PDF, which is exactly how the pack was circulated. Removed.

**A third of the questions could not be answered.** The reviewer answered "can't tell" to *did it
perform the behaviour its name claims* across roughly half the cohort, correctly — `density_moderate`,
`combo09` and `01_single_text_prompt__factory_aisle__p0` claim no behaviour a person could check.
The question is now asked only when the name states one.

**Probe cells have no obstacle.** Every probe drew "probe has no visible obstacle, so contact cannot
be judged". They run on a bare plane; there is nothing to clear. The first question is no longer
asked of them, nor of the reduced-evidence cards where the body was never recorded.

Together these removed **105 of 294 questions** — a third of the reviewer's work, all of it
unanswerable by construction. Asking anyway does not merely waste time; it buries the cards where
the question is real.

## Disagreements with the gates — the actual finding

Three episodes the gates accepted were judged **semantically wrong** by the reviewer:

| episode | gate | reviewer | note |
|---|---|---|---|
| `local_rollouts/lc005_crouch18` | accepted | behaviour **no** | "does not visibly lower relative to ordinary gait despite `crouch18`" |
| `sweep_rollouts/w_tuckcap30` | accepted | behaviour **no** | "arm motion looks essentially like nominal gait; no visible tuck" |
| `sweep_rollouts/w_tuckcap40` | accepted | behaviour **no** | "arm motion looks essentially like nominal gait; no visible tuck" |

These are exactly the failures the four-label decomposition predicts and no numeric gate detects: the
clips are **physically fine** — cleared, upright, tracked — and do not contain the behaviour their
name asserts. The gates never claimed to check that, which is why `executed_semantic_valid` is a
separate column and why the corpus-level headline is 32% rather than 96%.

The two `w_tuckcap` cases are the more pointed. Both are arm-tuck clips at capped excursion, and a
cap that produces no visible tuck is a clip that satisfies its geometric target while failing its
semantic one. That is a real limit on how far the excursion cap can be tightened before the operator
stops meaning anything, and it was invisible to every measurement taken so far.

## What this changes

Nothing about the physics results. It changes the release: `executed_semantic_valid` will be
populated from human review rather than left null, and the three episodes above are the first
entries. It also sharpens the arm tuck's story — its yield problem is measured, and now there is
evidence that some of its *accepted* clips are not tucks either.
