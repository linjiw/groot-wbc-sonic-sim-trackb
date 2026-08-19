# What the first reviewer pass found

A full pass over the 98-card pack produced four defects in the pack. Three were obvious once
reported. The fourth was reported as a finding about the dataset, and turned out to be a finding
about the evidence — which is the more useful of the two.

## Defects in the pack

**Nineteen cards rendered no image.** Cards from roughly the eightieth onward showed only questions.
The cause was `loading="lazy"` on the contact sheets: images below the fold never load when a
reviewer prints to PDF, which is exactly how the pack was circulated. Removed.

**A third of the questions could not be answered.** The reviewer answered "can't tell" to *did it
perform the behaviour its name claims* across roughly half the cohort, correctly — `density_moderate`,
`combo09` and `01_single_text_prompt__factory_aisle__p0` claim no behaviour a person could check.
The question is now asked only when the name states one.

**Probe cells have no obstacle.** They run on a bare plane; there is nothing to clear. The first
question is no longer asked of them, nor of the reduced-evidence cards where the body was never
recorded. With the previous item this removed **105 of 294 questions** — a third of the reviewer's
work, all of it unanswerable by construction. Asking anyway does not merely waste time; it buries
the cards where the question is real.

**The strip could not show a tuck at all.** Three episodes were scored "does not perform its named
behaviour": `lc005_crouch18`, `w_tuckcap30` and `w_tuckcap40`, the latter two as "arm motion looks
essentially like nominal gait; no visible tuck". Read as a dataset finding, these were false
semantic labels the gates had accepted. They are not.

Measured against the nominal, the executed wrist in both `w_tuckcap` clips is drawn **0.155 m and
0.163 m laterally** — the tuck is present in the physics, and survives execution rather than being
tracked away by the controller. It was invisible because the strip drew capsules as `(x, z)`: a side
elevation, in which a lateral arm retraction has no projection. The reviewer was shown the one view
that cannot contain the behaviour, and reported accurately on what they saw.

Two things follow. The strip now carries a **plan view** beneath the elevation, which supplies the
missing axis; and where a cell has an unadapted counterpart the nominal body is drawn **underneath
in grey**, so the comparison happens inside one panel instead of against ordinary gait remembered
from a card seen forty places earlier. On `w_tuckcap30` the pale nominal arms now extend visibly
beyond the adapted ones at frames 79 and 118.

A frame-sampling fix was also attempted and **discarded**. Only one of the strip's six frames falls
inside the adaptation window, which looked like the cause until the projection was checked; and no
window detector survived contact with the data, because leg swing dominates every posture signal
tried at every smoothing scale. Concentrating frames on a window would not have helped when the
window's content was unprojectable anyway.

## What this changes

Nothing about the physics results, and nothing about the three episodes: they are **not** entered as
semantic failures, and `executed_semantic_valid` gains no rows from this pass. What it changes is
the pack. A review whose evidence omits the axis the behaviour lives on cannot measure semantic
validity — it measures the projection. The re-review runs on the corrected sheets.

The episode still worth watching is the arm tuck's cap. Its yield problem is measured, and the
question of whether a capped tuck stays *recognisable* as a tuck remains open — but it must be
asked of evidence that can show one.
