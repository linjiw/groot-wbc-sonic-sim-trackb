# What the controller keeps: measuring adaptations that survive execution

A clip named `crouch18` asserts a crouch. What the dataset can honestly claim is weaker — that a
crouch was **commanded**. The frozen controller decides how much of it happens, and a label
describing motion the robot largely did not perform is a mislabel however cleanly the physics gates
pass. This is the axis SONIC tracking can measure that the contact gates cannot.

## The measurement

Per clip, over the joints the operator actually moves and the frames where it is active:

```
survival = mean |executed_adapted − executed_nominal| / mean |reference_adapted − reference_nominal|
```

Both sides come from `reference_g1_qpos` and `dof_pos`, which the rollout writes on the same frame
grid — nothing is resampled, and the 30 fps / 50 Hz factor never enters. Operator joints are
discovered from the reference difference rather than hardcoded, so one implementation reads a
crouch, a tuck, or a combination.

Two rollouts of the same journey drift apart on their own, and that drift inflates the numerator.
The **control** column measures it on the joints the operator never touches. Across 23 clips the
operator signal stands 4.7×–13.6× above its own drift floor, so the ratios below are signal.

`scripts/research/measure_operator_survival.py <corpus dirs> --csv out.csv`

## Result

| operator | n | survival | median |
|---|---|---|---|
| crouch (lower body) | 11 | 46–67% | **58%** |
| combo (hip + waist) | 2 | 65% | 65% |
| tuck (upper body) | 10 | 57–116% | **89%** |

The split is not explained by how much was asked for. At matched commanded amplitude near 0.3 rad
the two operators diverge by a factor of two:

| clip | commanded | survival |
|---|---|---|
| `x000_crouch015` | 0.267 rad | 46% |
| `x000_crouch030` | 0.394 rad | 51% |
| `w_tuckcap30` | 0.282 rad | **105%** |
| `x000_tuck` | 0.313 rad | **91%** |
| `x002_tuck` | 0.331 rad | **88%** |

**The controller preserves arm departures and resists leg departures.** That is the expected shape
for a tracking policy whose legs carry the load and hold balance while the arms are comparatively
free — but it had not been measured, and it sets a hard limit on what a lower-body label can mean.

The arm's fidelity has its own ceiling. Both uncapped large tucks (0.741 and 0.869 rad commanded)
fall to 57%, matching the lower body. Full survival is a property of *small* arm adaptations, not of
arms as such.

One clip resists the story: `w_tuck06win18` commands only 0.072 rad and survives at 61%, where the
amplitude trend predicts near-total survival. It also carries the thinnest margin over drift in the
tuck group (4.9×), and its executed amplitude, 0.044 rad, is small enough that the ratio is
sensitive to noise the control column cannot fully remove. It is recorded, not explained.

## What this changes

**The excursion cap is doing more than protecting trackability.** Capped tucks survive at 101–105%
while uncapped ones survive at 57%. The cap was introduced to keep clips trackable; it also keeps
them *faithful*, which is the stronger justification and was not the reason given for it.

**Lower-body labels overstate by roughly 40%.** A clip commanding an 18 cm crouch executes about
58% of that departure. The release should carry executed amplitude beside commanded amplitude so a
consumer can see the difference rather than inherit the label's claim.

**It gives semantic validity a numeric partner.** Human review remains the arbiter of whether a
behaviour is recognisable — nothing here replaces it. But survival separates two failures review
conflates: a clip whose behaviour the controller discarded, and a clip whose behaviour is present
and merely hard to see. The `w_tuckcap` clips a first reviewer scored as showing no tuck survive at
101–105%; the behaviour was there and the [evidence could not show it](review_pilot_findings.md).

## Open

Survival is measured against a matched nominal, so it exists only for adapted clips with a
counterpart — 23 of the corpus. Clips mined as pairs rather than built by an operator have no
commanded departure to divide by, and need a different treatment.
