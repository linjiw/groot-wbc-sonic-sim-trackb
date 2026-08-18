# Predictions recorded before the result

Written before the measurement, so the record cannot be rewritten afterwards. Four of the
predictions below were wrong, which is the reason this file exists: without it, a refuted guess
quietly becomes a "finding we always suspected".

## Open

*(none)*

## P5, confirmed — after I got the intermediate reading wrong twice

The registered claim was that knee excursion governs crouch trackability. It does. Getting there
took two wrong intermediate readings, both recorded here because the sequence is the lesson.

**Reading 1 (too strong).** Four clips across three motions gave a monotone drift-versus-excursion
table with the boundary between 0.994 and 1.000 rad, and I reported trackability as predictable from
the clip.

**Reading 2 (too weak).** Rebuilding the failing clip at 0.980 rad left it rejected with drift 0.223
against 0.224 — twenty mrad changed nothing — so I withdrew the causal claim and said excursion
correlates across motions without controlling within one.

**Reading 3, from a wider within-motion sweep.** Excursion *does* control it. My 0.980 probe was
simply still inside x000's failure region, because that motion's threshold is nowhere near 0.99:

| excursion | drop | outcome | drift |
|---|---|---|---|
| 0.619 rad | 30.0 mm | **accepted** | 0.140 m/s |
| 0.980 rad | 69.6 mm | rejected | 0.223 m/s |
| 1.000 rad | 72.2 mm | rejected | 0.224 m/s |

So the threshold is real and **motion-specific**: x000's sits in (0.619, 0.980) where motion 005
holds at 0.994. A 20 mrad step was far too small to find it, and reading 2 mistook "no effect over
this range" for "no effect".

### What this means operationally, which is the useful part

x000 is not a failure of the operator. It is caught in a **two-sided squeeze**, and the two sides
are measured in different currencies:

| excursion | predicted window | trackable? |
|---|---|---|
| 0.619 rad | 43.1 mm | **yes** |
| 0.980 rad | 92.3 mm | no |
| 1.000 rad | 95.6 mm | no |

The strengths that buy a usable window are untrackable, and the strength that tracks buys a window
too narrow to trust — 43 mm predicted is 10 to 35 mm real, given that measured windows run 23–82% of
predicted, against 82–98 mm for the three families that work.

**So the crouch's yield is set by whether a motion's trackability threshold sits above the excursion
its window needs.** That is a screen worth one rollout per motion rather than a bisection: compute
the excursion a 60 mm window requires (CPU only), then test trackability at exactly that excursion.
Motions where the two do not overlap are dropped before any family is attempted.

### The original registration, unedited

**P5 — crouch trackability is set by knee excursion (registered 2026-08-18, 2 of 3 cells pending).**
Motion 005's crouch at 0.994 rad is accepted; x000's at 1.000 rad — sitting exactly on the cap — is
rejected for pure reference drift with zero contact, at 0.225 m/s against the accepted crouch's
0.100. If excursion is the governing variable, then **x002 (0.929 rad) and x003 (0.936 rad) should
both be accepted**, and the crouch's boundary sits just under 1.00 rad.

*Confidence: moderate, and deliberately stated because the analogous prediction failed for the arm
tuck.* Excursion did **not** predict tuck trackability there. What differs is the failure mode: the
tuck's rejections are collisions at low drift (0.051 and 0.003 m/s), while the crouch's is a
tracking failure with no contact at all — and a tracking failure is the kind of thing excursion
plausibly governs, where a collision is governed by geometry. If x002 or x003 is rejected, excursion
is not the variable for the crouch either, and per-clip rollout screening is the only method for
both operators.

## Resolved

| # | prediction | outcome |
|---|---|---|
| P1 | The arm tuck will track better than the crouch because it moves less mass | **wrong** — it holds the route (19.6 mm) but loses the effect, and yields 1 of 3 |
| P2 | Lower joint excursion improves retention | **wrong** — the smallest excursion gave the worst retention |
| P3 | Clip duration dominates trackability | **wrong** — refuted alongside P2 |
| P4 | The tuck fails by pressing the wrist into the hip | **wrong** — the gap is negative on every nominal and its change anti-correlates with the verdict; the failures are external contact, not self-contact |

The pattern in P1–P4 is one thing: each was an attempt to *infer* trackability from the clip instead
of measuring it. That is also what forced `family_eligible` out of the Stage 2 report after four
estimators gave four answers. P5 is the same species of claim, which is why it is registered rather
than assumed.
