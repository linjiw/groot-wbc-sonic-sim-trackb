# Predictions recorded before the result

Written before the measurement, so the record cannot be rewritten afterwards. Four of the
predictions below were wrong, which is the reason this file exists: without it, a refuted guess
quietly becomes a "finding we always suspected".

## Open

*(none)*

## P5, partly confirmed and partly refuted

Registered before the two pending cells returned. The **correlation** held; the **causal** reading
did not, and the causal reading is the one that mattered.

| excursion | motion | outcome | \|drift\| |
|---|---|---|---|
| 0.929 rad | x002 | accepted | 0.015 m/s |
| 0.936 rad | x003 | accepted | 0.025 m/s |
| 0.994 rad | 005 | accepted | 0.100 m/s |
| 1.000 rad | x000 | **rejected** | 0.225 m/s |

Drift magnitude does rise monotonically with knee excursion across those four clips. But the four
clips come from **three different motions**, so that table mixes between-motion and within-motion
variation — and the within-motion test refutes the causal claim:

| excursion | motion | outcome | \|drift\| |
|---|---|---|---|
| 1.000 rad | x000 | rejected | 0.224 m/s |
| **0.980 rad** | x000 | **rejected** | **0.223 m/s** |

Cutting 20 mrad on the motion that failed moved its drift by 0.001 m/s. **So excursion correlates
with trackability across motions but does not control it within one**, at least over this range, and
"crouch trackability is predictable from knee excursion" is too strong. x000 is a harder motion to
crouch, and the monotone table was largely reading motion identity.

What survives: the crouch fails by drift with zero contact where the tuck fails by collision, and
the crouch's yield is **2 of 3** valid nominals against the tuck's 1 of 3. What does not: that the
clip alone tells you which.

Two far weaker crouches on x000 — 0.619 rad for a 30 mm drop and 0.420 rad for 15 mm — are running
to settle whether it is crouchable at any strength. If both drift, x000 is simply not crouchable and
the yield stays 2 of 3. If one holds, the threshold is motion-specific and no single global cap can
express it.

`MAX_CROUCH_EXCURSION_RAD` was tightened from 1.00 to 0.98 on the strength of this prediction. The
bound is still defensible — a cap should sit below every excursion observed to fail — but the benefit
claimed for it, that it would turn an untrackable clip into a trackable one, **is false**, and the
commit message asserting it is wrong.

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
