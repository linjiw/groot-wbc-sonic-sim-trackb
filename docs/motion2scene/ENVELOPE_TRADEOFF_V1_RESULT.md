# The inherited placement envelope, not the proposer, closed the funnel

2026-09-07. The [registered sweep](ENVELOPE_TRADEOFF_V1.md) evaluates 11,421 station/height
centres per carrier on the recorded seed-8721 achieved transitions and counts 10 mm
contrast witnesses as the audit envelope is scaled from the nominal pose to the
inherited +-20 mm xy / +-10 mm z / +-0.02 rad set. It took 443 CPU seconds, ran no
physics and changed no M2S-ICRA-v1 assignment, refusal or score.

| Envelope (xy / z) | 41001 | 41002 | 41003 |
| --- | ---: | ---: | ---: |
| nominal pose | **200** | **168** | **337** |
| +-2 mm / +-1 mm | 164 | 132 | 295 |
| +-4 mm / +-2 mm | 126 | 102 | 260 |
| +-5 mm / +-2.5 mm | 120 | 86 | 238 |
| +-6 mm / +-3 mm | 94 | 70 | 220 |
| +-8 mm / +-4 mm | 70 | 49 | 185 |
| +-10 mm / +-5 mm | 50 | 27 | 150 |
| +-15 mm / +-7.5 mm | 13 | 4 | 81 |
| **+-20 mm / +-10 mm (inherited)** | **2** | **0** | **32** |

![Contrast support against placement envelope, and the acquisition funnel under each contract](assets/envelope-tradeoff.png)

The best nominal joint margin — the smaller of target clearance and walk interference
at the proposal's own pose — is **24.42 mm on 41001, 20.48 mm on 41002 and 26.99 mm on
41003**. The inherited envelope asks each accepted centre to keep a 10 mm two-sided
contrast while the beam moves +-20 mm horizontally and +-10 mm vertically. That demand
is of the same order as the entire available window, so it removes 99%, 100% and 90%
of the witnesses on the three carriers.

This is the measured cause of the M2S-ICRA-v1 funnel: 47/48 analytic and 48/48 learned
generated slots refused, almost all for `contrast_audit`. The proposers were asked for
centres that are extremely rare or, on 41002, absent on this grid.

## Two causes, not one

The envelope is not the whole story, and the earlier
[support diagnostic](TRANSITION_SUPPORT_DIAGNOSTIC_V1.md) already found the second
cause. On 41003 the inherited envelope still leaves 32 sampled witnesses, yet the
study accepted zero slots there. That carrier's failure is reachability: the frozen
learned initializer's unchanged +-0.05 station / +-0.03 m trust boxes covered none of
the measured robust centres, so no amount of the same bounded search reaches them.

So the funnel closed for two separable reasons — an envelope that consumes the window,
and proposal draws that do not cover what survives it. Only the first is addressed by
a different contract; the second is a generator question that this ICRA scope
deliberately leaves closed, since no refit is in scope.

## What this does and does not license

It licenses one thing: a **separately registered** contract whose acceptance geometry
matches the question being asked. The obstacle in this study is authored into the
simulator at an exact pose, so a placement-uncertainty envelope is guarding a quantity
that is identically zero here, while the execution variability that does matter is
measured by paired physics and two evaluation seeds. That contract is
[M2S-ICRA-nominal-v1](M2S_ICRA_NOMINAL_V1.md), registered with its predictions before
its proposals were drawn.

It licenses nothing else. These are sampled witnesses on one finite grid at one physics
seed per carrier: not a feasible-volume estimate, not proof that unsampled centres fail,
and not a physical outcome. A witness is geometry only; whether the executed command
clears the beam is decided by paired physics, and the carriers whose d040 the tracker
rejects for endpoint error are exactly where achieved and reference geometry diverge
most. M2S-ICRA-v1 keeps its original contract, its refusals and its results unchanged,
and no result under the nominal contract may be reported as uncertainty-robust.

The inherited envelope also remains the right instrument for the question it was built
for. A hardware claim, where the beam's real pose is uncertain, would need it — and
this table then states the price directly: at +-20 mm only 2, 0 and 32 centres survive,
so a physical deployment of this contrast would need either a much larger executed
window or much tighter obstacle placement.

Evidence: [figure and released records](evidence/envelope-tradeoff-20260907/exports.json),
[reading PDF](assets/envelope-tradeoff.pdf), and `envelope-tradeoff.json` under
`/home/linjiw/research-data/groot-wbc/m2s-envelope-tradeoff-v1`, marked `analysis_only`
so it never enters the physics spending ledger.

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_envelope_tradeoff.py --out /path/to/envelope-output
```
