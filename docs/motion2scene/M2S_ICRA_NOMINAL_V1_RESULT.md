# Matched contract, equal labels: contrast construction teaches, and the learned generator matches the analytic one

2026-09-08, complete. All **72/72 label commands** and all **144/144 evaluation
executions** are admitted. Every arm holds exactly **15 complete groups**, so this is
the equal-label comparison the parent study could not achieve. The registered
[nominal contract](M2S_ICRA_NOMINAL_V1.md) filed its predictions before any proposal
was drawn, and M2S-ICRA-v1 keeps its own contract, refusals and results unchanged.

![Acquisition at equal labels and passage on held-out layouts](assets/nominal-outcomes.png)

## Acquisition

| Arm | Groups | Generated | Both pass | **Useful contrast** | Both fail | Training BCE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 15 | 9 | 7 | **0** | 8 | 0.001762 |
| Analytic | 15 | 9 | 4 | **9** | 2 | 0.000542 |
| Target-only | 15 | 9 | 13 | **0** | 2 | 0.000175 |
| Motion2Scene | 15 | 9 | 4 | **9** | 2 | 0.139004 |

Under the parent study's inherited envelope the learned arm acquired **zero** generated
groups and analytic acquired **one**. Under a contract whose acceptance geometry matches
the question being asked, both acquire **nine of nine**, and every one of those nine is
a physically verified walk-fail/d040-pass contrast. Uniform and target-only acquire nine
groups each and none of them is useful, exactly as their constructions imply.

All four registered acquisition predictions hold: analytic and Motion2Scene each yield at
least one eligible group per carrier (three each); at least one arm besides analytic
acquires a useful contrast (Motion2Scene, nine); uniform yields at most one (zero); and
target-only yields predominantly both-pass (13 of 15). The 113-offset diagnostic was
recorded for every proposal and gated nothing: 17 of the 18 assigned contrast-arm
proposals fail it, which is the measured price of the envelope rather than a new finding.

## Passage on the twelve held-out layouts

| Policy | Passage | d040 requests | Refusals | Per-carrier passage |
| --- | ---: | ---: | ---: | --- |
| Uniform | 14/36 | 0 | 17 | 0.333 / 0.417 / 0.417 |
| **Analytic** | **24/36** | 24 | 0 | 0.667 / 0.667 / 0.667 |
| Target-only | 14/36 | 0 | 0 | 0.333 / 0.417 / 0.417 |
| **Motion2Scene** | **22/36** | 22 | 0 | 0.583 / 0.583 / 0.667 |

| Paired comparison | Both pass | A only | B only | Neither | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Motion2Scene vs uniform | 14 | **8** | 0 | 14 | 0.0078 |
| Motion2Scene vs target-only | 14 | **8** | 0 | 14 | 0.0078 |
| Analytic vs uniform | 14 | **10** | 0 | 12 | 0.0020 |
| Motion2Scene vs analytic | 22 | 0 | 2 | 12 | 0.50 |

Both contrast arms beat both non-contrast arms with no reverse case anywhere, and the
direction holds on all three carriers. **Motion2Scene and analytic are statistically
indistinguishable** at equal labels: two discordant conditions out of 36, both favouring
analytic, p = 0.50. This is outcome B as pre-declared in the guidance, with the addition
that the learned arm now has data to be compared with rather than an empty corpus.

The 36 conditions are twelve layouts on three carriers at one physics seed, so the
condition-level p values overstate independence. The carrier-level statement is the
honest one: contrast arms beat non-contrast arms on 3 of 3 carriers, and the
Motion2Scene-analytic difference is -0.083, -0.083 and 0.000 by carrier.

## Where adaptation helps

| Beam underside | Conditions | M2S requests | M2S passes | Analytic requests | Analytic passes | Walking-only passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.18 m | 12 | 8 | 0 | 6 | 0 | 0 |
| 1.27 m | 12 | 9 | 10 | 12 | 12 | 2 |
| 1.36 m | 12 | 5 | 12 | 6 | 12 | 12 |

The middle band carries the whole effect, as in the parent study. Walking alone passes
2 of 12 there; analytic requests the adaptation in all 12 and passes all 12, Motion2Scene
requests it 9 times and passes 10. Neither command ever succeeds at 1.18 m, and walking
already suffices at 1.36 m where both arms still make a few unnecessary requests that
cost nothing here.

## The one place the learned generator is measurably worse

Motion2Scene's training loss is **0.139** against analytic's **0.000542**, a difference
of more than two orders of magnitude on the same architecture, optimizer and label count.
The registered construction record explains it: among proposals critical at the nominal
pose, 44 of 47 analytic scenes but only 24 of 34 learned scenes are visible to the
decision-time sensor, and 8 of 12 are invisible on 41003, where the learned initializer
concentrates late-station draws. A contrast the robot cannot observe is a label the
learner cannot fit, and four of the nine assigned Motion2Scene groups record zero sensor
ray hits. Neither contract gates visibility and none was added after the fact.

So the learned proposal matches the analytic solver on acquisition yield and on
downstream passage, and is worse on observability. That is a specific, measured and
fixable deficiency rather than a global verdict.

## Cost and scope

The whole study cost **1.442 contended GPU-hours**: 72 label commands plus 144
evaluations. The six shared background groups were reused from M2S-ICRA-v1 by reference
and not re-executed, so no reused trace counts as a new example. The scripted-ray and
privileged-geometry comparators do not depend on the training arm and were likewise
reused at this seed rather than re-run.

This is three inspected development carriers, one beam family, two commands, ideal
simulator rays, one controller and one physics seed for the evaluation. It is unseen-layout
transfer, **not** source-held-out transfer, and nothing here may be described as
uncertainty-robust: acceptance was judged at the nominal pose and the inherited envelope
is reported only as a diagnostic.

Evidence: [figure, records, fits and models](evidence/nominal-results-20260908/exports.json)
· [reading PDF](assets/nominal-outcomes.pdf) · raw data under
`/home/linjiw/research-data/groot-wbc/m2s-icra-nominal-v1` and `-learning-v1`.

```bash
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_icra_nominal.py register --out /path/to/nominal
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_icra_nominal.py prepare --out /path/to/nominal
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/motion2scene_nominal_pipeline.py --max-wall-hours 12
PYTHONPATH=.:scripts/research .venv_research/bin/python \
  scripts/research/render_motion2scene_nominal_results.py
```
