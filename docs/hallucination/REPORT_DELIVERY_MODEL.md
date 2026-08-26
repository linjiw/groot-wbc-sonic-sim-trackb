# D_phi: executed reach from a reference clip — first validated fit

**Date:** 2026-08-26
**Status:** E-D1 (leave-one-motion-out validation of `D_phi`) has evidence for the first time.
The model beats its honest baseline. It remains proposal-only; physics is still the verdict.

---

## Why this model and not the other one

`docs/lfh/lfh.md` §4.1 defines `D_phi` as the cheap differentiable predictor that occupies the seat
LfH gives its fixed decoder, and marks its validation "still required". Meanwhile
`q_LFH_conditional_v1` was fitted to predict **archetype identity** from trajectory features, and a
2026-08-26 audit refuted it: feature-blind counting scores 1.46416 nats against the kernel's
1.51373 (`REPORT_AUDIT_2026-08-26.md` §3).

The diagnosis was a target problem. The executed pair determines **where** a face must be; it does
not determine **what the face looks like**. So this fit targets the quantity the trajectory
actually constrains, and the one the pipeline is blocked on:

> Given a reference clip, what overhead reach will the frozen controller actually execute?

That matters because the critical window is an order statistic over *executed* reach. Today,
proposing a scene for a freshly generated motion costs two rollouts before any geometry can be
authored — which is exactly where E9a stopped. A usable `D_phi` lets an amplitude be chosen, and a
window checked for emptiness, before the spend.

## Corpus

`scripts/research/hallucination/build_reach_response_corpus.py` walks all 21 immutable LFH run
records and, for each executed cell, measures overhead reach **twice through the same instrument at
the same face**: once on the reference clip by forward kinematics, once on the recorded execution.
Face is 0.10 m along route by 3.0 m across, at route progress 0.55, taken at the *commanded*
station — a scene is authored in world coordinates from the plan, so the station is a property of
the plan, not of the execution.

`docs/hallucination/reach_response_corpus.csv` — **154 paired rows** after LFH-E12, 0 refusals.
Because both sides use one instrument, the residual is delivery rather than a change of definition.
The table below reports the pre-E12 state, which is the population the original fit was made on.

| population | n | residual mean | sd |
|---|---:|---:|---:|
| all executed cells | 118 | +17.8 mm | 57.7 mm |
| accepted | 89 | +25.3 mm | 62.8 mm |
| **accepted, empty scene** | **28** | **−3.9 mm** | **13.8 mm** |

The heavy tail belongs entirely to in-scene cells: the worst residuals (+180 to +200 mm) are
`duck_003` adapted cells with 0.30–0.37 m endpoint error. A drifted execution is at a different
gait phase when it reaches the commanded station, so it is taller there than the plan said. That is
route drift, not a delivery property, and it is why the model is fitted on empty-scene accepted
rollouts only. (Executed captures carry ~199 frames against a 120-frame reference simply because
the rollout records at 50 Hz and the reference is 30 fps; that is resampling, not a join error.)

## The original fit, and holding it out against 18 unseen clips

The fit was made **before** LFH-E12 ran. E12 then executed 47 empty-scene cells over 12
motions the model had never seen, spanning body modes absent from the original fit — `squat_pick`,
`walk_look`, `reach_walk`, `carry_walk`, `backward`, `turn_in_place`, `side_step`,
`stand_to_walk`. Refitting on the enlarged corpus (154 paired rows, **46 distinct clips**):

| model | RMSE | median abs | q90 abs |
|---|---:|---:|---:|
| **linear in commanded reach** | **9.71 mm** | **6.84 mm** | **16.85 mm** |
| identity (baseline) | 15.95 mm | 10.69 mm | 22.29 mm |
| constant offset | 16.17 mm | 10.27 mm | 22.75 mm |
| executed mean (feature-blind) | 22.25 mm | 11.95 mm | 35.31 mm |

The advantage over identity **grew** from 35.4% to **39.2%** as the corpus nearly doubled and
diversified. That is the behaviour of a relation that holds, not of a fit chasing 28 points.

### The original pre-E12 fit

Leave-one-motion-out over **28 distinct reference clips**, empty-scene accepted rollouts. The
baseline is **identity** — `executed = commanded` — because that is what the pipeline implicitly
assumes whenever it reasons about a reference clip.

| model | RMSE | median abs | q90 abs |
|---|---:|---:|---:|
| **linear in commanded reach** | **9.07 mm** | **5.75 mm** | **14.78 mm** |
| constant offset | 14.00 mm | 11.38 mm | 19.83 mm |
| identity (baseline) | 14.04 mm | 10.69 mm | 20.15 mm |
| executed mean (feature-blind) | 22.82 mm | 9.27 mm | 35.03 mm |

**35.4% RMSE reduction against identity**, and the feature-blind control is the *worst* model — the
opposite of what the archetype study found, and the reason this result is reportable while that one
is not. The E12 refit above supersedes these numbers; they are kept because they are what the
prediction was made on. `docs/hallucination/reach_delivery_model.json` carries the fit and the scores.

Caveats, stated in the artifact itself: the predictive band is the leave-one-motion-out q90, not a
calibrated confidence interval; the fit does not describe a drifting execution; and a predicted
window still requires executed physics before any scene is promoted.

## The operator delivers 63–84% of its commanded window

Pairing every accepted empty-scene adapted clip with its accepted nominal gives the quantity that
actually decides whether a scene exists:

| source | commanded window | executed window | delivered |
|---|---:|---:|---:|
| `lfh_086_crouch` | 83.3 mm | 67.7 mm | 81% |
| `lfh_089_crouch` | 85.5 mm | 71.9 mm | 84% |
| `lfh_090_crouch` | 84.0 mm | 52.5 mm | 63% |

The reference over-states the window by 16–32 mm, consistently in the same direction. This sharpens
the draft's "the predicted window is optimistic on both sides" from one family to three independent
sources measured through one instrument.

## A null control, and what it says about the engineering margin

The eight accepted arm-tuck pairs are a genuine null on this axis: an arm tuck does not change
overhead reach, so their **commanded** overhead window is exactly 0.000 mm by construction. Their
**executed** windows are:

> −2.73, −1.35, −0.56, −0.52, −0.58, −0.28, +1.21, +1.85 mm
> (n = 8, mean −0.37 mm, sd 1.42 mm, max |.| **2.73 mm**)

So the executed-window estimator reproduces a true zero to within **2.73 mm** across paired
rollouts of clips that differ only in arm pose. That is a measured noise floor for this
instrument, and it is a much better negative control than anything currently in the record —
it also confirms the semantic partition attributes an upper-body edit to the upper body.

**This does not license changing the 18.044 mm engineering margin, and it is not being changed.**
That value is E1a's maximum observed three-run *in-scene clearance* range across eight overhead
cells — a different quantity, measured under contact and drift, where the spread is legitimately
larger. But the two numbers differ by a factor of six, they are both being used to reason about
overhead placement, and the margin is applied symmetrically so it consumes 36.1 mm of every raw
window. Since the raw windows above are 52–72 mm, the margin is currently spending half to
two-thirds of the available room. Whether an empty-scene-calibrated margin is admissible for
empty-scene window construction is a registered question, not a silent edit.

## What this changes

1. **E-D1 has evidence.** `D_phi` can be reported as fitted and cross-validated against a real
   baseline, with its population and caveats stated.
2. **The paper gains a learned component that survives audit.** After the archetype model was
   refuted, this is the model to report — and reporting both, with the baseline that killed the
   first, is stronger than reporting either alone.
3. **Motion-to-scene inference for a generated motion becomes possible to attempt.** The chain is:
   roll the generated nominal once, predict the adapted reach with `D_phi`, solve the window, and
   only author geometry if the predicted window survives its own uncertainty band. E12 supplies the
   amplitude ladder that makes the second step accurate.
4. **The instrument has a measured noise floor** (2.73 mm) and a measured optimism (16–32 mm), both
   from paired rollouts rather than assumption.

## Reproduce

```bash
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/build_delivery_corpus.py
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/build_reach_response_corpus.py
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/fit_reach_delivery_model.py
```
