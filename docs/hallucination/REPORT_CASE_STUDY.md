# Motion-to-scene inference over the whole gated pool — 94 clips, 376 proposals

**Date:** 2026-08-26
**Artifact:** `docs/hallucination/case_study.json`
**Runner:** `scripts/research/hallucination/run_case_study.py`
**Figure:** `docs/source/_static/lfh_e17/case-map.png`

CPU proposal support only. A proposable rung licenses empty-scene physics for that pair; it is not
a family and not a verdict. Where physics has been run it disagrees with this sweep in a specific
direction, recorded in §5.

---

## 1. What was run

Every clip the reference gate passes — **94 of the pool's 150** — swept at all four crouch
amplitudes (40/55/70/85 mm), giving **376 proposals**. For each, the route descriptors, the
predicted window from `D_phi`, and, when refused, the gate that refused it.

The pool contains exactly the motion families worth asking about: curving gently left and right,
walking then turning sharply and continuing, stepping sideways while facing forward, turning around
in place then walking, each at slow / steady / brisk pace, forwards and backwards, with and without
a carried object.

## 2. Headline

| | |
|---|---:|
| clips with at least one proposable rung | **39 / 94 (41%)** |
| proposable rungs | **56 / 376 (14.9%)** |
| refused: `window_below_min` | **234 / 376 (62.2%)** |
| refused: `operator_saturated` | 86 / 376 (22.9%) |

**Nearly two-thirds of all refusals are the engineering margin**, not geometry, not the controller,
and not the motion. This is LFH-E16b's finding reproduced at pool scale.

## 3. The margin is the binding constraint, and the effect is large

Holding everything else fixed and varying only the symmetric margin:

| margin, each side | proposable rungs | clips with a proposal |
|---|---:|---:|
| **18.044 mm (current)** | **56 / 376 (14.9%)** | **39 / 94** |
| 12 mm | 144 / 376 (38.3%) | 88 / 94 |
| 10 mm | 180 / 376 (47.9%) | **94 / 94** |
| **7.94 mm (E16b's measured worst three-seed range)** | **213 / 376 (56.6%)** | **94 / 94** |
| 5 mm | 249 / 376 (66.2%) | 94 / 94 |

At the margin the repeatability measurement actually supports, **every clip in the pool yields a
proposal** and the proposable-rung rate rises 3.8×. The current value was derived from in-scene
*clearance* ranges on overhead cells; the quantity it guards is an empty-scene reach difference
whose measured three-seed range is at most 7.94 mm. Changing it is a registered decision that has
not been taken — but this is what it is worth.

## 4. The motion families, by name

| family | clips | propose | median face misalignment |
|---|---:|---:|---:|
| curve gently **left** | 8 | 5 | 16.8° |
| curve gently **right** | 8 | 4 | 19.1° |
| walk, **turn sharply**, continue | 4 | 2 | 21.2° |
| **steps sideways** (hip-led, facing forward) | 2 | **0** | 17.3° |
| **turn around in place** then walk | 3 | **0** | 4.1° |
| slow-paced variants (all families) | 29 | 11 | 16.6° |

Curving motions do produce proposals. The two families that produce none are the two you asked
about specifically, and they fail for **different** reasons:

- **Sideways stepping** fails on `window_below_min`. The body's overhead silhouette barely changes
  when a crouch is applied to a sideways gait, so there is no separation to place a face in.
- **Turn-in-place** fails on `operator_saturated` *and* `window_below_min`, at a misalignment of
  only 4.1°. Its failure is not orientation — the crouch operator saturates against its excursion
  cap before delivering a usable drop, because a turning stance already consumes joint range.

That distinction matters: sideways needs a *different constraint axis* (a lateral gap, which the
repaired instrument can now compute), while turn-in-place needs a *different operator*.

## 5. Where this sweep is optimistic

This is CPU support, and LFH-E12 measured the gap. Backward walking proposes on 11 of 15 clips
here, yet E12's `backward` nominal was **rejected in physics** at 0.380 m endpoint error with a
disallowed foot contact, and its `side_step` nominal at 0.454 m. The controller refuses whole gaits
that the geometry is perfectly happy with.

So read 41% as an **upper bound on the fraction of the pool that could become sources**, not as a
yield. The physics funnel from E12 — 12 motions → 8 nominals accepted → 5 delivering an amplitude
→ (after E16b's seed correction) 2 stable — is the real conversion rate, and it is the number that
should be quoted.

## 6. Face orientation: measured, not argued

For each clip the sweep records the angle between the world route axis the face is authored on and
the executed tangent at the binding station. Faces are authored axis-aligned, so this angle *is*
the error in the face's orientation.

| misalignment | clips | propose |
|---|---:|---:|
| 0–5° | 31 | 15 (48%) |
| 5–15° | 30 | 11 (37%) |
| 15–30° | 22 | 11 (50%) |
| **30–90°** | 11 | **2 (18%)** |

The worst cases reach **52.6°**, on clips whose total heading change runs to 321°. The verified
source `lfh_089_crouch` sits at 0.74° — which is why the existing corpus is sound, and why it is
sound *only* for near-straight routes.

`case-map.png` draws this for six representative cases. On the straight walk the oriented face and
the axis-aligned face coincide. On the curving and sideways cases they are visibly rotated apart —
on clip 001 by 27.6°, so the authored face presents its 0.10 m along-route extent to a body
crossing at a completely different angle, and the crossing footprint the window depends on is
simply the wrong region.

`overhead_face_reach` now accepts `route_yaw_rad` and evaluates in the route frame; authoring and
keep-out still need the matching rotation before a curved-route family can be built.

## 7. What this changes

1. **The margin decision is now supported by two independent measurements** — E16b's three-seed
   window ranges (≤7.94 mm) and this sweep's sensitivity curve (15% → 57% of rungs). It remains an
   unspent, registered decision.
2. **Curving motions are not blocked by the method**, only by axis-aligned authoring. Half the
   curving clips already propose despite a mis-oriented face.
3. **Sideways and turn-in-place need different machinery**, and the sweep says which: a lateral
   axis for one, a different operator for the other. Neither is a tuning problem.
4. **41% is a ceiling, not a yield.** The controller, not the geometry, is the narrow part of the
   funnel, and E12 measured it.

## Reproduce

```bash
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/run_case_study.py
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/render_case_map.py \
  --index 8 --index 1 --index 125 --index 3 --index 64 --index 120 \
  --out docs/source/_static/lfh_e17/case-map.png
```
