# Multi-obstacle LfLH for humanoid counterfactuals: what works, and what does not

**Date:** 2026-08-27
**Code:** `gear_sonic/dataset_generation/hallucination/{lflh,motion_envelope}.py`,
`scripts/research/hallucination/{build_candidate_sets,train_lflh}.py` · **Tests:** 12

Supersedes the retracted `REPORT_LFLH_COMPARISON.md`. Every defect that retraction identified is
addressed here; the results are mixed, and reported as such.

---

## 1. What is different from the retracted attempt

| retracted | now |
|---|---|
| decoder was a soft indicator of a closed-form interval | decoder **re-decides** which of 8 candidate motions the scene prefers |
| 2 parameters, one overhead face | **5 obstacles x 6 parameters**, any direction |
| coordinate anchored on the answer's lower edge | no anchor derived from any feasibility answer |
| `min_log_sigma = -6`, and the headline was that clamp | `min_log_sigma = -20`; a reported sigma cannot be its own floor |
| no entropy/KL — collapse was a theorem | genuine Gaussian KL carrying `-log sigma`, plus obstacle-obstacle repulsion |
| compared against uniform-on-the-answer | compared against **random-from-prior**, **input-ablated**, and **no-KL** |

**Obstacles are now directional.** `motion_envelope.py` measures up/left/right body extents per
route station in the executed route frame. Overhead binding requires the box to cover the route
centreline; lateral binding requires it to sit off to one side and straddle mid-body height.

Finding that distinction took fixing a real bug: a waist-height side obstacle was being scored as
overhead, so every obstacle blocked every candidate and the decoder always returned the nominal.

Unit tests verify the decoder decides: an empty scene selects the nominal (the cheapest), a wide
overhead bar selects the crouch, a **left** obstacle selects the **left** arm tuck, and a right
obstacle the right one. That is the sidedness the previous videos could never show.

## 2. Per-clip: it works, for edits that open a wide enough band

One model per clip, 8 real clips, 500 steps, one observed edit each:

| observed edit | scenes selecting it |
|---|---:|
| `tuck_left_070` | **1.00** |
| `tuck_right_040` | **1.00** |
| `tuck_right_070` | **1.00** |
| `crouch_040`, `crouch_055`, `crouch_070`, `tuck_left_040` | 0.00 |
| mean | 0.375 |

With a softer decoder (`temperature_m` 0.02 → 0.15) the deepest crouch recovers:

| `temperature_m` | crouch_040 | crouch_055 | crouch_070 |
|---|---:|---:|---:|
| 0.02 | 0.00 | 0.00 | 0.00 |
| 0.06 | 0.03 | 0.00 | **1.00** |
| 0.15 | 0.14 | 0.00 | **1.00** |

**The failures are optimisation, not representation, and that is demonstrated rather than
asserted.** Hand-placing an overhead bar at the midpoint of the nominal/crouch gap selects
`crouch_040` at every obstacle width tested (0.04–0.80 m half-length). The target is reachable; the
optimiser does not find it from a random start, because outside the band both candidates are
equally blocked or equally clear and the gradient is flat.

**One variable explains the pattern: the width of the band the edit opens.** `crouch_070` separates
the body by ~102 mm at its best station and succeeds; `crouch_040` separates by ~65 mm and fails.
That is the same quantity the closed-form solver calls the executed window `|W|`, so a narrow
window is hard for the hallucinator to find for the same reason it is hard for physics to deliver.

## 3. Across clips: it fails, and its own control says why

One model over 24 clips, targets rotated across crouches and both tucks:

| arm | scenes selecting the observed motion | station sd | lateral sd | side entropy |
|---|---:|---:|---:|---:|
| **learned** | **9.3%** | 5.74 | 0.498 | 0.647 |
| learned, **input ablated** (fed the corpus-mean profile) | **27.2%** | 5.73 | 0.497 | 0.649 |
| random from the prior | 11.5% | 3.23 | 0.460 | 0.760 |
| learned without KL | 0.0% | 7.35 | 0.485 | 0.673 |

The learned arm is **beaten by random obstacles, and beaten badly by its own input-ablation**.
Feeding the model the corpus mean instead of each clip's own profile makes it three times better.
Tripling the training budget (700 → 2500 steps) makes it *worse*, 9.3% → 2.7%, so this is not
under-training:

| steps | learned | input ablated | random |
|---|---:|---:|---:|
| 700 | 9.3% | 27.2% | 11.5% |
| 2500 | **2.7%** | 22.3% | 11.6% |

**Conditioning on the trajectory is currently anti-informative.** This is the same shape of finding
as the archetype kernel earlier in this project: a model asked to condition on features that barely
distinguish its inputs does worse than ignoring them. The per-clip profiles differ mainly in one
channel over a few stations — a tucked arm changes the lateral extent by 40–70 mm out of ~350 mm —
and the encoder does not extract it.

Note the input-ablation control is exactly what exposed the retracted version as a constant. Here
it does its job again, in the other direction: the model *is* input-dependent, and that dependence
hurts.

## 4. Honest status

**Established.** A multi-obstacle hallucinator with a decoder that genuinely re-decides can place
obstacles that make a *specific, named* edit the preferred one — including choosing the correct
side for a left versus right arm tuck. That is the mechanism the paper needs, and it is verified
per direction by test and per clip on real data.

**Not established.** That a single model amortises this across clips. At 24 clips it loses to
random and to its own ablation. Until it beats both, no claim about a learned scene distribution
should appear in the paper.

**Diagnosed.** Success tracks the width of the band the edit opens. The narrow-band failures are
optimisation, proven by hand-placement recovering the target.

## 5. What to do next, in order

1. **Anneal the decoder temperature** during training — soft for gradient flow, sharpened for a
   faithful decision. LfLH anneals its own loss weights over 1000 epochs; we do not anneal at all.
   The 0.02 → 0.15 sweep already shows the direction.
2. **Initialise from the closed form.** We can compute a feasible overhead band exactly. Starting
   the obstacle mean inside it and letting the model refine turns a search problem into a
   refinement problem, and is legitimate provided the initialisation is reported.
3. **Train per-clip, then amortise.** Per-clip optimisation already works for wide bands. Fit the
   encoder to the per-clip solutions as a supervised regression, rather than asking it to discover
   them through the decoder.
4. **Only then report a distribution.** With reconstruction rate *and* diversity, against
   random-from-prior and input-ablated, at more than one training budget.

## Reproduce

```bash
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/build_candidate_sets.py --clips 24
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/train_lflh.py --steps 700 --draws 48
```
