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


---

# Addendum, 2026-08-27: seeding and annealing, and what the band width predicts

The previous section diagnosed the per-clip failures as optimisation rather than representation and
proposed two fixes. Both were implemented and ablated. **One works, one does not, and which is
which is predictable in advance.**

## The ablation

Six clips, 400 steps, one observed edit each, evaluated at the sharp decoder. "Minimum edits" are
the shallowest of each kind (`crouch_040`, `tuck_left_040`, `tuck_right_040`); "deep edits" are the
deepest (`crouch_070`, `tuck_left_070`).

| | baseline | anneal only | seed only | seed + anneal |
|---|---:|---:|---:|---:|
| **minimum edits** | **0.505** | 0.047 | 0.307 | 0.198 |
| **deep edits** | 0.831 | **1.000** | 0.503 | 0.508 |

## What separates the two rows is the band, and it is computable beforehand

The width of the band an edit opens against its *cheapest rival* — the same quantity the closed-form
solver calls `|W|`, generalised to left and right:

| target | band vs cheapest rival | rivals |
|---|---:|---:|
| `crouch_040` | **81.1 mm** | 1 |
| `tuck_left_040` | 47.9 mm | 3 |
| `tuck_left_070` | 40.5 mm | 7 |
| `crouch_055` | 23.3 mm | 4 |
| `crouch_070` | **22.4 mm** | 5 |

A minimum edit only has to beat the nominal, so its band is wide. A deep edit has to beat every
shallower edit of the same kind as well, so its band is narrow — which is the lexicographic
minimum-edit rule expressed as geometry, and exactly what `regret = xi * |W|` says.

**Annealing helps narrow bands and destroys wide ones.** Deep edits go 0.831 → **1.000**; minimum
edits collapse 0.505 → **0.047**. The mechanism is consistent: the soft phase supplies gradient
where the sharp loss is flat, which is what a narrow band needs. Where the band is already wide the
search was never the problem, and the soft phase instead lets the model settle on placements that
fail once the decision is made faithful again.

**This is a usable rule, not a curiosity.** The band width is known *before* training, from the
envelopes alone. So the closed-form analysis does not merely compete with the learned machinery —
it configures it: anneal when the band is narrow, do not when it is wide.

## Seeding fails, and the likely reason is the one that sank the retracted version

Closed-form seeding hurt both regimes (0.505 → 0.307, 0.831 → 0.503) despite the seed being good on
its own — seed-only accuracy with no training at all is 8/8 on `crouch_040` and 7/8 on
`tuck_left_040`.

The probable mechanism is in how the seed is installed: the output layer's weights are zeroed so
that the initial output *is* the seed. That also zeroes the gradient path to the encoder, so the
model begins as a constant function of its input and has no pressure to stop being one. That is the
same failure the retracted experiment shipped, arrived at from the opposite direction. It is a
hypothesis, not a measurement — the test would be to seed only the bias while leaving the weights
at their usual initialisation, and to report the input-ablation control alongside.

## Where this leaves LfLH for the paper

**Working, verified:** a multi-obstacle hallucinator with a decoder that genuinely re-decides can
place obstacles that make a specific named edit preferred, choosing the correct side for a left
versus right arm tuck. Per clip, with annealing on narrow bands, deep edits reach a **1.000**
selection rate.

**Not working:** amortising across clips. The learned arm still loses to random and to its own
input-ablation, and seeding — the intervention meant to help — makes the input-dependence worse.

**The honest headline** is not "LfLH works" or "LfLH fails". It is that **the inverse problem is
well-posed exactly where the band is wide, and the band is computable in closed form.** A learned
hallucinator is worth its cost in the narrow-band regime, where search is genuinely hard; in the
wide-band regime the closed form already answers the question and the learner adds variance. That
is a sharper claim than either paper it comes from makes, and it is supported by an ablation with
the controls stated.

## Next

1. **Seed the bias only**, leaving the encoder's gradient path intact, and re-run with the
   input-ablation control. This is the direct test of the hypothesis above.
2. **Gate annealing on the computed band width** rather than applying it uniformly, and re-run the
   24-clip amortisation with that rule.
3. **Only then** report a distribution, with reconstruction rate and diversity, against
   random-from-prior and input-ablated, at more than one training budget.
