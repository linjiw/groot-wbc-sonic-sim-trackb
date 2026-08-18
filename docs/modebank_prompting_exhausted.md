# Prompting cannot build the mode bank

The plan was to ask the generator for whole-route motion *styles* rather than mid-clip
events, on the evidence that styles come back reliably and events never do. Seven new modes
were written, generated at two seeds each, and gated before any GPU rollout. **None is
usable**, and the three reasons are distinct enough to close the question.

Baseline for comparison, from the same batch: a plain walk reference has a silhouette peak of
1.248 m, a half-width of 0.203 m and a trailing-foot apex of 0.210 m.

| mode | reachable | separation from the walk | why it fails |
|---|---|---|---|
| `crouch_shallow` | 0 / 2 | +402 mm | waist at a limit on 100% of frames |
| `crouch_squat` | 0 / 2 | +371 mm | waist at a limit on 100% of frames |
| `crouch_nofold` | 0 / 2 | +405 mm | waist at a limit on 100% of frames |
| `crouch_upright` | **2 / 2** | **−8 mm** | reachable, and does not crouch |
| `arm_tuck` | 2 / 2 | **−64 mm** | *wider* than a plain walk |
| `shoulder_turn` | 2 / 2 | −18 mm | wider than a plain walk |
| `high_knee` | 2 / 2 | +19 mm | lifts, but not enough to place a bar in |

## The crouch is a clean dichotomy

Every prompt that actually lowers the body — by 371 to 405 mm, which is enormous — folds the
waist past the G1's range on **every single frame**. The one phrasing that keeps the waist
within limits produces a motion 8 mm *taller* than a plain walk: it does not crouch at all.

This includes a prompt written specifically to forbid the failure: *"walks forward crouched
down with the head low, keeping the back straight and never folding forward at the waist"*.
It folds the waist on 100% of frames and reaches the deepest silhouette of the batch, 0.800 m.

There is no third outcome in four attempts. The generator's crouch is a waist fold, and the
instruction not to fold at the waist removes the crouch rather than changing how it is made.
**Prompting is exhausted here**, and the embodiment-aware retargeter is now required rather
than a fallback — the honest reading of a knee-driven crouch is that it has to be constructed,
because it is not in the model's distribution to be asked for.

## Asking for tucked arms produced a wider robot

`arm_tuck` was written for the lateral regime, which has no prompt targeting it at all. Both
its clips are perfectly reachable and both are **64 mm wider** than a plain walk — the
opposite of what was asked for, and a larger error than the margin any family would need.

That is the `side_step` failure again, one level further along. There, a behaviour that
genuinely occurred did not serve the geometry regime, because side-stepping is translation
rather than narrowing. Here the behaviour does not occur at all, and the silhouette moves the
wrong way. The recurring lesson is the same: **a prompt names an intent, and the regime needs
a geometric quantity, and the two are related only loosely.**

## What this changes

- **The mode bank cannot be built from text alone.** Of the three regimes, prompting supplies
  no adapted motion for any of them: overhead needs a crouch it cannot make reachable,
  lateral needs a narrowing it makes wider, floor needs a lift it makes too small.
- **The retargeter moves onto the critical path.** Minimise FK deviation from the generated
  crouch subject to joint limits, a waist-pitch penalty, preserved root path and foot
  contacts. The target is a silhouette peak near 1.00–1.10 m with the saturation gone — the
  generated clips already reach 0.80–0.89 m, so there is depth to give away.
- **The gate paid for itself immediately.** Fourteen rollouts were not spent, and the
  diagnosis is specific enough to act on: three modes say *retarget*, three say the prompt
  produced the wrong geometry, and one says it produced too little of the right one.
- **The corpus ceiling is unchanged**: two disjoint overhead families and one lateral, until
  retargeting works.

## What was not tested

These are references, not rollouts. `crouch_upright`, `arm_tuck`, `shoulder_turn` and
`high_knee` are all trackable-looking and might roll out perfectly well — they are simply not
*separated* from a walk, so they cannot anchor a counterfactual family. They remain valid
motions for the corpus, and nothing here says they are bad clips.

Two seeds per prompt is a small sample. It is enough to close the crouch question, where the
outcome is binary and identical across all six clips of the three deep variants, and it is
thinner evidence for `high_knee`, whose +19 mm might reach the bar at another seed or a
different phrasing.
