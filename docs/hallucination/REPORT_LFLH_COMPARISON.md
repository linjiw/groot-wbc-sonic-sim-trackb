# Does a learned hallucinator recover the humanoid inverse set, or collapse onto it?

**Date:** 2026-08-26
**Code:** `gear_sonic/dataset_generation/hallucination/learned_hallucinator.py`,
`scripts/research/hallucination/run_lflh_comparison.py`
**Artifact:** `docs/hallucination/lflh_comparison.json` · **Tests:** 8

---

## 1. The question, and why it is answerable here

LfLH learns `q_psi(C | p)` by sampling obstacles, pushing them through a **fixed differentiable
planner**, and reconstructing the trajectory. Both LfLH and Dyna-LfLH report that the learned
distribution **mode-collapses** — it finds one obstacle configuration that explains the trajectory
and cannot cover the many others that would do equally well. Dyna-LfLH names this as a performance
limitation.

In the 2-D mobile-robot setting that is hard to quantify, because the true inverse set

```
C(p) = { C : p in argmin_p' J(p'; C) }
```

is not known. **In the humanoid overhead case it is.** The set of face coordinates that make the
nominal strike and the adaptation clear is exactly

```
[ R_adapted(u) + delta , R_nominal(u) - delta ]   at every route station u
```

So a learned distribution can be scored against ground truth instead of against itself. That is
what this experiment does.

## 2. Setup

Faithful to LfLH's shape, scaled to our constraint:

- **Trajectory `p`**: the pair of overhead reach profiles `R_nominal(u)`, `R_adapted(u)` sampled at
  24 route stations, measured with the same instrument the window solver uses.
- **Hallucinator `g_psi`**: three 1-D convolutions over the profile pair, temporal pooling, a fully
  connected head emitting Gaussian means and log-variances — the same architecture family as
  LfLH's, emitting two parameters (station, face coordinate) rather than ten ellipses, because the
  humanoid overhead constraint has two free parameters and not thirty.
- **Fixed decoder `d`**: a parameter-free differentiable surrogate that answers only what physics
  answers — at this face, does the nominal strike and does the adaptation clear — read from the
  reach profiles through a soft station gather. It has **no learnable parameters**, which is what
  stops encoder and decoder colluding, and a test asserts that.
- **Objective**: reparameterised sampling, reconstruction (`strike AND clear`), plus station and
  coordinate priors — LfLH's three terms.

**The surrogate is a study instrument, not a proposal mechanism.** Our real decoder is a frozen
SONIC policy in Isaac: not differentiable, ~50 s per evaluation, and not even deterministic
(LFH-E16b measured the same motion accepting at 3/3, 1/3 and 0/3 seeds). The LfLH training loop
cannot be run against it. The point of the surrogate is to ask whether the *method* would recover
the inverse set if the decoder were free — and the answer does not depend on the decoder's cost.

## 3. Result: the collapse reproduces, and it is severe

16 clips, 24 stations each, 800 steps, 400 samples per source.

| | start | end | contraction |
|---|---:|---:|---:|
| `sigma_coordinate` | **87.89 mm** | **0.248 mm** | **354x** |
| `sigma_station` | — | 0.0073 | — |

| sampler | valid placements | **occupancy of the feasible set** |
|---|---:|---:|
| learned LfLH-style hallucinator | **100.0%** | **0.7%** |
| uniform over the closed-form support | **100.0%** | **83.4%** |

`valid` is the share of sampled placements that actually separate the pair — LfLH's reconstruction
success. `occupancy` is the share of the exact feasible set's cells that any sample reaches.

**The learned hallucinator is perfectly correct and almost entirely non-diverse.** It finds a
placement that explains the motion and puts essentially all its mass there: 0.7% of the space of
placements that would have explained it equally well. Sampling uniformly over the interval we can
compute reaches 83.4% of that space at identical validity — a **119x coverage gap with no loss of
correctness**.

This is the same mechanism the user's 2-D toy shows, where the decisive parameter's variance
contracts to the model's floor while a weakly-coupled parameter keeps some spread. Widening the
distribution samples faces that fail to separate the pair, which raises expected reconstruction
loss, so the optimiser has a direct incentive to shrink. Nothing is wrong with the training; the
objective is simply not a coverage objective.

## 4. What this establishes for the paper

1. **The design choice is now evidence, not assertion.** We do not fit a distribution over face
   placement because, measured on our own data against known ground truth, fitting one costs 119x
   coverage for no gain in validity. Computing the support and sampling inside it dominates.
2. **The mode collapse is not specific to 2-D navigation.** It reproduces on humanoid reach
   profiles with a two-parameter obstacle, which is the smallest inverse problem where it could
   have gone away.
3. **It sharpens where a learned component *does* belong.** The failure is coverage of a
   *solvable* set. Where the set is not solvable in closed form — which station, which constraint
   axis, how many obstacles, dynamic obstacles — a learned distribution is the right tool, and the
   coverage metric here is how it should be judged.
4. **`valid_rate` alone is a misleading metric**, and this is the general lesson. A collapsed
   sampler scores 100% on it. Any future hallucinator in this project must report occupancy of the
   feasible set beside it, exactly as this comparison does.

## 5. Limits, stated plainly

- The surrogate decoder is not physics. It reproduces the *geometric* decision (strike / clear)
  and nothing about controller delivery, contact dynamics, or seed sensitivity. A hallucinator
  that looked good here would still have to face the real decoder.
- Single Gaussian only. A mixture or a normalizing flow would cover more; the comparison here is
  against the model class LfLH actually uses, and the point is that the *objective* rewards
  collapse regardless of the family's capacity.
- 16 clips, reference-side reach profiles, one commanded amplitude, overhead axis only.
- Occupancy is measured on a 24 x 24 cell discretisation of the feasible set; the absolute number
  moves with the binning, the 119x ratio does not.

## Reproduce

```bash
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/run_lflh_comparison.py --clips 16 --steps 800 --samples 400
```
