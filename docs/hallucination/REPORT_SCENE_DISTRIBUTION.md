# q(S | executed pair): the scene distribution and how to score it

**Date:** 2026-08-26
**Code:** `gear_sonic/dataset_generation/hallucination/scene_distribution.py`,
`scripts/research/hallucination/evaluate_scene_distribution.py`
**Tests:** `tests/dataset_generation/test_hallucination_scene_distribution.py` (10)

---

## 1. What is identifiable, and what is sampled

The inverse problem as posed — *given a humanoid motion, produce the scene that explains it* — is
badly non-identifiable. A crouch is explained by a beam, a lintel, a low branch, a table edge, or
by nothing at all: an empty room admits the motion too. No amount of model capacity fixes this,
because the information is not in the input.

Conditioning on a **counterfactual pair** does fix it. Given an executed nominal trajectory and an
executed adaptation of it, the set of overhead faces for which the nominal strikes and the
adaptation clears is a closed-form interval:

```
W = [ R_adapted + delta_clear , R_nominal - delta_strike ]
```

There is nothing to learn here and nothing to sample: it is an order statistic over two executed
reaches. So the model separates cleanly into a part that is *computed* and a part that is
*sampled*:

| | object | status |
|---|---|---|
| Support | `CriticalAtom` — station, route axis, finite extent, `W` | deterministic, closed form |
| Proposal | `SceneDistribution` — where in `W`, which archetype, how much context | sampled |
| Verdict | Isaac physics through the frozen controller | never modelled |

**Archetype choice is not conditioned on the trajectory.** The 2026-08-26 audit found that such
conditioning scored *worse* than feature-blind counting (1.51373 vs 1.46416 nats), because the
executed pair constrains where a face must be and not what it looks like. The archetype component
is therefore an explicit source-balanced categorical with a 10% exploration floor, documented as a
design choice rather than presented as a fit.

## 2. Regret and necessity have closed forms, and share one budget

The user's membership set is

```
S_{eps,delta}(tau) = { S : Feasible = 1, Regret <= eps, Necessity >= delta, Realism >= r0 }
```

`Regret` was the term the pipeline had never measured. It turns out to need no measurement at all.

Let `xi` be the normalised position of the face in the window, `c = L + xi*|W|`. Under the
lexicographic cost `J(tau; S) = infinity if infeasible else D(tau, tau_0)` with `D` monotone in
edit depth, the cheapest feasible edit is the one that *just* clears the face, so:

```
Regret(xi)    = xi * |W|                       (verified identically equal to the adapted
                                                motion's clearance under the face)
Necessity(xi) = delta_strike + (1 - xi) * |W|  (the nominal's penetration of the face)
```

Both were checked against their definitions in `test_regret_equals_xi_times_window_and_the_adapted_clearance`
and confirmed numerically on the authored `ladder_138` family:

| `xi` | regret | necessity | sum | adapted clearance |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00 mm | 52.12 mm | 52.12 mm | 18.04 mm |
| 0.25 | 8.52 mm | 43.60 mm | 52.12 mm | 26.56 mm |
| **0.50** | **17.04 mm** | **35.08 mm** | **52.12 mm** | 35.08 mm |
| 0.75 | 25.56 mm | 26.56 mm | 52.12 mm | 43.60 mm |
| 1.00 | 34.08 mm | 18.04 mm | 52.12 mm | 52.12 mm |

Three consequences worth stating in the paper.

1. **`xi` is normalised regret.** The docs describe it as "difficulty" or "position in the window";
   it is exactly the fraction of the window spent on an edit deeper than the scene required.
2. **Regret + necessity is invariant.** Their sum is `delta_strike + |W|`, fixed by the executed
   window. No placement strategy improves both; **only a deeper delivered adaptation does.** This
   is the precise sense in which the delivered amplitude (LFH-E12) and the engineering margin
   (LFH-E16) are the quantities that matter — they are the only things that widen the budget.
3. **The corpus convention costs half the window in regret.** Placing the hard face at the window
   centre, as every verified family does, sets `xi = 0.5` and therefore spends 17.0 mm of
   `ladder_138`'s 34.1 mm window on regret. What it buys is *execution margin*: at `xi -> 0` the
   adaptation clears by exactly `delta_clear` and any run-to-run variability turns a clear into a
   strike. That trade — explanatory tightness against execution robustness — is the real one, and
   it is now explicit and tunable via `regret_budget_mm`.

## 3. The metric, and the baselines that make it mean something

The refuted archetype model is the reason this section exists: a proposal distribution is worth
only what it beats. Four samplers are scored on identical atoms, 2000 samples each, admitting a
proposal when it is in support, geometrically feasible for the adaptation, and satisfies
`regret <= 15 mm`, `necessity >= 20 mm`, `realism = 1.0`.

| sampler | in support | **admitted** | median regret | median necessity |
|---|---:|---:|---:|---:|
| **`q_LFH` (shaped, regret-budgeted)** | 100.0% | **100.0%** | **7.7 mm** | **42.9 mm** |
| uniform in support | 100.0% | 45.9% | 16.3 mm | 33.6 mm |
| route-midpoint station | 19.4% | 8.4% | 16.8 mm | 33.5 mm |
| random placement (no pair knowledge) | 8.1% | **3.8%** | 16.3 mm | 34.5 mm |

Read it in three steps, because each row isolates one thing:

- **Random placement admits 3.8%.** Drawing a face height from the range the corpus's obstacles
  actually occupy, with no knowledge of the pair, almost never produces a counterfactual: the face
  either clears both motions or strikes both. This is the "put an object next to the motion"
  strawman, quantified.
- **Knowing the support is most of the work** — uniform-in-support jumps to 45.9%. That step is the
  closed-form window, which is *computed*, not learned. The method's power lives here, and it is
  honest to say so.
- **Shaping adds the rest**, taking admitted proposals to 100% while halving median regret. The
  shaping is a regret budget, not a fitted density; it is a design choice with a stated objective,
  and its effect is exactly what the closed forms in §2 predict.

Everything in this table is a **geometric necessary condition**. None of it is a physics verdict,
and every promoted scene still goes through the unchanged Isaac scorer.

## 4. Realism, scored rather than asserted

`realism_score` checks an archetype's **non-binding** dimensions against a published-dimension
prior (`ARCHETYPE_DIMENSION_PRIORS`). The binding coordinate is deliberately excluded: it is set by
the trajectory, so scoring it would measure the solver rather than the plausibility of the object.
This is a crude dimension check by design — it answers "would a person call this a door lintel",
which is the part of realism this corpus can defend without a user study.

## 5. Limits

- Overhead axis only. Lateral support exists on the repaired instrument but is refused by the
  imported margin (see the register's lateral re-derivation entry).
- The geometric feasibility flag is a necessary condition read off swept volumes, not a verdict.
  The corpus's own proxy-versus-physics agreement is 16/20, and where they disagree physics wins.
- Four atoms is a small corpus. The baselines are what make the comparison meaningful at this
  scale; the absolute admitted rate of a shaped sampler on four atoms is not a population claim.
- The scorecard says nothing about whether a *learner* benefits. That is the separate E5 question,
  and it is not answered here.

## Reproduce

```bash
env -u PYTHONPATH ~/miniconda3/envs/env_isaaclab/bin/python \
  scripts/research/hallucination/evaluate_scene_distribution.py
```
