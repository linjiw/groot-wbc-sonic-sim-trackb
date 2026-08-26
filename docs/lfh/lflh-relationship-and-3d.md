# LfH, LfLH, LfH-CP, and what SweepCF actually is

**2026-08-26.** Written against the user's LfLH walkthrough. The purpose is to place our method
precisely relative to that line of work, say honestly where it is ahead and where it is behind, and
derive from that what the 3D extension has to be.

---

## 1. The four methods, in the terms of the walkthrough

| | inverse map | decoder | what is learned |
|---|---|---|---|
| **LfH** | hand-built: the executed C-space tube is free, the rest is obstacle | none | nothing |
| **LfLH** | `q_psi(C \| p)` — Gaussian over ellipse centres and sizes | **fixed differentiable planner** (Ego-Planner as a differentiable layer) | the hallucinator, by trajectory reconstruction |
| **LfH-CP** | critical configuration, then procedural realisation | classical planner | the critical constraint only |
| **SweepCF (ours)** | **closed-form interval** over face coordinate | **frozen SONIC + Isaac physics** | `D_phi` (executed reach), and nothing about the verdict |

The walkthrough's central point is correct and is the thing most easily missed: **the fixed decoder
is what gives hallucinated obstacles their semantics.** If the decoder were also trained, encoder
and decoder can collude — the encoder emits numbers with no geometric meaning and the decoder
learns to invert them, reconstruction loss goes to zero, and the "obstacle distribution" describes
nothing. Pinning the decoder to a planner whose cost function is fixed means an obstacle parameter
can only move the output by actually changing planning cost.

We inherit that principle and pay a different price for it.

## 2. Our decoder is physics, and it is not differentiable

`d(C)` for us is: author a USD scene, run a frozen SONIC policy in Isaac Lab, and read the verdict.
Three properties follow, and they drive every design difference.

**It is not differentiable.** There is no `d p_hat / d C`. The LfLH gradient chain

```
dL/dpsi = dL/dp_hat * dp_hat/dC * dC/dpsi
```

has no middle term for us. We cannot train a hallucinator by trajectory reconstruction, and no
amount of engineering makes a contact-rich rigid-body simulation with a frozen neural controller
into a differentiable layer we would trust.

**It is expensive.** One evaluation is a GPU rollout, roughly 50 s contended. LfLH can afford to
sample ten obstacle configurations per trajectory per training step. We can afford four cells per
family. This is a factor of thousands, and it is the single most important practical difference.

**It is stochastic.** LFH-E16b measured it: the same motion at the same amplitude accepted at 3/3
simulator seeds for one clip, 1/3 for another, and 0/3 for a third. `d(C)` is not a function.

A learned `q_psi` trained by reconstruction through this decoder is therefore not merely hard, it
is the wrong shape of solution. What we did instead was ask where the inverse set is available
**in closed form**, and it turns out that for the overhead axis it is:

```
W = [ R_adapted + delta_clear , R_nominal - delta_strike ]
```

an exact interval, computed from two executed reaches, with no sampling and no fitting.

## 3. Why this dodges the mode-collapse the walkthrough diagnoses

The walkthrough's toy shows `sigma_cy -> 0.0186` — the parameter that decides which side the
obstacle goes collapses almost to the floor, because widening it samples obstacles that push the
planner the wrong way and raise expected loss. Dyna-LfLH reports the same failure and calls it a
performance limitation.

We do not fit a variance, so there is nothing to collapse. The support is computed exactly and the
proposal samples inside it. The audit noted that our "400/400 samples in support" check is an
affine identity and cannot fail — presented as a criticism of that check as *evidence*, which it
is. But as a *property* it is the whole point: staying inside an exactly-known support is trivial,
whereas an amortised Gaussian has to learn it and provably contracts while trying.

The honest accounting is therefore:

- **LfLH learns a distribution and loses coverage.** Its σ contracts to the safest single
  explanation.
- **We compute the support and keep full coverage of it** — but only along the one parameter we
  have solved, and only on the one axis where we have solved it.

Ours is a narrower claim, more exactly held. That is the trade, and it should be stated that way
in the paper rather than as a straightforward improvement.

## 4. The three LfLH losses are our three gates

The correspondence is exact enough to be worth a table, because it shows we did not skip the
regularisers — we moved them from a loss to a gate.

| LfLH loss | what it prevents | our gate |
|---|---|---|
| `l_rec = ||p - d(C)||^2` | obstacles that do not change the plan | **necessity**: the hard cell's nominal must be *rejected*, with contact attributed to the binding face |
| `l_coll` (clearance to the target plan) | obstacles sitting on the trajectory, which change the plan but make it unexecutable | **feasibility**: Tier-2 keep-out at authoring, then the hard cell's adapted motion must be *accepted* with zero external contact |
| `l_prior` (plausible location and size) | obstacles at the workspace edge, or with absurd radius | **realism**: archetype dimension priors, scored on non-binding dimensions only |

The walkthrough's summary — reconstruction makes the obstacle *matter*, collision keeps it from
*destroying* the target path, prior keeps it *sane* — is precisely our accepted/rejected/accepted
pattern plus keep-out plus the dimension check. Our version is discrete and adjudicated by physics
rather than continuous and differentiated, which is worse for optimisation and better for trust.

## 5. What we have that LfLH does not, and what it has that we do not

**Ours, additionally:**

- *Regret in closed form.* `regret = xi * |W|`, identically the adapted motion's clearance under
  the face. LfLH's reconstruction loss says the trajectory is *near-optimal* under the sampled
  obstacles but cannot say by how much. We can, exactly, with no extra rollout.
- *A necessity/regret budget.* `regret + necessity = delta_strike + |W|`, invariant in placement.
  Neither improves except through a deeper delivered adaptation. This is a statement about the
  inverse problem itself, not about a particular model.
- *A verified appearance-freedom result.* LFH-E18 put four visually distinct obstacles at one
  coordinate inferred from one pair and got the same 2x2 from all four, with the adapted cell's
  endpoint error identical to fifteen decimal places. That is LfH-CP's phase-2 claim, measured.

**Theirs, which we lack:**

- *Genuine multimodality.* Their inverse set contains "cylinder ahead", "small cylinder right",
  "short wall", "two narrow obstacles", "a dynamic obstacle". Ours is one scalar on one axis at one
  station. We do not represent the alternative explanations at all — we fix the station at route
  progress 0.55 and the axis to overhead, and call the rest unsupported.
- *An end-to-end trainable pipeline.* Their stage two trains a deployment planner on ray-cast
  hallucinated scenes. Our equivalent (claim 5 / E5) has never run.
- *Cheap iteration.* They can ablate the decoder, the prior, the collision term in an afternoon.
  Every ablation of ours costs GPU-hours and can be lost to another user's job.

## 6. The 3D extension: what actually blocks turning motions

This is the concrete answer to "how do we handle walk-left-then-straight, or a hip-led turn".

The representation already admits it. `docs/lfh/lfh.md` §2.1 defines the route frame
`T(u) = [tangent(u), lateral(u), z]` and a face normal `n_world = T(u*) n_route`. The **code does
not**. Three places assume the world axes:

1. `overhead_face_reach(tracks, station, route_axis, ...)` takes `route_axis` in `{"x", "y"}` and
   builds an **axis-aligned** rectangle. For a curved route the face is therefore mis-oriented by
   the angle between the world axis and the executed tangent at the station.
2. Scene authoring emits axis-aligned boxes with no rotation, and `stage_geometry` **refuses**
   rotations outright as a safety measure.
3. Keep-out validates in world axes for the same reason.

`run_case_study.py` measures the first directly, per clip: the angle between the authored face
normal and the executed tangent at the binding station. On near-straight walks it is a few degrees;
on the curving and turning clips it is tens of degrees. A face mis-oriented by 40 degrees does not
present its finite along-route extent to the body at all — the crossing footprint is wrong, so both
reaches are wrong, and the window computed from them is meaningless. **This, not physics, is why
the corpus filters to straightness >= 0.95.**

The fix is well-scoped and is the next piece of engineering:

- **Reach.** Rotate capsule coordinates into the route frame at `u*` before the rectangle test.
  `capsule_height_over_rectangle` is frame-agnostic — the transform is linear, so a transformed
  capsule is still a capsule, exactly as the fixed `lateral_face_reach` already does when it
  projects onto the per-frame heading. No new mathematics.
- **Authoring.** Emit a yaw on the binding prim, and teach `stage_geometry` to *read and verify* a
  rotation rather than refuse it. The refusal must be replaced by a check (the authored yaw equals
  the spec's route-frame yaw to a tolerance), not simply removed — it exists because an unnoticed
  rotation silently invalidates a measurement.
- **Keep-out.** Same transform, applied to both swept volumes.

Until then, every result we have is conditioned on a near-straight route, and the paper must say
so. The case study quantifies exactly how large that restriction is.

## 7. Obstacle inventory in 3D: which items can realise a critical point

The user's question — given a list of sample obstacles of different sizes and shapes, how do we
place them in 3D — has a precise answer in this framework, and it is a **filter followed by a
designed distribution**, not a learned placement.

An inventory item `o` with geometry `G_o` is *admissible* for a critical atom `a` at normalised
position `xi` when all four hold:

1. **It has a face of the right kind.** `G_o` contains a planar face whose outward normal is
   `-n_route` (for overhead, downward), which can be positioned with its underside at
   `a.coordinate_at(xi)`.
2. **That face spans the crossing footprint.** Its extent covers the finite along-route exposure
   and the across-route width the body occupies while crossing. If it does not span, the body
   passes beside the face and the window does not apply.
3. **Its remaining geometry clears both swept volumes.** Every part of `G_o` other than the binding
   face — jambs, webs, brackets, skirts — must lie outside `dilate(S(tau_0) U S(tau), delta_ko)`.
   This is the existing Tier-2 keep-out, applied to the whole item.
4. **Its non-binding dimensions are plausible.** The dimension prior, as now.

Conditions 1–2 are what make an item *usable*; 3 is what stops it from becoming a second,
unlabelled cause; 4 is the LfLH prior in another form. All four are computable per item without
physics, which matters because the inventory can be large and the decoder cannot.

The distribution then factorises exactly as it does today:

```
q(S | tau_0, tau) = q_bind(xi)                      # regret budget, closed-form support
                  * q_item(o | admissible(a, xi))   # designed, source-balanced
                  * q_pose(free DOF of o | o, a)    # rotation about the face normal,
                                                    #   slide along the face, non-binding depth
                  * q_ctx(G_ctx | tau_0, tau, o)    # keep-out-certified clutter
```

`q_pose` is the genuinely new term for 3D and is worth naming: once an item's binding face is
pinned to the coordinate, it usually retains free degrees of freedom that do not affect the
counterfactual — an I-beam can slide along its own length, a panel can rotate about the vertical if
it still spans the footprint. Those are free diversity, exactly analogous to LfH-CP's `g(K)`
generating unlimited realisations through a fixed critical point, and E18 is the first evidence
that this freedom is real rather than assumed.

**What we should not do** is learn `q_item` from trajectory features. The audit already settled
that: archetype identity is the part of the scene the executed pair does not constrain, and a
kernel conditioned on trajectory features scored *worse* than counting. The same argument applies
to any inventory: which object realises the face is not determined by the motion, so a model asked
to predict it is being asked to predict noise.

## 8. Where a learned component genuinely belongs

Three places, in order of how well the evidence supports them:

1. **`D_phi`: reference clip -> executed reach.** Already fitted and validated — 39.2% RMSE
   reduction against identity under leave-one-motion-out CV over 46 clips, holding up on 18 clips
   it never saw. This is our analogue of the differentiable decoder: not the decoder itself, but a
   cheap differentiable *surrogate* of the one quantity the expensive decoder is needed for.
2. **Trackability: will this commanded amplitude accept, and at what seed rate?** E16b showed
   acceptance is neither a property of the motion nor monotone in a simple way, and it is the gate
   that actually stops the pipeline. A model here would pay for itself immediately; the corpus is
   `delivery_corpus.csv` and is now 154 cells.
3. **The station and the axis.** Currently fixed at route progress 0.55 and overhead by fiat. Once
   oriented faces exist, *where along the route* and *from which direction* the binding face should
   come is a real inverse-problem question with genuine multimodality — and that is the point at
   which a LfLH-style distribution, or LfH-CP's critical-set factorisation, becomes the right tool
   rather than an unnecessary one.

Note the ordering. We should reach for a learned obstacle distribution when we have a
multi-parameter inverse set we cannot solve in closed form — not before. Today we have a
one-parameter set that we can.
