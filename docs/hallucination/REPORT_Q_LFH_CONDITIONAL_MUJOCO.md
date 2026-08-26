# Conditional q_LFH and MuJoCo Cross-Render Report

**Date:** 2026-08-22  
**Status:** conditional-inference mechanism retained; cross-render verified; E8 physics remains open

## Result

LFH now has an explicit conditional proposal mechanism for the user's core inverse problem:

`(executed nominal trajectory, executed adapted trajectory, critical condition)`
`-> q_theta(obstacle archetype, critical coordinate | trajectory-local support)`.

The deterministic support remains upstream of learning. It fixes the constraint axis, binding
keypoint, route phase, finite exposure, across-route extent, and engineering-safe coordinate
interval. The fitted component distributes mass only inside that interval. Every sample is marked
proposal-only; neither the distribution nor the MuJoCo renderer predicts a physics verdict.

The small-data implementation uses a source-balanced trajectory kernel rather than a neural model.
This is the appropriate complexity for four independent sources: each source receives equal prior
mass, nearby route/exposure/support features reweight the evidence, verified archetypes define the
empirical categorical component, and 10% explicit exploration prevents a closed support from being
mistaken for certainty.

## Leave-one-source-out mechanism check

The evaluation holds out every source in turn, excludes self-source evidence, and scores the
verified archetypes of the held-out trajectory.

| quantity | result |
|---|---:|
| independent motion sources | 4 |
| held-out verified labels | 12 |
| conditional mean log loss | 1.51373 nats |
| uniform-compatible log loss | 1.60944 nats |
| **feature-blind marginal-counting log loss** | **1.46416 nats** |
| top-3 recall | 11/12 = 91.67% |
| sampled coordinates inside exact support | 400/400 = 100% |

**Corrected 2026-08-26 — the conditioning mechanism is refuted at this scale.** The comparison
above originally reported only the uniform baseline and credited the 0.09571-nat gap to trajectory
conditioning. An audit replaced the kernel with a constant, holding the source balancing and the
0.10 exploration floor fixed, and scored **1.46416 nats** — better than the fitted model. The
trajectory kernel therefore *costs* 0.0496 nats relative to counting archetype frequencies; the
entire gain over uniform belongs to marginal frequency plus the exploration floor. A Laplace-
smoothed count baseline (alpha = 0.25) reaches 1.43543, and frequency counting wins at every
exploration setting tested.

Two of the three supporting numbers are true by construction rather than by fit:

- **Top-3 of 5 is not informative here.** Every source has `ibeam` and `shelf_plank` verified, so
  both sit at exactly 0.32 in every fold — 8 of the 12 labels are automatic hits. Feature-blind
  counting also scores 11/12, and shuffled features score 10–11/12. The single miss is source
  086's `hanging_panel`, the one label that would require genuine transfer.
- **400/400 in-support is an affine identity.** `sample()` bounds the quantile to [0, 1] and
  `coordinate_at()` maps [0, 1] onto the same held-out interval the check tests against. It
  cannot fail for any seed, or for any exploration width.

The mechanism is not leaking (held-out atoms are excluded, bandwidths are constants, the archetype
vocabulary is global). The problem is the **learning target**: archetype identity is close to
arbitrary appearance, and the executed pair does not determine it — while the quantity the pair
*does* determine, the placement window, is already available in closed form and is not learned.
Predicting archetype from trajectory features is predicting the one part of the scene the
trajectory does not constrain. See `docs/hallucination/REPORT_AUDIT_2026-08-26.md`.

For the three incomplete source-archetype crosses (`cf_005_056`, `lfh_089_crouch`, and
`lfh_090_crouch`), excluding already verified archetypes gives `hanging_panel` probability 0.95
and `hvac_duct` probability 0.05. This ranks the already planned E8 hanging-panel transfer first.
It is an experiment-prioritization result only; E8 still needs easy/context recertification and
hard-cell Isaac physics.

## MuJoCo cross-render

The E10b hard hanging-panel scene and both recorded SONIC/Isaac trajectories were independently
materialized in MuJoCo. The bridge parses the certified USDA axis-aligned cubes, compiles the
binding panel and four context objects as MuJoCo boxes, converts the declared IsaacLab 29-DoF order
to the audited MuJoCo order, preserves `wxyz` root quaternions, and applies every recorded state
with `mj_forward`.

| replay | authoritative Isaac verdict | max non-foot force | MuJoCo frames |
|---|---|---:|---:|
| nominal hard | rejected: disallowed robot contact | 106.23 N | 100 at 25 fps |
| adapted hard | accepted | 29.44 N | 100 at 25 fps |

The compiled MuJoCo boxes reproduce USDA centres within `2.22e-16 m` and full sizes exactly. Visual
inspection confirms that the same context-rich scene is present in both views, the nominal motion
stays taller at the binding station, and the adapted motion crouches under the panel. The combined
artifact is `docs/source/_static/lfh_progress/e10b-mujoco-cross-render.mp4`.

This is a **kinematic cross-render**, not a MuJoCo physics rollout. The collision/acceptance labels
remain the recorded Isaac verdicts. That distinction makes the video useful visual evidence
without creating a false cross-simulator physics claim.

## Proposal and paper positioning

The strongest defensible position is **counterfactual inverse scene design from motion**, not
generic scene generation:

1. Exact executed trajectory pairs define an identifiable critical-support interval.
2. A conditional proposal distribution selects scene semantics and difficulty inside that support.
3. Deterministic scene construction and keep-out checks preserve the intended causal obstacle.
4. Physics supplies necessity/sufficiency labels; rendering supplies auditable visual evidence.

This separation is the novelty: LFH does not ask a generator to hallucinate both geometry and its
own success label. It infers which scene distribution would make a demonstrated motion adaptation
necessary, then subjects that inverse explanation to simulator falsification.

The current evidence supports geometry-conditioned proposal feasibility, finite-exposure effects,
multi-archetype transfer, and one seed-matched multi-obstacle composition. It does not yet support
a natural-world obstacle density, simulator invariance, non-overhead generality, or downstream
learner improvement.

## Highest-value next work

1. Complete E8's hanging-panel transfers to form a four-source by three-common-archetype crossed
   matrix. Keep source/seed fixed and recertify easy/context before hard.
2. Run the registered equal-budget E5 comparison against uniform-feasible, DCS-ranked, and
   visual-only sampling. This is the experiment that can support the paper's learning-value claim.
3. Add more independent motion sources and non-overhead axes before replacing the interpretable
   kernel with a learned trajectory encoder. Select models by nested leave-one-motion-out testing,
   calibration, and support-violation rate—not training likelihood.
4. Treat simulator seeds and context layouts as explicit random effects. Report per-source and
   per-seed estimates instead of pooling failures into a single success rate.
5. Restore the NVIDIA driver, rerun the new inferred E8 scenes in SONIC/IsaacLab, and retain the
   MuJoCo bridge as a renderer/geometry cross-check. Do not promote it to a physics oracle without
   a separately registered cross-simulator dynamics study.
