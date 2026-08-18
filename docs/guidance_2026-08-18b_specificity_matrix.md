# Guidance: the 3×3 specificity matrix is the decisive physics experiment

Apparatus frozen at `sweepcf_apparatus_v0.1` (`fb3c956`). The next research unit is not
another validator, and test count stops being a measure of progress here.

## The claim ladder, scoped more tightly

"Established" was doing too much work. Corrected:

| claim | status |
|---|---|
| A swept-volume window exists | demonstrated across 3 independent families |
| Geometry changes the physics outcome | demonstrated across 3 families |
| Start-pose outcome robustness | demonstrated on **one** family (`cf_005_056`), 3/3 jitters, 12/12 cells — **cross-family replication pending** |
| The scene reverses the preferred behaviour | established only once matched operators pass stages 2 and 3 |
| A learner uses geometry | not started |
| It generalises to scene-first environments | not started |

## Preference reversal does not need a calibrated cost

Use a lexicographic rule, which sidesteps the uncalibrated weights entirely:

```
m*(S) = argmin_m  D(m, m₀)   subject to   m succeeds in S
```

with `D(m₀, m₀) = 0` and `D(T_k(m₀), m₀) > 0` for any operator `T_k`. In the easy scene both
succeed and the nominal has zero edit cost, so it wins; in the hard scene the nominal fails,
so the adaptation wins. That is a strict reversal without appealing to 0.776 against 1.091. A
continuous cost is still wanted later, for comparing two *different* adaptations — but it must
not block this claim.

## Stage 2: three sentinels, not a batch

In order: **strongest arm-tuck**, **strongest local-crouch**, then the **weak arm-tuck as a
refusal control**. The third is the important one — it should show that an operator can be
physically executable and still lack the geometric effect a family needs.

Two concepts, kept apart:

```
operator_trackable      the motion survives physics
family_eligible         its executed effect is large enough to build a family on
```

Per pair, report the paired differences rather than aggregate acceptance:

```
reference_effect_m, executed_effect_m, effect_retention_ratio
root_xy_p95_difference, root_yaw_p95_difference
gait_phase_difference, foot_contact_transition_difference
outside_window_joint_difference, outside_window_envelope_difference, recovery_error
new_self_contact, foot_skate, joint_limit_margin, tracking_error_difference
```

Retention is `(E_nom^exec − E_adp^exec) / (E_nom^ref − E_adp^ref)`. It need not equal 1. What
must hold: the effect does not reverse sign, it exceeds the natural spread of repeated nominal
rollouts, route and duration and gait phase are unchanged, and the motion returns to nominal
after the window.

Duration was previously the strongest single predictor of rejection. Matched operators hold
duration, route and gait phase fixed, which removes that confound — so results must be read as
paired differences.

## The lateral operator needs signed envelopes

Symmetric half-width `w(s) = max_b |y_b(s)|` dilutes the effect: arms swing out of phase, so at
a given station one side may be wide and the other not. Compute both sides:

```
w_L(s) = max_b  y_b(s)        w_R(s) = −min_b  y_b(s)
```

and jointly optimise station **and side**:

```
(s*, side*) = argmax over s, side  [ w_nominal,side(s) − w_tuck,side(s) ]
```

The first lateral family should use a **one-sided** obstacle — a rack, a cabinet edge, a wall
protrusion — not a symmetric corridor. For the clip that only narrowed 21 mm, report
`left_reduction`, `right_reduction`, `nonarm_width_floor`, and the critical capsule before and
after; if one side is still insufficient, the generator refuses it rather than shrinking the
safety margin to manufacture a family.

## The 3×3 specificity matrix

Once both operators survive, build one shared candidate bank from a single nominal motion —
`{nominal, local_crouch, local_arm_tuck}` — and run it through easy, overhead and lateral
scenes:

| candidate | easy | overhead | lateral |
|---|---|---|---|
| nominal | success, **preferred** | fail | fail |
| local crouch | success | success, **preferred** | fail or non-preferred |
| local arm tuck | success | fail or non-preferred | success, **preferred** |

This is stronger than two separate 2×2s, because it shows geometry deciding **which part of
the body** must change: nothing in the open, the legs and root under a shelf, only the arms
past a one-sided obstruction. Overhead and lateral are orthogonal embodiment adaptations; two
done cleanly beat three with a fragile high-knee.

Boundaries must be mined from the **executed** envelope, not the reference — SONIC may retain
60%, 80% or 100% of the intended effect, and what collides is the executed body.

## The learner, in parallel on CPU

Score compatibility, do not classify scene identity:

```
f_θ(S, m) → [ p̂_success , d̂_min ]
```

Score every candidate, drop the predicted-infeasible, take the smallest edit among the rest,
and **allow abstention** when none is feasible rather than forcing a collision.

- **Phase A, privileged geometry** — route-relative ceiling-height profile, signed left/right
  clearance profile, and each candidate's own envelope profile. A small MLP or even logistic
  regression. This validates the data structure and the supervision principle.
- **Phase B, ego depth** — depth history or a local point cloud, which is what makes "uses
  geometry from deployable perception" true. Depth before RGB: these scenes are primitive, and
  colour would be a shortcut to asset identity.

Controls, all required: **no-scene**, **scene-shuffle**, **candidate-order randomisation**,
**family holdout**, **nominal-motion holdout**.

## Three families are a smoke test, not evidence

They are enough to check the loader, candidate-order leakage, whether the model can overfit,
whether scene-shuffle bites, and whether the family split leaks. They are **not** enough for a
go/no-go: three methods tying on three families is underpowered, and cannot distinguish "no
signal" from "not enough data", "learner too weak", or "wrong input".

First paper-level learning result needs, at minimum: **24–30 verified families** across
overhead and lateral, 6+ distinct nominal motions, family-level split, nominal-motion holdout,
5+ training seeds, and family-level bootstrap intervals. A realistic first matrix from the six
Stage-1 candidates already in hand:

```
6 nominal motions × 3 scene types × 3 candidates = 54 rollouts
adding a second difficulty per hard regime:  6 × 5 × 3 = 90
```

That serves claims 5 and 6 more directly than 120 undifferentiated calibration rollouts.

## The primary metric is triplet choice accuracy

Per nominal motion, all three must be right:

```
TCA_i = 1[ m̂(S_easy) = nominal  ∧  m̂(S_overhead) = crouch  ∧  m̂(S_lateral) = tuck ]
```

Always-crouch scores zero. Always-adapt scores zero. Detecting an obstacle without choosing
*which* adaptation scores zero. Report alongside it: false-safe rate, unnecessary-adaptation
rate, missed-feasible rate, choice cost regret, scene-shuffle degradation, no-scene gap. All
intervals bootstrapped over families, never over frames.

## The sealed scene-first set

1. Split into `scene_first_dev` and `scene_first_test_sealed`. If all 30 are already declared
   test, sample a separate small dev set rather than carving one out.
2. **Never drop a scene because its physics outcome was unfavourable.** Report the true
   distribution: nominal-only, crouch-only, tuck-only, multiple-safe, none-safe. If most are
   all-safe or all-fail, that is a fact about the scene-first distribution.
3. Declare supported regimes *before* training. Floor scenes may be marked `unsupported_ood`
   from pre-existing metadata, never after seeing outcomes.

Every family must also satisfy `t_first_visible < t_adaptation_onset < t_bottleneck`, or it
demonstrates map-conditioned selection rather than ego perception.

## GPU order

1. arm-tuck Stage 2, crouch Stage 2, weak arm-tuck control
2. matched overhead 3×3, matched lateral 3×3, both with registered jitters
3. six-motion easy/overhead/lateral matrix; three-regime learning smoke; no-scene and shuffle
4. **stratified** calibration — 24–36 samples across near-threshold, medium and large windows
   in both regimes, *before* spending 120 on a predictor that may be systematically wrong

## Go / no-go

- **A, operators survive** — each operator trackable on ≥4/6 Stage-1 candidates, executed
  effect large enough for a usable window, route/gait/recovery intact. Below that, keep the
  working operator and fill the other regime from existing natural pairs rather than
  endlessly repairing a retargeter.
- **B, matched specificity exists** — one overhead and one lateral 3×3 as predicted, holding
  under registered jitters.
- **C, learning signal exists** — SweepCF beats decorated and random on family-level choice
  accuracy and false-safe, without raising unnecessary adaptation; scene-shuffle clearly
  degrades; no-scene is clearly worse.
- **D, scene-first generalisation exists.**

Only C and D are the paper's learning evidence.

**A correction to something previously written here:** "if all three are equal it is still a
paper" is not automatic. Equal on three families is merely underpowered. Equal on 24–30
families with motion holdout, a sealed scene-first test and an analytic oracle control would
be a meaningful negative result — *when does counterfactual supervision not beat explicit
geometric collision reasoning* — and only then.

## Not now

Floor-level adaptation waits until claims 5 and 6 are answered.
