# Guidance, 2026-08-17: settle the path gate before scaling

Received after the 13-behaviour review. Recorded verbatim in substance, with my status notes
added under each item as they are worked.

## The decision to make before the next batch: what is the path gate for?

Acceptance is a function of duration, not behaviour: 83% at 1.2 s falling to 64% at 4.8 s,
r = −0.47, with path-tracking error causing 40 of 54 rejections. The tracker starts on its
reference and drifts. Scaling to 1,000 episodes under this gate bakes in two biases — the
accepted corpus over-represents short clips and early-episode dynamics, and behaviour
families get penalised for prompt length rather than difficulty.

Step back to what the gate protects. In scene-around-motion episodes the clutter is built
around the executed corridor, so a drifted-but-stable episode is collision-free,
well-supported, and the scene fully explains the path the robot actually took. The Kimodo
reference was a means of producing behaviour, not the label.

1. **Re-anchor labels to the executed trajectory for scene-around-motion data.** Route, goal
   and language describe what the robot *did*; path-vs-reference error becomes a
   tracking-quality diagnostic in provenance, not an acceptance gate.
2. **Replace absolute p95 path error with a drift-rate gate** (m/s of divergence, or error
   growth relative to duration), calibrated from the 83%→64% curve. Catches genuinely
   unstable tracking without punishing length.
3. **Gate differently by generation direction.** For the scene-first subset the *planned*
   route is the label, so there the right gates are goal-region success plus drift rate.
   Corridor-hugging p95 over a whole episode is the wrong instrument in both directions.

Do the rework first, then re-grade the existing 157 for free before spending GPU nights.

## Three integrity items before anything freezes

1. **The page contradicts itself.** The stat box says worst accepted scene contact is 0.0 N
   while the caveat reports two accepted episodes with non-zero lateral contact, one at
   2.8 N — on a page whose stated guarantee is that recomputed numbers cannot disagree with
   the data. Almost certainly the review analyser and the acceptance gate are two
   implementations of "lateral scene contact" that diverge. Unify into one shared function,
   then re-grade.
2. **Quarantine those two episodes until the 2.8 N is explained.** Leading suspect: the
   documented decomposition edge case where a body touching self *and* scene in the same
   frame leaves an unpaired residual, or a pairing-tolerance failure in a three-body frame.
   Log the frame, bodies, and nearest-miss pairing at looser tolerance.
3. **The pinned GR00T loader smoke** is the last M0 box and a silent schedule risk if it has
   not run.

## Cheap wins from assets already owned

- **The squat failure is a finding, not just a bug.** Kimodo's human prior commands a waist
  fold the G1 cannot make, the joint clamps, and the reference itself becomes
  self-intersecting — which is why the joint-limit prefilter did not fire. Run
  capsule–capsule self-intersection on the reference kinematics using the existing 29-capsule
  collision model, and the whole failure class dies before touching the GPU.
- **The 0.55 m cliff defines a crouch-depth envelope** (0-for-12 below it, 88–90% in
  0.55–0.66 m). Do not hard-reject low-root references or the envelope can never be shown to
  have moved; route them into a quota-capped **frontier** bucket so envelope data keeps
  accruing cheaply.
- **The review card captions are informal specs** for the per-behaviour success predicates
  §9.4 always required. Turn them into code — pause: root speed below ε for ≥T; duck: head
  height dips through the shelf's z-band while inside its footprint; stop: deceleration
  without overshoot into a hold window. With 13 behaviours these predicates are the
  benchmark's success criteria, and they replace eyeballs for the episodes nobody reviewed.

## Shaping the next batch

- **Rebalance the prompt sampler by inverse frequency.** 88 of 157 attempts are forward
  walks, while walk-to-a-stop, turn-in-place and walk-and-reach sit at n = 2–3. Target a
  floor of 20–30 accepted episodes per family, using each family's measured acceptance rate
  to set attempt quotas.
- **Add the two missing Phase B label axes** before the big batch: goal-conditioned task
  language derived from the executed endpoint plus the nearest labelled proxy object (current
  prompts are motion-style strings, not tasks a navigation VLA is commanded with), and the
  scene-first subset for genuine geometry→action causality.
- **New evaluation axis unlocked by the 13-behaviour library:** a held-out-*behaviour* split
  (train on ten families, test on three) alongside held-out scenes and density.

## Status notes

- **Integrity 3 is already closed.** The pinned GR00T loader smoke ran against the real
  `ShardedSingleStepDataset` at `ab88b50c` on 2026-08-16: `ok: true`, action horizon 40, ego
  view [1, 480, 640, 3]. Env `/data/robotixx/envs/groot_loader`. It is not a schedule risk.
