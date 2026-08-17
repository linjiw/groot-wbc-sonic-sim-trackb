# Screening untrackable motions before simulating them

**Claim.** Joint-limit saturation in a generated motion predicts how hard the tracker will
drive the robot into itself, well enough to skip motions that cannot be tracked before
spending a rollout on them.

**Status: supported.** Correlation **0.933** over 11 distinct source motions (78 rollouts),
measured on the representation available before any GPU time is spent. The threshold is read
off the data, and the screen reports by default rather than skipping.

## Why saturation would predict anything

Kimodo generates human motion, which is then mapped onto the G1's joint ranges.
Configurations the robot cannot reach are clamped at their limits. Clamping does not merely
approximate the pose — it changes it, and a crouch clamped at the hips puts the thigh
through the pelvis. The physics then reports enormous self-contact force, which is a real
consequence of an unreachable reference rather than an artifact of a strict gate.

The mechanism is: **unreachable configuration → clamped joints → self-intersection →
self-contact force → rejection.** Saturation measures the first link, acceptance the last.

## The scale trap

The same motion measures differently depending on which representation you screen:

| Motion | Raw Kimodo CSV | Recorded `reference_g1_qpos` |
|---|---|---|
| `np_fastwalk` | 0.0000 | 0.0422 |
| `np_pausewalk` | 0.0000 | 0.0214 |
| `np_crouchwalk` | 0.0444 | 0.1391 |

SONIC transforms the reference on the way in, so the recorded values sit on a different
scale. This matters because the first validation done here used recorded references, and
applying that threshold (0.058) to raw CSVs would have **passed the crouch** — raw 0.0444 is
below 0.058 — the exact motion that produced 9686 N of self-contact. Two constants are
defined, and they are not interchangeable.

Everything below is the **pre-simulation** scale, because that is where screening pays.

## What was measured

Saturation is the fraction of (frame, joint) cells within 1% of a joint's range end. Joint
ranges come from the G1 MJCF, resolving `default` class ranges as well as explicit ones —
only 18 of the 29 joints carry an explicit `range`, and reading only those leaves a third of
the robot unscreened while appearing to work.

Each source motion is paired with the peak self-contact force over every rollout that used it:

| Source motion | Pre-sim saturation | Rollouts | Peak self-contact | Accepted |
|---|---|---|---|---|
| `np_fastwalk` | 0.0000 | 1 | 202 N | 1/1 |
| `np_pausewalk` | 0.0000 | 1 | 35 N | 1/1 |
| `07_text_terrain` | 0.0000 | 2 | 104 N | 0/2 |
| `05_root_path` | 0.0006 | 10 | 93 N | 0/10 |
| `04_ee_constraint` | 0.0016 | 8 | 61 N | 0/8 |
| `06_root_waypoints` | 0.0021 | 12 | 27 N | 0/12 |
| `01_single_text_prompt` | 0.0030 | 18 | 318 N | 16/18 |
| `03_full_body_keyframes` | 0.0071 | 16 | 283 N | 14/16 |
| **`02_multi_text_ee_constraint`** | **0.0116** | 8 | **712 N** | 0/8 |
| `08_text_object` | 0.0199 | 2 | 1101 N | 0/2 |
| `np_crouchwalk` | 0.0444 | 1 | **9686 N** | 0/1 |

Correlation over distinct source motions: **0.933**. Every motion that produced an accepted
episode sits at or below 0.0071.

## Where the threshold comes from

`PRESIM_SATURATION_LIMIT = 0.009`, placed in the observed gap between:

- **0.0071** (`03_full_body_keyframes`), peak 283 N — under the 343 N gate, and the source of
  14 accepted episodes, so it must not be screened; and
- **0.0116** (`02_multi_text_ee_constraint`), peak 712 N — more than double the gate, and the
  source of 8 rollouts that all failed.

At that threshold the screen would have skipped **11 rollouts** — 8 of
`02_multi_text_ee_constraint`, 2 of `08_text_object`, and the crouch — costing roughly
45 minutes of GPU, while losing **zero** accepted episodes.

## Two limits on the claim

**It predicts self-contact, not acceptance.** Most rejections in the table are *not*
self-contact rejections: `04_ee_constraint`, `05_root_path` and `06_root_waypoints` all sit
near zero saturation with low contact forces and were still rejected, for path error and
other reasons this signal knows nothing about. Screening on saturation removes untrackable
motions; it does not rank motions by quality, and it cannot replace the acceptance gates.

**The evidence base is behaviourally narrow.** 11 motions, of which 3 exhibited the failure
mode, from a corpus whose accepted episodes span a single behaviour. The gap between the
highest passing value (0.0071) and the lowest failing one (0.0116) is 0.0045 wide. That is
enough to set a screening threshold and not enough to call it calibrated; it should be
re-derived once the behaviour library widens, as part of the Phase C calibration pilot.

Because of that, the screen **reports by default and moves nothing** unless `--quarantine` is
passed. A rollout wrongly run costs four minutes of GPU; a rollout wrongly skipped is a
behaviour permanently absent from the corpus, and crouching is a behaviour the corpus
explicitly wants. Note also that root height is reported but never screened on: being low is
not a reason to skip a motion.

## The motion that motivated this

`clutter_np_crouchwalk_s0`, one of the first three motions ever generated here from a new
prompt: *"a person crouches down low, then stands up and walks forward"*. It reached
saturation 0.0444 and 9686 N of self-contact between `left_hip_roll_link` and `pelvis`, with
zero foot support — the robot was not standing on the ground. The acceptance gate rejected it
correctly. Screening would have saved the rollout and, more usefully, flagged at generation
time that this prompt needs a shallower crouch to be trackable at all.
