# What the first diverse batch broke

The corpus ran on two motions for its whole life. Forty varied ones went through the same
pipeline and broke five things that two motions could never have exposed. This records what
broke, the measurement that found it, and what changed — including one hypothesis that was
wrong.

## 1. Every capture was truncated at 59% of its motion

**Found by:** comparing `motion_time_s` against motion length across the batch.

39 of 40 captures reached 2.94 s of a 5.00 s motion. The rollout runner took a fixed step
count, inherited from a corpus whose two motions happened to be the same length.

This is worse than losing footage. The taxonomy's distinctive behaviours are *sequences* —
"walks, then squats to pick something up", "walks, pauses and looks around, then continues",
"stands still, then begins walking". Truncation cut the second clause off every one of them,
so the corpus would have contained the setup of each behaviour and none of the payoff, while
the prompt label claimed otherwise.

**Fixed:** `--max-steps` defaults to `auto` and reads the length from the motion library, so
a capture is one full pass of *that* motion.

**And the truncation was flattering the acceptance rate.** Grading the same episodes twice —
over the first 148 frames, then over the full pass — flips at least one from accepted to
rejected, and adds reasons to another:

| Episode | first 148 frames | full pass | reasons only in the full pass |
|---|---|---|---|
| `clutter_065` (side-step, brisk) | accepted | **rejected** | endpoint + path tracking error |
| `clutter_056` (duck under) | rejected | rejected | endpoint + path tracking error |

This is a same-episode comparison, not a batch-rate comparison, so it isolates the effect:
the robot starts on its reference and drifts, so the discarded later portion is exactly where
tracking is hardest. Every acceptance rate this project has quoted was measured over roughly
the first three-fifths of each motion.

One follow-on, found by running it: asking for one step *past* the pass puts the first frame
of the next pass in every capture, so every episode then contains a reset to split back out.
The step count is now exactly the pass length, since the recorder writes
`max_render_steps - 1` frames and stopping one frame short is harmless where starting a
second pass is not.

## 2. "Unevaluable" was indistinguishable from "rejected"

**Found by:** an episode reporting `accepted=False` with an **empty** rejection-reason tuple.

That episode carried 1776 N of scene contact that no gate had seen, because no gate ran: the
capture spanned an environment reset and the evaluator refused it. The batch logged it as
`RUN-OK`. So a recording bug was wearing the costume of a controller failure, and a real
collision was invisible.

Reported as one boolean, three different facts had been collapsed:

| Outcome | What it says |
|---|---|
| accepted | evaluated, every gate passed |
| rejected | evaluated, a gate failed, and the reasons name it |
| **unevaluable** | not assessed at all — the *capture* is the problem |

**Fixed:** `trajectory_segments` performs the split the evaluator's own error message asks
for; `episode_outcome` makes the three-way distinction first-class and divides the acceptance
rate by *evaluated* rather than total, so a recording bug cannot depress it. Applied to the
batch, the reset-spanning capture is recovered and its rejection correctly attributed to
`disallowed_robot_contact` — every rejection now has a stated reason.

A second effect showed up immediately: measured over the stitched file, that episode's
tortuosity was **3.309**, because the path length summed a teleport across the reset. Over
the recovered pass it is 1.359, and the batch maximum drops with it.

## 3. The body radius described the pelvis, not the robot

**Found by:** running the existing swept-volume machinery over every episode carrying
per-body pose.

`BODY_RADIUS_M` was 0.45 m. The measured swept half-width is **0.434–0.664 m** (median
0.500), so the assumption was exceeded on **98%** of 48 episodes. The old corpus survived it
because its two motions walked with arms tucked; reaching, carrying, squatting and looking
around swing wider.

What a room must actually leave free is that half-width **plus** how far the executed path
drifts from the reference the furniture was placed against. Measured per frame across the
batch, that sum spans **0.482–0.819 m** (median 0.582).

**Fixed:** radius is the measured 0.664 m and the default route clearance 0.85 m — not the
0.96 m that adding the two maxima would suggest, because the widest swept volume and the
largest drift rarely coincide.

## 4. A hypothesis that was wrong, recorded so it is not retried

The obvious explanation for the two contacts was that the clearance budget was too small.
**It was not.** The two episodes with scene contact had the *lowest* clearance requirement
(0.482 m and 0.532 m), while the three episodes that exceeded the provided 0.750 m had no
contact at all.

The budget is not tight, it is *probabilistic*: exceeding it only matters if furniture
happens to sit in the direction the robot drifted. That is why the corrected numbers above
are an honesty fix rather than a fix for the collisions — and why the remaining contact is
still unexplained by geometry alone.

## What this cost, and what it bought

Acceptance over the batch was **34/40 (85%) at the 2.94 s horizon** — and that horizon is
load-bearing, as §5 shows: the same pipeline over full-length captures accepts 5/12 (42%).
Rejections, all now carrying reasons: three `reference_path_tracking_error` (all brisk
curving motions, the hardest to track), two `disallowed_robot_contact`, one
`excessive_self_contact`.

Diversity moved where it was supposed to. Between-episode effective rank, the measure that
answers "how many distinct behaviours are here", went from **1.16 to 6.31**. Speed spans
0.205–1.816 m/s against the old corpus's 0.653–1.278, and heading change spans −6.86 to
+4.01 rad against essentially zero.

## 5. Episode length is a hidden parameter of every acceptance rate

**Found by:** grading the same episodes at increasing horizons, which isolates the effect —
no difference in scene, motion, or seed, only in how much of the trajectory the gates see.

| Horizon | Acceptance | Median p95 path error |
|---|---|---|
| 1.2 s | 83% | 0.098 m |
| 2.0 s | 75% | 0.138 m |
| 3.0 s | 67% | 0.151 m |
| 3.5 s | 67% | 0.180 m |
| 4.8 s | 64% | 0.248 m |

The tracker starts on its reference and drifts, so a longer episode is a harder one, and the
p95 path error crosses the 0.25 m threshold near 5 s. **The usable horizon of the SONIC
tracker on novel motions is about 4-5 s at the current threshold** — a property of the
controller, not of the corpus, and worth reporting as such.

The practical consequence: an acceptance rate quoted without its horizon is not comparable
to another one. Over the full 5 s pass the batch accepted 5/12 (42%); over the first 59% it
would have reported 73%.

**Acted on:** generation now defaults to **4.0 s**, chosen off the curve rather than
inherited — on the plateau before the decline steepens, still ~3.6 m of travel at a steady
pace against the old corpus's 3.5 m average, and long enough for a composite behaviour to
finish. This is not the same as capturing 4 s of a 5 s motion: Kimodo fits a *complete*
behaviour into the duration it is given, so truncating a 5 s "walk then squat" cuts
mid-squat while generating at 4 s produces a whole one.

`analyze_horizon_acceptance.py` restricts itself to horizons every episode can reach, because
a horizon that silently drops the shorter episodes compares a different population at each
row — which an earlier ad-hoc version of this table did.

## Still open after this round

- **The 2.8 N contact on `clutter_033`** is unexplained. It is marginal against a 1.0 N
  threshold, but "marginal" is not "understood".
- **Why an episode terminated at 1.2 s** without the robot falling (root height held at
  0.649 m) is not diagnosed. The split recovers the data; it does not explain the reset.
- **Every scene above was built with the old 0.75 m clearance.** The corrected 0.85 m and
  the full-length captures are validated on a fresh batch, not retro-fitted to these forty.
