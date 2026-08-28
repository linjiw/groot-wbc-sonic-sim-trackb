# The scene a motion can justify has a ceiling, and it is set by the edit, not by the model

**Date:** 2026-08-28
**Artifacts:** `docs/hallucination/scene_ceiling.json`, `docs/hallucination/lflh_sdf_c070.json`,
`docs/hallucination/lflh_sdf_c040.json`
**Runners:** `scripts/research/hallucination/measure_scene_ceiling.py`,
`scripts/research/hallucination/train_lflh_sdf.py`

A hallucinated scene is only evidence for a motion if it does two things at once: it must **admit**
the motion that was executed, and it must **exclude** the cheaper motion that was not. Report only
the first and a scene with no obstacles in it scores perfectly. Report only the second and a solid
wall does.

This note measures how much of both any scene can hold at once — before any model is trained — and
then measures how close a learned hallucinator gets to that bound. The bound turns out to be a
property of the *edit*, and it explains a result the previous run of this pipeline could not.

---

## 1. The quantity, and the three ways to read it

For a clip with executed motion `tau`, candidate set `C` with edit costs, and a scene `S`:

```
clear(c, S) = min over the body of candidate c of the signed distance to S    (> 0 means c fits)
margin(S)   = min( clear(tau, S) , how deeply the alternative is excluded )
```

Everything turns on what "the alternative" means, and the project's own definition of the inverse
set already answers it:

```
S_{eps,delta}(tau) = { S : Feasible = 1, Regret <= eps, Necessity >= delta, Realism >= r0 }
```

* **Feasible** — `clear(tau, S) > 0`. The motion that was executed has to fit.
* **Necessity** — the *cheapest* candidate, the motion that would have been executed in an empty
  scene, must be excluded by at least `delta`. Here that is the nominal.
* **Regret** — no cheaper candidate may be *feasible* and cheaper by more than `eps`. This is what
  handles the intermediate rungs of an amplitude ladder.

Two degenerate readings bracket it, and both are wrong in an instructive way:

| reading | "the alternative is excluded" means | verdict |
|---|---|---|
| **any-rival** | *some* cheaper candidate is struck | too weak: a box that strikes the nominal can leave `crouch_055` untouched, and the decoder would then prefer that |
| **all-rival** | *every* cheaper candidate is struck by the full margin | too strong: it demands the intermediate rungs be struck as hard as the nominal, which no scene can do |
| **`S_{eps,delta}`** | the nominal is excluded by `delta`; nothing cheaper is feasible within `eps` of the observed cost | the definition the project already committed to |

`measure_scene_ceiling.py` reports all three, by exhaustive search over one box (station x lateral
offset x underside height, fixed 0.20 x 0.90 x 0.30 m slab, hard minima over the capsule cloud).
The best it finds is the **ceiling**: no hallucinator, learned or otherwise, can exceed it.

## 2. Under the weak reading the ceiling is half the edit amplitude — and that is a mirage

| edit target | cheaper candidates | any-rival ceiling, median (range) |
|---|---:|---:|
| `crouch_040` | 1 | **33.2 mm** (25.6–49.1) |
| `crouch_055` | 4 | **41.7 mm** (33.5–59.1) |
| `crouch_070` | 5 | **50.5 mm** (39.3–73.6) |

Twelve clips per target, exhaustive search, hard minima. `ceiling_mm ~= 0.578 * amplitude_mm + 10.0`.

The any-rival law is close to perfectly linear in the crouch amplitude, which is what the
closed-form window predicts: a deeper crouch separates the two bodies further, and the extra
separation is what the scene spends on the two margins.

It is also **the wrong number**, and the size of the error is the point. On `crouch_070` the
candidate set contains five cheaper candidates — the nominal, `crouch_040`, `crouch_055`, and both
40 mm tucks — because a deeper edit costs more and so acquires its own shallower versions as rivals.
A box placed to clear `crouch_070` and strike the nominal must also deal with `crouch_055`, whose
body sits only 15 mm above `crouch_070`'s. The admissible band collapses from the 70 mm amplitude to
roughly the 15 mm rung spacing:

| edit target | rivals | all-rival | eps = 0 | eps = 0.1 | eps = 0.25 | eps = 0.5 | any-rival |
|---|---:|---:|---:|---:|---:|---:|---:|
| `crouch_040` | 1 | 33.2 | 33.2 | 33.2 | 33.2 | 33.2 | 33.2 |
| `crouch_055` | 4 | 8.3 | 15.2 | 15.2 | 41.7 | 41.7 | 41.7 |
| `crouch_070` | 5 | 8.3 | 15.5 | 15.5 | 31.3 | 50.5 | 50.5 |

All medians in millimetres over the same twelve clips.

Three things to read off it:

* **`crouch_040` is invariant across every column.** Its only cheaper candidate is the nominal, so
  all three readings coincide by construction. That the numbers agree is a check on the instrument,
  not a finding.
* **The amplitude law inverts once the ladder is taken seriously.** At `eps = 0` a 40 mm crouch
  admits a **33.2 mm** margin and a 70 mm crouch only **15.5 mm** — the shallowest edit in the
  ladder is the most explainable one, and going deeper *halves* the room available. The any-rival
  law's tidy 0.578 slope describes a quantity no sound scene can claim.
* **The jumps sit exactly at the rungs' cost gaps.** `crouch_055` costs 0.214 less than
  `crouch_070` and `crouch_040` costs 0.429 less, so `crouch_070`'s ceiling steps up at
  `eps = 0.25` (releasing `crouch_055`) and again at `eps = 0.5` (releasing `crouch_040` and the
  tucks). `eps` is not a smoothing knob; it selects which rungs the scene must argue against.

**The ceiling is set by the spacing of the edit ladder, not by the amplitude of the edit.** A deeper
crouch buys room against the nominal and spends it on the rungs it passes through. This is
registered as LFH-E19 / P5 in the prediction register, before the runs were read.

The `eps` column is the usable one, and it is a design knob rather than a fact about the world:
`eps` is how much edit cost the robot is allowed to have wasted. At `eps = 0` the scene must make
the executed motion the cheapest feasible one outright; as `eps` grows, intermediate rungs are
tolerated and the ceiling rises toward the any-rival figure.

## 3. Why the previous training run could not have worked

The earlier run of `train_lflh_sdf.py` targeted `crouch_040` and demanded
`m_clear = 40 mm`, `m_hit = 30 mm`. Against a measured ceiling that is a **median of 31.6 mm** at
that amplitude, that objective is unsatisfiable on nearly every clip: it asks for more separation
than the two bodies have.

The training trace says so directly. The final barrier was `0.01099` per clip-sample, and
`sqrt(0.01099) = 0.1048`, so the residual violation was `0.05 - 0.105 = -0.055 m` — which is
exactly the `-0.054 m` median worst clearance the same run reported. The optimiser was not stuck
and the geometry was not wrong. **The target was outside the feasible set, and the reported
18.8% robot-clear rate was measuring that, not the model.**

Margins are now derived from the amplitude (`default_margins`) rather than fixed, and every run
prints them.

## 4. A second defect: the reported clearance was the training surrogate

`SdfChoiceDecoder.clearance` returns a *soft* minimum, so that gradients reach the nearest few
points of the cloud instead of a single one. A soft minimum is a weighted average, so it is never
below the true minimum — and a scene can therefore score positive on it while a point of the robot
is inside an obstacle. Every reported "robot clear" number was computed this way.

`exact_clearance` now returns the hard minimum and every reported number uses it; the smooth form
is retained for the loss only, and both share one signed-distance implementation so they cannot
drift apart. `tests/dataset_generation/test_sdf_exact_clearance.py` pins the ordering and exhibits
the failure: one point 2 mm inside a box, inside a cloud of 1200 points that are 50 mm clear, is
reported clear by the soft minimum and penetrating by the hard one.

## 5. What the learned model achieves against the ceiling

<!--MODEL_TABLE-->

## 6. What this changes

**1. The paper cannot claim that a learned hallucinator finds scenes explaining a motion.** An
exhaustive box search saturates the bound on every clip, in seconds, with no training. Existence is
not the contribution and a reviewer will say so. What remains is **amortisation and conditioning**:
a near-ceiling scene for an unseen motion in one forward pass, *because* the model conditions on the
motion. `compare_amortisation.py` measures that as search-equivalence — how many random draws match
one forward pass — with the motion-blind ablation from the first retraction as a standing
diagnostic. If the answer is one or two draws, the closed form is the method and the paper should
say so.

**2. Never report a margin without its ceiling.** A raw margin is uninterpretable: a small one may
be the task rather than the model, which is exactly the error that produced the retracted 18.8%.
Efficiency — achieved margin over the per-clip ceiling — is the reportable quantity, and
`render_lflh_sdf_scenes.py --with-oracle` renders the model's scene beside the provable optimum for
the same clip so a figure cannot overstate it either.

**3. `eps` is a design parameter and has to be declared.** It is how much edit cost the robot is
allowed to have wasted, and the achievable margin depends on it by more than an order of magnitude.
A paper that reports a margin without stating its regret tolerance has not reported anything. The
default here is 0.25, which tolerates the neighbouring rung of the same operator and excludes
everything coarser.

**4. The edit ladder is part of the problem specification, not a sampling convenience.** Adding
rungs between the nominal and the target shrinks every scene's achievable margin, because each rung
is a candidate the scene must rule out. A denser ladder is not a free improvement in coverage; it
buys resolution in the edit space and pays for it in scene expressiveness. That trade is worth a
figure.

**5. Two habits, both learned the hard way here.** Never report the quantity you optimised through a
smoothing — report the exact one. And check feasibility before attributing a failure to the model:
the residual of an unconverged barrier is a number, and decoding it took one line of arithmetic that
would have saved the previous run entirely.

## Reproduce

```bash
PY="env -u PYTHONPATH $HOME/miniconda3/envs/env_isaaclab/bin/python"
$PY scripts/research/hallucination/measure_scene_ceiling.py --clips 12
$PY scripts/research/hallucination/train_lflh_sdf.py --target crouch_070 \
    --clips 16 --holdout 5 --steps 1500 --stride 8 --out docs/hallucination/lflh_sdf_c070.json
```
