# Indexes

Three CSVs at three grains, joined by stable ids. The JSON release beside them is faithful but
nested, and nested is the wrong shape for the question someone actually asks — *show me every
episode where the torso came within 20 mm of a shelf and the controller still held.*

| file | one row per | join key |
|---|---|---|
| `episodes.csv` | graded rollout | `episode_id` = `family_id/cell` |
| `families.csv` | counterfactual family | `family_id` |
| `motions.csv` | reference clip | `clip` |

## Columns worth knowing

**`min_clearance_mm`** — the smallest gap between any collision capsule and the obstacle, over the
whole episode. Negative means they overlap. This is the number that decides the outcome, and it is
measured on the executed body rather than on the reference.

**`closest_body`** vs **`contact_body`** — the first is whatever came nearest, measured
geometrically; the second is what the physics gate recorded as touching. They agree when there is
contact and the first is still meaningful when there is not.

**`overhead_force_n`** vs **`lateral_force_n`** — a shelf pushes down, a wall pushes sideways, and
the same collision reads very differently between them. Never quote a contact force without saying
which component.

**`route_reaches_obstacle`** — whether the executed path entered the obstacle's footprint at all.
Zero means the episode measures nothing about that obstacle, however its outcome reads.

**Empty is not zero.** An unmeasured value is left blank, because an unmeasured force and a zero
force are different facts.

## Example queries

```python
import pandas as pd
e = pd.read_csv("episodes.csv")

# Near misses the controller still held — the most informative cells in the corpus
e[(e.min_clearance_mm < 20) & (e.outcome == "accepted")]

# Every overhead collision, hardest first
e[e.overhead_force_n > 1].sort_values("overhead_force_n", ascending=False)

# The matched negatives: the cells a counterfactual claim rests on
e[(e.cell_role == "nominal_hard") & (e.outcome == "rejected")]

# Episodes whose obstacle the robot never reached, which measure nothing
e[e.route_reaches_obstacle == 0]

# Everything with video, for review
e[e.video_room.notna()][["episode_id", "outcome", "min_clearance_mm", "video_room"]]
```

```python
f = pd.read_csv("families.csv")
f[f.status == "verified"]                     # families that are evidence
f[f.status != "verified"]                     # and the ones that cost rollouts and are not

m = pd.read_csv("motions.csv")
m[m.embodiment_feasible == 0]                 # clips the body cannot hold
m.groupby("behaviour_class").embodiment_feasible.agg(["count", "sum"])
```

## A caution on `motions.csv`

`prompt_intent` is an **intent**. Of 75 prompts admitting a semantic predicate, only 24 contained
the behaviour they named, so `reference_semantic_valid` and `executed_semantic_valid` are separate
columns and are currently empty — they will be filled by human review, not inferred from the prompt.
