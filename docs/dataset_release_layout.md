# SweepCF release folder

Everything the project has produced, in two views that are deliberately kept apart.

```
MANIFEST.json          counts, byte size, and a SHA-256 over every file
motions/
  index.jsonl          one row per reference clip -- the qualification view
  clips/*.csv          the reference motions themselves, 36-column qpos at 30 fps
families/
  index.jsonl          one row per counterfactual family
  <family>/
    family.json        geometry, per-cell verdicts, contact attribution
    scenes/*.usda      the exact geometry physics loaded, not a description of it
    cells/<cell>/
      outcome.json     verdict, rejection reasons, contact body/direction/frame, drift
      trajectory.pkl   the raw capture, authoritative
    renders/           room camera, to-scale side view, robot's own head camera
splits/
  scene_first_v1.json  30 evaluation scenes, frozen before any operator existed
  review_cohort_v1.json 100 episodes stratified for two-rater human review
```

## Reading it honestly

**A prompt is an intent, not a label.** `motions/index.jsonl` carries `prompt_intent` and keeps
`reference_semantic_valid` / `executed_semantic_valid` as separate fields, currently null. Of 75
prompts that admitted a semantic predicate, only 24 produced the behaviour they named. Treating the
prompt as ground truth would mis-label roughly two thirds of them.

**Physics acceptance, behaviour semantics and scene compatibility are three different labels.**
A clip can track perfectly and not perform the behaviour asked for; it can perform the behaviour and
still be infeasible in a given room.

**Rejected cells are kept.** A counterfactual family's negative *is* its evidence. A corpus holding
only successes cannot support a claim about what fails.

**Scene geometry travels with the family.** Each family's claim is a claim about one shelf height in
one room, so the USDA that physics loaded is copied in rather than described. Families were once
built with the obstacle in the wrong coordinate frame, and the robot walked two metres clear of a
shelf it was supposed to meet; `obstacle.route_reaches_it` in each cell's `outcome.json` now records
whether the path entered the obstacle's footprint at all.

**Verified is not the same as present.** `families/index.jsonl` lists every family attempted. A
family counts as *verified* only when all four cells came out as a counterfactual requires: both
motions accepted in the easy scene, the nominal rejected in the hard one, the adapted motion
accepted there.

## Current honest state

One matched family (`mf_005_c08`) is fully verified, over one nominal motion. Three mined families
pair two separately generated clips, which is weaker evidence -- a reviewer can ask whether the
shelf separated two behaviours or two different journeys. `mf_x002_c08` and `mf_x003_c08` are
present but empty: their first build placed the obstacle in the wrong frame and is being re-run.

The dataset target is 24-30 verified families over 8 or more distinct nominal motions. Family count
and nominal-motion count are reported separately on purpose: five shelf heights on one motion are
five scene instances, never five independent behaviour families.

## Rebuilding

```bash
python scripts/research/build_dataset_release.py \
  --families-root .../matched .../counterfactual \
  --screen .../taxonomy/screen.json --clips .../taxonomy/motions_4s \
  --renders <render dir> --out .../sweepcf_release
```

Every number quoted elsewhere should be reproducible from `MANIFEST.json` and these files alone.
