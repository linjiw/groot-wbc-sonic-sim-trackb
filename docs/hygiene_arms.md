# Five training arms, one factor each

**Claim.** The hygiene × sampler matrix is single-factor *by construction*: arms B–E inherit
`hygiene_a_raw` wholesale and override exactly one functional leaf each. Nothing about compute,
seeding, callbacks or the resident-set size is restated per arm, so the arms cannot drift apart in a
way a reviewer would have to catch by reading five files.

**Status: composed, never run.** Every claim below about *what the configs say* was checked by
composing them (see [Verification](#verification)). Nothing below is a claim about training: no arm
has been launched, and two of the five banks do not exist yet.

## The matrix

| arm | file | bank | sampler | the one leaf that differs from A | the contrast it identifies |
|---|---|---|---|---|---|
| **A** | `hygiene_a_raw.yaml` | raw, 4950 clips | release default (`uniform_sampling_rate: 0.1`) | — (this is the reference) | baseline |
| **B** | `hygiene_b_pruned.yaml` | pruned (screen-flagged clips removed, N shrinks) | release default | `motion_lib_cfg.motion_file` | A vs B: is deleting unsupportable references worth it? Confounded with data volume by construction |
| **C** | `hygiene_c_repaired.yaml` | repaired (same clips, projected roots, **N preserved**) | release default | `motion_lib_cfg.motion_file` | A vs C: hygiene *without* the volume confound. B vs C: delete or fix? |
| **D** | `hygiene_d_uniform.yaml` | raw | `uniform_sampling_rate: 1.0`, adaptive path still **enabled** | `adaptive_sampling.uniform_sampling_rate` | A vs D: how much of the raw bank's cost is the sampler chasing clips it cannot learn? |
| **E** | `hygiene_e_cap.yaml` | raw | release default **plus** `max_prob_per_motion: 1.0101e-3` | `adaptive_sampling.max_prob_per_motion` | A vs E: does the one-line config remedy that already ships (but is `null` everywhere) buy what cleaning the bank buys? |

Priority if the GPUs only cover part of the matrix: **C > E > B > D**, all against A
(`docs/plan_feasibility_hygiene_v1.md`).

## Compute match

Identical in all five arms, and verified identical after composition:
`num_envs: 4096`, `seed: 0` (from `base.yaml`), `algo.config.num_learning_iterations: 100000`,
`algo.config.num_steps_per_env: 24`, `override_num_motions_to_load: 1024`, the `read_eval` and
`im_resample` callbacks (`motion_resample_frequency: 250`), and every `adaptive_sampling` knob other
than the one an arm is named for. Same rollout budget, same number of gradient steps, same
start-time machinery. What differs is which frames the samples land on.

To vary the seed, pass `seed=<s>` on the command line to **every** arm of a replicate. No arm
overrides `seed`.

## Launching

```bash
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/hygiene_c_repaired \
    +checkpoint=sonic_release/last.pt \
    headless=True
```

**Do not copy the README's launch line.** It ends with
`++manager_env.commands.motion.motion_lib_cfg.motion_file=...` and
`++...smpl_motion_file=...`; a `++` override outranks the overlay, so pasting it silently puts arms
B and C back on the raw bank and erases the entire bank axis. Pass the bank through the arm file or
not at all. `num_envs` is likewise already pinned; overriding it on one arm breaks the compute
match.

## Arm D: why the rate, not the switch

`uniform_sampling_rate: 1.0` and `adaptive_sampling.enable: false` are not the same experiment.
Disabling the adaptive path also changes the **start-time distribution**: the adaptive path draws a
bin, draws a frame in it, then shifts backward by a uniform integer in `[0, 200)` frames clamped at
zero (`motion_lib_base.py:3529-3548`), while the plain path draws continuous-uniform phase over the
whole clip (`:2249-2258`). Setting the rate to 1.0 leaves that machinery bit-identical and changes
only the weighting.

Note what "uniform" then means: uniform over *bins* weighted by `adp_samp_bin_weights`, which the
length-agnostic normalisation makes uniform over **clips** — not uniform over time within a clip.
Arm D answers "is failure-weighting better than equal weighting", not "is this the true uniform
reference". At iteration 0, A and D are the same sampler (`init_num_failures: 1` puts every bin at
failure rate 1.0); they separate only as episode outcomes accumulate.

## Arm E: what "5× fair share" can and cannot mean here

`max_prob_per_motion` is read as an **absolute probability** (`motion_lib_base.py:2462`), so a
multiple of fair share has to be divided out by hand: `5 / 4950 = 1.0101e-3` for the raw bank. That
number is bank-size specific and is not valid for the pruned bank. `"auto"` is not a substitute — it
resolves to `adp_samp_failure_rate_max_over_mean / N = 200/4950 = 4.04e-2`, a 200× ceiling that runs
but never binds, because the worst measured single-clip share on this sampler is 3.4–4.5% of the
mass.

The awkward part, stated rather than discovered in a log: **two scopes read this one number and they
have different N.**

| scope | N | code | at `1.0101e-3` |
|---|---|---|---|
| residency draw (which clips are resident) | `num_motions = 4950` | `:3456-3468`, consumed by `torch.multinomial(_sampling_prob, 1024, replacement=True)` at `:1116-1118` | **binds, at exactly 5× fair share**: ≤ 1.03 of 1024 resident slots against a fair 0.207 |
| frame draw (which frames, inside the resident set) | `num_active_motions` = unique clips among the 1024 draws ≈ **925** | `:3296`, guarded at `:3332` | **does not run**: the guard needs `N > 1/max_prob = 990` |

925 is not a guess: drawing 1024 times with replacement from 4950 gives 925.3 ± 8.6 unique clips
(400-trial simulation; analytically 925.1), and at iteration 0 that draw is exactly uniform, so 925
is the *ceiling* — concentration only lowers it.

And that gap cannot be closed from yaml. The frame-stage clamp needs `max_prob_per_motion > 1/925 =
1.081e-3`, i.e. **> 5.35× fair share of the bank**; at that value the ceiling equals the resident
set's own fair share and flattens the resident distribution to uniform, which is arm D. While the
resident window (1024) is smaller than the bank (4950), no single absolute number is both "5× of
bank fair share" and a non-degenerate cap inside the window.

**Honest limit, to be repeated in the write-up:** arm E bounds how *often* a clip is resident, not
how hard the sampler leans on it once it is there. Expect its exposure ledger to still show a top-1
clip share well above 5× fair share. The alternative reading — `5/1024 = 4.8828e-3`, 5× fair share
of the resident *window* and 24.2× of the bank end to end — is a defensible arm but a different one,
and the two must never both be labelled "5× fair share". Supporting a relative spec
(`{fair_share_multiple: 5}`) is a code change in `motion_lib_base.py`; this matrix does not touch
shipped training code, so it stays a follow-up.

Watch the resident size: the frame-stage guard switches **on** once `override_num_motions_to_load`
passes ≈1105 (where the expected unique count exceeds 990), which would change arm E's mechanism
mid-matrix. Lowering it is safe.

## Placeholders

`hygiene_b_pruned` and `hygiene_c_repaired` point at banks that **do not exist yet**, and both files
say so at the key:

- `data/motion_lib_hygiene/pruned/robot_filtered` — to be produced by
  `scripts/research/hygiene_screen_bank.py`, which is not on disk yet either.
- `data/motion_lib_hygiene/repaired/robot_filtered` — produced by
  `scripts/research/hygiene_repair_bank.py`, which is on disk.

Arm A's own path, `data/motion_lib_bones_seed/robot_filtered`, is the canonical release path used by
the README and every shipped config, and it is **also not materialised in this checkout**; the
4950-clip bank on this machine lives at
`/data/robotixx/groot-wbc-sonic-research/datasets/bones_seed_official_headline_scale4950/robot_filtered`.
Stage or symlink it once, for all five arms.

Two traps recorded at the keys they belong to:

- **Resident-set size vs a small bank.** If a bank has fewer clips than
  `override_num_motions_to_load`, `load_motions_for_training` takes the `max_num_seqs >=
  num_unique` branch (`motion_lib_base.py:1000-1002`), sets `all_motions_loaded`, and the resident
  set stops rotating for the rest of the run. That is a second mechanism change riding on the bank
  change, invisible in the logs. Before launch, set `override_num_motions_to_load` in **all five**
  files below the clip count of the smallest bank in the matrix.
- **The SMPL bank stays shared.** All five arms read `data/bones_seed_smpl`. The repair writes one
  column (`root_trans_offset[:, 2]`) and leaves `smpl_joints` alone, and the SMPL joints are
  consumed root-relative, so a repaired robot bank does not desynchronise them; pointing arm C at a
  repaired SMPL directory that nothing produces would resolve every key to `None`
  (`motion_lib_base.py:283-292`) and delete SMPL conditioning from that arm alone.

## Verification

Ran, on CPU, with no Isaac Lab import:

```python
from hydra import compose, initialize_config_dir
with initialize_config_dir(version_base="1.1", config_dir="gear_sonic/config"):
    cfg = compose(config_name="base",
                  overrides=["+exp=manager/universal_token/all_modes/hygiene_e_cap"])
```

All five arms compose (574 leaves each). Flattening `OmegaConf.to_container(cfg, resolve=False)` and
diffing every leaf against arm A gives, per arm, exactly one functional difference plus the
`exp_var` / `hygiene_matrix` metadata: `motion_file` for B and C, `uniform_sampling_rate` for D,
`max_prob_per_motion` for E. The compute-match keys listed above are identical across all five.
`max_prob_per_motion` parses as a float (`0.0010101`), and `0.0010101 × 4950 = 5.00000` fair shares.

Not verified, and not claimed:

- **No motion library was built and no training was run.** Composition proves the configs resolve;
  it says nothing about a `MotionLib` accepting these values at runtime.
- **The banks do not exist**, so no arm has been loaded end to end.
- The 925-unique-clips figure is a simulation of the documented draw plus its closed form, not an
  observation of the running sampler; the arm-E scope table is arithmetic over the code path, not a
  measurement.
- The `hygiene_matrix` block is inert metadata. Training ignores it; it exists so
  `scripts/research/analyze_hygiene_matrix.py` can name an arm from a run's `.hydra/config.yaml`
  (`arm` is spelled exactly as that script's `REQUIRED_ARMS` expects).
