# What the next steps cost on this card

Measured, not estimated. One RTX 5090, 32 GB, shared with other jobs.

## Per-unit costs

| unit | GPU memory | wall time | source |
|---|---|---|---|
| One SONIC rollout (1 env, ego render, trajectory) | **4.8 GB** | **~375 s** | the process that OOM'd reported 4.76 GiB in use; six consecutive counterfactual rollouts took 361–391 s each |
| One Kimodo motion generation | ~2 GB | ~15 s | the mode-bank batch, 16 clips |
| Prompt encoding (LLM2Vec) | 0 — **CPU only** | ~4 s/prompt | run with `CUDA_VISIBLE_DEVICES=""`; the 15 GB encoder must never share the card |
| Reference gate, envelope, clearance, operators | 0 — **CPU only** | ~1 s/clip | forward kinematics in MuJoCo |

The 375 s figure is **under contention** — three `minimax.train` and two `rsl_rl` jobs were
resident. On a quiet card the same rollout took ~40 s, a 9× swing, so any schedule below is a
ceiling rather than a forecast.

**The wait threshold is 6 GB**, not 4.8. PhysX asks for a 256 MB block up front and the
renderer and scene follow it; starting at 5 GB free is how the last batch died.

## What each remaining step needs

| step | rollouts | GPU-hours (contended) | notes |
|---|---|---|---|
| **Stage 2** — trackability of the two operators | 3 | 0.3 | blocking G2; nothing else waits on it |
| **G2-B** — one matched family 2×2 | 6 | 0.6 | 2 probes + 4 cells |
| **G2-B robustness** — 3 start-pose jitters | 12 | 1.3 | only after the 2×2 holds |
| **Track A** — calibrate the predicted window on existing pairs | ~120 | **12.5** | 20 pairs (5 overhead, 15 lateral) × 6 |
| **G3** — 3 nominals × 3 routes | 54 | 5.6 | shows the operator is not a one-off |
| **Physics labels for the frozen test set** | 30 scenes × 4 motions = 120 | **12.5** | the ground truth the selector is scored against |
| **Selector training** | 0 rollouts | ~1–2 GB, minutes | a small classifier, not a VLA |

**Total to a complete decisive experiment: roughly 33 GPU-hours contended**, or about 4 hours
if the card were free. Nothing here needs more than one rollout resident at a time, so it
fits alongside other work at 6 GB — it is a scheduling problem, not a capacity one.

## The cheapest ordering

Everything that can be answered without the GPU already is. The reference gate refuses 56 of
150 clips for ~1 s each, and both adaptation operators, the envelope mining, the station
ablation and the clearance measurement are pure CPU. That is deliberate: GPU is the scarce
resource here, and the gate exists so it is only spent on clips that can still fail for an
interesting reason.

If the card stays busy, the order that buys the most per rollout is:

1. **Stage 2** (3 rollouts) — decides whether the operators are usable at all.
2. **G2-B** (6) — the matched family, which is a stronger result than the current one.
3. **Track A on near-threshold pairs only** (~36, not 120) — the calibration question is
   answered by pairs near the 50 mm bar, not by the comfortable ones.
4. Everything else.
