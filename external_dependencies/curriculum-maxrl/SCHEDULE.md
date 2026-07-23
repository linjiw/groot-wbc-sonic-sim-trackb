# Experiment schedule & tracking

*Living document — updated as runs complete. Times are A10G wall-clock.*

## Currently executing (GPU queue, in order)

| # | run | duration | purpose | decision it feeds |
|---|---|---|---|---|
| E1 ✅ | ck_uniform_maxrl (2400 s + ckpt) | done | efficiency baseline | — |
| E2 ✅ | ck_uniform_grpo (2400 s + ckpt) | done | efficiency baseline | — |
| E3 ✅ | ck_frontier_alp_maxrl_hsd (2400 s + ckpt) | done | champion checkpoint | — |
| E4 ✅ | eval_efficiency over E1–E3 | done | samples-to-coverage | **RESULT: up to 11× vs GRPO at level 5, speedup grows with difficulty (1.2×/2.7×/11×); GRPO curves flatten at large k. Decision: efficiency leads the benefits table on site + PAPER.** |
| F1 ✅ | long_falp_hsdense (9600 s) | done | is level 6+ a duration question? | **NO** — mean 0.258→0.269, L5 doubles, L6 stays ≈0.01. Mechanism revision needed at depth; CPU-validate depth-scaled move budgets / param-sharing check first |
| F2 ✅ | matched_falp_p4_hsdense (2400 s) | done | γ=4 on GPU | **does not transfer** (AUC 0.231 vs 0.236) — ODE model predicted it (weak compounding at 13 broad levels); γ stays 1 on GPU/verl, CPU/chain-only effect |
| F3 ✅ | dense-hindsight seed 1 (2400 s) | done | champion multi-seed | see F4 |
| F4 ✅ | dense-hindsight seed 2 (2400 s) | done | champion multi-seed | champion 0.252±0.005 final / 0.229±0.009 AUC; paired deltas vs plain teacher 6/6 positive but final margin mostly seed-0; **honest headline: reliable AUC gain + never worse; final edge small in infinite-data regime** |

**GPU QUEUE DRAINED (all E and F runs complete).** Next wave now unblocked.

**Parallel CPU (done): MountainCar categorical result** — flag-only 0.000 →
uniform-mix 0.889 → teacher 0.944 → **full stack 1.000 every seed**; plus
the transfer lesson (per-bin params never reach the flag: curricula operate
through shared parameters).

Watcher: `watch_gpu.sh` (running) — notifies on stall (>25 min no log growth) or queue completion.

## Decision tree after the queue drains

```
E4 efficiency table
├─ champion shows ≥2× samples-to-coverage on frontier levels
│    → add efficiency chart to website; lead REPORT with it
└─ no separation → coverage parity note; keep AUC as headline

F1 long-horizon
├─ level 6 leaves 0 by 9600 s
│    → run F1b: same budget, uniform baseline (is it the schedule or just time?)
└─ still 0 → implement depth-scaled move budgets (hindsight-min-depth
     curriculum); CPU-validate first, then one 2400 s GPU run

F2 γ=4
├─ AUC ≥ dense-hindsight baseline +0.003 → set teacher-power=4 default for
│    level-structured tasks in maze + frontier_rl docs
└─ tie/worse → document as CPU-only effect (compounding weaker at 13 levels
     than 36 tasks); keep γ=1 GPU default

F3–F4 seeds
├─ ordering holds → REPORT tables get ±std; done
└─ champion within noise of frontier_alp → soften "champion" claim to tie
```

## Next wave (planned, not yet queued)

| priority | experiment | est. | prerequisite |
|---|---|---|---|
| P1 | **Efficiency eval of F1's long-horizon checkpoint** — does 4× training turn into inference-time speedup at deep levels? | 30 min | F1 |
| P2 | **MountainCar scaled benchmark** (steps 120→600, 5 seeds, γ ablation) — does hindsight reach the flag (hardest bin > 0)? First *external* env where the full stack could show a categorical win | ~2 h CPU (parallel, no GPU) | none |
| P3 | **Best-config maze run with all validated knobs** (frontier_alp + dense hs + γ=4 + greedy rollout allocation if F2 passes) — the "everything on" run | 40 min | F2 |
| P4 | **Streaming-pool teacher prototype** (parametric density over a continuous difficulty axis, ALP-GMM-style) — unblocks procedural/generative task sources; CPU-validate on a continuous-difficulty variant of grid_reach | CPU | none |
| P5 | SmolLM2-360M + GSM8K 2×2 via `verl_integration/` | 8-GPU node | **blocked on hardware** |

## Standing cadence

- After every completed run: analyze → update EXPERIMENTS.md/REPORT.md →
  sync logs to this repo → push.
- Website results tables refreshed when a headline number changes.
- Every negative result gets written up with its diagnosis (the H6 reversal
  and conditioning-rewrite lesson were the two most valuable findings so far).

## Risks being tracked

- **Shared GPU:** another user's job took the card once already; the final
  sweep waits politely (`pgrep` loop) and the watcher distinguishes
  "waiting" from "stalled".
- **Seed noise:** finals vary ±0.01–0.015 across seeds; no single-seed claim
  goes in REPORT.md without either multi-seed confirmation or an explicit
  single-seed caveat.
- **Toy→real gap:** every CPU win must re-prove itself on GPU (γ=4 is the
  current test case); two CPU pre-registrations have transferred correctly
  so far.
