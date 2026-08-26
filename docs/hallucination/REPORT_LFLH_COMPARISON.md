# RETRACTED: "Does a learned hallucinator recover the humanoid inverse set?"

**Date:** 2026-08-26 · **Status: the headline result of this report is withdrawn.**

An adversarial review of the first version found that every number in it was an artifact of the
setup rather than a measurement. The numbers reproduce exactly; they do not mean what the report
said they meant. This file now records what was claimed, why it was wrong, and the one part that
survives.

The retracted claims were: that a learned LfLH-style hallucinator's `sigma_coordinate` collapses
from 87.89 mm to 0.248 mm ("354x"), that this reproduces LfLH's reported mode collapse on humanoid
data, and that it therefore constitutes *evidence* for computing the support in closed form rather
than learning it.

---

## 1. Why the result was invalid

**The "learned" model is a constant function of its input.** Across all 16 clips the latent means
have standard deviation 0.0015, which is **0.15 mm** of clip-to-clip variation, against feasible
windows that differ by tens of millimetres between clips and whose lower edge moves 45–136 mm
*within* a clip. Controls settle it:

| arm | valid | occupancy |
|---|---:|---:|
| trained model | 100.0% | 0.66% |
| trajectory input replaced by the corpus-mean profile | 100.0% | 0.63% |
| **no network at all** — constant `adapted.mean() + 4 mm` | 100.0% | **0.50%** |

A constant scores within noise of the trained model. Nothing about `q(C | p)` was tested.

**The headline sigma is a hard-coded constant.** `learned_hallucinator.py` clamps `log_sigma` at
`-6.0`; `100 * exp(-6) = 0.24788`, which is the reported value to float32 precision, pinned
bit-for-bit from step 120 onward. `clamp` has zero gradient outside its range, so the parameter was
dead. Moving the floor moves the "result": `-3.0 -> 4.98 mm`, `-4.5 -> 1.11 mm`, `-8.0 -> 0.034 mm`.
The numerator, 87.89 mm, is untrained random initialisation and ranges 87.9–108.4 mm across seeds.
**"354x" is one seed divided by one magic number.**

**The 100% valid rate is the parameterisation, not the model.** The coordinate is emitted as
`sample * 0.1 + adapted.mean(dim=-1)`, and the true window's lower edge is `adapted + margin`. The
anchor is the answer, handed to the model by hand. This is exactly the "affine identity, cannot
fail" criticism the audit levelled at the 400/400 in-support check — it applies with equal force
here, and the report did not say so.

**The collapse is a theorem, not an experiment.** `-log sigmoid(z)` is `softplus(-z)`, which is
convex, so `E[L(mu + sigma * eps)]` is strictly increasing in `sigma` for every temperature, and
there is no entropy or KL term opposing it. `sigma -> 0` is provable before any data is loaded.
Confirmed by ablation: temperature (0.004 → 0.5) does not change it; `prior_weight = 0` does not
change it; **adding the missing KL does**:

| | sigma | valid | occupancy |
|---|---:|---:|---:|
| as implemented | 0.248 mm | 100.0% | 0.66% |
| + KL, weight 0.01 | 3.15 mm | 99.7% | 5.80% |
| + KL, weight 0.1 | 7.19 mm | 92.2% | **34.3%** |

**So the report's central claim is falsified by its own model class.** "A single Gaussian cannot
hold both high" is untrue: hand-setting sigma gives 97.4% valid at 9.5% occupancy and 89.1% at
28.3%. The 0.66% was the *objective's* choice under a missing regulariser, not the family's limit.

**The occupancy metric is broken.** Feasible cells are admitted by their centre, visited cells by
containment, so the two use different rules and occupancy can exceed 1 — measured at **124.7%** for
the uniform sampler at 6 bins. And 83.4% is `1 - exp(-N/K)`, a coupon-collector reading of
`--samples 400`, not a property of the sampler.

**The report's own robustness claim was false.** §5 stated "the absolute number moves with the
binning, the 119x ratio does not." Measured, the ratio moves 58x–137x with binning and 22x–158x
with sample budget. It is a free parameter of the report.

## 2. The citation was also wrong

- **LfLH (arXiv 2108.09793) does not report mode collapse.** The words "collapse", "diversity",
  "mode", "entropy" and "KL" do not appear. The claim belongs to **Dyna-LfLH v2**
  (arXiv 2403.17231) §IV-E, in a single-obstacle dynamic regime.
- **LfH-CP (arXiv 2509.26513) states the opposite of what was implied**: "LfLH, in comparison, can
  partially overcome mode collapse by hallucinating more obstacles in static environments." The
  reduction to one obstacle with two parameters removes the very mechanism the literature credits
  with mitigating collapse — and then reports collapse.
- LfLH emits **10 ellipses = 40 obstacle parameters** (UAV: 15 ellipsoids = 90), not "thirty".
- The implementation kept only a location-NLL analogue. It omits the trajectory MSE through a
  differentiable planner, the genuine closed-form `size_kl_loss`, obstacle–obstacle repulsion,
  the 0.5 m obstacle–plan clearance, loss annealing, and the five extra random obstacles LfLH
  injects per plan **specifically to increase sample variance**.

Calling it "an LfLH-style hallucinator" was not defensible. It is a two-parameter Gaussian trained
through a soft indicator of the closed-form window.

## 3. What survives

One claim, and it is worth keeping:

> **`valid_rate` alone is a misleading metric.** A collapsed sampler — indeed a constant — scores
> 100% on it. Any proposal distribution in this project must report coverage of the feasible set
> beside validity.

The irony is instructive: the report demonstrated this by accidentally shipping a constant that
scored 100% valid, and then read its own artifact as a finding about learned models.

The limits paragraph (§5 of the original: surrogate is not physics, single Gaussian, 16 clips,
one amplitude, overhead only) was accurate. It was simply not sufficient — the flaws were upstream
of the limits.

## 4. What a defensible version requires

1. **Prove the collapse; do not "measure" it.** Softplus is convex and LfLH's location term is an
   NLL with no `-log sigma`, so `E[L]` increases in sigma. One paragraph, no seeds, no clamp,
   unfalsifiable by tuning — a *stronger* claim than the retracted one.
2. **Remove the clamp from any headline** (`min_log_sigma = -20`) and report where sigma settles,
   mean ± sd over at least five seeds. Never report a start value drawn from initialisation.
3. **Fix `coverage`**: one cell rule for numerator and denominator, a test asserting
   `occupancy <= 1`, occupancy reported *as a curve in N* rather than at a single budget, and the
   station axis decomposed from the coordinate axis — a single-station sampler's ceiling is ~11%,
   so most of the retracted gap was the station axis, not the phenomenon.
4. **Add the controls that decide it**: no-network constant, the same Gaussian with a proper KL, a
   mixture or flow, and the valid-versus-occupancy Pareto frontier instead of two points.
5. **Remove the ground-truth leak**: parameterise the coordinate in absolute metres, and report the
   input-ablation as a standing diagnostic. If replacing the trajectory with the corpus mean does
   not change the output, the model is not conditional and nothing was learned.
6. **Implement LfLH's actual loss and multi-obstacle setting, or drop the name.**

Until at least 1–5 are done, this project should make **no claim** about learned hallucinators
versus closed-form support. The closed-form window remains the right engineering choice here for
the reasons in `docs/lfh/lflh-relationship-and-3d.md` §2 — a non-differentiable, expensive,
stochastic decoder — and those reasons stand on their own. They never needed this experiment, and
they are weakened rather than strengthened by an experiment that does not hold up.

## 5. Disposition of the code

`gear_sonic/dataset_generation/hallucination/learned_hallucinator.py` and
`scripts/research/hallucination/run_lflh_comparison.py` are retained, with the clamp, the leak and
the metric documented in place, so the retraction is reproducible. They must not be cited as
evidence for anything until rebuilt. `docs/hallucination/lflh_comparison.json` is retained as the
record of the retracted run.
