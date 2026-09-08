# Fable — guidance for the Motion2Scene ICRA paper (experiments complete, 2026-09-08)

Written 2026-09-07 (00:30 EDT), after reading the full `docs/motion2scene/` trail
(all `*_RESULT.md` through `SELECTOR_BREAKPOINT_RESULT.md`), the August LfH line
(`docs/lfh/`, `docs/hallucination/`, the earlier `fable.md`, now archived at
`docs/guidance/fable_2026-08-26_lfh.md`), `docs/paper/`, the uncommitted transition-bank
work, and the machine state. Where I make a judgement rather than restate a measurement,
I say so.

Deadline per the team's own check of the call: **ICRA 2027, 15 Sep 2026, 11:59 PST.
Eight pages including references.** Video upload reopens 17–22 Sep, so the video is
not on the critical path. That leaves eight days, of which at least four must be writing.

---

## Final — 2026-09-08. Both experiments are complete.

The plan below was written at 00:30 on Sep 7 and has been executed in full. Two
registered studies ran to completion for **6.6 contended GPU-hours** total. The result
is **outcome B**, exactly as pre-declared in §2.3.

### What was measured

| Construction | Groups | Useful contrasts | Passage (held-out) | Adapts |
|---|---:|---:|---:|---:|
| Uniform | 15 | 0 | 14/36 | 0 |
| **Analytic** | 15 | **9** | **24/36** | 24 |
| Target-only | 15 | 0 | 14/36 | 0 |
| **Motion2Scene** | 15 | **9** | **22/36** | 22 |

Both contrast arms beat both non-contrast arms with **no reverse case on any carrier**
(8–0 and 10–0). Motion2Scene and analytic are **indistinguishable**: 0–2 discordant of
36, p = 0.50. Contrast construction is what teaches the selector; learning the proposal
neither helps nor hurts at this label count.
[Result](docs/motion2scene/M2S_ICRA_NOMINAL_V1_RESULT.md).

The larger 540-run panel stands behind it: 72 traversal conditions, analytic 39/72
against 28/72 for the three non-adapting arms, 11–0 paired, all wins in the band its one
training contrast came from, plus the background suites showing no unnecessary adaptation
anywhere and blocked-scene refusal at 6/6 for every learner against 0/6 for the scripted
rule. [Result](docs/motion2scene/M2S_ICRA_540_RESULT.md).

### The one thing that nearly sank it

The first contract accepted **zero** learned and **one** analytic scene. The cause was
measured, not guessed: the inherited audit demanded each contrast survive a ±20 mm beam
displacement while the entire executed window is 20–27 mm wide. Witnesses fall from
200/168/337 at the nominal pose to 2/0/32. A separately registered nominal contract, with
predictions filed before its proposals were drawn, moved eligibility from 1/48 to 41/48
(analytic) and 0/48 to 33/48 (learned).
[Result](docs/motion2scene/ENVELOPE_TRADEOFF_V1_RESULT.md).

### The learned generator's one measured deficiency

Its training loss is 0.139 against analytic's 0.00054 on identical architecture and label
count, because only 24 of 34 of its critical scenes are visible to the decision-time
sensor against 44 of 47 for analytic. Four of its nine assigned groups record zero ray
hits. Report it; no visibility gate was added after the fact.

### Final thesis

*A training scene is useful only if it contains a decision-relevant contrast that
survives execution and is visible at decision time. Such contrasts are geometrically
scarce, the scarcity is measurable before any simulation, and how the contrast is
proposed matters far less than whether the acceptance geometry lets it exist at all.*

### What remains for the submission

Writing only. The abstract can now be written; the decision §2.3 asked for is made.
Sections I–VII of `docs/motion2scene/ICRA_MANUSCRIPT.md` carry the final numbers, five
pages with figures. Remaining: cut to eight pages in the ICRA template, internal review
against the `xiao-paper-review` rubric, and the video from existing replays during the
17–22 Sep window. **No further physics is required for this submission.**

### Beyond the paper

Hardware is a separate program, not an extension of this one. The five gates in
dependency order are on the [project page](docs/index.html): re-price the contrast
against real obstacle-pose uncertainty (the envelope curve already gives its cost),
replace ideal rays with the deployed sensor, run source-held-out transfer on the eight
reserved ancestors, extend to sequential decisions with matched data aggregation, and
qualify a genuine abort — a refusal here continues walking and is not a protective stop.

---

## 0. Verdict in six lines (written 00:30; see the update above)

1. The project has a working embodied pipeline, exact reproducibility, and an unusually
   honest failure ledger. It does **not** have a paper, and its intended headline
   ("a learned generator supplies better training scenes than analytic construction")
   is currently *contradicted* by its own data: all nine Motion2Scene training scenes
   are both-fail, and the only measured learned rescue came from the analytic arm.
2. The learner tie was a **bug, not a finding**: the 0.001 normalization clamp on 70
   constant dimensions blew test inputs up to 1952.75. The linear control on physical
   scales fixed it. The M2S data failure has a **diagnosed, cheap cause**: proposals
   were screened against the *reference* trajectory, and the achieved-transition
   screen catches all 13 false-clears. Both fixes exist. Neither has been run as data.
3. Compute is not the constraint. A cell costs ~33 s (0.0094 GPU-h); the whole project
   has spent 4.3 GPU-h. The whole ICRA experiment below costs ~6 GPU-h. The constraint
   is **scope discipline and GPU contention** (the 7500 MiB floor has already stalled
   a batch to 0/20 while another job holds 4.2 GB).
4. **Change the thesis now.** The paper is not "learned beats analytic". It is
   *"training scenes constructed from executed motion contrasts, with paired physical
   labels, teach a perceptive selector when to adapt — and here is which part of the
   construction does the work."* Analytic vs learned proposal becomes one factor in a
   4-arm ablation, and every outcome of that factor is reportable.
5. **Cut the plan to one budget, four arms, one learner, at most three carriers, the
   twelve frozen layouts.** Drop 24/48/96, drop the fifth arm, drop optimizer seeds
   (the learner is deterministic), drop the 9490xxx final-source acquisition. Those
   are the RA-L/IROS extension, not the ICRA paper.
6. Decision point is **Sep 10, 18:00**. If the transition-aware M2S arm has ≥1
   useful contrast per carrier on ≥2 carriers, write the four-arm paper. If a second
   carrier never qualifies or M2S is again all both-fail, still write the paper with
   the narrowed claim in §2.3 — but expect a modest acceptance probability and
   decide then, in writing, whether the same manuscript goes to RA-L instead.

---

## 1. Where the work actually stands

### 1.1 Established (physics-verified, reproducible, quotable)

| Claim | Evidence | Where |
|---|---|---|
| A generated/analytic overhead beam separates walk from crouch in closed loop under frozen SONIC | absent 3/3 both; present d055 3/3 at 0 N, upright 0/3 (73.6–762.1 N) | `BEAM_EXECUTION_V3_RESULT` |
| Generated constraints separate motions across sources | 6/8 pairs qualify, 18/24 requested slots separate, upright peaks 93.7–1542 N | `SOURCE_EXECUTION_V1_RESULT` |
| The deeper adaptation is *not* necessary: d040 clears the selected beam | d040 3/3 at 0 N | `BEAM_D040_EXECUTION_V1_RESULT` |
| The command timing window is narrow and measurable | seed 8042: 0.20 s fail (429 N), **0.30 s pass**, 0.40 s fail (565 N) | `ACTION_LABEL_COMPLETION_V1_RESULT` |
| A scripted upper/lower-ray selector works and its limits are mapped | 42 cells: reactive & oracle 9/10 poses; blind 0/10; 500 ms delay 0/2; 6/6 negative controls reject | `OVERHANG_VARIATION_V1_RESULT` |
| Execution-aware analytic construction is a strong, cheap competitor | 127/128 unique accepted, 99.0 % station coverage vs learned 37.3 % at equal queries | `COUNTERFACTUAL_STAGE_V1_RESULT` |
| Learned local-event proposals transfer to fresh sources | 384/384 vs uniform 30/384 on 43001–43008, −68.7 % search time; ties gradient | `FRESH_SOURCE_V1_RESULT` |
| A learned selector can realise a measured action advantage from rays | analytic-trained linear control requests d040 at 1.27 m/8512: 0 N vs walking 57.9 N | `SELECTOR_BREAKPOINT_RESULT` P3 |
| Reference-based proposal screening is the cause of useless M2S scenes | reference screen: 13 false-clears (9 in M2S); achieved-transition screen: 0 | `SELECTOR_BREAKPOINT_RESULT` P2 |

### 1.2 Contradicted or retracted (must not appear as claims)

- "d055 is the minimum required adaptation" — d040 clears 3/3.
- "Learned generator beats analytic / gradient" — ties at best (384/384 both); analytic has 2.7× coverage.
- "Distillation removes search cost without loss" — 377/384 vs 384/384; only 47 % total query reduction.
- Pose-robustness prediction (reactive > oracle) — both 9/10.
- The original 20 MLPs' independent-layout tie — a normalization defect, retained as a failure, not a comparison.
- Everything in `docs/paper/sweepcf_draft.md` (Aug 19) — different apparatus, Claim 5 never ran. Do not merge it into this paper.
- The August `q_LFH` conditional result (refuted by its own feature-blind baseline).

### 1.3 Unproved (the gap the paper must close or explicitly narrow)

- Any training-data advantage of *transition-aware* M2S scenes over analytic, no-contrast, or uniform scenes with the repaired learner.
- Any result on more than one carrier under the 0.30 s switching contract (41002 was the only qualified one until tonight; the transition-bank v2 batch finished at 00:30 with **41001, 41002 and 41003 all qualified at both seeds**, 12/12 cells, 0.105 GPU-h. K = 3 is available.)
- Source-held-out transfer (9490001–9490008 are reserved IDs, nothing generated).

### 1.4 The August LfH line

Sound geometry (closed-form window, bit-exact reproduction, `D_phi` at −39 % RMSE), two learned results retracted, Claim 5 never run, two causal families. It shares no apparatus with Motion2Scene and the README correctly refuses to pool it. **Keep it out of this paper** beyond one motivating sentence. It is a separate short paper or the RA-L appendix, not eight-page material.

---

## 2. The paper you can honestly write by Sep 15

### 2.1 Title and thesis

**Motion2Scene: Constructing Training Scenes from Executed Humanoid Motion Contrasts.**
(Keep "Transition-Aware Counterfactual Environments" as the subtitle only if the
transition-aware arm is the winner; otherwise it over-promises.)

> A humanoid's executed motion contrast — the same approach with and without an
> adaptation, both run through the frozen controller — identifies where an obstacle
> must sit to make the adaptation *useful*. We construct training scenes at those
> locations, label each with the paired physical outcome of both commands, and show
> that a fixed perceptive selector trained on such scenes learns when to adapt, while
> selectors trained on uniform or contrast-free scenes do not. We ablate which parts
> of the construction matter: execution-verified vs reference-based conditioning,
> contrast vs target-only objectives, and learned vs analytic proposal.

This thesis is true today for the analytic arm on one carrier, and the experiment in
§3 tests it on two to three carriers with all four arms. Its strength is that **every
outcome of the learned-vs-analytic factor is a finding**, so the paper cannot be
killed by that result.

### 2.2 Claim ladder (each row needs a number in the paper)

| # | Claim | Evidence | Have it? |
|---|---|---|---|
| C1 | Executed contrasts identify useful scenes; reference-based conditioning does not | P2 forecast (13 vs 0 false-clears); corpus yield (M2S-ref 0/9 useful vs analytic 4/8) | **yes** |
| C2 | Constructed scenes separate commands physically across sources | 6/8 sources, 18/24 slots; d040 vs d055; 0.30 s window | **yes** |
| C3 | Contrast-labelled scenes train a selector that adapts correctly on independent layouts; uniform and no-contrast do not | §3 experiment, four arms, frozen 12 layouts × 2 seeds × K carriers | **no — Sep 8–10** |
| C4 | The learned proposal model does / does not add value over execution-aware analytic construction at matched labels | same experiment, M2S vs analytic paired by layout | **no — same run** |
| C5 | Limits: sensing/timing failures are the binding constraint, not scene supply | 42-cell panel, both-fail low beams (1.18 m), 500 ms delay | **yes** |

Ship C1, C2, C5 with certainty; C3 is the paper's spine and is cheap; C4 is reported
whichever way it falls.

### 2.3 Pre-declare the three outcomes of C3/C4 now (write them into the register)

- **A. M2S-transition ≥ analytic > uniform, no-contrast.** Headline: transition-aware
  learned construction; analytic is the matched strong baseline.
- **B. analytic ≥ M2S-transition > uniform, no-contrast.** Headline: *execution-verified
  contrast construction* is what matters; the learned proposal buys search time
  (−69 %) and fresh-source transfer (384/384) but not label quality at this scale.
  This is a perfectly good ICRA paper and, on current evidence, the likeliest outcome.
- **C. all arms tie or all fail.** No selector paper. Report C1/C2/C5 as a
  data-construction and physical-validation study, and send it to RA-L with the
  larger study rather than to ICRA. Decide on Sep 10.

---

## 3. The one experiment: M2S-ICRA-v1

Everything below is a *narrowing* of `TRANSITION_AWARE_NEXT_STAGE.md` and
`LEARNING_UTILITY_PLAN_V1.md`, not a new design. Register it as one document before
spending, house style, predictions first.

### 3.1 Freeze

- **Controller, sensing, contract:** SONIC frozen; ideal 48-ray packet (144 values);
  single decision at 0.30 s; walk-commit vs request-d040; 3.3–3.5 s return; first-episode
  scorer (≤1 N, body origins cross 0.1 m, 0.3 s upright). Unchanged.
- **Learner (one, for all arms):** the deterministic regularized logistic control from
  P3 — physical scales, zero init, Adam 0.01, 2000 full-batch updates, 0.01 weight
  penalty, 214 inputs, two heads, walk-preferred rule, refusal→walk. The MLP is a
  retained failure in the appendix. Because the learner is deterministic, **optimizer
  seeds are removed**; uncertainty comes from carriers, layouts and physics seeds
  (§3.5), and from leave-4-out refits scored offline against the paired label table.
- **Carriers:** 41001, 41002, 41003 — all three qualified at both seeds tonight. Cap at three. No 9490xxx acquisition for ICRA. Say plainly:
  "unseen-layout transfer on qualified development carriers; source-held-out transfer
  is future work."
- **Generator:** the frozen all8/8421/1200 initializer + pattern-17 search, with the
  proposal *evaluator* switched from the complete-reference trajectory to the achieved
  empty-scene entry+exit transition (seed 8721 supplies proposals; 8722 is the
  predeclared repeat). No refitting.

### 3.2 Arms (four, not five)

| Arm | Construction | Role |
|---|---|---|
| Uniform | station [0.1, 0.9], underside [1.1, 1.45] m, common validity checks only | untargeted |
| Analytic | execution-aware capsule–box solver, global distinct search, on the *achieved* transition | claim-bearing comparator |
| No-contrast | same generator/search, target-clearance objective only, achieved transition | contrast ablation |
| Motion2Scene | frozen learned initializer + pattern search + independent audit, achieved transition | full method |

The reference-based M2S arm is **not re-run**: its 0/9 useful yield and the P2 forecast
are C1 and go in the paper as measured. Identical common background quota
(absent/raised/blocked, one quarter) acquired once per carrier and shared.

### 3.3 Budget

One budget: **24 complete encounter groups per arm**, pooled across carriers (8 per
carrier at K=3, 12 at K=2), 18 generated + 6 shared background, both commands labelled.
Charge every proposal, rejection and both-fail. No refills.

### 3.4 Cost (measured 33 s/cell, 375 s reserved, serial, 7500 MiB floor)

| Stage | Cells | GPU-h (measured) |
|---|---|---|
| Carrier qualification (running) | 12 | 0.11 |
| Label acquisition: 4 arms × 18 × 2 + background 6 × 2 × K(3) | 180 | ~1.7 |
| Evaluation: 12 layouts × 2 physics seeds × K(3) × (4 learned + scripted-ray + oracle-analytic) | 432 | ~4.1 |
| Reserve for retries / a fourth carrier | — | ~1.5 |
| **Total** | ~620 | **~7.5** (fits one week's 24 h envelope with margin) |

CPU: 4 fits + 4 × 6 jackknife refits, seconds. Generation: minutes.

### 3.5 Endpoints (register before the first label run)

- **Primary:** carrier-averaged contact-qualified passage on the traversal suite,
  arm vs arm **paired by (carrier, layout, physics seed)**. Report the paired 2×2
  counts (both pass / only A / only B / both fail) for M2S–analytic, M2S–uniform,
  analytic–uniform, M2S–no-contrast. No p-value hunting; with ~72 paired cells at K=3
  use exact binomial on discordant pairs and report the interval.
- **Secondary, own suites:** unnecessary d040 on absent/raised; blocked-scene
  refusal/infeasibility; d040 request rate; refusals that then walked successfully;
  peak force, resets, returns.
- **Comparators:** scripted upper/lower-ray rule and privileged analytic oracle,
  executed on the same cells.
- **Yield:** useful-contrast groups per requested scene and per GPU-h, per arm.
- **Predictions to register:** (i) analytic and M2S-transition each produce ≥1 useful
  contrast (walk-fail/d040-pass) per carrier; (ii) uniform ≤1 in total; (iii) no-contrast
  ≥6/9 both-pass; (iv) the achieved-transition screen has 0 false-clears among executed
  cells; (v) passage: contrast arms > uniform by ≥3 discordant cells net; (vi) M2S vs
  analytic: **no prediction** — declare it a two-sided comparison.

### 3.6 What to cut, explicitly

- 24/48/96 budgets and 60–75 fits → one budget, four fits.
- Fifth arm (reference-based M2S) → reported from existing data.
- Five optimizer seeds → deterministic learner, jackknife.
- Final-source acquisition (9490xxx) → future work; two sentences in Limitations.
- Any generator refit, sensing change, new skill, continuous certificates, hardware.
- The 480 paused original-MLP evaluations → stay paused forever; appendix row.

---

## 4. Schedule, Sep 7 → Sep 15

| When | GPU | Work | Gate |
|---|---:|---|---|
| Sep 7 (tonight) | 0.1 | Transition-bank v2 **done: 41001/41002/41003 qualified at both seeds, 12/12**. Write `DEVELOPMENT_TRANSITION_BANK_V1_RESULT.md`. Commit the untracked bank scripts/tests (test passes). Register `M2S_ICRA_V1.md` with §3.5 predictions. | **passed: K = 3** |
| Sep 8 | 1.7 | CPU: instantiate the 12 frozen layouts in each carrier's route frame; run all four constructions on achieved transitions; audit; freeze the 24-group lists. GPU: acquire labels (180 cells). **Start writing Intro / Method / Related work / C1–C2 sections in parallel — they do not depend on the result.** | all 24 groups labelled per arm |
| Sep 9 | 4.1 | Fit 4 controls + jackknife; command-check integration (recorded 12/12 protocol); run the 432-cell evaluation. | evaluation complete |
| Sep 10 | 0.5 | Analysis, paired tables, figures 2–4. **18:00 decision: outcome A / B / C (§2.3).** Freeze evidence hashes. | claim chosen |
| Sep 11–13 | 0 | Full draft: results, limitations, figure 1 composite; internal review with the `xiao-paper-review` rubric; cut to 8 pages incl. references. | draft complete |
| Sep 14 | 0 | Final read, hash-pinned evidence bundle, source tarball, README pointers. | — |
| Sep 15 | 0 | Submit before 11:59 PST. Video during 17–22 Sep from existing replays. | — |

Risk that actually threatens this: **GPU contention.** The other resident Isaac job holds
4.2 GB on a 16 GB card, and the floor has already produced a 0/20 launch. Coordinate the
GPU for the Sep 8–9 windows now, or run overnight; do not lower the floor mid-study.

---

## 5. Paper skeleton (8 pages including references)

1. **Introduction (¾ p).** Decorated scenes vs *useful* scenes; a single motion does
   not identify its obstacle, an executed contrast does; three questions the paper
   answers (does construction matter, which part, does learning the proposal help).
   Figure 1: one carrier — walk and d040 executions, the achieved transition, the
   beam placed by each arm, the paired outcomes, and the selector's request on an
   independent layout. All assets exist (`selector-breakpoint.mp4`, beam renders).
2. **Related work (½ p).** LfLH / LfH / LfH-CP (motion→scene already includes downstream
   learning: our distinction is executed humanoid contrasts and paired physical
   labels); HumanoidPF (generation utility as generalisation standard); Perceptive
   Humanoid Parkour (perceptive composition — we scope to data construction); SONIC
   (inherited). Cite rliable for paired small-n reporting. Five to seven more cites
   on scene generation for embodied AI. No novelty over-claim.
3. **Problem and construction (1½ p).** The contrast pair; the 0.30 s switching
   contract; the four constructions; achieved-transition screening; independent audit
   and refusal; labels as Y(scene, state, phase, history, command). One table.
4. **Physical validation (1 p).** C2 numbers; d040/d055; timing window; sensing panel
   summary; the P2 false-clear result as the bridge to C1. Figure 2: force/outcome
   panel across sources and heights.
5. **Learning-utility study (2 p).** Setup, arms, learner, frozen layouts, endpoints;
   paired 2×2 tables; yield per GPU-h; comparators; jackknife stability. Figure 3:
   per-(carrier, layout, seed) outcome heatmap by arm. Figure 4: yield and cost.
6. **Limitations and retained failures (½ p).** One beam family, two commands, ideal
   rays, K ≤ 3 development carriers, no source-held-out transfer, one controller;
   the MLP normalization failure; low-beam both-fail; 500 ms delay.
7. **Conclusion (¼ p).** References (~1 p).

Do not spend pages on the CPU generator lineage (inverse → uncertainty → margin →
carrier → event → refinement → station → fresh → distillation). One paragraph plus a
supplementary pointer to `docs/motion2scene/`.

---

## 6. Reviewer objections you will get, and the honest answers

- *"It's one beam and two commands."* Yes; say so in the first paragraph of Limitations
  and in the abstract ("a controlled minimal instance"). The paper's value is the
  construction-and-label protocol and the ablation, not breadth.
- *"Ideal rays, no perception."* Declared. The scripted-ray comparator bounds what
  perception could add; the 500 ms delay result shows the selector is timing-limited.
- *"n=2–3 sources."* Paired design, all cells shown, carrier-grouped reporting, no
  pooled percentages. Source-held-out transfer is named as the next study.
- *"The learned generator doesn't beat the analytic one."* If outcome B: that is the
  finding, and it is more useful to the field than a marginal win. State what the
  learned model does buy (search time, fresh-source transfer) with numbers.
- *"Why not RL / a diffusion baseline / hardware?"* Out of scope by design; each is
  one sentence.

---

## 7. Rules that this project has already paid for (keep)

- Register before spend; predictions first; infra failure ≠ rejection; no retries
  without charge.
- Physics is the verdict; screens, forecasts and MuJoCo replays never re-grade.
- One decision time (0.30 s); old 0.20 s labels are not 0.30 s labels.
- Fixed learner across arms; any learner change is applied to every arm and declared.
- Derivatives, physics repeats and layouts stay with their carrier; report grouped.
- Retained failures go in the paper, not only in the ledger.
- Never let a test-bank failure tune generation, sensing or the controller.

## 8. Do not

- Do not run 24/48/96, five arms, or 75 fits before Sep 15.
- Do not acquire or "qualify" the 9490xxx sources this week.
- Do not resume the 480 paused MLP evaluations.
- Do not refit the generator on d040 or on achieved transitions.
- Do not merge the August SweepCF/LfH apparatus or numbers into this manuscript.
- Do not lower the 7500 MiB floor or the 375 s timeout to make a batch fit.
- Do not write the abstract until the Sep 10 decision.

---

## 9. If I had to pick one thing

Write up the transition-bank result tonight, register M2S-ICRA-v1 with §3.5's predictions,
and start the Sep 8 label acquisition on achieved transitions. Everything else in the
paper is already measured. The single thing standing between this project and a
submittable manuscript is **one 180-cell labelled corpus built the right way**, and it
costs under two GPU-hours.

---

## 中文摘要

1. 现状：pipeline 完整、可精确复现、失败记录诚实，但**没有稿子**；原定主张"学习式生成器优于解析式"被自己的数据否定（M2S 9/9 both-fail，唯一的学习式 rescue 来自 analytic arm）。
2. 两个已诊断的原因都是可修的：selector 的 0.001 归一化 bug（线性 control 已修）；生成器用 reference 轨迹而非实际执行的 transition 做筛选（achieved-transition screen 抓住全部 13 个 false-clear）。修好后的数据一个都还没采。
3. 算力不是瓶颈（每 cell 33 s，全项目共用 4.3 GPU-h，整个 ICRA 实验约 6–8 GPU-h）；瓶颈是 scope 和 GPU 抢占（7500 MiB 门槛已让一批 0/20 启动）。
4. **立刻改论点**：不是"学习 vs 解析"，而是"从执行过的动作对比构造训练场景 + 成对物理标签，能教会感知型 selector 何时适应；并消融构造中哪一部分起作用"。学习 vs 解析变成四臂消融里的一个因子，无论结果如何都可写。
5. **砍到最小**：一个 budget（24 组/臂）、四臂、一个确定性线性 learner、≤3 个 carrier、12 个冻结 layout。去掉 24/48/96、第五臂、optimizer seeds、9490xxx 新 source。
6. 时间：9/7 晚完成 carrier 资格 + 注册；9/8 采标签（~1.7 h）并同步开写不依赖结果的章节；9/9 评估（~4.1 h）；**9/10 18:00 决定 A/B/C**；9/11–14 写；9/15 提交。若只有单 carrier 或全臂平局，则同一稿改投 RA-L。
