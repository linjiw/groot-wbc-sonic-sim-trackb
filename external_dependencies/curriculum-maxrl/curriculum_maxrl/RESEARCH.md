# Deep-research synthesis: modern curriculum learning for RL (2020–2026) and where MaxRL fits

Produced by a multi-agent research pass (5 search angles → 15 sources → 3-vote
adversarial verification per claim; all "high" findings survived 3-0 against
primary sources). Condensed here; task-relevant implications at the end.

## The shared architecture of modern curricula

Every modern method = **a per-task signal estimating learning value for the
current policy** + **a mechanism adapting the sampling distribution toward the
frontier of competence**. The lineages differ in the signal:

### 1. Regret-based teacher–student / UED (high confidence, primary-source verified)

- **PAIRED** (arXiv:2012.02096): environment-generating adversary maximizes
  *regret* = antagonist return − protagonist return. Key mechanism: unsolvable
  environments give regret exactly 0, so the teacher is structurally
  disincentivized from proposing them; motivated by the failures of domain
  randomization (no adaptation) and pure minimax (generates unsolvable levels
  with no learning signal). Produces an emergent curriculum "within the
  agent's zone of proximal development".
- **PLR** (arXiv:2010.03934): selective *replay* of procedurally generated
  levels, priority = time-averaged L1 value loss (|GAE|);
  `P_replay = (1−ρ)·P_score + ρ·P_staleness`.
- **Robust PLR / Dual Curriculum Design** (arXiv:2110.02439): unifies PLR and
  PAIRED with minimax-regret guarantees at Nash equilibria; uses
  regret-approximating scores (Positive Value Loss, MaxMC); provably robust
  *because* it refuses to train on uncurated levels — "train on less data".
- **ACCEL**: evolves/edits previously high-regret levels instead of sampling
  fresh — fixes random search's curse of dimensionality in big design spaces.

### 2. Learning-progress (LP) / ZPD lineage (high confidence)

- **TSCL** (Matiisen et al.): bandit over the estimated *derivative* of
  per-task performance; optimal under concave learning profiles (Lopes &
  Oudeyer 2012).
- **ALP-GMM** (arXiv:1910.07224): GMM over (task-parameter, |ΔLP|) pairs on
  continuous task spaces. *Absolute* LP detects competence loss → directly
  counters forgetting. Empirically beats a hand-designed difficulty-sliding
  Oracle (80% vs 68% mastered on 12D Hexagon Tracks, p<0.02) precisely because
  the Oracle forgets easy tasks; most robust LP teacher as the fraction of
  unlearnable tasks grows.
- Survey position (arXiv:2003.04664): LP beats fixed intermediate-difficulty
  (ZPD-band) targeting because it needs no threshold and is robust to tasks
  stuck at intermediate scores where no improvement is possible.

### 3. Learnability / success-variance (high confidence — most relevant to us)

- **SFL / "No Regrets" critique** (NeurIPS 2024, arXiv:2408.15099): showed the
  practical regret proxies (MaxMC, PVL) actually correlate with *success
  rate*, not regret — SOTA UED curricula spend most experience on
  already-mastered levels. Proposes direct **learnability = p(1−p)** (Bernoulli
  variance of success; zero at p=0 and p=1), buffers inconsistently-solved
  levels, trains on a buffer/random mix.

### 4. RLVR-specific curricula (LLM reasoning)

- GRPO zero-gradient problem: all-fail or all-pass groups give zero advantage
  → wasted compute; **DAPO dynamic sampling** oversamples and filters groups
  with 0 < K < N; **ADARFT** targets a pass-rate band around a difficulty
  target. Structurally the same zero-signal phenomenon as regret-0 unsolvable
  levels in UED (verifier-endorsed analogy).

## Where MaxRL sits in this taxonomy (synthesis finding, medium confidence)

MaxRL's w(p) ≈ 1/p reweighting is an **implicit, objective-level difficulty
curriculum**:

- (a) It *subsumes* explicit hard-prompt upweighting by folding it into the
  loss — analogous in intent to regret's frontier targeting.
- (b) Unlike learnability p(1−p), the 1/p weight grows monotonically as p→0 —
  favoring near-unsolvable prompts that the SFL evidence says carry ~zero
  learning signal; the Maclaurin truncation (T=N) is exactly what bounds the
  resulting p≈0 variance explosion.
- (c) It **cannot rescue truly all-fail prompts** (K=0 → group dropped): the
  same zero-signal structure as regret-0 levels (UED fixes via teacher
  disincentive) and all-fail groups (DAPO fixes via resampling/filtering).
- (d) As a static pass-rate signal, it inherits the two failure modes LP
  methods target: can't distinguish hard-but-learnable from stagnant/
  unlearnable prompts, and has no anti-forgetting mechanism (though its w(p)
  keeps weight ≈1 on easy prompts rather than zeroing them like GRPO).

Suggested integrations from the literature: PLR-style prompt replay buffers
scored by learnability or LP with MaxRL weighting inside the curated batch;
DAPO-style p=0 filtering before likelihood reweighting.

## Key implication discovered during our design work

Our MaxRL-native frontier utility

```
u_N(p) = (1 − (1−p)^N) · (1−p) = pass@N · (1−p)
```

**reduces exactly to SFL's learnability p(1−p) at N = 1** and, as N grows,
shifts its mass toward harder prompts (any p ≳ 1/N stays near-max utility). So
it is a *compute-indexed generalization of learnability* that widens the ZPD
band at exactly the rate the MaxRL estimator can absorb it (a group of N
rollouts produces signal with probability pass@N). The teacher and the
objective are indexed by the same compute knob — which is the conceptual core
of the curriculum-MaxRL integration.

Remaining gap (from lineage 2): u_N(p) is still a static difficulty signal —
it cannot detect stagnation or forgetting. A production version should add an
ALP-style |Δp̂| term (detect regression on mastered prompts, re-inject them)
on top of the uniform-floor replay we already include.

## Primary sources

- PAIRED: https://arxiv.org/abs/2012.02096
- PLR: https://arxiv.org/abs/2010.03934
- Robust PLR / DCD: https://arxiv.org/abs/2110.02439
- SFL / learnability critique: https://arxiv.org/abs/2408.15099
- ACL survey: https://arxiv.org/abs/2003.04664
- ALP-GMM: https://arxiv.org/abs/1910.07224
- MaxRL: https://arxiv.org/abs/2602.02710
