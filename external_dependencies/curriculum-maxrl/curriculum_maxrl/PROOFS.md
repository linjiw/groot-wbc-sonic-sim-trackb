# Formal results for curriculum-MaxRL

Self-contained statements and proofs of the identities the algorithm rests
on. Every result is also verified numerically (`run_validation.py`,
THEORY.md §5 snippet). Notation: a prompt has pass rate `p ∈ (0,1)`; a group
is `N` i.i.d. rollouts with `K ~ Bin(N, p)` successes; the MaxRL
variance-reduced weights (paper eq. 10 / Algorithm 1) are

```
w_i = r_i/K − 1/N   for K ≥ 1,      w_i = 0 for all i when K = 0.
```

---

## Proposition 1 (Exact expected advantage mass of MaxRL)

**Claim.** `E[Σᵢ|wᵢ|] = 2·(pass@N(p) − pass@1(p)) = 2·((1−(1−p)ᴺ) − p).`

**Proof.** Condition on K. For K ≥ 1 there are K successes with weight
`1/K − 1/N > 0` (since K ≤ N) and N−K failures with weight `−1/N`, so

```
Σ|w| = K(1/K − 1/N) + (N−K)/N = 1 − K/N + 1 − K/N = 2(1 − K/N).
```

For K = 0 the group is dropped: Σ|w| = 0. Therefore

```
E[Σ|w|] = 2·E[(1 − K/N)·1{K≥1}]
        = 2·(P(K≥1) − E[K·1{K≥1}]/N)
        = 2·(P(K≥1) − E[K]/N)          (K·1{K≥1} = K a.s.)
        = 2·((1−(1−p)ᴺ) − p).          ∎
```

**Interpretation.** The learning signal a prompt commands equals twice the
probability it is *solvable within N attempts but not within one*. This is
the estimator's own zone-of-proximal-development functional — the teacher
utility is not a heuristic added on top of MaxRL but a quantity MaxRL already
computes implicitly.

---

## Proposition 2 (Peak of the utility)

**Claim.** `u(p) = (1−(1−p)ᴺ) − p` on [0,1] is strictly concave with unique
maximizer `p* = 1 − N^(−1/(N−1))`, and `p* = ln N / N + O((ln N / N)²)`.

**Proof.** `u′(p) = N(1−p)^{N−1} − 1`, `u″(p) = −N(N−1)(1−p)^{N−2} < 0` on
(0,1), so u is strictly concave; setting u′ = 0 gives
`(1−p*)^{N−1} = 1/N`, i.e. `p* = 1 − N^{−1/(N−1)}`. For the asymptotic,
`N^{−1/(N−1)} = exp(−ln N/(N−1)) = 1 − ln N/(N−1) + O((ln N/N)²)`, hence
`p* = ln N/(N−1) + O((ln N/N)²) ≈ ln N / N`. ∎

**Interpretation.** Doubling the group size moves the optimal difficulty
band harder by a factor ≈ `ln 2N / ln N · N/(2N) ≈ 1/2` in pass-rate terms:
the curriculum and the objective share one compute knob, with an exact rate.

---

## Proposition 3 (Greedy rollout allocation is optimal)

**Claim.** For fixed prompt set with pass rates p₁..p_m and budget
B = Σᵢ Nᵢ, total mass `M = Σᵢ u_{Nᵢ}(pᵢ)` is maximized by greedy
water-filling on the marginal `Δᵢ(N) = pᵢ(1−pᵢ)ᴺ`.

**Proof.** The increment of one more rollout on prompt i is

```
u_{N+1}(pᵢ) − u_N(pᵢ) = (1−pᵢ)ᴺ − (1−pᵢ)^{N+1} = pᵢ(1−pᵢ)ᴺ > 0,
```

which is strictly decreasing in N. So M is a sum of separable concave
functions of the integer allocation; for such problems the greedy algorithm
(repeatedly assign the next unit to the largest current marginal) is exact —
the standard exchange argument: swapping any unit from a prompt with a
smaller marginal to one with a larger marginal cannot decrease M, and
marginals only shrink as units are added. ∎

**Interpretation.** `pᵢ(1−pᵢ)ᴺ = P(rollout N+1 is the group's first
success)`. Optimal compute allocation = "give the next rollout to the prompt
where it is most likely to flip a dead group live."

---

## Proposition 4 (RLOO advantage mass = learnability)

**Claim.** For RLOO weights `wᵢ = (rᵢ − r̄₋ᵢ)/N` (leave-one-out baseline),
`E[Σ|w|] = 2p(1−p)·N/(N−1) → 2p(1−p)`.

**Proof.** With K successes: a success has weight `(1 − (K−1)/(N−1))/N =
(N−K)/(N(N−1))` and a failure `−K/(N(N−1))` in magnitude. Summing:

```
Σ|w| = K(N−K)/(N(N−1)) + (N−K)K/(N(N−1)) = 2K(N−K)/(N(N−1)).
E[K(N−K)] = N·E[K] − E[K²] = Np − (Np(1−p) + N²p²) = N(N−1)p(1−p).
⇒ E[Σ|w|] = 2p(1−p).                                        ∎
```

**Interpretation.** SFL's "learnability" curriculum objective p(1−p)
(Rutherford et al. 2024) *is* the advantage mass of the RLOO estimator. The
curriculum literature and the estimator algebra converge on the same
functional from opposite directions — and our Prop. 1 shows MaxRL
generalizes it to a compute-indexed family (u_N → learnability at N=1).

---

## Proposition 5 (Signal ordering: why MaxRL dominates on the frontier)

**Claim.** For all N ≥ 2 and p ≤ p*(N):
`u_MaxRL(p) ≥ u_RLOO(p)` with ratio `→ N` as p → 0.

**Proof.** `u_MaxRL(p)/2 = (1−(1−p)ᴺ) − p = Σ_{k=1}^{N} C(N,k)pᵏ(−1)^{k+1}... `
simpler: for small p, `1−(1−p)ᴺ = Np − C(N,2)p² + O(p³)`, so
`u_MaxRL/2 = (N−1)p + O(p²)` while `u_RLOO/2 = p(1−p) = p + O(p²)`.
Ratio → N−1 ≈ N. Both vanish at p ∈ {0,1}; on (0, p*] the MaxRL mass is
strictly larger since `(1−(1−p)ᴺ) − p ≥ p(1−p)` ⇔ `1−(1−p)ᴺ ≥ 2p−p²
= 1−(1−p)²`, true for N ≥ 2. ∎

**Interpretation.** On frontier prompts (p small but nonzero) MaxRL's
estimator concentrates ~N× more expected signal than RLOO's — the
finite-sample mechanism behind the paper's "MaxRL extracts more learning
signal" (their Fig. 7), and the reason the same teacher helps MaxRL more
safely than GRPO (whose mass has a √p singularity in *ratio* terms but is
throttled by dead groups in absolute terms; see THEORY.md §2).

---

## Proposition 6 (Hindsight relabeling: characterization of the update)

Setting: dead group (K = 0) on task τ; each rollout i has an achieved
prefix/goal g(zᵢ); relabel to goal g* achieved by at least one rollout, with
r̃ᵢ = 1{g(zᵢ) reaches g*}, and apply the MaxRL weights w̃ to the truncated
trajectories.

**Claim.** The relabeled update equals the success-conditioned ML gradient of
the *relabeled* task τ(g*), estimated under the conditional sampling law
`z ~ m_θ(·|τ) | {K_τ = 0, g* achieved}` instead of `z ~ m_θ(·|τ(g*))`. It is
therefore (a) an exact ML-gradient direction whenever achieving g* from τ's
prompt and from τ(g*)'s prompt induce the same conditional trajectory
distribution (true on the skill chain, where the prompt does not enter the
policy; approximately true when prompts share the relevant context), and
(b) biased in general, with bias controlled by the divergence between those
two conditional laws.

**Proof sketch.** By the paper's Theorem 1,
`∇J_ML(τ(g*)) = E[∇log m_θ(z) | success on τ(g*)]`. The relabeled average is
the same functional applied to samples from the *other* conditional law. On
the skill chain both laws are products of the same per-skill categoricals
restricted to "prefix correct," hence identical — the update is exactly the
ML gradient in expectation (V1 verifies: cosine of the *mean* relabeled
gradient to the true gradient ≈ 1). In general the gap is
`E_ν[∇log m] − E_μ[∇log m]` for two conditionals ν, μ over successful
trajectories, bounded by `sup‖∇log m‖ · TV(ν, μ)`. ∎

**Practical reading.** Hindsight is not "biased noise" — it is the right
gradient *for a shifted task distribution*. The failure mode to watch is not
gradient direction but *coverage*: relabeled goals are those the current
policy stumbles into, so pure hindsight would drift toward self-reachable
goals (the HER drift). The teacher's floor + the original-task groups anchor
against this.

---

## Proposition 7 (Explore/exploit: what the Thompson teacher pays)

The teacher faces a nonstationary bandit: arm = prompt, payoff = advantage
mass `u_N(p_t(a))` where `p_t` moves as the student learns. Two structural
facts shape the design:

1. **Bounded regret-per-step against a static oracle.** With Thompson
   sampling on a Beta posterior with decay (effective sample size
   `ESS = 1/(1−γ)` at decay γ), the posterior tracks a drifting p with lag
   `O(1/ESS)`; the sampling distribution differs from the oracle's by
   `O(|û−u|/Σu)`. V2 measures this gap end-to-end (oracle vs Thompson AUC).
2. **The floor is not a tuning nicety but a lower bound on information.**
   With floor f, every prompt is sampled at rate ≥ f/m, so the posterior's
   staleness is bounded and *mastered-then-forgotten* prompts are re-detected
   within `O(m/f)` groups. Setting f = 0 makes forgetting undetectable
   (a prompt with p̂ ≈ 1, u ≈ 0 is never revisited) — V3's f = 0 arm tests
   exactly this failure.

These are design constraints rather than theorems; V2/V3 quantify them.
