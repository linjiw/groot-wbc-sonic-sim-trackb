# LACE competitive-landscape verification

Verified against primary sources on 2026-08-13. These facts constrain the LACE thesis and
baseline table; they are not evidence that LACE's own hypotheses are true.

## SONIC release baseline

- The current SONIC paper is [arXiv:2511.07820 v4](https://arxiv.org/abs/2511.07820), dated
  2026-08-13 and associated with Science Robotics 11(117), 2026.
- The official repository's base motion configuration enables adaptive sampling by default,
  with 50-frame bins, a pseudocount of one, a 0.1 uniform component, and a 200-frame
  pre-failure window. See the pinned
  [motion configuration](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/gear_sonic/config/manager_env/commands/terms/motion.yaml#L16-L25).
- The release experiment raises the relative failure-rate cap to 200; see the pinned
  [release configuration](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/c374bae5b9039cd0ee71377e654d11ce1bc69e1d/gear_sonic/config/exp/manager/universal_token/all_modes/sonic_release.yaml#L68-L73).
- The implementation estimates early-termination frequency per temporal bin, caps it relative
  to the global mean, normalizes it, mixes in uniform exploration, applies duration weighting,
  and samples near failure. Therefore the primary RQ2 baseline is the exact released scalar
  failure-rate sampler. Uniform sampling is only a floor. RQ1 intervention runs must freeze or
  disable the native sampler so it cannot move underneath a causal upweighting row.
- The repository documentation calls `filter_and_copy_bones_data.py` a physical-feasibility
  filter, but the implementation only rejects filename keywords such as beds, stairs, chairs,
  handstands, and obstacle jumps. It computes no joint-, contact-, torque-, or dynamics-based
  feasibility quantity. Passing the release filename filter is therefore an eligibility control,
  not evidence that a reference is physically feasible; RQ1 still needs an independent
  reference-feasibility covariate and explicit corruption checks.

## Direct novelty constraints

### EGM / BCCAS

[Error-Guided Motion Learning](https://arxiv.org/abs/2512.19043) v1, dated 2025-12-22,
already globally indexes one-second motion bins, maintains an EMA of composite scalar tracking
error, and anneals sampling toward high-error bins while preserving uniform exploration.
Consequences:

- LACE cannot claim the first cross-motion or global temporal-bin curriculum.
- The baseline suite requires an EGM-style composite-error sampler at matched budget.
- No official code was linked by the paper at verification time, so this arm must be labelled a
  paper-spec reimplementation and expose underspecified choices as sensitivity analyses.

### Athena-WBC

[Athena-WBC](https://arxiv.org/abs/2607.04837) v2, dated 2026-07-07, selects a residual hard
set with a general teacher, trains dynamic and balance specialists on that set, routes each
motion to the best teacher by rollout success, and then distils/fine-tunes. Its structured
specialization includes relaxed effort/temporal regularization and reduced-to-nominal gravity
continuation. Consequences:

- LACE cannot claim the first failure-guided specialization, capability routing, or structured
  curriculum intervention.
- An Athena-style non-language structured controller/oracle is a relevant RQ3 comparator if the
  live configuration can apply and read back the same interventions.
- Athena-WBC does not estimate a pairwise source-to-target training-transfer geometry; its
  diagnostic manifests use semantics and reference-motion proxies. That leaves LACE's narrow
  H1 distinct.

## Adjacent work that narrows rhetoric

- [ASPIRE](https://arxiv.org/abs/2607.00272), 2026-06-30, maps multimodal execution traces to
  failure diagnosis, repair code, validation, and reusable skills in manipulation. Language
  diagnosis plus verified structured repair is therefore not itself novel.
- [OmniTrack](https://arxiv.org/abs/2602.23832), 2026-02-27, explicitly addresses reference
  feasibility. LACE needs a feasibility filter/covariate so mechanism signatures are not merely
  detecting corrupt or infeasible references.
- [ZEST](https://arxiv.org/abs/2602.00401), 2026-01-30, combines adaptive hard-segment sampling
  with a model-based assistive-wrench curriculum. It is relevant structured-curriculum prior art.
- [FAST](https://arxiv.org/abs/2602.11929), 2026-02-12, studies parameter-efficient residual
  adaptation to hard and unseen motions with retention constraints. Avoid broad claims about
  efficient hard-motion adaptation.
- [GTP-FA](https://arxiv.org/abs/2606.03385), 2026-06-02, studies stable failure-mode
  distributions and diagnosis-guided optimization in manipulation; cite it as cross-domain
  precedent.

## Defensible thesis after verification

> A frozen, policy-conditioned, multi-mechanism representation predicts measured
> source-to-target training transfer better than semantic, reference-kinematic, feasibility,
> and scalar-difficulty representations, and can support fixed-budget allocation without adding
> experts or model capacity.

No primary-source humanoid whole-body-control work found in this verification performed that
exact transfer-geometry test. This is a bounded search conclusion, not proof of universal
novelty; repeat the search immediately before submission.
