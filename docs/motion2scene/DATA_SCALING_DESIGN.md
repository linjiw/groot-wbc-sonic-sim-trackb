# Scaling the motion data without inflating the evidence

The [training-source distillation pilot](DISTILLATION_V1_RESULT.md) now completes a teacher bank, twelve student/control fits and 3,456 independently checked outputs. Hybrid raw accepts **334/384**, original raw **292/384**, and the stronger query-budget control **310/384**. Hybrid five-evaluation search accepts **377/384**, versus **384/384** for original seventeen-evaluation search. These are results on observed development sources; the 430xx pool remains excluded.

The recipe misses at least one registered source-wise raw-proposal or reduced-search criterion. Prioritize diagnosing teacher coverage, source-specific geometry and proposal concentration on development data before another fresh acquisition. Do not treat pooled gains or cheaper search alone as preserved performance.

Status: The [fresh-source audit](FRESH_SOURCE_V1_RESULT.md) now completes eight new source motions and 16 derivatives, with all seven methods frozen before acquisition. Learned pattern accepts **384/384**, uniform pattern **30/384**, original gradient **384/384** and the raw model **296/384**. Learned pattern improves over uniform on every source, but its prediction of higher acceptance than gradient fails because the counts tie. Probe/refinement retains one failure. The complete source-wise predicates and costs are retained. These are finite reference checks on CPU-generated straight walking, not physical execution or a collision guarantee.

The original inventory and source/phase records remain unchanged. The acquisition comparison is complete. Move the learning work toward [train-only search distillation](TRAIN_ONLY_DISTILLATION_DESIGN.md): test whether synthetic geometric targets improve raw proposals and reduce online query cost. Preserve all 43001–43008 sources and derivatives as excluded from training and selection. Another confirmatory transfer comparison requires a separately registered source pool. Develop imported-body and temporal clearance contracts alongside the learning work.

The equal-query eight-parent comparison improves unseen-phase yield by 12.5 percentage
points, but one test source regresses. The two six-parent subsets differ by 35/192 valid
draws. The subsequent [matched-query study](REFINEMENT_V1_RESULT.md) raises hybrid yield to
189/192 through bounded geometric correction. The [station-search comparison](STATION_SEARCH_V1_RESULT.md) then removes five
original-gradient failures on fresh draws. Use the [fresh-source result](FRESH_SOURCE_V1_RESULT.md)
and source-composition evidence before the proposed 4/8/16 expansion below. This is evidence for testing selective data
acquisition, not evidence that the better observed subset will generalize to fresh data.

## Available data and the actual independent unit

The table below preserves the pre-study inventory state; the completed source/phase
registry records the subsequent 80 target constructions separately.
The [inventory](evidence/data-scale-inventory.json) verifies the saved factorial design,
its reference summary, and the eight additional route-retention CSV hashes. It covers
these named collections, not every motion anywhere in the workspace.

| Collection | Motion files/cases | Source groups | Use and limitation |
|---|---:|---:|---|
| Original factorial study | 144 references: 6 body modes × 3 routes × 8 seeds | 8 generation groups | All prompts/routes sharing a seed stay in the same split |
| Current edited crouch bank | 24 target cases, with 8 neutral parents | The same 8 groups | Three event edits add variation, not independent parents |
| Additional route-retention set | 8 neutral walking references | 8 additional seeds, 42001–42008 | Previously observed; no new matched crouch pair is qualified here |

The factorial study therefore cannot be treated as 144 independent training/test sources.
The additional neutral set is a possible development acquisition input, not eight fresh
confirmatory pairs. The completed source/phase study assigns this set new development roles 4/2/2 before
constructing targets, while preserving the historical study. Its prior empty-scene tracking outcomes do not predict whether a new
crouch will execute or whether a generated obstacle is compatible with that execution.
A genuinely fresh final test requires new source identities and outcomes left uninspected
until the model and protocol are fixed.

## Longer-term data-size comparison

First use the current 2-versus-4-parent result to decide which bottleneck needs study. It
has only one nested subset: a gain or regression can reflect which parents were added,
rather than a general effect of dataset size. More repetitions of those parents cannot
resolve this ambiguity.

For the next acquisition, target a training pool large enough for nested **4/8/16-parent**
curves, with separate validation and test pools. This is a proposed capacity requirement,
not a qualification-yield promise. Fix candidate counts, source identities, generation
settings, complete derivative lineage, stop rules and the source split before acquisition.
Use multiple predeclared nested subset orders inside the training pool to measure subset
sensitivity; keep test parents identical across those comparisons.

Perform two compute contracts separately:

- **Equal geometry-query budget:** bigger datasets receive fewer visits per parent. This
  measures the trade-off under a fixed training allocation.
- **Equal visits per parent:** larger datasets receive proportionally more queries. This
  measures the combined benefit/cost of more data and more optimization.

Do not label their difference a pure data-size effect. Report candidate, reference-gate,
geometric-feasibility and execution-qualification funnels separately. Preserve rejected
candidates and empty feasible sets instead of replacing them until the benchmark looks easy.
The reference-only learner may be studied separately from the execution-qualified dataset,
with the distinction carried on every result.

## Add information, not just edits

Withhold event stations or event durations as a separate generalization axis. The current
three event positions appear in every split, so a good carrier result alone cannot prove
novel phase interpolation. Keep all withheld-event derivatives attached to the original
parent group and avoid fitting normalization or selecting checkpoints on them.

Start with the straight-route beam family while isolating data effects. Curved routes
require a specified local heading/frame decoder; the existing end-to-end chord heading
is not a validated general curved-route orientation model. Later vary motion speed and
body envelopes, then add lateral-gap and step-over families with appropriate alternatives
and contact contracts. More objects in one scene require a joint whole-sequence check;
per-object acceptance does not imply scene-level acceptance.

Acquire additional executed target/alternative pairs only with the frozen controller,
predeclared route/event retention gates, and explicit treatment of failed attempts.
Execution seeds estimate tracking variation, not new motion-source diversity. The next
collision work still needs actual imported body geometry, between-frame clearance and
continuous placement uncertainty before accepting an executable scene guarantee.

## Decision rule

Spend on larger data when the completed comparisons indicate that local motion features
are useful and when the next source pool adds a distinct tested axis. If extra updates or
parameters fail, keep that failure visible and inspect the representation/objective before
allocating an indiscriminate larger run. Any proposed downstream policy benefit requires
its own fixed-learner, independent-scene experiment; reference proposal yield cannot stand
in for that result.
