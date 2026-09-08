# The completed 540-run panel: one contrast, and where it helps

**September 7, complete.** All **82/82 labeling runs** and all **540/540 evaluation
runs** are admitted: 432 traversal executions over 72 conditions with six policies
each, plus the 108 background control runs. Four primary deterministic learners and
24 prescribed leave-four-out refits are complete. Nothing remains pending.

## Data quality and the failed acquisition quota

| Data arm | Complete / requested groups | Both pass | d040 only | Walk only | Both fail |
| --- | --- | --- | --- | --- | --- |
| uniform | 24/24 | 10 | 0 | 0 | 14 |
| analytic | 7/24 | 4 | 1 | 0 | 2 |
| no_contrast | 22/24 | 20 | 0 | 0 | 2 |
| motion2scene | 6/24 | 4 | 0 | 0 | 2 |

The six shared background pairs are counted in each arm's fitting set but acquired
once. Generated groups are uniform 18/18, analytic 1/18, target-only 16/18 and M2S 0/18.
**Exactly one acquired group in the whole study is a useful walk-fail/d040-pass
contrast**, from analytic construction on 41001. The registered uniform <=1 useful and
target-only >=12/18 both-pass predictions hold; the prediction of at least one useful
generated contrast per carrier for analytic and M2S fails. The analytic and M2S
training sets therefore differ by exactly that one group and are otherwise identical.

The equal-24-complete-label goal fails in three arms. These are outcomes at equal
requested slots with unequal acquired data and costs. **The M2S-arm model is trained
only on shared backgrounds.** Its performance cannot establish the quality of accepted
M2S-generated training scenes, because no such scenes were acquired.

## Actual policy executions over all 72 traversal conditions

| Policy / data arm | Passage | d040 requests | Refusals followed by walking |
| --- | --- | --- | --- |
| Uniform | 28/72 | 0/72 | 32/72 |
| Analytic | 39/72 | 29/72 | 0/72 |
| Target-only | 28/72 | 0/72 | 0/72 |
| M2S: background only | 28/72 | 0/72 | 0/72 |
| Scripted rays | 46/72 | 46/72 | 0/72 |
| Privileged geometry | 39/72 | 14/72 | 34/72 |

Analytic versus uniform has 28 both-pass,
11 analytic-only, 0 uniform-only
and 33 both-fail conditions, with the same
11-to-0 table against target-only
and against background-only M2S. The carrier-averaged difference is
15.278 percentage points, positive on all three
carriers (6 on 41001, 2 on 41002, 3 on 41003). All 11 additional passages request d040 and measure
0 N beam force; their matched walking comparators record 385.1-1685.5 N.
The condition-level discordance p is 0.000977, but
72 conditions are twelve layouts and two seeds inside three carriers, so the honest
statement is the carrier-level one: same direction on all three, n = 3.

Uniform, target-only and background-only M2S are **behaviourally identical**: each
issues walking on every condition and passes 28/72.
Adding eighteen untargeted groups or sixteen target-only groups changes the fitted
policy not at all; adding one contrast changes it completely. Both statements rest on
a single acquired example, so the effect size is not estimable here.

## Where the adaptation helps, and where nothing does

| Beam underside | Conditions | Analytic d040 | Analytic passes | Scripted d040 | Scripted passes | Walking-only passes |
| --- | --- | --- | --- | --- | --- | --- |
| 1.18 m | 24 | 11 | 0 | 11 | 0 | 0 |
| 1.27 m | 24 | 14 | 15 | 23 | 22 | 4 |
| 1.36 m | 24 | 4 | 24 | 12 | 24 | 24 |

The panel has a hard ceiling. In the 47 conditions where both commands
were actually executed there are 19 d040-only successes,
16 both-pass, 12 both-fail and
**0 walking-only successes**, so the best outcome available
from the observed commands is 47/72. The scripted rule reaches
46/72, one below that ceiling; analytic reaches
39/72. Under-adaptation, not misfiring adaptation,
separates the learners from the ceiling: analytic issues d040
29 times against the script's
46.

Every one of the 11 analytic-only wins falls in the middle band, which is where
the single training contrast sat (underside 1.2755 m). At the high band walking already
passes 24/24 and the learner mostly leaves it alone; at the low band neither command
ever succeeds. One example transferred across carriers within its own geometry band and
not beyond it.

## Background controls: no unnecessary adaptation, and a scripted blind spot

| Background suite | Analytic passage | Analytic d040 requests | Analytic refusals |
| --- | --- | --- | --- |
| absent | 6/6 | 0 | 0 |
| raised | 6/6 | 0 | 0 |
| blocked | 0/6 | 0 | 6 |

Across the absent and raised suites **no policy requests d040 even once**
(0 requests in 72 runs), so no arm pays an
unnecessary-adaptation cost on this panel; the learned adaptation is not indiscriminate.
The blocked suite separates the methods in the opposite direction from traversal: every
learner and the privileged forecast refuse 6/6, while the scripted rule
refuses 0/6. The script is the stronger traversal baseline and the
weaker infeasibility detector. No blocked scene is passable, so refusal there is correct
classification, not successful avoidance, and it is reported separately from passage.

## One useful contrast explains the adaptation requests in the registered refits

The analytic fitting set is the six shared backgrounds plus one useful generated
contrast. The one leave-four-assignment-out fold that withholds that label produces a
model whose training IDs, weights, biases and scales equal the background-only M2S
primary fit exactly. Over 90 recorded inputs it requests d040
**0** times, against **29** for the full analytic
fit, changing 29 selected actions. 4 of the
5 folds that withhold no useful label leave the request count unchanged, and
2 withhold no available label at all and are counted as such.

This is a controlled data-removal diagnostic of the fixed learner. It is not a new
physical evaluation of the fold models and it does not establish reliable learning from
one example in a population of sources.
[Exact equality check](evidence/icra-results-540-20260907/contrast-removal-check.json).

## Cost and what this settles

Label acquisition costs 0.824108 contended GPU h and the 540 admitted
policy executions cost 4.310964 h. These exclude separately recorded
generator/teacher costs and shared bank acquisition; no complete end-to-end cost claim
is made from physics time alone.

The completed panel establishes contrast-construction benefit over untargeted and
target-only construction, and establishes **no** learned-generator benefit, because the
learned arm acquired nothing to test. That is neither outcome A nor outcome B from the
guidance. The measured cause of the empty funnel is the inherited placement envelope
rather than the proposal model, and the separately registered nominal contract tests
whether a matched envelope changes the acquisition and the learning.

[All outcomes](assets/icra-540-outcomes.pdf) · [Yield and passage](assets/icra-540-yield.pdf) ·
[Envelope trade-off](ENVELOPE_TRADEOFF_V1_RESULT.md) ·
[Full records and saved primary/refit models](evidence/icra-results-540-20260907/exports.json) ·
[Working manuscript](ICRA_MANUSCRIPT.pdf).
