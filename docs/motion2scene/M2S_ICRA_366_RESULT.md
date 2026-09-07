# Completed paired labels and the first 366 fixed-policy evaluations

**September 7 snapshot:** all **82/82 labeling runs** complete and all 41 unique
encounter pairs pass measurement admission. Four primary deterministic learners and
24 prescribed leave-four-out refits complete. **366/540 evaluation runs** are admitted:
61 of 72 traversal conditions, each with all six policies. The remaining 66 traversal
and 108 background runs are pending. This is an ordered partial panel, not completion.

## Data quality and the failed acquisition quota

| Data arm | Complete / requested groups | Both pass | d040 only | Walk only | Both fail |
| --- | --- | --- | --- | --- | --- |
| uniform | 24/24 | 10 | 0 | 0 | 14 |
| analytic | 7/24 | 4 | 1 | 0 | 2 |
| no_contrast | 22/24 | 20 | 0 | 0 | 2 |
| motion2scene | 6/24 | 4 | 0 | 0 | 2 |

The six shared background pairs are counted in each arm's fitting set but acquired
once. Generated groups are uniform 18/18, analytic 1/18, target-only 16/18 and M2S 0/18.
Only the analytic generated group is physically useful. Uniform's generated pairs
contain six both-pass and twelve both-fail cases; all sixteen target-only generated
pairs are both-pass. Thus the registered uniform <=1 useful and target-only >=12/18
both-pass predictions hold, while the prediction of at least one useful generated
contrast per carrier for analytic and M2S fails. Three carriers qualify, but useful
contrast acquisition remains confined to one analytic scene on 41001.

The equal-24-complete-label goal fails in three arms. These are outcomes at equal
requested slots with unequal acquired data and costs. **The M2S-arm model is trained
only on shared backgrounds.** Its later performance cannot establish the quality of
accepted M2S-generated training scenes, because no such scenes were acquired.

## Actual policy executions, with paired conditions preserved

| Policy / data arm | Passage | d040 requests | Refusals followed by walking |
| --- | --- | --- | --- |
| Uniform | 21/61 | 0/61 | 24/61 |
| Analytic | 30/61 | 22/61 | 0/61 |
| Target-only | 21/61 | 0/61 | 0/61 |
| M2S: background only | 21/61 | 0/61 | 0/61 |
| Scripted rays | 36/61 | 38/61 | 0/61 |
| Privileged geometry | 29/61 | 10/61 | 33/61 |

Analytic versus uniform has 21 both-pass, nine analytic-only, zero uniform-only and
31 both-fail conditions. The carrier-averaged difference is +14.603 percentage points
on this completed slice. Per-carrier extra passes are five on 41001 and two each on
41002/41003. The evaluated denominator is 21 conditions for 41001 and twenty each for
the other carriers; repeats and layout variants do not create independent ancestors.
All nine analytic-only passages requested d040 with 0 N measured beam force; their
walking comparators record 385.1–1685.6 N. These are matched policy executions under
the recorded-state and source-bank audit, not counterfactual outcome predictions.

The script passes six conditions that analytic misses, with no reverse difference.
The privileged forecast and analytic disagree in both directions (three analytic-only,
two privileged-only). Privileged means a known-scene achieved-capsule forecast, not
an optimal transition-time oracle. The script's 36/61 is therefore a stronger observed
baseline than the analytic learner in this slice. Uniform, target-only and background-
only M2S all execute walking, although uniform also reports 24 refusals. Three of those
refusals subsequently pass by walking; they are not protective stopping behavior.

Thirty-eight conditions include actual executions of both commands across the six
policies. Their outcomes agree whenever the issued command agrees. Fifteen of those
conditions show d040-only success, and none shows walking-only success. The other
23 conditions have walking observations only; their unexecuted d040 outcomes remain
unknown. Do not infer that d040 could rescue every failed walk.

## One useful contrast explains the adaptation requests in the registered refits

The analytic fitting set is the six shared backgrounds plus one useful generated
contrast. Its first leave-four-assignment-out fold removes that one available label
and three already refused assignments. The remaining six training IDs and all weight,
bias and scale arrays equal the background-only M2S primary fit exactly. On the 61
recorded inputs, this refit requests d040 zero times; the full analytic fit requests
it 22 times. All five other analytic folds retain the same 22 requests. Two folds
remove no available labels and are explicitly counted as such.

This is a controlled data-removal diagnostic of the fixed learner. It is not a new
physical evaluation of the fold models, and it does not establish reliable learning
from one example in a population of sources. It does connect one physical contrast
to changed learned requests, while the main admitted policy runs measure nine passage
improvements. [Exact equality check](evidence/icra-results-366-20260907/contrast-removal-check.json).

## Cost, current limit and remaining experiments

Label acquisition costs 0.824108 contended GPU h. The 366 admitted
policy executions cost 3.146765 h. These exclude separately recorded
generator/teacher costs and shared bank acquisition; no complete end-to-end cost claim
is made from physics time alone. Full query records and shared prior costs remain in
the acquisition registration and source-bank record.

The next six-cell batch reserves 0.625 GPU h. At the budget check, rolling daily use
is 7.409639 h, so 8.034639 h would exceed the standing 8 h/day envelope. The supervisor
correctly waits; it has neither lowered the reservation nor counted the pending 174
runs as failures. The first relevant cost expires at approximately 14:39:32 UTC on
September 7. Later batches remain subject to fresh rolling-window checks.

Finish the unchanged 174 evaluation assignments, including all background controls,
before the final paired result and September 10 claim decision. The current finding
is analytic data benefit on an inspected partial panel and learned-construction
acquisition failure. It establishes neither outcome A nor outcome B from the guidance,
both of which expected M2S to outperform untargeted data. The manuscript must retain
this distinction and the unequal-data shortfall.

[All outcomes](assets/icra-366-outcomes.pdf) · [Yield and passage](assets/icra-366-yield.pdf) ·
[Full records and 28 saved primary/refit models](evidence/icra-results-366-20260907/exports.json) ·
[Working manuscript](ICRA_MANUSCRIPT.pdf).
