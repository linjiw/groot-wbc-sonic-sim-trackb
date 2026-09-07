# ICRA scope adopted from fable.md and fable.html

Dated snapshot: 2026-09-07T05:11:43.867263+00:00. This is ongoing work, not a completed learning comparison.

The user's guidance supersedes the earlier five-arm 24/48/96 and generator-refit plan
for this submission. The new [registered four-arm study](M2S_ICRA_V1.md) retains the
frozen generator, one deterministic linear learner, three qualified development
carriers and the twelve existing layouts. No new-source acquisition, new skill or
resumption of the 480 original MLP assignments is included.

## Completed construction, incomplete labels

| Data arm | Eligible / requested generated groups | Maximum complete groups including shared backgrounds |
| --- | --- | --- |
| Uniform | 18/18 | 24/24 |
| Analytic | 1/18 | 7/24 |
| Target-only | 16/18 | 22/24 |
| Motion2Scene | 0/18 | 6/24 |

Six shared background groups are identical across arms, pooled across three carriers.
The guidance's 18 generated + 6 background quota implies at most 156 unique command
executions, not 180. Refusals reduce the actual assignment to **82 commands**. All
192 raw outputs and 72 assigned generated slots remain in the proposal record. The
24-complete-label target already cannot be reached in three arms; no refill repairs
it. Any downstream comparison must disclose the unequal acquired counts.

At this snapshot, **3/82 physical command executions completed**;
0 executions belong to fully measurement-admitted batches. Partial batches
are not training-eligible. Acquisition has spent 0.034432 contended GPU h. The
first matched pair completed before the driver yielded at 4982 MiB free, below the
unchanged 7500 MiB floor. The resource supervisor resumes only never-started cells
when memory and rolling budget gates admit them. Failed cells, changed hashes or
rejected measurement audits stop the pipeline for review.

No new selector has been fitted, and no M2S-ICRA evaluation is completed at this
snapshot. Four primary fits and 24 offline leave-four-out refits wait for all admitted
labels. The fixed evaluation has 432 traversal assignments and 108 separately reported
background assignments. The extra background panel implements unnecessary-adaptation
and blocked-refusal measurements; it is not pooled into traversal success.

## What changed in the research judgment

The smaller experiment is appropriate, but the guidance's causal wording is stronger
than the evidence. The original MLP repair changed scaling, capacity and regularization
together; it did not isolate normalization as the sole cause. The achieved forecast
caught thirteen reference false-clears but missed two other contact-qualified failures.
Most importantly, the completed transition-construction pilot and this larger frozen
proposal pass both yield zero accepted learned scenes. A nominal refit might be a
later project; the user has explicitly excluded it from this ICRA study.

The manuscript now tests which construction factors matter, rather than presuming
that learned proposals beat analytic construction. It must retain the possibility
that the current pipeline supplies no useful learned training data. Analytic results,
reference-level transfer and proposal timing are different measurements and cannot
be combined into an unmeasured transition-aware learning advantage.

## Manuscript and remaining work

[The working manuscript](ICRA_MANUSCRIPT.md) ([reading PDF](ICRA_MANUSCRIPT.pdf)) now contains the introduction, verified
related work, command-bound labels, four construction methods, common learner,
completed physical evidence and limitations. Learning results remain marked pending;
the abstract is deferred to the registered result decision.

Remaining: finish the 82 assigned labels; fit and compile the four shared controls;
execute and audit the 432+108 fixed evaluation assignments; compute grouped paired
tables and costs; select the supported claim; finish the eight-page submission and
its final evidence audit. Source-held-out transfer and multi-budget curves are later
work. Completion of the project is not claimed by preparing these assignments.

[Execution supplement](M2S_ICRA_EXECUTION_V1.md) ·
[Proposal and execution snapshot](evidence/icra-stage-20260907/status.json) ·
[Raw records and hashes](evidence/icra-stage-20260907/exports.json) ·
[Latest completed physical demo](assets/transition-construction.mp4).
