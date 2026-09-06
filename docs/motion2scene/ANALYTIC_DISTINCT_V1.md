# Distinct-output analytic baseline

2026-09-06, registered development correction before these outputs are selected/audited.
Retain ANALYTIC_GLOBAL_V1.md's exact geometry proposal and 136-candidate search budget.
Its first selection rule produced 84 distinct placements in 128 requests. Inspection
of SEARCH traces finds 12–26 distinct search-passing candidates per case, so padding
with repeated station winners is unnecessary. These are already observed development
cases; this correction is not fresh confirmation and the original result remains intact.

Change only selection: rank stations by best search slack, ties by station index;
rank passing candidates within each station by decreasing slack, then candidate index.
Cycle through stations, selecting the next globally distinct passing placement each time,
until eight are selected. If fewer than eight pass search, fill the remaining requests
with the highest-slack distinct failed candidates. Preserve those refusals. Never use
independent audit scores or reference maps to select outputs. Rerun candidate geometry
from scratch, then audit all eight outputs; charge exactly 6,884 queries per job.

Retain the 16 cases, CPU float64, two threads, 180-second cap, no model fitting and
output freeze before map scoring. Prediction: 128/128 independently accepted UNIQUE
within-case placements and mean reference station coverage above original_pattern17.
Record raw traces and every rejection. This only resolves duplicate padding; fresh
source confirmation, wall-time comparisons and obstacle-present execution remain separate.
