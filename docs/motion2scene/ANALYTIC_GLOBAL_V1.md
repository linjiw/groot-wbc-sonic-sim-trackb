# Event-aware bounded global beam baseline

2026-09-06, development follow-up registered after the coverage diagnostic and before
these solver outputs. The first analytic arm trades yield for coverage (103/128 accepted,
68.96% reference station coverage). Test a stronger solver at the learned pattern arm's
same 136 candidate evaluations per eight-output job. No learning or fresh-source claim.

Use the same 16 observed development cases and permitted target/upright geometry.
At each of the existing 20 analytic stations, propose six heights at envelope midpoint
plus {-0.025,-0.015,-0.005,0.005,0.015,0.025} m, clipped to the original domain.
Add the nominal midpoint at the 16 stations with greatest envelope separation. This
gives exactly 136 candidates. Evaluate each at the frozen 17 SEARCH offsets, 4,624
whole-motion queries. Independent final audit offsets cannot rank candidates.

For each station retain the earliest maximum worst-margin-slack candidate. Sort these
station winners by decreasing slack, then station index. If any pass search-side checks,
request eight outputs by cycling through passing station winners (use the first eight
when more are available). If none pass, use the best eight failed station winners and
preserve all refusals. Repeated outputs are declared; they improve no coverage metric.
Independently audit all eight at 113 offsets, plus the existing two-output Torch crosscheck
(1,808+452 queries). Total online cost is 6,884 queries per job, as for learned pattern17.
Also report unique outputs, passing station winners, candidate arrays, preprocessing and
search/audit times. CPU float64, two threads, 180-second experiment cap, zero GPU time.

Freeze all 16 outputs before loading the independent reference maps for coverage scoring.
Maps were already inspected to motivate this development comparison; they are not held-out
confirmation. The solver function receives geometry and route only, never map values or
event metadata. Compare to the historical original_pattern17 outputs at eight requests/job.
Report every source. Prediction: match 100% pooled reference yield while increasing mean
reference-relative station coverage; ties pass yield, a strict increase is needed for coverage.
Failure remains a result. Success would prioritize analytic beam synthesis for this domain;
its absence of fitting cost is real, but contemporary matched wall-time trials remain needed
before a latency superiority claim. Preserve learning research for richer constraints or
settings where this inexpensive analytic solver fails.
