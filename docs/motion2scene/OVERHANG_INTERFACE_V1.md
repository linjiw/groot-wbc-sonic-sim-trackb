# Overhang/free-space observer and legal-transition pilot v1

Registered 2026-09-06 before physics. This is the next development experiment
after reactive v3's failed specificity predicate. That result and all four
wall-triggered false switches remain immutable. This is a changed observer and
guard, not a retry of the old scientific configuration. No learned policy claim.

## Design and information contract

Retain carrier 41002, frozen SONIC, neutral/d040 references, imported physical beam,
raised control at 2 m underside, absent control and the far wall. Use new physics
seeds **8031,8032**, not new sources. Seven conditions per seed = **14 cells**:
absent/reactive, critical/reactive, raised/reactive, critical/blind,
absent/oracle, critical/oracle, critical/late_oracle.

Keep the old twelve upper rays, 3 m range, heading-aligned mounting, and head-height
candidate band [1.15,1.50] m. For every upper candidate, cast three horizontal
rays from the same sensor XY at known flat-floor heights 0.35, 0.75, 1.05 m toward
the upper hit's XY. Range = min(3 m, horizontal upper-hit distance + 0.1 m).
Require all three lower rays to have no non-robot collision. A wall should block
these lower rays; a traversable beam should not. Preserve nearest-hit occlusion.
Only ignore the robot itself; obstacle identity and authored pose are not selector
inputs. Record upper hits, all lower queries, full-horizon raw classification and
actual decisions. This is sparse ideal collision lidar, not a volume certificate.

Keep the >=0.2 s trigger and the 3.3 s return request, but authorize onset only in
[0.2,0.4] s and return in [3.3,3.5] s. At each requested switch additionally require
maximum joint-reference position difference <=0.05 rad and reference anchor-position
difference <=0.01 m at the same clock. Do not blend, teleport or reset; a refused
request leaves the active reference unchanged and remains logged. This guard does
not certify velocity/torque continuity or arbitrary gait transitions. The oracle
requests d040 at 0.2 s; late_oracle first requests it at 2.6 s, outside the legal
window. Late requests are intentional failures, never labelled successful avoidance.

## Predictions and denominators

P1 (sensor specificity, independent of guard): all four absent/raised reactive
cells have zero raw overhang-positive frames over the entire capture, zero switches,
but nonzero upper candidates with lower-ray obstruction. Both critical/reactive
cells observe an overhang before 0.4 s and make exactly 0→1 then 1→0, onset in
[0.2,0.4] s and return at 3.3 s. A timing cutoff alone cannot pass P1.

P2 (physics): the ten normal reactive/oracle cells all complete contact-free
first-episode passage with no observed fall/reset. Both blind cells exceed 1 N.
The two late_oracle cells are excluded only from this positive prediction and
reported as explicit late-refusal task outcomes in the full fourteen-row table.

P3 (guard): every executed switch preserves root/joint state and clock exactly,
and obeys both phase and reference-jump limits. Both late_oracle cells record at
least one denied onset and no actual switch. Their inability to avoid contact is
not counted as successful traversal; report contacts/falls/resets independently.

P4 (measurement): all trajectories retain 199 control frames / 796 consecutive
200 Hz physics samples, four substeps per control frame and <=1e-5 N discrepancy
between last-substep force and the retained 50 Hz sensor; blind/absent provide
positive/negative beam-force controls. Keep the existing first-episode score (>1 N
rejects), body-origin crossing and stabilization contract. No full-collider claim.

## Resources, lineage and stopping

Hash-pin this protocol, new implementation, original runtime and references, scene
assets, prior v3 result/failed-specificity audit and orphan-cleanup supplement.
The prior interrupted v2 process was found retaining GPU memory and killed after
verifying its exact lineage. Charge its additional reservation time separately;
never overwrite the historical record. No other workload is terminated.

14 serial cells × 375 s = 1.45834 GPU-hour ceiling, within the standing 8/day,
24/week budget. Use the previously validated 7500 MiB free-memory floor and no
cameras. Yield under contention. Stop on infrastructure failure, retain all
scientific failures without replacement, and verify no rollout descendants remain
at completion. No training promotion; no use of 430xx sources for fitting.
