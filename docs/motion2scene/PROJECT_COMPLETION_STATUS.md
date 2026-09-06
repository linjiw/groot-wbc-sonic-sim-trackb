# Motion2Scene: progress toward the complete project

**Completed action study (September 6):** [all 12 new runs](ACTION_LABEL_COMPLETION_V1_RESULT.md)
complete 16/16 matched 0.20 s action-label pairs. In the earlier-beam development case,
seed 8042 passes at 0.30 s but fails at 0.20 and 0.40 s. The [v2 learning contract](LEARNING_CONTRACT_V2.md)
therefore selects one common 0.30 s decision for new data in every arm; the old labels
remain at 0.20 s and cannot be reused as 0.30 s outcomes. The matched no-contrast generator
baseline is fitted, the shared outcome learner is implemented, and robot-data selector
fits remain zero. Next acquire the comparative corpus and evaluate learning utility.


**Learning-utility priority (September 6):** [the revised comparison plan](LEARNING_UTILITY_PLAN_V1.md)
freezes the current generator, makes Motion2Scene versus strong analytic data the
primary question, and uses complete outcomes for explicit encounter commands.
The [registered 12-cell label/timing study](ACTION_LABEL_COMPLETION_V1.md) is the
only added diagnostic prerequisite. Continuous certificate expansion, new skills
and new generator architectures are deferred. The plan also corrects the evaluation
cost and current ICRA submission/video schedule.


Assessment dated 2026-09-06. **We have an embodied research prototype, but not yet a
complete ICRA experimental contribution.** Geometry generation and simulator execution
have evidence. The central claim—that generated environments improve policy learning—
has no comparative result yet. Counting finished scripts or rollouts would overstate
completion, so the status below uses evidence gates instead of a percentage.

| Workstream | Current evidence | Completion assessment |
| --- | --- | --- |
| Inverse scene generation | 384/384 fresh-source requests accepted by proxy screening; conditioning/search ablations; explicit rejection | Strong offline foundation. Learned value against the strong analytic baseline remains to be earned downstream. |
| Collision and time realism | 384/384 nominal authored-native interval bounds, minimum 21.623 mm; separate static proxy pose domains | Bounded progress. Joint pose/time uncertainty, cooked-shape and numerical enclosure guarantees remain open. |
| Embodied scene validation | 52-cell generated-source pilot; 6/8 source pairs qualify; 18/24 requested slots separate | Demonstrated in Isaac Lab within the development pool. Source refusals remain; walk contacts but can still cross. |
| Sensing and legal commands | Complete 42-cell stress panel: reactive/oracle 9/10 poses each, 6/6 negative specificity, 500 ms delay fails 2/2 | One observed source. Earlier-beam robustness prediction fails; noisy sensing, later decisions and source transfer remain open. |
| Downstream learning benefit | 16/16 matched development label pairs at 0.20 s; shared 214-input learner and no-contrast baseline implemented; comparison designed | Not demonstrated. This is the main scientific bottleneck. |
| Paper and release | Public notebook, protocols, complete result tables, failed experiments, source snapshots and recorded-state demos | Research documentation exists. Final comparative figures, claim audit and a reproducible end-to-end benchmark remain. |

Two of the three intended contribution pillars—generation and closed-loop feasibility—
have bounded evidence. That is **not “two-thirds complete”**: the remaining learning
comparison could reject the motivating thesis. The project is beyond an offline geometry
demo and still well short of submission readiness. A strong analytic method tying or
beating the learned generator must change the contribution statement, not be hidden.

The final transfer ancestors are **not yet reserved**: the first conservative ID inventory
reported collisions, and the second exceeded its 180 s limit. No sources were generated
or declared fresh. The twelve independently specified layout tests remain reserved.
[Reservation failure record](evidence/transfer-reservation-failure.json).

## Remaining experiments, in decision order

1. **Acquire the comparative corpus at the frozen 0.30 s decision.** The six missing
   controls are complete; 16/16 development pairs at 0.20 s retain all measured outcomes.
   They are not a training corpus and cannot be relabeled for 0.30 s. Bind the exact
   d040 generator input and measure both deployed commands on every assigned scene.
2. **Keep timing support and measurement scope explicit.** The 12-cell experiment
   reproduces the old 0.20 s failure and rescues seed 8042 at 0.30 s; 0.40 s fails.
   This motivates the common future decision phase, not a general safe window. Preserve
   all historical failures. Use direct decision-state capture for the learning corpus;
   further geometry, new skills and extended timing sweeps are deferred.
3. **Run the smallest useful learning-data comparison.** Train the same exteroceptive
   selector with Motion2Scene, uniform, strong analytic/grid, and an explicitly defined
   unconstrained same-family generator. Hold policy architecture, loss, physics-label
   budget and optimizer steps fixed. Test every legal alternative for labels; retain
   infeasible scenes. Report acquisition, rejection, synthesis and training costs.
4. **Test transfer with the split frozen before fitting.** Keep motion ancestors,
   derivatives and scene perturbations together. Acquire a new test bank; 430xx remains
   excluded from tuning and the inspected 410xx/420xx sources are not fresh tests. Use
   five optimizer seeds, source-level outcomes and fixed learning-curve checkpoints.
5. **Stress the supported policy and complete the paper evidence.** Test range/depth
   noise, latency and tracking disturbances in the frozen supported domain; report
   missed detections, contacts, falls, refusals and time-to-traverse. Add hardware only
   if the final claim requires physical deployment; no hardware result is currently
   claimed. Finish relevant literature positioning, ablations and artifact instructions.

The next go/no-go decision is whether a fixed legal selector can produce dependable
physics labels over the declared scene support. After that, prioritize the learning
comparison over further nominal-demo polishing. Better sensing alone does not show that
Motion2Scene environments are better training data.

## What the demos mean

The newest videos reconstruct measured Isaac Lab joint/root states with the repository's
MuJoCo visual robot model and the registered beam transform. Contact readouts come from
the recorded 200 Hz Isaac sensor. Rendering uses `mj_forward`, never `mj_step`: these
are visual replays, not independent MuJoCo dynamics validation. The visual mesh differs
from the Isaac collision asset; numerical validity comes from the measurements and
audits, not apparent pixel clearance. Older reference-only animations remain labeled.

See the [ICRA gate ledger](ICRA_COMPLETION_PLAN.md), [latest completed interface
result](OVERHANG_INTERFACE_V2_RESULT.md), [variation registration](OVERHANG_VARIATION_V1.md),
and [downstream design](DOWNSTREAM_SENSOR_POLICY_PILOT_DESIGN.md).
