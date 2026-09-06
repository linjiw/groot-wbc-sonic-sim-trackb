# Motion2Scene: progress toward the complete project

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
| Downstream learning benefit | 144-value sensor features packaged; 10/16 complete skill-label pairs, six missing comparators; matched comparison designed | Not demonstrated. This is the main scientific bottleneck. |
| Paper and release | Public notebook, protocols, complete result tables, failed experiments, source snapshots and recorded-state demos | Research documentation exists. Final comparative figures, claim audit and a reproducible end-to-end benchmark remain. |

Two of the three intended contribution pillars—generation and closed-loop feasibility—
have bounded evidence. That is **not “two-thirds complete”**: the remaining learning
comparison could reject the motivating thesis. The project is beyond an offline geometry
demo and still well short of submission readiness. A strong analytic method tying or
beating the learned generator must change the contribution statement, not be hidden.

## Remaining experiments, in decision order

1. **Complete the six missing negative-control skill labels.** Register oracle d040
   executions for absent/raised/blocked × seeds 8041/8042 before spending physics.
   Reuse the measured neutral traces only where no switch occurred. Preserve the
   42-cell panel and its 9/10 pose result; neither skill passing is a valid label,
   not a row to discard. This closes an interface-data contract, not a training set.
2. **Freeze the measurement and action contract.** Diagnose the seed-8042 earlier-beam
   contact using its measured trajectory; do not change the frozen switch window or
   relabel the failure. The native outer-envelope crossing audit preserves all 14
   prior classifications (10 passes, four failures), with completion 20–40 ms later.
   Extend that audit to the new panel before choosing a common learning label horizon.
   Keep cooked geometry and continuous-time guarantees separate. Qualify any d055
   addition independently; the present selector remains binary.
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
