# Motion2Scene: progress toward the complete project

Assessment dated 2026-09-06. **We have an embodied research prototype, but not yet a
complete ICRA experimental contribution.** Geometry generation and simulator execution
have evidence. The central claim—that generated environments improve policy learning—
has no comparative result yet. Counting finished scripts or rollouts would overstate
completion, so the status below uses evidence gates instead of a percentage.

| Workstream | Current evidence | Completion assessment |
| --- | --- | --- |
| Inverse scene generation | 384/384 fresh-source requests accepted by proxy screening; conditioning/search ablations; explicit rejection | Strong offline foundation. Learned value against the strong analytic baseline remains to be earned downstream. |
| Collision and time realism | Native asset inventory, one-carrier temporal bounds, 384 conditional local pose certificates | Partial. These are separate scopes; no joint native continuous-time certificate over the 384-output batch. |
| Embodied scene validation | 52-cell generated-source pilot; 6/8 source pairs qualify; 18/24 requested slots separate | Demonstrated in Isaac Lab within the development pool. Source refusals remain; walk contacts but can still cross. |
| Sensing and legal commands | Corrected 14-cell interface panel: 4/4 negative controls, 2/2 critical reactive passages; late refusals remain failures | Working ideal-sensor prototype on one source. Pose/delay panel registered and partially executed; noisy sensing, later decisions and source transfer remain open. |
| Downstream learning benefit | A common sensor/command/loss and matched generator comparison are designed | Not demonstrated. This is the main scientific bottleneck. |
| Paper and release | Public notebook, protocols, complete result tables, failed experiments, source snapshots and recorded-state demos | Research documentation exists. Final comparative figures, claim audit and a reproducible end-to-end benchmark remain. |

Two of the three intended contribution pillars—generation and closed-loop feasibility—
have bounded evidence. That is **not “two-thirds complete”**: the remaining learning
comparison could reject the motivating thesis. The project is beyond an offline geometry
demo and still well short of submission readiness. A strong analytic method tying or
beating the learned generator must change the contribution statement, not be hidden.

## Remaining experiments, in decision order

1. **Finish the frozen interface stress panel.** Complete 42 registered cells for five
   poses, three packet delays, three negative controls and two seeds. Preserve all
   failures and the reused nominal comparators. A failed delayed decision identifies
   unsupported timing; do not widen the switch window without a new registered test.
2. **Freeze the measurement and action contract.** The new native outer-envelope audit
   retains 10/14 original outcomes and shifts crossing completion 20–40 ms later.
   Extend that independently to future runs; keep cooked geometry and continuous-time
   guarantees separate. Validate any added d055 reference as a third skill, with no
   training augmentation and explicit transition tests. The present selector is binary.
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
