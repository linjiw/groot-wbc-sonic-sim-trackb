# Next bounded study: variation before policy training

Planning record, 2026-09-06; not an executed or registered batch. Use the
[current overhang result](OVERHANG_INTERFACE_V2_RESULT.md) and immutable realized
reference banks to prepare a concrete manifest before spending GPU time.

Test the smallest variation panel that can expose the remaining dependence on a
single beam pose and early observation. Start with the nominal scene plus one-factor
changes: underside h0±10 mm, center ±0.15 m along the route, and observation delays
of 0.10, 0.25 and 0.50 s. Keep yaw, beam dimensions and the room fixed. These are
eight distinct critical-scene/observation conditions, not a Cartesian-grid guarantee.
Retain absent and raised controls, add an explicitly blocked-underpass control, and
use two new frozen physics seeds (proposed 8041/8042). Final scene hashes and budget
must be registered before launch. Do not acquire or fit on 430xx sources.

Delay the observation packet, including its original timestamp; never substitute
the current scene pose into a delayed sample. Continue to log current raw sensing,
delivered observations, requested/allowed commands, denied reasons, actual reference
and physics-step forces separately. The existing onset window is deliberately narrow.
A late refusal is expected to fail some traversal requests and must remain a failure
in the task denominator; do not call guard compliance successful avoidance.

Compare the sensor selector with blind walking and continuously selected feasible
references under the same scene support. The label bank should eventually include
neutral, d040 and d055, each loaded explicitly for evaluation and checked against
its input route. Validate adding the third reference as a separate interface change;
do not silently reinterpret the current binary selector as a three-skill policy.
Where no alternative succeeds, retain an infeasible label. Do not replace lower-beam
failures or assume deeper crouching will always rescue them.

Report a complete scene×method×seed table: first-episode contact-free passage,
noncontact traversal, reset/fall, raw sensor errors, legal/denied transitions,
200 Hz force peak/duration/impulse, progress, and source-reference fidelity. Keep
full-collider crossing and sensor noise/depth transfer as distinct uncompleted gates.
The prediction register should separate nominal reproduction, negative specificity,
finite perturbation response and delayed-decision failures.

Then freeze the API and the policy-learning experiment. The concrete comparison
remains generated training environments versus uniform placement, a strong analytic
curriculum and an unconstrained same-family generator, with identical policy/loss,
source ancestry splits, five optimizer seeds, held-out test scenes and matched full
cost accounting. Better sensing alone cannot establish the generator's learning value.
