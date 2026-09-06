# Learning input binding: result and retained failures

The first 0.20 s decision packet is bound to its **same-index** recorded post-physics
robot state in all 42 prior cells. Worst reconstructed upper-ray-origin discrepancy:
**2.98e-10 m**. These are paired sensing/state measurements, not policy performance.
The command's reference phase has advanced when sensing runs; it must not be used as
an array index into the next physical state. The frozen planning note's proposed
next-index alignment is superseded by this audit.

Requested and realized delay are measured separately. Every delivered packet has the
registered monotonic capture age: 100 ms → 5 frames → 100 ms; 250 ms → 13 frames →
260 ms; 500 ms → 25 frames → 500 ms. Both index and elapsed-timestamp checks pass all
42 cells, including unavailable warmup packets. The four 250/500 ms cells have no
available sensor packet at the fixed first decision; no future observation is imputed.

The initial audit aborted because the single-episode dataset loader rejected a
reset-spanning failure capture. The reset-aware adapter validates every segment,
including short failed first episodes and tails, while retaining all raw frames.
It does not reinterpret a reset as success or select a better episode.

The second audit's next-index physical-state prediction fails. The third audit's
whole-capture same-index prediction also fails at reset/terminal boundaries where
command and recorder lifecycle do not expose the same state. Those attempts remain
available. The fourth audit tests the **already prespecified first decision** only;
it does not claim whole-capture alignment. Future sequential decisions require their
own capture contract. No robot-data policy fit occurred.

[Decision binding](evidence/decision-binding.json) · [Decision registration](evidence/decision-binding-registration.json) · [Delay and failed next-index audit](evidence/learning-input-audit-v2.json) · [Failed whole-capture audit](evidence/learning-input-audit-v3.json) · [Initial loader failure](evidence/learning-input-audit-v1-failure.json)
