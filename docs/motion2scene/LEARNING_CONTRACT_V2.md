# Learning contract v2: select one common 0.30 s decision for new data

Registered after the development timing result on September 6, before any robot-data
selector fitting or final test execution. The generator and learner architecture,
normalization rule, optimizer, supported actions, test-layout reservation and comparison
arms in [v1](LEARNING_UTILITY_PLAN_V1.md) are unchanged. This revision changes the
future common-bank decision phase from 0.20 to **0.30 s**, for every arm and comparator.

The bounded timing experiment found seed 8042's earlier-beam encounter fails at 0.20 s
(429.159 N), passes at 0.30 s (zero measured force), and fails at 0.40 s (564.862 N).
Seed 8041 passes all three times. The new explicit 0.20 s path reproduces the old
scripted privileged trace exactly. The result supports a timing-dependent outcome
within this development encounter; it does not establish an optimal or robust timing
window, and later is not generally better.

The 16 completed label pairs remain bound to **0.20 s** and remain training-ineligible.
Do not transfer their success labels to 0.30 s. Acquire new paired neutral-commit and
immediate-d040-request executions from matched pre-decision histories at 0.30 s for
the comparative corpus. Both-fail remains a legitimate measured outcome. The neutral
command means commitment for the remainder of this encounter, not waiting one tick.
No stop action or repeated reconsideration is introduced. The common recovery request
and phase/jump guard are retained. Do not let the scripted overhang rule override the
learned decision; it remains a separately measured baseline.

Bind sensor and proprioceptive inputs at the actual first decision callback. The
recorded-state origin audit on the old first 0.20 s decision passes all 42 runs;
full-capture/terminal alignment failed, so do not apply an assumed global frame shift.
The new acquisition must record the decision state directly and check paired histories,
including prior actions/tokens. After this contract revision, proceed to the learning
comparison rather than extending the timing or collision-certification campaign.
