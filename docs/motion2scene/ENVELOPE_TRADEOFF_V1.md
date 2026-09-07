# Envelope/contrast trade-off diagnostic v1

Post hoc CPU diagnostic registered September 7, after the M2S-ICRA-v1 acquisition
funnel completed and before any later construction contract is chosen. It adds no
physics, no proposal, no label and no acceptance-rule change, and it leaves every
assigned M2S-ICRA-v1 outcome and refusal exactly as recorded.

## Question

The completed funnel refused 47/48 analytic and 48/48 learned generated slots, almost
all for `contrast_audit`. The [finite support diagnostic](TRANSITION_SUPPORT_DIAGNOSTIC_V1.md)
already showed the two-sided cause on a 20 x 71 grid: nominal witnesses exist
(28/19/43 per carrier) while the inherited 113-offset audit leaves 1/0/3. That
established *that* the perturbation requirement binds. It did not measure *how much*
envelope the recorded transitions can afford, so it cannot tell a later contract
which envelope to declare.

This diagnostic measures the trade-off curve: surviving witnesses as a function of
the audit envelope, scaled from zero to the inherited size.

## Method

Fixed grid over the registered proposal domain: stations 0.10 to 0.90 in 0.01 steps
and undersides 1.100 to 1.450 m in 0.0025 m steps (7,181 centres). For each qualified
carrier, use the same recorded seed-8721 achieved walk and d040 transitions and the
authored reference route already used by the acquisition.

Keep the 10 mm criterion unchanged: a witness needs target clearance >= +10 mm and
walk interference <= -10 mm, jointly, at every offset in the set. Scale the inherited
113-offset set by 0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75 and 1.0, so scale 1.0
reproduces the inherited +-20 mm xy, +-10 mm z, +-0.02 rad audit and scale 0.0 is the
nominal pose alone. Report surviving witness counts and the best joint margin, defined
as min(target clearance, -walk clearance), at every scale.

Because the inherited offset set contains the zero offset, survival is monotone in
the envelope, so only nominal witnesses are re-evaluated at larger scales. The script
asserts that zero offset is present rather than assuming it.

CPU ceiling 1800 s. Record every scale including those with zero survivors.

## What this cannot establish

Sampled witnesses on one finite grid are not a feasible-volume estimate and are not
proof that unsampled centres fail. Scaling an audit envelope is a diagnostic sweep,
not a proposed acceptance rule: any later contract must be separately registered,
must state its envelope before search and physics, and cannot inherit the original
robust certificate by quietly shrinking its offsets. A surviving witness is geometry
only. It is not a physical label, not a tracker outcome and not evidence that the
executed command would clear the beam. Only paired physics can decide that, and the
carriers whose d040 the tracker rejects for endpoint error remain the cases where
achieved and reference geometry diverge most.

The diagnostic uses one physics seed per carrier for the achieved transitions, so it
describes those recorded executions, not a distribution over seeds.

Output: `envelope-tradeoff.json` under
`/home/linjiw/research-data/groot-wbc/m2s-envelope-tradeoff-v1`, marked
`analysis_only` so it never enters the physics spending ledger.
