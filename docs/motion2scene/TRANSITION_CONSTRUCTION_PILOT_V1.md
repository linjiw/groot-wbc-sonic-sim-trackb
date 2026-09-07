# Transition-aware construction pilot v1

This is a development mechanism study, not the final learning comparison. Freeze
before computation: eligible carriers are exactly 41001/41002/41003 that pass both
seeds of DEVELOPMENT_TRANSITION_BANK_V1. Keep every refused carrier in the requested
six arm/carrier groups (two requested physical scenes each; twelve scene slots total).

Both analytic and learned pipelines receive the same achieved empty-scene walk and
0.30 s d040 trajectories from seed 8721, including entry and return. Use their full
50 Hz capsule clouds for the proposal evaluator. Keep the authored neutral-reference
route for station coordinates. The learned initializer remains the frozen local model
conditioned on the complete d040 reference, with the same 16 draws and bounded
17-query pattern search. This deliberately isolates evaluator replacement; it does
not claim that the reference-trained initializer already models achieved transitions.
The strong analytic initializer uses achieved envelope heights over twenty stations
and the existing 136-candidate global search. Charge both actual query counts; no
claim of equal generator compute. No model fitting or teacher acquisition in this pilot.

Use the original 17 search offsets and 113 independent offsets, 10 mm target clearance
and 10 mm alternative interference. Keep 16 output proposals per arm/carrier and
assign their first two slots, without refill from later outputs. The analytic output
uses the existing search-only distinct selection rule. Reject domain violations,
duplicates within arm, reserved original-layout neighborhoods, pre-decision capsule
clearance below 10 mm, and failure of independent contrast screening. Preserve all
proposals and all reasons, including more than one reason for a slot.

Also require a causal visibility proposal flag: at least one recorded upper-ray
origin/direction from the 0.30 s empty-scene packet intersects the proposed beam
within 3 m and the observer's 1.15–1.50 m upper-hit band. This analytic box-only check
omits room occlusion and is not a sensor observation. The actual physics packet must
independently show the beam and measured free space below; report that separate
outcome. Do not supply object identity or true beam coordinates to a learned selector.

For each eligible first-two placement: seeds 8721 and 8722 × both commands, same
0.30 s decision and frozen scorer. At most 48 physical runs if all twelve slots
survive. Create one source batch at a time (at most 16 runs, 1.666667 reserved GPU h)
with a fresh budget check; preserve unstarted assignments on an operational stop.
All source-bank and direct capture audits apply; compare the first-decision features
and history between paired actions. Reuse existing empty-scene records only as
explicit controls and bank inputs, never count them as new executions. No scene-
dependent script controls these forced commands.

Prediction: each qualified carrier yields at least one observable walk-fail/d040-pass
pair from each arm. Report physical useful-contrast yield per assigned slot and per
physics seed; retain both-fail and both-success outcomes. Do not count rejected slots
as measured physics or as successful placements. The four outcome heads' labels,
physical visibility, return/reset and cost remain separate. No selector is fitted;
the common learner's isolation factorial and full five-arm 24/48/96 comparison remain
next steps. CPU preparation ceiling 600 s. Inherit 7500 MiB/375 s/serial execution
and the rolling 8 h/day, 24 h/week limits. Original 120/600 evaluation remains paused.
