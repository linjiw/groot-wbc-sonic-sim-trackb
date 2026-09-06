# Registered uncapped verification of the native temporal bound

Registered 2026-09-06 after native-temporal v1. Preserve v1: sampled native separation
passes 384/384 at 30 and 120 Hz, but **0/384** pass its capped continuous bound.
The minimum bounds range from -2.253 to +3.463 mm, below the required 10 mm.
At the minimum-bound witness of the first request, both endpoint values are capped
at 20 mm. This motivates an independent measurement check, not changing any scene.

An exact distance capped from above remains a valid lower bound, but can make interval
verification unnecessarily weak even for a distant moving shape. Use uncapped exact
capsule-to-box distances for all 45 target outer shapes at all 120 Hz interpolated
poses of all 384 frozen placements. Keep the same displacement bound and 1e-5 m
allowance. No scene movement, new generation, training, data selection or gate changes.
The original P2 failure remains authoritative for the original checker.

Predictions: **U1** capping the new arrays at 20 mm reproduces every stored v1 target
clearance within 1e-8 m; **U2** all 384 uncapped interval minima are >=10 mm and retain
the already measured native primitive neutral-interference witnesses. All placements
remain in the denominator, and any failures remain failures. Report uncapped sampled
and interval minima, timing, witness ownership, and the discrepancy from v1 arrays.

This is validation of a measurement contract on the frozen audit set. It does not
select a generator, fit its parameters, alter a threshold, or make 430xx training
eligible. The same nominal-only, authored-envelope, declared-interpolant limitations
apply. No original 113-offset domain guarantee, actual cooked-mesh CCD, or new physics.
CPU ceiling 300 s, two Torch threads, no GPU. Pin source result, registrations,
geometry, reference assets and code before computation; verify afterward.
