# Carrier 41002 original-clock repeatability v1

Registered before execution, 2026-09-05. Development-only follow-up to the complete paired
timing diagnostic. Carrier selection is post-outcome: 41002 was its only retained original-clock
neutral/d055 pair. This is one carrier, with physics seeds as repeated measurements.

Execute exactly six registered cells: original neutral and d055 at seeds 7901, 7902 and 7903.
Preserve motion/reference hashes, controller, empty scene, trajectory capture and all thresholds.
Execute neutral before d055 at each seed. A rejected neutral skips its dependent crouch but
both remain in the denominator. Scientific rejections are never retried. Require 9000 MiB free
VRAM, serial execution, 375-second per-cell timeout, and stop on infrastructure failure.
The projected ceiling is 0.625 GPU-hours, within the standing 8/day and 24/week budget.

Predictions, assessed only after the complete batch:

1. All three neutrals pass tracker and relative-route checks.
2. All three crouches pass tracker and relative-route checks.
3. All three pairs retain the localized crouch with the existing 50 mm minimum effect,
   60% reference-magnitude retention and 30% event-IoU thresholds.

Strict development repeatability requires all three predictions. Report all scalar route,
tracking, height and event metrics, achieved trajectory fingerprints and seed-to-seed spread.
Identical trajectories cannot establish perturbation robustness. No tolerances are adjusted
from these observations. The previous route-validation failure remains unchanged. These seeds
do not replace the separate registered Q4 seeds (7401–7403); this experiment does not admit Q4.

If the pair repeats, investigate a third original-clock reference level and finite beam
placement. Use all full trajectories in world coordinates, not just root-height extrema or
separately normalized motion phases. Check both target clearance and weaker interference with
finite obstacle extent and explicit placement jitter. Exact static capsule–box distances are
still a collision-model and sampled-time geometric diagnostic, not Isaac contact verdicts or
continuous-time certification. Keep all proposals ineligible for learning until full qualification
and obstacle-present preference reversal. If repeatability fails, report and diagnose the failed
seed without replacing it.
