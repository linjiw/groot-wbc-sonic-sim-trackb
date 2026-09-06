# Native collider frame binding: registered CPU check

Registered 2026-09-06 before computation. The planned 384-output native audit needs
evidence that the reference FK model and Isaac's recorded rigid-body frames agree.
Matching body names alone is insufficient. Use all fourteen completed evaluation-bank
captures, never the excluded fresh-source outcomes for fitting or method selection.

Convert each measured Isaac root/joint state into the existing canonical G1 qpos
order, then run `mj_forward` with the hash-pinned reference FK MJCF used in the original
fresh-source audit. Compare every named native collision owner against the measured
Isaac body position and wxyz quaternion at every recorded frame. Do not refit joint
offsets, translate bodies independently, or drop discrepant links.

Prediction: maximum body-position discrepancy <=1 mm and maximum geodesic orientation
discrepancy <=0.1 degree for all fourteen captures. If this fails, the native batch
extension cannot assume a shared frame contract. Report the maximum, worst owner,
per-owner errors, full-frame denominator, and a conservative bound on displacement of
each cached collider's local endpoints. This is a model/frame agreement diagnostic,
not a new collision, controller or transfer result. No new physics; CPU ceiling 120 s.

Hash-pin this protocol, driver, source results, source trajectories, MJCF and cached
native geometry before computation. Preserve original results and all failures. The
430xx sources remain excluded from training and selection; only their registered
FK asset identity is read to identify the existing measurement contract.
