# Recorded-transition forecast audit v1

This post hoc CPU development audit tests a proposal model, not a new label or a
new generator. Register its input hashes before computation. Keep all 35 physically
labeled generated training scenes (9 uniform, 8 analytic, 9 no contrast, 9 Motion2Scene).
Do not include final-source candidates or use independent layouts to rank proposals.

Compare the minimum sampled 29-capsule beam clearance from (a) the complete neutral
and d040 references, and (b) the two already recorded empty-scene command executions
at the common 0.30 s decision, seed 8602, source 41002. These empty-scene trajectories
include entry and return. Require zero resets, legal commands, and matched recorded
pre-decision state/history before using them. Preserve the authored scene frame;
report first root positions instead of silently realigning achieved trajectories.

For each fixed physical beam, report both clearances and the observed paired outcomes.
A clearance >=10 mm is only a proposal-screen flag. Compare that flag to measured
contact <=1 N separately from the complete task score (crossing and first-episode
stability). Count false clear predictions and missed physically contact-qualified
commands for each model and data arm. The capsule screen is sampled at each input's
native rate (references 30 Hz, executions 50 Hz), not mesh truth or continuous-time
certification; neither its successes nor failures replace physical labels.

CPU ceiling 180 seconds; no new physics, training or search. Include code, motion,
corpus and empty-transition provenance. No assumption that achieved transitions
improve all metrics. Empty-scene tracking under one seed cannot forecast contact
responses or establish generalization. Next construction must give the learned and
execution-aware analytic arms the same transition bank and charge its acquisition.
