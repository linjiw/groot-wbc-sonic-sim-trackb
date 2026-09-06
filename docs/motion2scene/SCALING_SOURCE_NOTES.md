# Source check: what should scale in learned hallucination?

Reading date: 2026-09-05. This is a focused main-text recheck, not an independent
reproduction or a current-literature novelty survey.

LfLH separates learning the hallucination distribution from training the eventual
navigation policy. Its fixed differentiable planner supplies reconstruction feedback;
priors and collision penalties shape proposals, and invalid samples are filtered before
policy training. Increasing generated scene count therefore changes a later dataset
stage, distinct from enlarging the hallucination model or its motion-source bank.
[Original LfLH, §§III-B/C and IV-A.1](https://arxiv.org/html/2108.09793v1).

LfH-CP factorizes dynamic hallucination into critical obstacle location/time and procedural
trajectories through those critical points. Its scene construction still checks plan
collision. The paper reports 30.83% versus 22.5% navigation success over 120 simulated
trials in 60 environments, despite much stronger individual-obstacle coverage. It explicitly
identifies simultaneous obstacles as a remaining difficulty and distinguishes individual
obstacle coverage from joint coverage of several obstacles.
[LfH-CP, §§III-D/E and V-B/C](https://arxiv.org/html/2509.26513v1).

**Our inference:** scaling a useful local geometric representation is a better hypothesis
to test than assuming a larger global latent will resolve the earlier conditioning
failure. The current eight route intervals are hand-designed proposal supports, not a
reproduction of learned critical times in LfH-CP. Nor does a target-versus-upright preference
test establish global optimality among all humanoid motions. These distinctions remain
explicit in the [registered comparison](EVENT_SCALING_V1.md).

For this project, measure four separate quantities: distinct motion-source count, model
capacity, optimization queries and accepted scene diversity. Later, add object count and
joint scene validity as another axis. Larger raw sample count can yield more accepted
copies without adding meaningful scene support. A downstream fixed-policy experiment
must determine whether generated diversity helps execution. Neither the cited navigation
results nor the present reference-geometry study guarantees that outcome for humanoids.

The [data acquisition design](DATA_SCALING_DESIGN.md) and
[event-conditioned model design](EVENT_CONDITIONED_GENERATOR_DESIGN.md) record the next
experiments. The current study changes no protocol after this source recheck.
