# Positioning the learning-data comparison

Primary-source check, September 6. This is a focused comparison with four relevant
works, not an exhaustive novelty search. The conclusions about Motion2Scene below
are our interpretation of the implemented experiment and its current limitations.

**Learning environments from motion already includes downstream learning.** LfLH
learns an obstacle distribution from open-space motion plans through a planner-based
reconstruction objective, samples environments, renders observations and trains a
planner by supervised learning. Its ground and aerial instantiations also retain
collision-checking control components. Motion2Scene therefore cannot claim novelty
merely from reversing the scene-to-motion direction or from adding a downstream
learner. Its proposed distinction concerns humanoid motion contrasts and measured
outcomes of the actual transition command. [LfLH, sections III–IV](https://arxiv.org/html/2108.09793v1).

**Perceptive skill composition is also established.** Perceptive Humanoid Parkour
uses motion matching to compose skills, trains tracking experts, and distills a
depth-conditioned student with DAgger and reinforcement learning. It reports physical
G1 traversal. Our two-command selector with ideal collision rays is a narrower
experimental interface; it does not establish a more general perceptive locomotion
system. [Perceptive Humanoid Parkour, abstract and method](https://arxiv.org/html/2602.15827v2).

**Scene-generation utility has a close humanoid comparator.** HumanoidPF combines
body-part potential-field observations and reward guidance with realistic indoor
scene crops and procedural obstacles. Section IV-B compares training-scene mixtures
on thirty artist-designed test scenes. This makes downstream generalization from the
generated data a relevant standard for our contribution. Its policy and scene family
differ from ours, so cross-paper success percentages would not be a controlled
comparison. [HumanoidPF, sections III-B and IV-B](https://arxiv.org/html/2601.16035v1).

**The motion executor is inherited.** SONIC studies motion-tracking scale and supplies
a generalist controller with multiple command interfaces. We freeze that foundation
and change the scene-training distribution and supervised decision module. The
tracking controller's capabilities are not a new contribution of Motion2Scene.
[SONIC, abstract and system description](https://arxiv.org/html/2511.07820v3).

| Work or component | Changed component relevant here | Consequence for our experiment |
| --- | --- | --- |
| LfLH | Learned obstacle distribution used to train a planner | Show what transition-aware humanoid supervision adds |
| Perceptive Humanoid Parkour | Skill composition and perceptive control | Scope our claim to training-data construction |
| HumanoidPF | Body–obstacle representation and hybrid scene generation | Evaluate independent scenes with a fixed learner |
| SONIC | Scaled motion-tracking foundation and control interfaces | Hold the executor fixed and credit its role |
| Motion2Scene development study | Four scene-data sources, paired physical command labels | Analytic-versus-learned data comparison carries the learned-value claim |

The defensible question is whether motion-contrast data construction improves the
same deployed selector relative to strong analytic construction at matched complete
label counts or matched acquisition cost. The current first-wave evaluation is
partial and uses one observed source. The earlier corpus's nine both-fail Motion2Scene
scenes, the missing independent-source evidence and the outstanding larger data
budgets prevent a present claim of learned-generator superiority.

A favorable result would support a bounded data-construction contribution. A tie or
loss to analytic construction would separate the usefulness of motion-conditioned
constraints from an unproved benefit of learning their distribution. Completing the
fixed comparison is therefore more informative than adding another successful demo.
