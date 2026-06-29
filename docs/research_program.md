# GR00T-SONIC Humanoid Research Program

## Thesis

This project tests whether a staged humanoid curriculum is safer and more
sample-efficient than direct end-to-end VLA fine-tuning for loco-manipulation.
The control split is fixed:

```text
language + vision + proprioception
        -> Isaac-GR00T N1.7 VLA
        -> 64D SONIC motion token + 7D left hand + 7D right hand
        -> SONIC / gear_sonic_deploy whole-body controller
        -> Unitree G1
```

GR00T should choose task-conditioned actions. SONIC should keep the body
balanced, smooth, and executable. The MVP must not ask GR00T to produce raw
joint torques or replace SONIC's deploy observation/action contract.

## MVP

Start with a tabletop task, not walking:

```text
Instruction: "Pick up the red cup and place it on the tray."
Initial state: G1 stands in front of a table.
Required behavior: reach, grasp, lift, place.
Excluded from MVP: walking carry, synthetic data, SONIC retraining.
```

This is deliberately narrow. It proves the vertical path from data collection
through GR00T fine-tuning and SONIC inference before adding locomotion.

## Milestones

| Milestone | Goal | Exit gate |
|---|---|---|
| M0 schema smoke test | Environments, assets, and interfaces load. | G1 RobotModel, 64D token action, 7D hands, deploy obs config, and one dataset sample validate. |
| M1 tiny vertical slice | Collect and train on 10 tabletop demos. | One cleaned LeRobot dataset loads in GR00T and one inference rollout runs in sim. |
| M2 direct baseline | Fine-tune GR00T on 50-100 successful demos. | Open-loop eval passes and C++ SONIC sim inference is stable. |
| M3 curated curriculum | Train with curated stage metadata and stricter data QA. | Beats direct baseline on success/drop/fall metrics. |
| M4 walking extension | Add short table-to-tray walking carry. | Tabletop metrics remain stable and walking fall/drop rates are acceptable. |
| M5 augmentation | Add BONES-SEED/SONIC fine-tuning or GR00T-Dreams. | Improves unseen objects/layouts without harming safety metrics. |

## Baselines

| Baseline | Purpose |
|---|---|
| SONIC teleop/data replay | Confirms the controller and recording path are healthy. |
| Direct GR00T fine-tune | Measures foundation-model benefit without curriculum. |
| Curated curriculum GR00T fine-tune | Main method. |
| SONIC motion tracking only | Body-competence lower bound without semantic task policy. |
| GR00T-Dreams augmentation | Phase-2 generalization baseline, not an MVP dependency. |

## Stage Gates

Do not advance to real robot until the previous gate has passing logs:

1. Dataset gate: no malformed modality config, missing video, NaN action, stale
   SMPL lead-in, or timestamp drift beyond tolerance.
2. Open-loop gate: held-out action predictions have plausible token/hand ranges
   and no action spikes.
3. Sim gate: C++ SONIC sim inference starts paused, blends to initial pose, and
   completes tabletop rollouts without falls.
4. Hardware canary gate: emergency stop, policy pause, temperature monitor,
   joint-limit monitor, and manual recovery are verified before autonomous runs.

## Safety Rules

- The first real-robot autonomous run starts with the robot physically guarded
  and policy inference paused.
- The operator must verify `k`, `i`, and `p` controls before unpausing the VLA.
- Trials progress in this order: hands, reach, touch, grasp, lift 5 cm, place
  nearby, full tabletop task, then walking.
- A policy that occasionally sends unsafe commands is not ready for hardware,
  even if aggregate success is high.

## Repository Boundary

This repo owns the SONIC/WBC substrate. Research additions should be thin:
configuration, validation, task/eval scripts, and docs. Avoid changing
`gear_sonic_deploy` command writing, ZMQ protocol layout, observation ordering,
or released ONNX configs unless a later milestone explicitly requires it.
