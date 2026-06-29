# GR00T-SONIC Evaluation Protocol

## Evaluation Layers

Use four separate evaluation results. Do not collapse them into one score.

| Layer | Question | Required output |
|---|---|---|
| Dataset QA | Is the training/eval data structurally valid? | Validator JSON and terminal log. |
| Open-loop | Does GR00T predict plausible actions on held-out demos? | Token/hand error and action smoothness report. |
| Closed-loop sim | Does the policy complete the task through SONIC? | Success, fall, drop, timeout, completion time. |
| Real canary | Is the hardware behavior safe at slow speed? | Trial log, aborts, temperatures, operator notes. |

## MVP Scenario Matrix

| Scenario | Object | Layout | Perturbation | Minimum trials |
|---|---|---|---|---:|
| seen_tabletop | red cup | seen tray pose | none | 20 |
| unseen_layout | red cup | shifted tray/table pose | none | 20 |
| distractor | red cup + distractors | seen tray pose | visual clutter | 20 |
| lighting_shift | red cup | seen tray pose | lighting/camera shift | 20 |

Walking, push disturbance, and low-friction scenarios are deferred until the
tabletop task is stable.

## Metrics

Primary metrics:

- `task_success_rate`
- `fall_rate`
- `drop_rate`
- `operator_abort_rate`
- `completion_time_s`

Secondary metrics:

- `action_token_l2`
- `left_hand_l1`
- `right_hand_l1`
- `action_jerk`
- `joint_limit_violations`
- `policy_latency_ms`
- `camera_frame_age_ms`

## MVP Pass Gate

The tabletop MVP is considered working when:

```text
sim task_success_rate >= 0.80
sim fall_rate <= 0.05
sim drop_rate <= 0.10
real slow trials >= 10 successful trials, if hardware is available
```

If real hardware is unavailable, record that explicitly and treat the result as
a simulation-only milestone.

## Real-Robot Canary Order

1. Start stack with VLA paused.
2. Verify emergency stop and manual stop.
3. Verify open/close hands.
4. Reach near object without touching.
5. Touch object.
6. Grasp object.
7. Lift 5 cm.
8. Place nearby.
9. Run full tabletop fetch-place.
10. Add walking only after tabletop logs pass review.

## Reporting

Every eval report should include:

- Git commit and local dirty-state summary.
- Dataset path and split.
- Model checkpoint path.
- SONIC deploy checkpoint and observation config.
- Exact commands run.
- Metrics table.
- Failure examples and whether they are perception, grasp, balance, timing, or
  schema failures.
