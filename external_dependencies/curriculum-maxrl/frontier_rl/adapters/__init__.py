"""Reference TaskSpace adapters, in increasing realism:

- skill_chain: the validated CPU testbed re-expressed against the framework
  interfaces (regression anchor — must reproduce curriculum_maxrl numbers).
- grid_reach: a gridworld reach task with a REINFORCE softmax policy —
  the pattern for goal-conditioned robotics-style envs (goal = position,
  relabel = final position reached).
- gym_goal: a gym-style adapter skeleton showing where env.reset/step go
  and how to bin a continuous goal space into task ids (imports gymnasium
  lazily; the rest of the package never needs it).
"""
