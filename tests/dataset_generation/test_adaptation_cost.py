"""Tests for measuring what an adaptation costs.

The point of the cost is to make "minimal adaptation" mean something. A policy that crouches
through an empty corridor collides with nothing and is still wrong, and without a cost there
is no way to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.dataset_generation.adaptation_cost import (
    DEFAULT_WEIGHTS,
    adaptation_cost,
    prefers,
)

FPS = 50.0


def motion(frames: int = 200, pelvis: float = 0.75, amplitude: float = 0.10,
           offset: float = 0.0, speed: float = 1.0) -> dict:
    """A body walking forward with a swinging gait, optionally lowered and offset in pose."""
    root = np.zeros((frames, 3))
    root[:, 0] = np.arange(frames) * speed / FPS
    root[:, 2] = pelvis
    phase = np.linspace(0, 8 * np.pi, frames)
    dofs = np.zeros((frames, 29))
    dofs[:, :6] = amplitude * np.sin(phase)[:, None] + offset
    return {"root_pos_w": root, "dof_pos": dofs, "fps": FPS, "total_frames": frames}


def test_a_lowered_motion_costs_more_than_an_upright_one():
    upright = adaptation_cost(motion(), "walk")
    crouched = adaptation_cost(motion(pelvis=0.55), "crouch")
    # The crouch must actually register as lowered before the cost can mean anything.
    assert crouched.total > upright.total


def test_a_permanently_crouched_clip_scores_its_full_depth():
    """The case that makes an external reference necessary.

    Measuring against the clip's own upright height looks more robust and is wrong here: a
    motion crouched for its whole duration never rises, so its own 90th percentile *is* the
    crouch and the most expensive adaptation available scores zero. A sustained crouch is
    precisely the adapted motion this work is built on.
    """
    crouched = adaptation_cost(motion(pelvis=0.55), "crouch")
    assert crouched.pelvis_lowering_m == pytest.approx(0.20, abs=1e-6)
    assert crouched.pelvis_height_m == pytest.approx(0.55, abs=1e-6)


def test_a_different_robot_can_supply_its_own_walking_height():
    tall = adaptation_cost(motion(pelvis=0.90), "tall", reference_pelvis_m=0.90)
    assert tall.pelvis_lowering_m == pytest.approx(0.0, abs=1e-6)


def test_a_dip_partway_through_registers_as_lowering():
    payload = motion()
    payload["root_pos_w"][80:120, 2] = 0.55
    assert adaptation_cost(payload, "duck").pelvis_lowering_m > 0.02


def test_more_joint_motion_costs_more():
    calm = adaptation_cost(motion(amplitude=0.05), "calm")
    busy = adaptation_cost(motion(amplitude=0.40), "busy")
    assert busy.joint_travel_rad_per_s > calm.joint_travel_rad_per_s
    assert busy.total > calm.total


def test_a_held_posture_reads_as_deviation_while_a_stride_does_not():
    """A walk returns to its median pose every stride; a sustained crouch does not.

    This is what lets the postural term separate them without a reference pose.
    """
    swinging = adaptation_cost(motion(amplitude=0.10), "walk")
    held = adaptation_cost(motion(amplitude=0.10, offset=0.5), "held", reference_pose=np.zeros(29))
    assert held.postural_deviation_rad > swinging.postural_deviation_rad


def test_an_explicit_reference_pose_is_used_when_given():
    payload = motion(amplitude=0.0, offset=0.3)
    against_zero = adaptation_cost(payload, "x", reference_pose=np.zeros(29))
    against_self = adaptation_cost(payload, "x")
    assert against_zero.postural_deviation_rad > against_self.postural_deviation_rad


def test_duration_is_part_of_the_cost():
    quick = adaptation_cost(motion(frames=100), "quick")
    slow = adaptation_cost(motion(frames=400), "slow")
    assert slow.duration_s > quick.duration_s
    assert slow.total > quick.total


def test_components_are_reported_so_the_weights_can_be_changed_without_rerunning():
    """The weights are a convention, so a reader must be able to reweight them."""
    cost = adaptation_cost(motion(), "walk")
    assert set(cost.components()) == {
        "pelvis_lowering_m", "pelvis_height_m", "joint_travel_rad_per_s",
        "postural_deviation_rad", "duration_s",
    }
    assert set(DEFAULT_WEIGHTS) == {
        "pelvis_lowering", "joint_travel", "postural_deviation", "duration",
    }


def test_a_preference_needs_a_margin_to_count():
    """Two motions differing in the fourth decimal are interchangeable, and calling one
    optimal overstates the measurement."""
    a = adaptation_cost(motion(), "a")
    b = adaptation_cost(motion(), "b")
    assert not prefers(a, b, margin=0.05)
    assert not prefers(b, a, margin=0.05)


def test_the_preference_reversal_the_family_rests_on():
    """In the easy scene both motions succeed, so the cheaper one should win outright.

    On the measured family the walk totals 0.822 against the crouch's 1.117 -- the crouch
    costs 1.36x -- which is what makes the hard scene a *reversal* rather than simply a
    constraint.
    """
    walk = adaptation_cost(motion(pelvis=0.75, amplitude=0.10), "walk")
    duck = adaptation_cost(motion(pelvis=0.60, amplitude=0.18), "duck")
    assert prefers(walk, duck, margin=0.05)
    assert duck.total / walk.total > 1.1


def test_a_payload_without_dof_positions_falls_back_to_the_reference_qpos():
    payload = motion()
    payload["reference_g1_qpos"] = np.concatenate(
        [np.zeros((len(payload["root_pos_w"]), 7)), payload.pop("dof_pos")], axis=1
    )
    assert adaptation_cost(payload, "x").joint_travel_rad_per_s >= 0.0
