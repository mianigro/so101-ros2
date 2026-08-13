"""Focused tensor tests for pickup reward geometry and state transitions."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from so101_rl.tasks.common.mdp.pickup import (
    bilateral_same_step_contact,
    bounded_closure_increment,
    episode_best_increment,
    first_event_increment,
    grasp_targets_from_fixed_pad,
    object_between_jaws,
    projected_half_extent,
    retryable_event_increment,
)
from so101_rl.tasks.object_in_cup.mdp.rewards import (
    approach_progress as ApproachProgress,
    closure_progress as ClosureProgress,
    grasp_acquired as GraspAcquired,
    grasp_held as GraspHeld,
    lift_progress as LiftProgress,
)
from so101_rl.tasks.three_boxes_in_cups.mdp import rewards as three_box_rewards


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


class GraspTargetTests(unittest.TestCase):
    def test_cube_support_and_nominal_target_at_zero_and_45_degrees(self):
        yaw = torch.tensor([0.0, math.pi / 4.0])
        quaternion = torch.zeros((2, 4))
        quaternion[:, 2] = torch.sin(0.5 * yaw)
        quaternion[:, 3] = torch.cos(0.5 * yaw)
        normal = torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1)

        support = projected_half_extent(
            quaternion, normal, (0.0125, 0.0125, 0.0125)
        )
        torch.testing.assert_close(
            support,
            torch.tensor([0.0125, 0.0125 * math.sqrt(2.0)]),
            atol=1e-7,
            rtol=0.0,
        )

        pad_center = torch.tensor([[-0.00805, -0.000218, -0.0925]]).expand(2, -1)
        pad_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1)
        target = grasp_targets_from_fixed_pad(
            pad_center,
            pad_quaternion,
            quaternion,
            (0.0125, 0.0125, 0.0125),
            pad_thickness=0.0005,
            clearance=0.0005,
        )
        self.assertAlmostEqual(target[0, 0].item(), 0.0052, places=7)
        self.assertAlmostEqual(target[0, 1].item(), -0.000218, places=7)
        self.assertAlmostEqual(target[0, 2].item(), -0.0925, places=7)
        self.assertGreater(target[1, 0].item(), target[0, 0].item())

    def test_cube_edge_needs_one_millimetre_of_jaw_insertion(self):
        objects = torch.tensor(
            [
                [0.05, 0.0, -0.024000],
                [0.05, 0.0, -0.024001],
                [-0.001, 0.0, 0.0],
                [0.101, 0.0, 0.0],
                [0.05, 0.013, 0.0],
                [0.05, 0.0, 0.013],
            ],
            dtype=torch.float64,
        ).unsqueeze(0)
        object_quaternions = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=torch.float64
        ).expand(1, objects.shape[1], -1)
        fixed = torch.zeros((1, 3), dtype=torch.float64)
        fixed_quaternion = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]], dtype=torch.float64
        )
        moving = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64)

        between = object_between_jaws(
            objects,
            object_quaternions,
            fixed,
            fixed_quaternion,
            moving,
            (0.0125, 0.0125, 0.0125),
            fixed_pad_length=0.025,
            minimum_insertion=0.001,
        )

        self.assertEqual(
            between.tolist(), [[True, False, False, False, False, False]]
        )


class BoundedProgressTests(unittest.TestCase):
    def test_best_progress_cannot_be_farmed_by_backing_off(self):
        best = torch.zeros((1, 1))
        initialized = torch.ones((1, 1), dtype=torch.bool)
        eligible = torch.ones((1, 1), dtype=torch.bool)
        cumulative = 0.0

        for value in (0.2, 0.8, 0.3, 0.7, 1.0, 0.4):
            increment, best, initialized = episode_best_increment(
                torch.tensor([[value]]), best, initialized, eligible
            )
            cumulative += increment.item()

        self.assertAlmostEqual(cumulative, 1.0, places=6)
        self.assertAlmostEqual(best.item(), 1.0, places=6)

        # Resetting the bookkeeping restores the full bounded episode budget.
        best.zero_()
        initialized.fill_(True)
        first, best, initialized = episode_best_increment(
            torch.tensor([[0.6]]), best, initialized, eligible
        )
        second, _, _ = episode_best_increment(
            torch.tensor([[0.9]]), best, initialized, eligible
        )
        self.assertAlmostEqual(first.item(), 0.6, places=6)
        self.assertAlmostEqual(second.item(), 0.3, places=6)

    def test_closure_can_reward_an_aligned_reclose(self):
        credited = torch.zeros((1, 1))
        initialized = torch.zeros(1, dtype=torch.bool)
        eligible = torch.ones((1, 1), dtype=torch.bool)
        alignment = torch.ones((1, 1))

        first, credited, initialized = bounded_closure_increment(
            torch.tensor([1.2]),
            torch.tensor([1.2]),
            alignment,
            credited,
            eligible,
            initialized,
            closure_range=1.2,
        )
        close, credited, initialized = bounded_closure_increment(
            torch.tensor([1.2]),
            torch.tensor([0.0]),
            alignment,
            credited,
            eligible,
            initialized,
            closure_range=1.2,
        )
        reopen, credited, initialized = bounded_closure_increment(
            torch.tensor([0.0]),
            torch.tensor([1.2]),
            alignment,
            credited,
            eligible,
            initialized,
            closure_range=1.2,
        )
        reclose, credited, _ = bounded_closure_increment(
            torch.tensor([1.2]),
            torch.tensor([0.0]),
            alignment,
            credited,
            eligible,
            initialized,
            closure_range=1.2,
        )
        self.assertEqual(first.item(), 0.0)
        self.assertEqual(close.item(), 1.0)
        self.assertEqual(reopen.item(), 0.0)
        self.assertEqual(reclose.item(), 1.0)
        self.assertEqual(credited.item(), 2.0)


class BilateralContactTests(unittest.TestCase):
    def test_requires_same_box_in_same_stored_substep_and_masks_placed_box(self):
        fixed = torch.zeros((5, 4, 1, 3, 3))
        moving = torch.zeros_like(fixed)

        fixed[0, 0, 0, 0, 0] = 0.2  # unilateral
        fixed[1, 0, 0, 0, 0] = 0.2  # different boxes
        moving[1, 0, 0, 1, 0] = 0.2
        fixed[2, 0, 0, 1, 0] = 0.2  # asynchronous substeps
        moving[2, 1, 0, 1, 0] = 0.2
        fixed[3, 2, 0, 2, 0] = 0.2  # valid same box/substep
        moving[3, 2, 0, 2, 0] = 0.2
        fixed[4, 0, 0, 0, 0] = 0.1  # threshold must be exceeded
        moving[4, 0, 0, 0, 0] = 0.1

        contact = bilateral_same_step_contact(
            fixed, moving, force_threshold=0.1
        )
        self.assertEqual(
            contact.tolist(),
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [False, False, False],
            ],
        )

        credited = torch.zeros_like(contact, dtype=torch.bool)
        eligible = torch.ones_like(contact, dtype=torch.bool)
        eligible[3, 2] = False
        impulse, credited = first_event_increment(contact, credited, eligible)
        self.assertEqual(impulse.sum().item(), 0.0)
        self.assertFalse(credited.any().item())


class RetryableEventTests(unittest.TestCase):
    def test_rising_edge_re_arms_after_contact_loss(self):
        in_contact = torch.zeros((1, 2), dtype=torch.bool)
        eligible = torch.ones((1, 2), dtype=torch.bool)
        impulses = []
        # Box 0 contact over time: absent, appear, hold, drop, reappear.
        # Box 1 never contacts. Expected impulses fire only on rising edges.
        events = [
            torch.tensor([[False, False]]),
            torch.tensor([[True, False]]),
            torch.tensor([[True, False]]),
            torch.tensor([[False, False]]),
            torch.tensor([[True, False]]),
        ]
        expected = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]
        for event in events:
            impulse, in_contact = retryable_event_increment(
                event, in_contact, eligible
            )
            impulses.append(impulse.tolist()[0])
        self.assertEqual(impulses, expected)

    def test_ineligible_entity_never_fires(self):
        in_contact = torch.zeros((1, 1), dtype=torch.bool)
        eligible = torch.zeros((1, 1), dtype=torch.bool)
        impulse, in_contact = retryable_event_increment(
            torch.tensor([[True]]), in_contact, eligible
        )
        self.assertEqual(impulse.item(), 0.0)
        self.assertFalse(in_contact.item())


class ScriptedPickupSequenceTests(unittest.TestCase):
    def test_pickup_sequence_and_retryable_between_jaws_closure(self):
        object_position = torch.tensor([[0.30, 0.0, 0.0125]])
        object_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        gripper_position = torch.tensor([[1.2]])
        pad_position = torch.tensor([[[0.18675, 0.0, 0.0125]]])
        pad_quaternion = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        moving_pad_position = torch.tensor([[[0.28675, 0.0, 0.0125]]])
        moving_pad_quaternion = pad_quaternion.clone()
        fixed_forces = torch.zeros((1, 4, 1, 1, 3))
        moving_forces = torch.zeros_like(fixed_forces)

        env = SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=1.0 / 30.0,
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pos_w=_proxy(object_position),
                        root_quat_w=_proxy(object_quaternion),
                    )
                ),
                "robot": SimpleNamespace(
                    data=SimpleNamespace(joint_pos=_proxy(gripper_position))
                ),
                "fixed_jaw_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        pos_w=_proxy(pad_position),
                        quat_w=_proxy(pad_quaternion),
                        force_matrix_w_history=_proxy(fixed_forces),
                    )
                ),
                "moving_jaw_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        pos_w=_proxy(moving_pad_position),
                        quat_w=_proxy(moving_pad_quaternion),
                        force_matrix_w_history=_proxy(moving_forces)
                    )
                ),
            },
        )
        robot_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0]))
        fixed_cfg = SimpleNamespace(name="fixed_jaw_contact")
        moving_cfg = SimpleNamespace(name="moving_jaw_contact")
        cfg = SimpleNamespace()
        approach = ApproachProgress(cfg, env)
        closure = ClosureProgress(cfg, env)
        grasp = GraspAcquired(cfg, env)
        lift = LiftProgress(cfg, env)

        initial_approach = approach(
            env,
            (0.0125, 0.0125, 0.0125),
            0.0005,
            fixed_sensor_cfg=fixed_cfg,
        )
        self.assertGreater(initial_approach.item(), 0.0)
        self.assertEqual(
            approach(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                fixed_sensor_cfg=fixed_cfg,
            ).item(),
            0.0,
        )
        object_position[:, 0] = 0.20
        self.assertGreater(
            approach(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                fixed_sensor_cfg=fixed_cfg,
            ).item(),
            0.0,
        )

        self.assertEqual(
            closure(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                robot_cfg=robot_cfg,
                fixed_sensor_cfg=fixed_cfg,
            ).item(),
            0.0,
        )
        gripper_position[:, 0] = 0.0
        self.assertAlmostEqual(
            closure(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                robot_cfg=robot_cfg,
                fixed_sensor_cfg=fixed_cfg,
            ).item(),
            30.0,
            delta=1e-3,
        )

        gripper_position[:, 0] = 1.2
        self.assertEqual(
            closure(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                robot_cfg=robot_cfg,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        object_position[:, 0] = 0.30
        gripper_position[:, 0] = 0.0
        self.assertEqual(
            closure(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                robot_cfg=robot_cfg,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        gripper_position[:, 0] = 1.2
        closure(
            env,
            (0.0125, 0.0125, 0.0125),
            0.0005,
            robot_cfg=robot_cfg,
            fixed_sensor_cfg=fixed_cfg,
            moving_sensor_cfg=moving_cfg,
        )
        object_position[:, 0] = 0.20
        gripper_position[:, 0] = 0.0
        self.assertAlmostEqual(
            closure(
                env,
                (0.0125, 0.0125, 0.0125),
                0.0005,
                robot_cfg=robot_cfg,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            30.0,
            delta=1e-3,
        )

        self.assertEqual(
            grasp(
                env,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        fixed_forces[0, 0, 0, 0, 0] = 0.2
        moving_forces[0, 0, 0, 0, 0] = 0.2
        self.assertAlmostEqual(
            grasp(
                env,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            30.0,
            delta=1e-3,
        )
        self.assertEqual(
            grasp(
                env,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        # Retryable grasp: dropping contact re-arms the landmark, so a fresh
        # pinch after a drop pays again.
        fixed_forces.zero_()
        moving_forces.zero_()
        self.assertEqual(
            grasp(
                env,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        fixed_forces[0, 0, 0, 0, 0] = 0.2
        moving_forces[0, 0, 0, 0, 0] = 0.2
        self.assertAlmostEqual(
            grasp(
                env,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            30.0,
            delta=1e-3,
        )

        # Dense lift: pays a constant once the held cube clears the table.
        self.assertEqual(
            lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        object_position[:, 2] = 0.0134  # below the 1 mm clearance
        self.assertEqual(
            lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )
        object_position[:, 2] = 0.0135  # exactly at the inclusive threshold
        self.assertEqual(
            lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            1.0,
        )
        object_position[:, 2] = 0.20
        self.assertEqual(
            lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            1.0,
        )
        moving_forces.zero_()
        self.assertEqual(
            lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,
        )

    def test_three_box_lift_is_constant_and_masks_ineligible_boxes(self):
        positions = torch.tensor(
            [[[0.0, 0.0, 0.0134], [0.0, 0.0, 0.0135], [0.0, 0.0, 0.20]]]
        )
        eligible = torch.tensor([[True, True, False]])
        contact = torch.tensor([[True, True, True]])
        pickup_tensors = (positions, torch.empty(0), torch.empty(0), eligible)
        env = SimpleNamespace(num_envs=1, device="cpu")
        lift = three_box_rewards.lift_progress(SimpleNamespace(), env)

        with (
            patch.object(
                three_box_rewards, "_pickup_tensors", return_value=pickup_tensors
            ),
            patch.object(
                three_box_rewards, "_bilateral_contact", return_value=contact
            ),
        ):
            reward = lift(
                env,
                lift_clearance=0.001,
                object_rest_height=0.0125,
                placement={},
            )

        self.assertEqual(reward.item(), 1.0)

    def test_grasp_held_rewards_clamped_geometry(self):
        object_position = torch.tensor([[0.025, 0.0, 0.0]])
        object_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        gripper_position = torch.tensor([[0.0]])  # closed
        pad_position = torch.tensor([[[0.0, 0.0, 0.0]]])
        pad_quaternion = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        moving_pad_position = torch.tensor([[[0.05, 0.0, 0.0]]])
        moving_pad_quaternion = pad_quaternion.clone()

        env = SimpleNamespace(
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pos_w=_proxy(object_position),
                        root_quat_w=_proxy(object_quaternion),
                    )
                ),
                "robot": SimpleNamespace(
                    data=SimpleNamespace(joint_pos=_proxy(gripper_position))
                ),
                "fixed_jaw_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        pos_w=_proxy(pad_position),
                        quat_w=_proxy(pad_quaternion),
                    )
                ),
                "moving_jaw_contact": SimpleNamespace(
                    data=SimpleNamespace(
                        pos_w=_proxy(moving_pad_position),
                        quat_w=_proxy(moving_pad_quaternion),
                    )
                ),
            }
        )
        robot_cfg = SimpleNamespace(name="robot", joint_ids=torch.tensor([0]))
        fixed_cfg = SimpleNamespace(name="fixed_jaw_contact")
        moving_cfg = SimpleNamespace(name="moving_jaw_contact")

        common = {
            "robot_cfg": robot_cfg,
            "fixed_sensor_cfg": fixed_cfg,
            "moving_sensor_cfg": moving_cfg,
            "closed_threshold": 0.15,
        }

        # Gripper clamped on a cube sitting between the pads -> dense 1.0.
        gripper_position[:, 0] = 0.0
        self.assertEqual(
            GraspHeld(env, (0.0125, 0.0125, 0.0125), **common).item(), 1.0
        )
        # A loose hold (gripper at 0.20 rad => ~32 mm gap, wider than the
        # 25 mm cube) does not count as holding under the tightened threshold.
        gripper_position[:, 0] = 0.20
        self.assertEqual(
            GraspHeld(env, (0.0125, 0.0125, 0.0125), **common).item(), 0.0
        )
        # Opening the gripper fully also drops the reward.
        gripper_position[:, 0] = 1.2
        self.assertEqual(
            GraspHeld(env, (0.0125, 0.0125, 0.0125), **common).item(), 0.0
        )
        # Cube pushed out of the gap -> 0 even with the gripper clamped.
        gripper_position[:, 0] = 0.0
        object_position[:, 0] = 0.20
        self.assertEqual(
            GraspHeld(env, (0.0125, 0.0125, 0.0125), **common).item(), 0.0
        )


if __name__ == "__main__":
    unittest.main()
