"""Focused tensor tests for the non-farmable pickup reward contract."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from so101_rl.tasks.common.mdp.pickup import (
    bilateral_same_step_contact,
    bounded_closure_increment,
    episode_best_increment,
    first_event_increment,
    grasp_targets_from_fixed_pad,
    projected_half_extent,
)
from so101_rl.tasks.object_in_cup.mdp.rewards import (
    approach_progress as ApproachProgress,
    closure_progress as ClosureProgress,
    grasp_acquired as GraspAcquired,
    lift_progress as LiftProgress,
)


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

    def test_closure_is_bounded_and_reopening_does_not_refill_it(self):
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
        self.assertEqual(reclose.item(), 0.0)
        self.assertEqual(credited.item(), 1.0)


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


class ScriptedPickupSequenceTests(unittest.TestCase):
    def test_rewards_fire_once_in_approach_close_grasp_lift_order(self):
        object_position = torch.tensor([[0.30, 0.0, 0.0125]])
        object_quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        gripper_position = torch.tensor([[1.2]])
        pad_position = torch.tensor([[[0.18675, 0.0, 0.0125]]])
        pad_quaternion = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
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

        # Dense lift: pays the lifted fraction each step while contact holds,
        # and keeps paying (does not extinguish like the old episode-best form).
        self.assertEqual(
            lift(
                env,
                lift_height=0.075,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.0,  # at rest height -> score 0
        )
        object_position[:, 2] += 0.0375  # half of lift_height
        self.assertAlmostEqual(
            lift(
                env,
                lift_height=0.075,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            0.5,  # score 0.5, dense
            delta=1e-6,
        )
        object_position[:, 2] += 0.0375  # full lift_height
        self.assertAlmostEqual(
            lift(
                env,
                lift_height=0.075,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            1.0,
            delta=1e-6,
        )
        self.assertAlmostEqual(  # dense: repeats every step, does not extinguish
            lift(
                env,
                lift_height=0.075,
                object_rest_height=0.0125,
                fixed_sensor_cfg=fixed_cfg,
                moving_sensor_cfg=moving_cfg,
            ).item(),
            1.0,
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
