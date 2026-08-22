"""Focused tests for boundaries and state transitions that define task success."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from isaaclab.managers import ObservationTermCfg, RewardTermCfg
from so101_rl.tasks.object_in_cup.mdp.geometry import (
    placement_mask,
    update_settle_counter,
)
from so101_rl.tasks.object_in_cup.mdp.rewards import transport_object
from so101_rl.tasks.object_in_cup.mdp.terminations import object_dropped
from so101_rl.tasks.common.mdp.observations import camera_rgb


class PlacementLogicTests(unittest.TestCase):
    def setUp(self):
        self.relative = torch.tensor(
            [
                [0.003, 0.0, 0.020],
                [0.0031, 0.0, 0.020],
                [0.0, 0.0, 0.014],
                [0.0, 0.0, 0.020],
                [0.0, 0.0, 0.020],
                [0.0, 0.0, 0.020],
            ]
        )
        self.linear_velocity = torch.zeros((6, 3))
        self.angular_velocity = torch.zeros((6, 3))
        self.gripper_position = torch.full((6,), 1.2)
        self.linear_velocity[3, 0] = 0.051
        self.angular_velocity[4, 2] = 1.01
        self.gripper_position[5] = 1.19

    def test_insertion_release_and_velocity_boundaries(self):
        valid = placement_mask(
            self.relative,
            self.linear_velocity,
            self.angular_velocity,
            self.gripper_position,
            xy_tolerance=0.003,
            center_z_min=0.015,
            center_z_max=0.030,
            linear_velocity_max=0.05,
            angular_velocity_max=1.0,
            released_position_min=1.2,
        )
        self.assertEqual(valid.tolist(), [True, False, False, False, False, False])

    def test_settling_window_requires_consecutive_steps(self):
        counter = torch.zeros(2, dtype=torch.long)
        for valid in (
            torch.tensor([True, True]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
        ):
            counter = update_settle_counter(counter, valid)
        self.assertEqual(counter.tolist(), [3, 1])
        self.assertEqual((counter >= 3).tolist(), [True, False])

    def test_drop_threshold(self):
        root_positions = torch.tensor([[0.0, 0.0, -0.0201], [0.0, 0.0, -0.0200]])
        proxy = SimpleNamespace(torch=root_positions)
        object_asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=proxy))
        env = SimpleNamespace(scene={"object": object_asset})
        self.assertEqual(
            object_dropped(env, minimum_height=-0.02).tolist(), [True, False]
        )

    def test_transport_credit_scales_with_lift_height(self):
        object_positions = torch.tensor(
            [
                [0.0, 0.0, 0.0134],
                [0.0, 0.0, 0.0135],
                [0.0, 0.0, 0.0285],
                [0.0, 0.0, 0.0435],
                [0.0, 0.0, 0.0125],
            ],
            dtype=torch.float32,
        )
        cup_positions = torch.zeros_like(object_positions)
        env = SimpleNamespace(
            device="cpu",
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pos_w=SimpleNamespace(torch=object_positions)
                    )
                ),
                "cup": SimpleNamespace(
                    data=SimpleNamespace(
                        root_pos_w=SimpleNamespace(torch=cup_positions)
                    )
                ),
            }
        )
        term = transport_object(
            RewardTermCfg(func=transport_object),
            SimpleNamespace(num_envs=5, device="cpu"),
        )

        reward = term(
            env, std=0.08, minimum_height=0.0135, lift_height=0.03
        ).tolist()

        # The cube sits directly above the cup, so the proximity score is 1.0
        # and the reward isolates the lift conditioning: nothing at or below
        # the off-table gate (pushing the cube cannot farm the term), half
        # credit halfway up the ramp, full credit at carry height.
        self.assertEqual(reward[0], 0.0)
        self.assertEqual(reward[1], 0.0)
        self.assertAlmostEqual(reward[2], 0.5, places=5)
        self.assertAlmostEqual(reward[3], 1.0, places=5)
        self.assertEqual(reward[4], 0.0)


class VisualLatencyResetTests(unittest.TestCase):
    def test_camera_delay_accepts_full_and_indexed_resets(self):
        env = SimpleNamespace(num_envs=4, device="cpu")
        term = camera_rgb(
            ObservationTermCfg(
                func=camera_rgb,
                params={"randomize": False, "max_delay_steps": 1},
            ),
            env,
        )

        term.reset()
        term.reset(torch.tensor([1, 3], dtype=torch.long))
        self.assertEqual(tuple(term._delay.time_lags.shape), (4,))


if __name__ == "__main__":
    unittest.main()
