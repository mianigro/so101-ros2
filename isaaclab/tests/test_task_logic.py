"""Focused tests for boundaries and state transitions that define task success."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from isaaclab.managers import ObservationTermCfg
from so101_rl.tasks.object_in_cup.mdp.geometry import (
    placement_mask,
    update_settle_counter,
)
from so101_rl.tasks.object_in_cup.mdp.terminations import object_dropped
from so101_rl.tasks.object_in_cup.mdp.vision_observations import camera_rgb


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
