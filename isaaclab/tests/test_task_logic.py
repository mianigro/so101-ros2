"""Focused tests for boundaries and state transitions that define task success."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from isaaclab.managers import ObservationTermCfg
from so101_rl.tasks.object_in_cup.mdp.critic_observations import (
    CRITIC_STATE_COMPONENTS,
    CRITIC_STATE_DIM,
    critic_task_state,
)
from so101_rl.tasks.object_in_cup.mdp.geometry import (
    placement_mask,
    update_settle_counter,
)
from so101_rl.tasks.object_in_cup.mdp.terminations import object_dropped
from so101_rl.tasks.common.mdp.observations import camera_rgb


class CriticObservationTests(unittest.TestCase):
    def test_critic_task_state_layout(self):
        joint_position = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        joint_velocity = joint_position + 10.0
        last_action = joint_position + 20.0
        object_position = torch.tensor([[31.0, 32.0, 33.0]])
        gripper_position = torch.tensor([[1.0, 2.0, 3.0]])
        cup_position = torch.tensor([[4.0, 5.0, 6.0]])
        object_quaternion = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        linear_velocity = torch.tensor([[41.0, 42.0, 43.0]])
        angular_velocity = torch.tensor([[51.0, 52.0, 53.0]])

        def tensor(value):
            return SimpleNamespace(torch=value)

        robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=tensor(joint_position),
                joint_vel=tensor(joint_velocity),
            )
        )
        object_asset = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=tensor(object_position),
                root_quat_w=tensor(object_quaternion),
                root_lin_vel_w=tensor(linear_velocity),
                root_ang_vel_w=tensor(angular_velocity),
            )
        )
        env = SimpleNamespace(
            scene={
                "robot": robot,
                "object": object_asset,
                "cup": SimpleNamespace(
                    data=SimpleNamespace(root_pos_w=tensor(cup_position))
                ),
                "ee_frame": SimpleNamespace(
                    data=SimpleNamespace(
                        target_pos_w=tensor(gripper_position[:, None, :])
                    )
                ),
            },
            action_manager=SimpleNamespace(action=last_action),
        )

        def entity(name):
            return SimpleNamespace(name=name)

        state = critic_task_state(
            env,
            robot_cfg=SimpleNamespace(name="robot", joint_ids=torch.arange(6)),
            object_cfg=entity("object"),
            cup_cfg=entity("cup"),
            ee_frame_cfg=entity("ee_frame"),
        )
        expected_components = (
            joint_position,
            joint_velocity,
            last_action,
            object_position - gripper_position,
            object_position - cup_position,
            object_quaternion,
            linear_velocity,
            angular_velocity,
        )

        self.assertEqual(CRITIC_STATE_DIM, 34)
        self.assertEqual(tuple(state.shape), (1, CRITIC_STATE_DIM))
        offset = 0
        for (name, width), expected in zip(
            CRITIC_STATE_COMPONENTS, expected_components, strict=True
        ):
            self.assertTrue(
                torch.equal(state[:, offset : offset + width], expected), name
            )
            offset += width
        self.assertEqual(offset, CRITIC_STATE_DIM)


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
