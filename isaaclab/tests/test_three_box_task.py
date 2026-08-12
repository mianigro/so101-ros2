"""Focused contracts for the three-box/three-cup scenario."""

from __future__ import annotations

import itertools
import unittest
from types import SimpleNamespace

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from so101_rl.tasks.three_boxes_in_cups.agents.rsl_rl_vision_ppo_cfg import (
    SO101ThreeBoxesInCupsVisionPPOCfg,
)
from so101_rl.tasks.three_boxes_in_cups.mdp.critic_observations import (
    THREE_BOX_CRITIC_STATE_COMPONENTS,
    THREE_BOX_CRITIC_STATE_DIM,
    critic_task_state,
)
from so101_rl.tasks.three_boxes_in_cups.mdp.events import (
    sample_separated_positions,
)
from so101_rl.tasks.three_boxes_in_cups.mdp.geometry import (
    best_assignment_score,
    complete_assignment,
    placement_matrix,
    update_settle_counter,
)


def _tensor(value: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=value)


class ThreeBoxCriticTests(unittest.TestCase):
    def test_exact_84_value_layout(self):
        joint_position = torch.arange(6, dtype=torch.float32).reshape(1, 6)
        joint_velocity = joint_position + 10.0
        last_action = joint_position + 20.0
        gripper = torch.tensor([[2.0, 3.0, 4.0]])
        box_positions = torch.tensor(
            [[[31.0, 32.0, 33.0], [41.0, 42.0, 43.0], [51.0, 52.0, 53.0]]]
        )
        cup_positions = torch.tensor(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]
        )
        quaternions = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 60.0
        linear = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3) + 80.0
        angular = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3) + 100.0

        boxes = {
            f"box_{index + 1}": SimpleNamespace(
                data=SimpleNamespace(
                    root_pos_w=_tensor(box_positions[:, index]),
                    root_quat_w=_tensor(quaternions[:, index]),
                    root_lin_vel_w=_tensor(linear[:, index]),
                    root_ang_vel_w=_tensor(angular[:, index]),
                )
            )
            for index in range(3)
        }
        cups = {
            f"cup_{index + 1}": SimpleNamespace(
                data=SimpleNamespace(root_pos_w=_tensor(cup_positions[:, index]))
            )
            for index in range(3)
        }
        robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=_tensor(joint_position), joint_vel=_tensor(joint_velocity)
            )
        )
        env = SimpleNamespace(
            scene={
                "robot": robot,
                "ee_frame": SimpleNamespace(
                    data=SimpleNamespace(target_pos_w=_tensor(gripper[:, None, :]))
                ),
                **boxes,
                **cups,
            },
            action_manager=SimpleNamespace(action=last_action),
        )
        state = critic_task_state(
            env,
            robot_cfg=SimpleNamespace(name="robot", joint_ids=torch.arange(6)),
            ee_frame_cfg=SimpleNamespace(name="ee_frame"),
        )
        expected = (
            joint_position,
            joint_velocity,
            last_action,
            (box_positions - gripper[:, None]).reshape(1, -1),
            (box_positions[:, :, None] - cup_positions[:, None]).reshape(1, -1),
            quaternions.reshape(1, -1),
            linear.reshape(1, -1),
            angular.reshape(1, -1),
        )

        self.assertEqual(THREE_BOX_CRITIC_STATE_DIM, 84)
        self.assertEqual(tuple(state.shape), (1, THREE_BOX_CRITIC_STATE_DIM))
        offset = 0
        for (name, width), component in zip(
            THREE_BOX_CRITIC_STATE_COMPONENTS, expected, strict=True
        ):
            self.assertTrue(
                torch.equal(state[:, offset : offset + width], component), name
            )
            offset += width
        self.assertEqual(offset, THREE_BOX_CRITIC_STATE_DIM)

    def test_shared_mlp_produces_one_value_from_84_values(self):
        observations = TensorDict(
            {"critic_state": torch.zeros(4, THREE_BOX_CRITIC_STATE_DIM)},
            batch_size=[4],
        )
        groups = {"critic": ["critic_state"]}
        config = SO101ThreeBoxesInCupsVisionPPOCfg().critic
        critic = MLPModel(
            observations,
            groups,
            "critic",
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            obs_normalization=config.obs_normalization,
            distribution_cfg=config.distribution_cfg,
        )
        self.assertEqual(tuple(critic(observations).shape), (4, 1))
        self.assertIsNone(critic.distribution)


class ThreeBoxGeometryTests(unittest.TestCase):
    def test_success_is_permutation_invariant_and_requires_distinct_cups(self):
        for permutation in itertools.permutations(range(3)):
            permuted = torch.zeros((1, 3, 3), dtype=torch.bool)
            for box_index, cup_index in enumerate(permutation):
                permuted[0, box_index, cup_index] = True
            self.assertEqual(complete_assignment(permuted).tolist(), [True])
            self.assertEqual(best_assignment_score(permuted).tolist(), [1.0])

        duplicated = torch.zeros((1, 3, 3), dtype=torch.bool)
        duplicated[0, 0, 0] = True
        duplicated[0, 1, 0] = True
        duplicated[0, 2, 2] = True
        self.assertEqual(complete_assignment(duplicated).tolist(), [False])
        self.assertAlmostEqual(best_assignment_score(duplicated).item(), 2.0 / 3.0)

    def test_previous_placements_remain_released_while_next_box_is_grasped(self):
        cups = torch.tensor([[[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.20, 0.0, 0.0]]])
        boxes = cups + torch.tensor([0.0, 0.0, 0.02])
        zero_velocity = torch.zeros((1, 3, 3))
        closed_gripper = torch.tensor([0.40])
        near_first_box = boxes[:, 0]
        valid = placement_matrix(
            boxes,
            cups,
            zero_velocity,
            zero_velocity,
            near_first_box,
            closed_gripper,
            xy_tolerance=0.004,
            center_z_min=0.015,
            center_z_max=0.035,
            linear_velocity_max=0.025,
            angular_velocity_max=0.50,
            released_position_min=1.20,
            release_distance_min=0.050,
            require_stable=True,
        )
        self.assertFalse(valid[0, 0].any())
        self.assertTrue(valid[0, 1, 1])
        self.assertTrue(valid[0, 2, 2])

    def test_all_three_must_remain_valid_for_ten_steps(self):
        valid_pairs = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        counter = torch.zeros(1, dtype=torch.long)
        for _ in range(9):
            counter = update_settle_counter(counter, complete_assignment(valid_pairs))
        self.assertFalse((counter >= 10).item())
        counter = update_settle_counter(counter, complete_assignment(valid_pairs))
        self.assertTrue((counter >= 10).item())
        valid_pairs[0, 2, 2] = False
        counter = update_settle_counter(counter, complete_assignment(valid_pairs))
        self.assertEqual(counter.item(), 0)


class ThreeBoxResetTests(unittest.TestCase):
    def test_fixed_and_full_curriculum_layout_bounds_and_spacing(self):
        nominal = torch.tensor(
            [[0.15, -0.085], [0.22, -0.120], [0.29, -0.085]]
        )
        fixed = sample_separated_positions(
            nominal,
            count=8,
            progress=0.0,
            nominal_jitter=0.0,
            zone_low=(0.14, -0.13),
            zone_high=(0.30, -0.04),
            minimum_separation=0.040,
            max_attempts=128,
            device="cpu",
        )
        self.assertTrue(torch.equal(fixed, nominal.unsqueeze(0).expand(8, -1, -1)))

        torch.manual_seed(7)
        randomized = sample_separated_positions(
            nominal,
            count=256,
            progress=1.0,
            nominal_jitter=0.003,
            zone_low=(0.14, -0.13),
            zone_high=(0.30, -0.04),
            minimum_separation=0.040,
            max_attempts=128,
            device="cpu",
        )
        self.assertTrue((randomized[..., 0] >= 0.14).all())
        self.assertTrue((randomized[..., 0] <= 0.30).all())
        self.assertTrue((randomized[..., 1] >= -0.13).all())
        self.assertTrue((randomized[..., 1] <= -0.04).all())
        distances = torch.cdist(randomized, randomized)
        diagonal = torch.eye(3, dtype=torch.bool).unsqueeze(0)
        self.assertTrue((distances.masked_fill(diagonal, 1.0) >= 0.040).all())

        cups = sample_separated_positions(
            torch.tensor([[0.15, 0.085], [0.22, 0.120], [0.29, 0.085]]),
            count=256,
            progress=1.0,
            nominal_jitter=0.003,
            zone_low=(0.14, 0.04),
            zone_high=(0.30, 0.13),
            minimum_separation=0.060,
            max_attempts=128,
            device="cpu",
        )
        cup_distances = torch.cdist(cups, cups)
        self.assertTrue((cup_distances.masked_fill(diagonal, 1.0) >= 0.060).all())
        cross_distances = torch.linalg.vector_norm(
            randomized[:, :, None] - cups[:, None, :], dim=-1
        )
        self.assertTrue((cross_distances >= 0.055).all())

    def test_impossible_layout_fails_clearly(self):
        with self.assertRaisesRegex(RuntimeError, "could not sample"):
            sample_separated_positions(
                torch.zeros((3, 2)),
                count=1,
                progress=1.0,
                nominal_jitter=0.0,
                zone_low=(0.0, 0.0),
                zone_high=(0.0, 0.0),
                minimum_separation=0.1,
                max_attempts=3,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
