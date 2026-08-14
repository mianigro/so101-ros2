"""Focused tensor tests for pickup reward primitives and outcome rewards."""

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
    lift_progress,
    transport_object as TransportObject,
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


class ThreeBoxLiftTests(unittest.TestCase):
    """The sibling three-box task keeps its contact-gated lift constant."""

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


class ApproachProgressTests(unittest.TestCase):
    """The reaching shaping term pays only genuine episode-best improvements."""

    STEP_DT = 1.0 / 30.0
    CUBE_POSITION = (0.20, -0.065, 0.0125)

    def _make_env(self, grasp_position: torch.Tensor):
        cube = torch.tensor([self.CUBE_POSITION])
        return SimpleNamespace(
            num_envs=1,
            device="cpu",
            step_dt=self.STEP_DT,
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(root_pos_w=_proxy(cube))
                ),
                "ee_frame": SimpleNamespace(
                    data=SimpleNamespace(target_pos_w=_proxy(grasp_position))
                ),
            },
        )

    def _term(self, env):
        return ApproachProgress(SimpleNamespace(), env)

    @staticmethod
    def _score(distance: float) -> float:
        return (1.0 - torch.tanh(torch.tensor(distance / 0.08))).item()

    def test_first_step_pays_full_score_over_step_dt(self):
        # Grasp point 10 cm from the cube centre.
        env = self._make_env(torch.tensor([[0.30, -0.065, 0.0125]]))
        reward = self._term(env)(env, position_scale=0.08).item()
        self.assertAlmostEqual(
            reward, self._score(0.10) / self.STEP_DT, places=6
        )

    def test_holding_or_backing_off_pays_zero(self):
        grasp = torch.tensor([[0.30, -0.065, 0.0125]])
        env = self._make_env(grasp)
        term = self._term(env)

        def call() -> float:
            return term(env, position_scale=0.08).item()

        self.assertAlmostEqual(call(), self._score(0.10) / self.STEP_DT, places=6)
        # Holding the same distance pays nothing.
        self.assertEqual(call(), 0.0)
        # Backing away pays nothing.
        grasp.copy_(torch.tensor([[0.40, -0.065, 0.0125]]))
        self.assertEqual(call(), 0.0)
        # A genuine improvement pays only its delta.
        grasp.copy_(torch.tensor([[0.25, -0.065, 0.0125]]))
        self.assertAlmostEqual(
            call(),
            (self._score(0.05) - self._score(0.10)) / self.STEP_DT,
            places=6,
        )

    def test_cumulative_payoff_telescopes_to_the_best_score(self):
        grasp = torch.tensor([[0.40, -0.065, 0.0125]])
        env = self._make_env(grasp)
        term = self._term(env)
        cumulative = 0.0
        # Approach, retreat, and re-approach: only the improvements pay.
        for distance in (0.20, 0.14, 0.18, 0.10, 0.16, 0.06, 0.12, 0.02):
            grasp.copy_(torch.tensor([[0.20 + distance, -0.065, 0.0125]]))
            cumulative += term(env, position_scale=0.08).item()
        best = self._score(0.02) / self.STEP_DT
        self.assertAlmostEqual(cumulative, best, places=5)
        self.assertLessEqual(cumulative, best + 1e-6)

    def test_reset_re_arms_for_a_new_episode(self):
        grasp = torch.tensor([[0.30, -0.065, 0.0125]])
        env = self._make_env(grasp)
        term = self._term(env)
        first = term(env, position_scale=0.08).item()
        self.assertEqual(term(env, position_scale=0.08).item(), 0.0)
        term.reset()
        self.assertAlmostEqual(
            term(env, position_scale=0.08).item(), first, places=6
        )

    def test_reward_is_direction_agnostic(self):
        # Equal distances from different approach directions pay identically.
        rewards = []
        for grasp_position in (
            [0.30, -0.065, 0.0125],
            [0.20, 0.035, 0.0125],
            [0.20, -0.065, 0.1125],
        ):
            env = self._make_env(torch.tensor([grasp_position]))
            rewards.append(self._term(env)(env, position_scale=0.08).item())
        self.assertEqual(len(set(rewards)), 1)
        self.assertAlmostEqual(rewards[0], self._score(0.10) / self.STEP_DT, places=6)


class LiftProgressTests(unittest.TestCase):
    """The pick-up outcome pays a pure cube-height ramp, contact-free."""

    def _make_env(self, object_z: float):
        object_position = torch.tensor([[0.20, -0.065, object_z]])
        return SimpleNamespace(
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(root_pos_w=_proxy(object_position))
                ),
            },
        )

    def test_height_ramp(self):
        common = {"object_rest_height": 0.0125, "height_scale": 0.05}
        # At rest the ramp is exactly 0.
        self.assertEqual(
            lift_progress(self._make_env(0.0125), **common).item(), 0.0
        )
        # Below rest it clamps to 0.
        self.assertEqual(
            lift_progress(self._make_env(0.0), **common).item(), 0.0
        )
        # Half-way up the scale pays half credit.
        self.assertAlmostEqual(
            lift_progress(self._make_env(0.0125 + 0.025), **common).item(),
            0.5,
            places=6,
        )
        # One height_scale above rest saturates at 1.0.
        self.assertEqual(
            lift_progress(self._make_env(0.0125 + 0.05), **common).item(), 1.0
        )
        # Far above, still saturated.
        self.assertEqual(
            lift_progress(self._make_env(0.20), **common).item(), 1.0
        )


class TransportDecayTests(unittest.TestCase):
    """The transport reward must decay while held aloft and re-arm on a drop."""

    def _make_env(self, object_z: float, cup_xy=(0.24, 0.07)):
        object_position = torch.tensor([[0.20, -0.065, object_z]])
        cup_position = torch.tensor([[cup_xy[0], cup_xy[1], 0.0]])
        return SimpleNamespace(
            num_envs=1,
            device="cpu",
            scene={
                "object": SimpleNamespace(
                    data=SimpleNamespace(root_pos_w=_proxy(object_position))
                ),
                "cup": SimpleNamespace(
                    data=SimpleNamespace(root_pos_w=_proxy(cup_position))
                ),
            },
        )

    def _reward(self, env):
        # Recompute the static reward shape with a fresh term so the aloft
        # counter reflects the scripted height sequence.
        return TransportObject(SimpleNamespace(), env)(
            env,
            std=0.08,
            minimum_height=0.0135,
            grace_steps=30,
            decay_steps=120,
            decay_floor=0.2,
        ).item()

    def test_full_credit_during_grace_window(self):
        env = self._make_env(object_z=0.05)
        reward = self._reward(env)
        # First aloft step: full base (1.0) times the radial score.
        object_position = env.scene["object"].data.root_pos_w.torch
        cup_position = env.scene["cup"].data.root_pos_w.torch
        radial = torch.linalg.vector_norm(
            (object_position - cup_position)[:, :2], dim=-1
        ).item()
        expected = (1.0 - torch.tanh(torch.tensor(radial / 0.08))).item()
        self.assertAlmostEqual(reward, expected)

    def test_decays_after_grace_window_while_held(self):
        env = self._make_env(object_z=0.05)
        object_position = env.scene["object"].data.root_pos_w.torch
        cup_position = env.scene["cup"].data.root_pos_w.torch
        radial = torch.linalg.vector_norm(
            (object_position - cup_position)[:, :2], dim=-1
        ).item()
        radial_score = (1.0 - torch.tanh(torch.tensor(radial / 0.08))).item()

        term = TransportObject(SimpleNamespace(), env)
        call = lambda: term(
            env, std=0.08, minimum_height=0.0135,
            grace_steps=30, decay_steps=120, decay_floor=0.2,
        ).item()

        # Step through the grace window (30 steps): base stays at 1.0.
        last_grace = call()
        self.assertAlmostEqual(last_grace, radial_score)

        # Step 60 steps further into the decay ramp (halfway to floor).
        for _ in range(60):
            mid_decay = call()
        self.assertAlmostEqual(mid_decay, radial_score * (1.0 - 0.8 * 0.5))

        # Step to the end of the ramp: base reaches the floor (0.2).
        for _ in range(60):
            full_decay = call()
        self.assertAlmostEqual(full_decay, radial_score * 0.2)

    def test_drop_resets_aloft_counter_and_re_arms_full_reward(self):
        env = self._make_env(object_z=0.05)
        object_position = env.scene["object"].data.root_pos_w.torch
        cup_position = env.scene["cup"].data.root_pos_w.torch
        radial = torch.linalg.vector_norm(
            (object_position - cup_position)[:, :2], dim=-1
        ).item()
        radial_score = (1.0 - torch.tanh(torch.tensor(radial / 0.08))).item()

        term = TransportObject(SimpleNamespace(), env)
        call = lambda: term(
            env, std=0.08, minimum_height=0.0135,
            grace_steps=30, decay_steps=120, decay_floor=0.2,
        ).item()

        # Decay all the way to the floor.
        for _ in range(30 + 120):
            call()
        self.assertAlmostEqual(call(), radial_score * 0.2)

        # Drop the cube below minimum_height and step: reward is 0 and the
        # aloft counter resets.
        object_position[:, 2] = 0.0
        self.assertEqual(call(), 0.0)

        # Lift again: the very first aloft step pays full base credit.
        object_position[:, 2] = 0.05
        self.assertAlmostEqual(call(), radial_score)

    def test_zero_when_not_lifted(self):
        env = self._make_env(object_z=0.005)  # below minimum_height 0.0135
        self.assertEqual(self._reward(env), 0.0)


if __name__ == "__main__":
    unittest.main()
