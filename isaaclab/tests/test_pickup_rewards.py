"""Focused tensor tests for the object-in-cup outcome rewards."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from so101_rl.tasks.common.mdp.pickup import episode_best_increment
from so101_rl.tasks.object_in_cup.mdp.rewards import (
    approach_progress as ApproachProgress,
    lift_progress,
    transport_object as TransportObject,
)


def _proxy(tensor: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(torch=tensor)


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
                    # The ee_frame carries one target per frame slot; the
                    # reward reads slot 0 (the nominal grasp point).
                    data=SimpleNamespace(
                        target_pos_w=_proxy(grasp_position.unsqueeze(1))
                    )
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
        # Approach, retreat, and re-approach without leaving the retry radius:
        # only the improvements pay, and the payoff telescopes to the best.
        for distance in (0.20, 0.14, 0.18, 0.10, 0.16, 0.06, 0.05, 0.02):
            grasp.copy_(torch.tensor([[0.20 + distance, -0.065, 0.0125]]))
            cumulative += term(env, position_scale=0.08).item()
        best = self._score(0.02) / self.STEP_DT
        self.assertAlmostEqual(cumulative, best, places=5)
        self.assertLessEqual(cumulative, best + 1e-6)

    def test_retreat_re_arms_a_discounted_approach_budget(self):
        grasp = torch.tensor([[0.30, -0.065, 0.0125]])
        env = self._make_env(grasp)
        term = self._term(env)

        def call():
            return term(env, position_scale=0.08).item() * self.STEP_DT

        def at(distance):
            grasp.copy_(torch.tensor([[0.20 + distance, -0.065, 0.0125]]))

        # The first attempt pays the full telescoped score...
        at(0.02)
        self.assertAlmostEqual(call(), self._score(0.02), places=6)
        # ...retreating beyond the retry radius pays nothing...
        at(0.12)
        self.assertEqual(call(), 0.0)
        # ...but re-arms the budget, so the return leg pays again at half.
        at(0.02)
        recovery = (self._score(0.02) - self._score(0.06)) * 0.5
        self.assertAlmostEqual(call(), recovery, places=6)
        # A second cycle pays a quarter: cycling decays geometrically.
        at(0.12)
        self.assertEqual(call(), 0.0)
        at(0.02)
        self.assertAlmostEqual(call(), recovery * 0.5, places=6)
        # Holding position still pays nothing.
        self.assertEqual(call(), 0.0)

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
            lift_height=0.03,
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
            env, std=0.08, minimum_height=0.0135, lift_height=0.03,
            grace_steps=30, decay_steps=120, decay_floor=0.2,
        ).item()

        # Step through the grace window (30 steps): base stays at 1.0.
        for _ in range(29):
            call()
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
            env, std=0.08, minimum_height=0.0135, lift_height=0.03,
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
