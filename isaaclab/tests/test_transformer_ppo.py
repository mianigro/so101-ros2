"""Focused temporal transformer actor and export contract tests."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from models.transformer_ppo.models import TransformerActorCritic, _TransformerTemporalCore, _frame_diff_features
from so101_rl.visual_contract import SO101_TEMPORAL_LOOKBACK_FRAMES


def _history_observations(batch: int = 2) -> TensorDict:
    lookback = SO101_TEMPORAL_LOOKBACK_FRAMES
    return TensorDict(
        {
            "joint_state": torch.zeros(batch, 6),
            "wrist": torch.zeros(batch, lookback, 3, 120, 160),
            "overhead_1": torch.zeros(batch, lookback, 3, 120, 160),
            "overhead_2": torch.zeros(batch, lookback, 3, 120, 160),
            "critic_state": torch.zeros(batch, 34),
        },
        batch_size=[batch],
    )


def _observation_groups() -> dict[str, list[str]]:
    return {
        "actor": ["joint_state", "wrist", "overhead_1", "overhead_2"],
        "critic": ["critic_state"],
    }


class TransformerActorContractTests(unittest.TestCase):
    def test_three_camera_history_actor_and_torchscript_contract(self):
        observations = _history_observations()
        model = TransformerActorCritic(
            observations,
            _observation_groups(),
            "actor",
            output_dim=6,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7},
            lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
            d_model=64,
            num_heads=4,
            num_layers=1,
            d_ff=128,
            dropout=0.1,
        )
        model.eval()
        actions = model(observations)
        self.assertEqual(tuple(actions.shape), (2, 6))
        scripted = torch.jit.script(model.as_jit())
        exported_actions = scripted(
            observations["joint_state"],
            [
                observations["wrist"],
                observations["overhead_1"],
                observations["overhead_2"],
            ],
        )
        self.assertEqual(tuple(exported_actions.shape), (2, 6))
        self.assertTrue(torch.allclose(actions, exported_actions, atol=1.0e-6))

        latent = model.get_latent(observations)
        # 6 normalized joint positions plus one d_model embedding per camera.
        self.assertEqual(tuple(latent.shape), (2, 6 + 3 * 64))

    def test_attention_pooling_variant_matches_export(self):
        observations = _history_observations()
        model = TransformerActorCritic(
            observations,
            _observation_groups(),
            "actor",
            output_dim=6,
            hidden_dims=[256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7},
            lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
            d_model=64,
            num_heads=4,
            num_layers=1,
            d_ff=128,
            dropout=0.0,
            final_layer_pooling=True,
            final_pool_skip=True,
        )
        model.eval()
        actions = model(observations)
        scripted = torch.jit.script(model.as_jit())
        exported_actions = scripted(
            observations["joint_state"],
            [
                observations["wrist"],
                observations["overhead_1"],
                observations["overhead_2"],
            ],
        )
        self.assertTrue(torch.allclose(actions, exported_actions, atol=1.0e-6))

    def test_lookback_mismatch_is_rejected(self):
        observations = TensorDict(
            {
                "joint_state": torch.zeros(2, 6),
                "wrist": torch.zeros(2, SO101_TEMPORAL_LOOKBACK_FRAMES + 1, 3, 8, 8),
            },
            batch_size=[2],
        )
        with self.assertRaises(ValueError):
            TransformerActorCritic(
                observations,
                {"actor": ["joint_state", "wrist"], "critic": ["joint_state"]},
                "actor",
                output_dim=6,
                distribution_cfg=None,
                lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
                d_model=64,
                num_heads=4,
                num_layers=1,
                d_ff=128,
            )

    def test_privileged_mlp_critic_over_history_observations(self):
        observations = _history_observations()
        critic = MLPModel(
            observations,
            _observation_groups(),
            "critic",
            output_dim=1,
            hidden_dims=[256, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=None,
        )
        values = critic(observations)
        self.assertEqual(tuple(values.shape), (2, 1))
        self.assertIsNone(critic.distribution)

    def test_actor_parameter_budget(self):
        # The spatial-softmax reduction replaced a flattened Linear(38400, 64)
        # projection (~2.46M parameters per camera, ~9.9M actor). The reduced
        # actor must stay well inside a 3M budget.
        model = TransformerActorCritic(
            _history_observations(),
            _observation_groups(),
            "actor",
            output_dim=6,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7},
            lookback_frames=SO101_TEMPORAL_LOOKBACK_FRAMES,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 2_000_000)
        self.assertLess(parameter_count, 3_000_000)

    def test_frame_diff_features_are_zero_for_static_windows(self):
        constant = torch.ones(2, SO101_TEMPORAL_LOOKBACK_FRAMES, 64)
        diff = _frame_diff_features(constant)
        self.assertTrue(torch.all(diff == 0.0))
        moving = torch.arange(5.0).repeat(2, 1).unsqueeze(-1) * torch.ones(2, 1, 64)
        diff = _frame_diff_features(moving)
        # Zero for the oldest frame, unit deltas afterwards.
        self.assertTrue(torch.all(diff[:, 0] == 0.0))
        self.assertTrue(torch.all(diff[:, 1:] == 1.0))

    def test_causal_mask_restricts_attention_to_the_past(self):
        torch.manual_seed(5)
        lookback = SO101_TEMPORAL_LOOKBACK_FRAMES
        # Directional perturbation of the newest frame (a constant offset
        # would be stripped by the pre-LayerNorm). Several inputs so the
        # bidirectional control cannot pass by chance.
        x = torch.randn(16, lookback, 64)
        perturbed = x.clone()
        perturbed[:, -1] += 5.0 * torch.randn_like(perturbed[:, -1])

        causal = _TransformerTemporalCore(
            lookback, 64, 4, 1, 128, 0.0, causal_mask=True
        ).eval()
        self.assertTrue(
            torch.allclose(causal(x)[:, :-1], causal(perturbed)[:, :-1], atol=1.0e-6)
        )

        bidirectional = _TransformerTemporalCore(
            lookback, 64, 4, 1, 128, 0.0, causal_mask=False
        ).eval()
        bidirectional_change = (
            (bidirectional(x)[:, :-1] - bidirectional(perturbed)[:, :-1])
            .abs()
            .max()
            .item()
        )
        self.assertGreater(bidirectional_change, 1.0e-4)


if __name__ == "__main__":
    unittest.main()
