"""Recurrent state-space actor contract tests."""

from __future__ import annotations

import unittest

import torch
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from models.mamba_ppo.models import MambaActorCritic

IMAGE_SHAPE = (3, 120, 160)


def _observation_groups() -> dict[str, list[str]]:
    return {
        "actor": ["joint_state", "wrist", "overhead_1", "overhead_2"],
        "critic": ["critic_state"],
    }


def _single_frame_observations(batch: int = 2, image_shape=IMAGE_SHAPE) -> TensorDict:
    return TensorDict(
        {
            "joint_state": torch.randn(batch, 6),
            "wrist": torch.randn(batch, *image_shape),
            "overhead_1": torch.randn(batch, *image_shape),
            "overhead_2": torch.randn(batch, *image_shape),
            "critic_state": torch.zeros(batch, 34),
        },
        batch_size=[batch],
    )


def _sequence_observations(steps: int = 5, batch: int = 2, image_shape=IMAGE_SHAPE) -> TensorDict:
    return TensorDict(
        {
            "joint_state": torch.randn(steps, batch, 6),
            "wrist": torch.randn(steps, batch, *image_shape),
            "overhead_1": torch.randn(steps, batch, *image_shape),
            "overhead_2": torch.randn(steps, batch, *image_shape),
            "critic_state": torch.zeros(steps, batch, 34),
        },
        batch_size=[steps, batch],
    )


def _build_model(obs: TensorDict, **overrides) -> MambaActorCritic:
    defaults = dict(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7},
        d_model=64,
        num_layers=2,
    )
    defaults.update(overrides)
    return MambaActorCritic(obs, _observation_groups(), "actor", output_dim=6, **defaults)


class MambaRecurrentActorTests(unittest.TestCase):
    def test_forward_contract_and_hidden_state(self):
        model = _build_model(_single_frame_observations())
        model.eval()
        self.assertTrue(model.is_recurrent)
        self.assertIsNone(model.get_hidden_state())

        observations = _single_frame_observations()
        actions = model(observations)
        self.assertEqual(tuple(actions.shape), (2, 6))

        state = model.get_hidden_state()
        self.assertIsNotNone(state)
        self.assertEqual(tuple(state.shape), (1, 2, model._state_dim))

        latent = model.get_latent(observations)
        # 6 normalized joint positions plus one d_model embedding per camera.
        self.assertEqual(tuple(latent.shape), (2, 6 + 3 * 64))

    def test_frame_history_windows_are_rejected(self):
        windowed = TensorDict(
            {
                "joint_state": torch.zeros(2, 6),
                "wrist": torch.zeros(2, 4, 3, 120, 160),
            },
            batch_size=[2],
        )
        with self.assertRaisesRegex(ValueError, "single-frame"):
            _build_model(windowed)

    def test_stepping_matches_sequence_scan(self):
        torch.manual_seed(7)
        sequence = _sequence_observations()
        model = _build_model(
            _single_frame_observations(), obs_normalization=False
        )
        model.eval()

        def frame(t: int) -> TensorDict:
            return TensorDict(
                {key: value[t] for key, value in sequence.items()}, batch_size=[2]
            )

        model.reset()
        stepped_actions = torch.stack(
            [model(frame(t)) for t in range(sequence.batch_size[0])], dim=0
        )
        scanned_actions = model(
            sequence,
            masks=torch.ones(*sequence.batch_size, dtype=torch.bool),
            hidden_state=None,
        )
        max_error = (stepped_actions - scanned_actions).abs().max().item()
        self.assertLess(max_error, 1.0e-4)

    def test_gradients_flow_through_the_full_scan(self):
        model = _build_model(_single_frame_observations(), hidden_dims=[128])
        sequence = _sequence_observations()
        actions = model(
            sequence,
            masks=torch.ones(*sequence.batch_size, dtype=torch.bool),
            hidden_state=None,
        )
        actions.pow(2).mean().backward()
        recurrent_params = [
            p for name, p in model.named_parameters() if "pipelines" in name
        ]
        self.assertTrue(recurrent_params)
        self.assertTrue(all(p.grad is not None for p in recurrent_params))

    def test_reset_zeros_done_environments(self):
        model = _build_model(_single_frame_observations(), obs_normalization=False)
        model(_single_frame_observations())
        self.assertGreater(model.get_hidden_state()[0, 0].abs().sum().item(), 0.0)
        model.reset(dones=torch.tensor([1, 0]))
        state = model.get_hidden_state()
        self.assertEqual(state[0, 0].abs().sum().item(), 0.0)
        self.assertGreater(state[0, 1].abs().sum().item(), 0.0)

        model.reset()
        self.assertIsNone(model.get_hidden_state())

    def test_recurrent_storage_round_trip(self):
        torch.manual_seed(3)
        num_envs, num_steps = 4, 8
        image_shape = (3, 32, 48)
        model = _build_model(
            _single_frame_observations(image_shape=image_shape),
            hidden_dims=[128],
            obs_normalization=False,
            d_model=32,
        )
        observations = _single_frame_observations(num_envs, image_shape)
        storage = RolloutStorage(
            "rl", num_envs, num_steps, observations, [6], torch.device("cpu")
        )

        dones = torch.zeros(num_envs, 1, dtype=torch.long)
        dones[1, 0] = 1  # env 1 terminates mid-rollout

        class _Transition:
            pass

        for _ in range(num_steps):
            step_obs = _single_frame_observations(num_envs, image_shape)
            with torch.inference_mode():
                hidden_state = model.get_hidden_state()
                actions = model(step_obs, stochastic_output=True)
                log_prob = model.get_output_log_prob(actions)
            transition = _Transition()
            transition.hidden_states = (hidden_state, None)
            transition.observations = step_obs
            transition.actions = actions
            transition.values = torch.zeros(num_envs, 1)
            transition.actions_log_prob = log_prob
            transition.distribution_params = tuple(
                p for p in model.output_distribution_params
            )
            transition.rewards = torch.randn(num_envs, 1)
            transition.dones = dones.clone()
            storage.add_transition(transition)
            # Reset outside inference mode, like play-time callers do.
            model.reset(dones.squeeze(-1))

        storage.returns = torch.zeros(num_steps, num_envs, 1)
        storage.advantages = torch.zeros(num_steps, num_envs, 1)
        minibatches = 0
        for batch in storage.recurrent_mini_batch_generator(
            num_mini_batches=2, num_epochs=1
        ):
            hidden_state = batch.hidden_states[0]
            self.assertIsNotNone(hidden_state)
            self.assertEqual(hidden_state.shape[0], 1)
            self.assertEqual(hidden_state.shape[2], model._state_dim)

            actions = model(
                batch.observations,
                masks=batch.masks,
                hidden_state=hidden_state,
                stochastic_output=True,
            )
            self.assertEqual(tuple(actions.shape), tuple(batch.actions.shape))
            log_prob = model.get_output_log_prob(batch.actions)
            self.assertEqual(
                tuple(log_prob.shape),
                tuple(batch.old_actions_log_prob.squeeze(-1).shape),
            )
            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            (ratio.mean()).backward()
            model.zero_grad()
            minibatches += 1
        self.assertEqual(minibatches, 2)

    def test_torchscript_export_matches_stepwise_rollout(self):
        torch.manual_seed(11)
        sequence = _sequence_observations()
        model = _build_model(
            _single_frame_observations(), obs_normalization=False
        )
        model.eval()

        def frame(t: int) -> TensorDict:
            return TensorDict(
                {key: value[t] for key, value in sequence.items()}, batch_size=[2]
            )

        model.reset()
        stepped_actions = torch.stack(
            [model(frame(t)) for t in range(sequence.batch_size[0])], dim=0
        )
        scripted = torch.jit.script(model.as_jit())
        scripted.reset()
        scripted_actions = torch.stack(
            [
                scripted(
                    obs["joint_state"],
                    [obs["wrist"], obs["overhead_1"], obs["overhead_2"]],
                )
                for obs in (frame(t) for t in range(sequence.batch_size[0]))
            ],
            dim=0,
        )
        max_error = (stepped_actions - scripted_actions).abs().max().item()
        self.assertLess(max_error, 1.0e-5)

    def test_onnx_wrapper_step_parity(self):
        model = _build_model(_single_frame_observations(), obs_normalization=False)
        model.eval()
        observations = _single_frame_observations()
        model.reset()
        expected = model(observations)
        onnx_actor = model.as_onnx()
        deltas, new_state = onnx_actor(
            observations["joint_state"],
            torch.zeros(1, 2, model._state_dim),
            observations["wrist"],
            observations["overhead_1"],
            observations["overhead_2"],
        )
        self.assertEqual(tuple(deltas.shape), (2, 6))
        self.assertEqual(tuple(new_state.shape), (1, 2, model._state_dim))
        self.assertLess((deltas - expected).abs().max().item(), 1.0e-5)


if __name__ == "__main__":
    unittest.main()
