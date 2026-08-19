"""Focused deployable visual actor and export contract tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict

from so101_rl.camera_profile import (
    camera_profile_sha256,
    load_camera_profile,
    look_at_opengl_xyzw,
    wxyz_to_xyzw,
)
from models.ppo.models import SpatialSoftmaxCNNModel
from so101_rl.export_manifest import build_policy_manifest
from so101_rl.tasks.common.agents.rsl_rl_ppo_cfg import SO101VisualPPOCfg
from so101_rl.visual_contract import SO101_ACTOR_OBSERVATION_GROUPS


class VisualPolicyContractTests(unittest.TestCase):
    def test_yaml_quaternion_and_overhead_optical_axis_contract(self):
        profile = load_camera_profile()
        wrist_wxyz = profile["cameras"]["wrist"]["orientation_wxyz"]
        self.assertEqual(
            wxyz_to_xyzw(wrist_wxyz),
            (wrist_wxyz[1], wrist_wxyz[2], wrist_wxyz[3], wrist_wxyz[0]),
        )
        overhead = profile["cameras"]["overhead_1"]
        quaternion = look_at_opengl_xyzw(
            overhead["position_m"], overhead["look_at_m"]
        )
        self.assertAlmostEqual(sum(value * value for value in quaternion), 1.0, places=6)
        self.assertEqual(len(camera_profile_sha256()), 64)

    def test_three_encoder_actor_and_torchscript_contract(self):
        observations = TensorDict(
            {
                "joint_state": torch.zeros(2, 6),
                "wrist": torch.zeros(2, 3, 120, 160),
                "overhead_1": torch.zeros(2, 3, 120, 160),
                "overhead_2": torch.zeros(2, 3, 120, 160),
                "critic_state": torch.zeros(2, 34),
            },
            batch_size=[2],
        )
        observation_groups = {
            "actor": list(SO101_ACTOR_OBSERVATION_GROUPS),
            "critic": ["critic_state"],
        }
        cnn_cfg = {
            "output_channels": [16, 32, 32],
            "kernel_size": [8, 4, 3],
            "stride": [4, 2, 1],
            "activation": "elu",
        }
        model = SpatialSoftmaxCNNModel(
            observations,
            observation_groups,
            "actor",
            output_dim=6,
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.7},
            cnn_cfg=cnn_cfg,
        )
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

        critic_cfg = SO101VisualPPOCfg().critic
        critic = MLPModel(
            observations,
            observation_groups,
            "critic",
            output_dim=1,
            hidden_dims=critic_cfg.hidden_dims,
            activation=critic_cfg.activation,
            obs_normalization=critic_cfg.obs_normalization,
            distribution_cfg=critic_cfg.distribution_cfg,
        )
        values = critic(observations)
        self.assertEqual(tuple(values.shape), (2, 1))
        self.assertIsNone(critic.distribution)

    def test_manifest_contains_only_real_actor_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            policy = directory / "policy.pt"
            onnx = directory / "policy.onnx"
            policy.write_bytes(b"torchscript")
            onnx.write_bytes(b"onnx")
            limits = [[-2.0, 2.0] for _ in range(6)]
            manifest = build_policy_manifest(
                "SO101-Object-In-Cup-Vision-v0", limits, policy, onnx
            )
        actor = manifest["actor_observations"]
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(
            [camera["key"] for camera in actor["cameras"]],
            [
                "observation.images.wrist",
                "observation.images.overhead_1",
                "observation.images.overhead_2",
            ],
        )
        self.assertEqual(actor["joint_state"]["key"], "observation.state")
        self.assertNotIn("critic", str(actor).lower())
        self.assertNotIn("object", str(actor).lower())
        self.assertEqual(manifest["policy"]["frequency_hz"], 30.0)
        self.assertEqual(
            manifest["deployment"]["controller_command"]["representation"],
            "absolute_joint_position",
        )
        self.assertEqual(
            manifest["deployment"]["controller_command"]["units"], "rad"
        )


if __name__ == "__main__":
    unittest.main()
