# Copyright 2026 Dmitri Manajev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout server zero-shot adaptation tests (stub policy, no weights).

Requires torch (pixi lerobot env); skipped elsewhere.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover - outside the pixi env
    HAS_TORCH = False

from pi05_selfimprove import contract

if HAS_TORCH:
    from pi05_selfimprove.rollout_server import RolloutPolicyServer

RIG_FEATURES = {
    contract.STATE_FEATURE_KEY: {"dtype": "float32", "shape": (6,)},
    **{contract.camera_feature_key(camera):
       {"dtype": "image",
        "shape": (contract.IMAGE_HEIGHT, contract.IMAGE_WIDTH, 3)}
       for camera in contract.CAMERA_KEYS},
}

BASE_IMAGE_KEY_MAP = {
    "observation.images.wrist": "observation.images.left_wrist_0_rgb",
    "observation.images.overhead_1": "observation.images.base_0_rgb",
    "observation.images.overhead_2": "observation.images.right_wrist_0_rgb",
}


def _feature(shape):
    return SimpleNamespace(shape=tuple(shape))


def _pi05_base_stub_policy():
    """Policy config namespace mimicking lerobot/pi05_base's schema."""
    return SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.images.base_0_rgb": _feature((3, 224, 224)),
                "observation.images.left_wrist_0_rgb": _feature((3, 224, 224)),
                "observation.images.right_wrist_0_rgb": _feature((3, 224, 224)),
                contract.STATE_FEATURE_KEY: _feature((32,)),
            },
            output_features={contract.ACTION_FEATURE_KEY: _feature((32,))},
        ),
    )


@unittest.skipUnless(HAS_TORCH, "torch not importable")
class ZeroShotAdaptationTests(unittest.TestCase):
    def _server(self) -> RolloutPolicyServer:
        server = RolloutPolicyServer("stub")
        server.policy = _pi05_base_stub_policy()
        return server

    def test_hello_with_remap_pads_and_slices(self) -> None:
        server = self._server()
        reply = server._handle_hello({
            "features": RIG_FEATURES,
            "actions_per_chunk": 4,
            "image_key_map": BASE_IMAGE_KEY_MAP,
            "action_dim": 6,
        })
        self.assertEqual(reply["type"], "welcome")
        self.assertEqual(reply["action_dim"], 6)
        self.assertEqual(server._policy_state_dim, 32)

    def test_hello_without_remap_rejected_with_hint(self) -> None:
        server = self._server()
        reply = server._handle_hello({
            "features": RIG_FEATURES,
            "actions_per_chunk": 4,
        })
        self.assertEqual(reply["type"], "error")
        self.assertIn("image_key_map", reply["message"])

    def test_observation_remapped_and_padded(self) -> None:
        server = self._server()
        server._handle_hello({
            "features": RIG_FEATURES,
            "actions_per_chunk": 4,
            "image_key_map": BASE_IMAGE_KEY_MAP,
            "action_dim": 6,
        })
        images = {
            contract.camera_feature_key("wrist"):
                __import__("numpy").zeros((2, 4, 4, 3), dtype="uint8"),
        }
        observation = server._observation(
            images, __import__("numpy").zeros((2, 6)), "task", 2)
        self.assertIn("observation.images.left_wrist_0_rgb", observation)
        self.assertNotIn(contract.camera_feature_key("wrist"), observation)
        self.assertEqual(tuple(observation[contract.STATE_FEATURE_KEY].shape),
                         (2, 32))
        self.assertTrue(torch.all(
            observation[contract.STATE_FEATURE_KEY][:, 6:] == 0))

    def test_run_policy_slices_action_padding(self) -> None:
        import numpy as np

        server = self._server()
        server._handle_hello({
            "features": RIG_FEATURES,
            "actions_per_chunk": 4,
            "image_key_map": BASE_IMAGE_KEY_MAP,
            "action_dim": 6,
        })
        server.preprocessor = lambda observation: observation
        server.postprocessor = lambda tensor: tensor
        server.policy = SimpleNamespace(
            config=server.policy.config,
            predict_action_chunk=(
                lambda observation: torch.zeros(1, 4, 32)),
        )
        actions = server._run_policy({})  # stubs ignore the observation
        self.assertEqual(actions.shape, (1, 4, 6))

    def test_identity_map_for_finetuned_checkpoint(self) -> None:
        server = RolloutPolicyServer("stub")
        server.policy = SimpleNamespace(
            config=SimpleNamespace(
                input_features={
                    **{contract.camera_feature_key(camera):
                       _feature((3, 224, 224)) for camera in contract.CAMERA_KEYS},
                    contract.STATE_FEATURE_KEY: _feature((6,)),
                },
                output_features={contract.ACTION_FEATURE_KEY: _feature((6,))},
            ),
        )
        reply = server._handle_hello({
            "features": RIG_FEATURES,
            "actions_per_chunk": 4,
        })
        self.assertEqual(reply["type"], "welcome")
        self.assertEqual(reply["action_dim"], 6)


if __name__ == "__main__":
    unittest.main()
