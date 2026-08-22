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

"""RoundConfig YAML round-trip and derived-path tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve.config import RoundConfig


class RoundConfigTests(unittest.TestCase):
    def test_yaml_roundtrip(self) -> None:
        config = RoundConfig(
            round_index=3,
            phase="sim",
            checkpoint_in="rounds/round_2/checkpoint",
        )
        config.rollout.num_envs = 4
        config.train.steps = 8_000
        config.dataset.teleop_repo_ids = ["local/demo_a", "local/demo_b"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_3" / "config.yaml"
            config.save(path)
            loaded = RoundConfig.load(path)
        self.assertEqual(loaded.round_index, 3)
        self.assertEqual(loaded.rollout.num_envs, 4)
        self.assertEqual(loaded.train.steps, 8_000)
        self.assertEqual(loaded.dataset.teleop_repo_ids,
                         ["local/demo_a", "local/demo_b"])
        self.assertEqual(loaded.checkpoint_in, "rounds/round_2/checkpoint")

    def test_defaults(self) -> None:
        config = RoundConfig()
        self.assertEqual(config.phase, "sim")
        self.assertEqual(config.judge.name, "scripted")
        self.assertEqual(
            config.mixed_repo_id(),
            "local/so101_self_improve_round0_mixed",
        )
        self.assertEqual(config.round_name, "round_0")

    def test_round_dirs(self) -> None:
        config = RoundConfig(round_index=5)
        root = Path("/tmp/rounds")
        self.assertEqual(config.round_dir(root), root / "round_5")
        self.assertEqual(config.rollouts_raw_dir(root),
                         root / "round_5" / "rollouts_raw")
        self.assertEqual(config.checkpoint_dir(root),
                         root / "round_5" / "checkpoint")

    def test_joint_map_resolution(self) -> None:
        from self_improve import contract

        config = RoundConfig()
        default = config.resolve_joint_map()
        np.testing.assert_allclose(
            default.scale, contract.default_joint_map().scale)

        config.rollout.joint_map = {
            "scale": [1.0] * 6,
            "offset": [0.0] * 6,
        }
        identity = config.resolve_joint_map()
        np.testing.assert_allclose(identity.offset, 0.0)

if __name__ == "__main__":
    unittest.main()
