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

"""Tests for the training command assembly and round-loop bookkeeping."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve.config import RoundConfig
from self_improve.train_bc import (
    build_train_command,
    find_latest_checkpoint,
    init_weights_source,
    training_dataset_repo_id,
)


class TrainCommandTests(unittest.TestCase):
    def test_round_0_from_base_on_teleop(self) -> None:
        config = RoundConfig(round_index=0)
        command = build_train_command(config, Path("/tmp/run"))
        joined = " ".join(command)
        self.assertIn("--policy.path=lerobot/pi05_base", joined)
        self.assertIn(
            "--dataset.repo_id=algorithmtheworld/so101-pick-and-place",
            joined)
        self.assertIn("--policy.freeze_vision_encoder=true", joined)
        self.assertIn("--policy.push_to_hub=false", joined)
        self.assertIn("--output_dir=/tmp/run", joined)

    def test_round_n_from_previous_checkpoint_on_mixed(self) -> None:
        config = RoundConfig(round_index=2,
                             checkpoint_in="rounds/round_1/ckpt")
        config.train.init_from = "previous"
        command = build_train_command(config, Path("/tmp/run2"))
        joined = " ".join(command)
        self.assertIn("--policy.path=rounds/round_1/ckpt", joined)
        self.assertIn("--dataset.repo_id=local/so101_self_improve_round2_mixed",
                      joined)

    def test_round_n_from_base_on_mixed_with_rename_map(self) -> None:
        config = RoundConfig(round_index=1)
        config.train.init_from = "base"
        config.rollout.image_key_map = {
            "observation.images.wrist":
                "observation.images.left_wrist_0_rgb",
            "observation.images.overhead_1":
                "observation.images.base_0_rgb",
        }
        command = build_train_command(config, Path("/tmp/run1"))
        joined = " ".join(command)
        self.assertIn("--policy.path=lerobot/pi05_base", joined)
        self.assertIn("--dataset.repo_id=local/so101_self_improve_round1_mixed",
                      joined)
        self.assertIn(
            '--rename_map={"observation.images.overhead_1":'
            '"observation.images.base_0_rgb","observation.images.wrist":'
            '"observation.images.left_wrist_0_rgb"}',
            command,
        )

    def test_round_n_requires_checkpoint(self) -> None:
        config = RoundConfig(round_index=3, checkpoint_in=None)
        config.train.init_from = "previous"
        with self.assertRaises(ValueError):
            init_weights_source(config)

    def test_unknown_init_source_is_rejected(self) -> None:
        config = RoundConfig(round_index=1)
        config.train.init_from = "latest"
        with self.assertRaisesRegex(ValueError, "expected 'base' or 'previous'"):
            init_weights_source(config)

    def test_extra_args_passthrough(self) -> None:
        config = RoundConfig(round_index=0)
        config.train.extra_args = ["--policy.num_inference_steps=4"]
        command = build_train_command(config, Path("/tmp/run"))
        self.assertIn("--policy.num_inference_steps=4", command)

    def test_find_latest_checkpoint(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run" / "checkpoints"
            root.mkdir(parents=True)
            self.assertIsNone(find_latest_checkpoint(root.parent))
            for step in (5000, 15000, 10000):
                (root / str(step)).mkdir()
                (root / str(step) / "model.safetensors").write_bytes(b"x")
            latest = find_latest_checkpoint(root.parent)
            self.assertEqual(latest.name, "15000")


class RealSessionCommandTests(unittest.TestCase):
    def test_configured_policy_and_chunk_size_reach_launch_command(self) -> None:
        from self_improve.real_rollout import session_commands

        config = RoundConfig()
        config.train.policy_type = "smolvla"
        config.train.base_repo_id = "lerobot/smolvla_base"
        config.rollout.actions_per_chunk = 8

        commands = session_commands(
            config,
            setup="monomanual_dual_overhead",
            experiment="self_improve",
            policy_server_address="127.0.0.1:8090",
        )

        inference_command = commands[1]
        self.assertIn("policy_type:=smolvla", inference_command)
        self.assertIn("actions_per_chunk:=8", inference_command)


class LoopImportTests(unittest.TestCase):
    def test_stage_validation(self) -> None:
        from self_improve.loop import ALL_STAGES, DEFAULT_STAGES

        self.assertEqual(DEFAULT_STAGES, ("rollout", "build", "train", "eval"))
        self.assertIn("judge", ALL_STAGES)

    def test_missing_previous_checkpoint_fails_before_server_start(self) -> None:
        from self_improve.loop import run_round

        config = RoundConfig(round_index=2)
        config.train.init_from = "previous"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "self_improve.loop.PolicyServerProcess") as server:
            with self.assertRaisesRegex(ValueError, "has no checkpoint_in"):
                run_round(config, Path(tmp), ["rollout"])
        server.assert_not_called()

    def test_visualizer_is_forwarded_to_sim_rollout(self) -> None:
        from self_improve.loop import _run_sim_rollout

        config = RoundConfig(round_index=1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "self_improve.loop.subprocess.run") as subprocess_run:
            _run_sim_rollout(config, Path(tmp), eval_mode=False,
                             visualizer="kit", device="cuda:0")

        command = subprocess_run.call_args.args[0]
        self.assertIn("--device", command)
        self.assertEqual(command[command.index("--device") + 1], "cuda:0")
        self.assertEqual(command[-2:], ["--visualizer", "kit"])
        subprocess_run.assert_called_once_with(command, check=True)

    def test_run_round_routes_policy_and_sim_to_separate_devices(self) -> None:
        from self_improve import loop

        config = RoundConfig(round_index=1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                loop, "PolicyServerProcess") as policy_server, mock.patch.object(
                loop, "_run_sim_rollout") as sim_rollout:
            loop.run_round(
                config, Path(tmp), ["rollout"],
                policy_device="cuda:1", sim_device="cuda:0",
            )

        policy_server.assert_called_once_with(
            "lerobot/pi05_base",
            host=config.rollout.server_host,
            port=config.rollout.server_port,
            policy_type=config.train.policy_type,
            actions_per_chunk=config.rollout.actions_per_chunk,
            device="cuda:1",
            dtype="bfloat16",
        )
        sim_rollout.assert_called_once_with(
            config, Path(tmp), eval_mode=False,
            visualizer=None, device="cuda:0", num_envs=None,
        )

    def test_real_phase_rejects_sim_stages(self) -> None:
        from self_improve.loop import run_round

        config = RoundConfig(round_index=2, phase="real",
                             checkpoint_in="rounds/round_1/ckpt")
        config.train.init_from = "previous"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "self_improve.loop.PolicyServerProcess") as server:
            for stages in (["rollout"], ["eval"], ["rollout", "build"]):
                with self.assertRaisesRegex(ValueError, "phase=real"):
                    run_round(config, Path(tmp), stages)
        server.assert_not_called()

    def test_real_phase_allows_refine_stages(self) -> None:
        from self_improve import loop

        config = RoundConfig(round_index=2, phase="real",
                             checkpoint_in="rounds/round_1/ckpt")
        config.train.init_from = "previous"
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "self_improve.dataset_tools.build_round_dataset"
                ".build_from_config",
                return_value={"repo_id": "local/x"}) as build, mock.patch(
                "self_improve.train_bc.run_training", return_value={}):
            loop.run_round(config, Path(tmp), ["build", "train"])
        build.assert_called_once()

    def test_build_stage_receives_real_dataset_repo_id(self) -> None:
        from self_improve import loop

        config = RoundConfig(round_index=2, phase="real",
                             checkpoint_in="rounds/round_1/ckpt")
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
                "self_improve.dataset_tools.build_round_dataset"
                ".build_from_config",
                return_value={"repo_id": "local/x"}) as build:
            loop.run_round(config, Path(tmp), ["build"],
                           real_dataset_repo_id=(
                               "local/so101_self_improve_round2_real"))
        build.assert_called_once_with(
            config,
            Path(tmp),
            real_dataset_repo_id="local/so101_self_improve_round2_real",
        )


if __name__ == "__main__":
    unittest.main()
