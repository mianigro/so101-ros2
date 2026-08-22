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

"""Tests for the pure parts of the sim rollout loop (no Isaac Sim needed)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve import contract, sim_rollout


class EnvActionBufferTests(unittest.TestCase):
    def test_submit_pop_in_order(self) -> None:
        buffer = sim_rollout.EnvActionBuffer(
            sim_rollout.AGGREGATE_FUNCTIONS["latest_only"])
        chunk = np.arange(12, dtype=float).reshape(4, 3)
        buffer.submit(chunk, start_timestep=10)
        self.assertEqual(buffer.pending_from(10), 4)
        for index in range(4):
            np.testing.assert_allclose(buffer.pop(10 + index), chunk[index])
        self.assertIsNone(buffer.pop(14))

    def test_aggregation_merges_overlapping_timesteps(self) -> None:
        buffer = sim_rollout.EnvActionBuffer(
            sim_rollout.AGGREGATE_FUNCTIONS["weighted_average"])
        first = np.zeros((2, 6)) + 1.0
        second = np.zeros((2, 6)) + 2.0
        buffer.submit(first, 0)
        buffer.submit(second, 1)  # overlaps at timestep 1
        np.testing.assert_allclose(buffer.pop(0), first[0])
        merged = buffer.pop(1)
        np.testing.assert_allclose(merged, 0.3 * 1.0 + 0.7 * 2.0)
        np.testing.assert_allclose(buffer.pop(2), second[1])

    def test_past_timesteps_dropped(self) -> None:
        buffer = sim_rollout.EnvActionBuffer(
            sim_rollout.AGGREGATE_FUNCTIONS["latest_only"])
        buffer.submit(np.zeros((2, 6)), 0)
        buffer.pop(0)
        buffer.submit(np.ones((2, 6)), 0)  # starts in the past
        self.assertIsNone(buffer.pop(0))
        np.testing.assert_allclose(buffer.pop(1), np.ones(6))

    def test_reset_clears(self) -> None:
        buffer = sim_rollout.EnvActionBuffer(
            sim_rollout.AGGREGATE_FUNCTIONS["latest_only"])
        buffer.submit(np.zeros((2, 6)), 0)
        buffer.pop(0)
        buffer.last_action = np.ones(6)
        buffer.reset()
        self.assertEqual(buffer.pending_from(0), 0)
        self.assertIsNone(buffer.last_action)
        # After reset the timestep counter restarts; a chunk at t=0 is
        # accepted again because _latest_executed was cleared.
        buffer.submit(np.zeros((1, 6)), 0)
        self.assertIsNotNone(buffer.pop(0))


class TerminationClassificationTests(unittest.TestCase):
    def _terms(self, *, success=0, dropped=0, invalid=0, time_out=0):
        return {
            "dones": np.array([True]),
            sim_rollout.SUCCESS_TERM: np.array([success], dtype=bool),
            sim_rollout.TIMEOUT_TERM: np.array([time_out], dtype=bool),
            "dropped": np.array([dropped], dtype=bool),
            "invalid": np.array([invalid], dtype=bool),
        }

    def test_success(self) -> None:
        self.assertEqual(
            sim_rollout.classify(self._terms(success=1), 0), (True, "success"))

    def test_failure_precedence(self) -> None:
        self.assertEqual(
            sim_rollout.classify(self._terms(dropped=1), 0), (False, "dropped"))
        self.assertEqual(
            sim_rollout.classify(self._terms(invalid=1), 0), (False, "invalid"))
        self.assertEqual(
            sim_rollout.classify(self._terms(time_out=1), 0), (False, "timeout"))
        self.assertEqual(
            sim_rollout.classify(self._terms(), 0), (False, "unknown"))


class EpisodeRecorderTests(unittest.TestCase):
    def test_record_save_and_cleanup(self) -> None:
        steps = 4
        height, width = 8, 12
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recorder = sim_rollout.EpisodeRecorder(
                root / "rollouts_raw" / "tmp", env_id=2, episode_index=7,
                max_steps=steps, image_height=height, image_width=width,
            )
            recorder.start()
            for step in range(steps):
                images = {
                    camera: np.full((1, height, width, 3), step * 10,
                                    dtype=np.uint8)
                    for camera in contract.CAMERA_KEYS
                }
                recorder.add(
                    images,
                    state=np.full(6, step, dtype=np.float32),
                    action=np.full(6, -step, dtype=np.float32),
                    sim_state=np.full(6, step + 0.5, dtype=np.float32),
                    sim_action=np.full(6, -step - 0.5, dtype=np.float32),
                )
            path = recorder.save(
                root / "rollouts_raw",
                task="pick up the cube",
                success=True,
                reason="success",
                source="sim",
                joint_map=contract.default_joint_map(),
            )
            recorder.discard()
            self.assertEqual(path.name, "episode_000007.npz")
            self.assertFalse(path.parent.joinpath("tmp").exists())
            with np.load(path) as data:
                self.assertEqual(data[contract.NPZ_STATE].shape, (steps, 6))
                self.assertEqual(data[contract.NPZ_ACTION][3, 0], -3)
                wrist_key = contract.camera_feature_key("wrist")
                self.assertEqual(data[wrist_key].shape, (steps, height, width, 3))
                self.assertEqual(data[wrist_key][2, 0, 0, 0], 20)
                self.assertTrue(bool(data[contract.NPZ_SUCCESS]))

    def test_max_steps_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = sim_rollout.EpisodeRecorder(
                Path(tmp), env_id=0, episode_index=0, max_steps=1,
                image_height=2, image_width=3)
            recorder.start()
            frames = {camera: np.zeros((1, 2, 3, 3), dtype=np.uint8)
                      for camera in contract.CAMERA_KEYS}
            recorder.add(frames, np.zeros(6), np.zeros(6),
                         np.zeros(6), np.zeros(6))
            with self.assertRaises(IndexError):
                recorder.add(frames, np.zeros(6), np.zeros(6),
                             np.zeros(6), np.zeros(6))


if __name__ == "__main__":
    unittest.main()
