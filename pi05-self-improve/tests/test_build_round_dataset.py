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

"""End-to-end dataset builder test — requires the pixi lerobot env
(numpy + lerobot with video encoding).  Skipped automatically elsewhere."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from lerobot.datasets import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:  # pragma: no cover - outside the pixi env
    HAS_LEROBOT = False

from pi05_selfimprove import contract

if HAS_LEROBOT:
    from pi05_selfimprove.dataset_tools.build_round_dataset import (
        build_round_dataset,
        rollout_features,
        select_sim_successes,
    )

HEIGHT, WIDTH = 64, 48  # encoder-friendly small frames
FRAMES_PER_EPISODE = 4


def _write_npz_episode(round_dir: Path, index: int, success: bool) -> None:
    images = {
        camera: np.full(
            (FRAMES_PER_EPISODE, HEIGHT, WIDTH, 3), index * 20 % 255,
            dtype=np.uint8)
        for camera in contract.CAMERA_KEYS
    }
    path = round_dir / "rollouts_raw" / f"episode_{index:06d}.npz"
    contract.save_episode_npz(
        path,
        images=images,
        state=np.linspace(-0.5, 0.5, FRAMES_PER_EPISODE * 6).reshape(
            FRAMES_PER_EPISODE, 6).astype(np.float32),
        action=np.linspace(0.5, -0.5, FRAMES_PER_EPISODE * 6).reshape(
            FRAMES_PER_EPISODE, 6).astype(np.float32),
        task="pick up the cube and place it in the cup",
        success=success,
        reason="success" if success else "timeout",
        source="sim",
        joint_map=contract.default_joint_map(),
    )
    contract.append_jsonl(round_dir / contract.ROLL_FILENAME, {
        "episode_index": index,
        "file": f"rollouts_raw/episode_{index:06d}.npz",
        "source": "sim",
        "task": "pick up the cube and place it in the cup",
        "success": success,
        "reason": "success" if success else "timeout",
        "steps": FRAMES_PER_EPISODE,
        "fps": contract.CONTROL_FREQUENCY_HZ,
    })


def _make_teleop_dataset(root: Path, repo_id: str, episodes: int) -> None:
    from lerobot.configs import RGBEncoderConfig

    features = rollout_features(HEIGHT, WIDTH)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        fps=int(contract.CONTROL_FREQUENCY_HZ),
        robot_type="so101",
        features=features,
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
    )
    for episode in range(episodes):
        for frame in range(FRAMES_PER_EPISODE):
            payload = {
                contract.STATE_FEATURE_KEY: np.full(
                    6, 0.1 * episode, dtype=np.float32),
                contract.ACTION_FEATURE_KEY: np.full(
                    6, -0.1 * episode, dtype=np.float32),
                "task": "pick up the cube and place it in the cup",
            }
            for camera in contract.CAMERA_KEYS:
                payload[contract.camera_feature_key(camera)] = np.full(
                    (HEIGHT, WIDTH, 3), 40 + 10 * episode, dtype=np.uint8)
            dataset.add_frame(payload)
        dataset.save_episode()
    dataset.finalize()


@unittest.skipUnless(HAS_LEROBOT, "lerobot not importable")
class BuildRoundDatasetTests(unittest.TestCase):
    def test_filter_and_mix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "rounds" / "round_1"
            _write_npz_episode(round_dir, 0, success=True)
            _write_npz_episode(round_dir, 1, success=False)
            _write_npz_episode(round_dir, 2, success=True)

            successes = select_sim_successes(round_dir)
            self.assertEqual(len(successes), 2)

            _make_teleop_dataset(root / "teleop", "local/test_teleop",
                                 episodes=2)

            summary = build_round_dataset(
                repo_id="local/test_round1_mixed",
                round_dir=round_dir,
                teleop_repo_ids=["local/test_teleop"],
                min_successes=2,
                root=root / "mixed",
                dataset_roots={"local/test_teleop": root / "teleop"},
                vcodec="h264",
                image_height=HEIGHT,
                image_width=WIDTH,
            )
            self.assertEqual(
                summary["sources"]["sim_rollouts"]["frames"],
                2 * FRAMES_PER_EPISODE)
            self.assertEqual(
                summary["sources"]["teleop"]["frames"],
                2 * FRAMES_PER_EPISODE)
            self.assertEqual(summary["total_frames"], 4 * FRAMES_PER_EPISODE)

            mixed = LeRobotDataset("local/test_round1_mixed",
                                   root=root / "mixed",
                                   return_uint8=True)
            self.assertEqual(len(mixed), 4 * FRAMES_PER_EPISODE)
            self.assertEqual(len(mixed.meta.episodes), 4)
            item = mixed[0]
            self.assertEqual(item[contract.STATE_FEATURE_KEY].shape, (6,))
            wrist_key = contract.camera_feature_key("wrist")
            self.assertEqual(tuple(item[wrist_key].shape[-2:]), (HEIGHT, WIDTH))

    def test_min_successes_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "rounds" / "round_2"
            _write_npz_episode(round_dir, 0, success=False)
            with self.assertRaises(RuntimeError):
                build_round_dataset(
                    repo_id="local/test_round2_mixed",
                    round_dir=round_dir,
                    teleop_repo_ids=[],
                    min_successes=1,
                    root=root / "round2_mixed",
                )

    def test_overwrite_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            round_dir = root / "rounds" / "round_3"
            _write_npz_episode(round_dir, 0, success=True)
            build_round_dataset(
                repo_id="local/test_round3_mixed",
                round_dir=round_dir,
                teleop_repo_ids=[],
                min_successes=1,
                root=root / "round3_mixed",
                vcodec="h264",
                image_height=HEIGHT,
                image_width=WIDTH,
            )
            with self.assertRaises(RuntimeError):
                build_round_dataset(
                    repo_id="local/test_round3_mixed",
                    round_dir=round_dir,
                    teleop_repo_ids=[],
                    min_successes=1,
                    root=root / "round3_mixed",
                )


if __name__ == "__main__":
    unittest.main()
