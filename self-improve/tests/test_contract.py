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

"""Contract tests: joint map math, npz episode format, and cross-checks
against so101_rl's visual contract (when its import path is available)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from self_improve import contract


class JointMapTests(unittest.TestCase):
    def test_default_map_is_identity_for_arm(self) -> None:
        joint_map = contract.default_joint_map()
        arm = np.array([0.1, -1.2, 0.8, 0.6, 0.05])
        dataset_row = np.concatenate([arm, [0.0]])
        sim_row = joint_map.to_sim(dataset_row)
        np.testing.assert_allclose(sim_row[:5], arm, atol=1e-12)

    def test_default_map_gripper_endpoints(self) -> None:
        joint_map = contract.default_joint_map()
        d_lo, d_hi = contract.DATASET_GRIPPER_RANGE
        s_lo, s_hi = contract.SIM_GRIPPER_RANGE
        np.testing.assert_allclose(joint_map.to_sim([0.0, 0.0, 0.0, 0.0, 0.0, d_hi])[5],
                                   s_hi, atol=1e-9)
        np.testing.assert_allclose(joint_map.to_sim([0.0, 0.0, 0.0, 0.0, 0.0, d_lo])[5],
                                   s_lo, atol=1e-9)

    def test_roundtrip(self) -> None:
        joint_map = contract.default_joint_map()
        # Dataset values inside the sim URDF limits round-trip exactly; the
        # real shoulder_lift can reach -1.836 rad, past the sim's -1.74533,
        # so stay in-range here (clamping is covered separately).
        dataset_positions = np.array([[0.05, -1.5, 0.9, 0.7, 1.0, 0.4],
                                      [-0.1, -1.7, 0.3, 0.5, 0.2, -0.5]])
        sim_positions = joint_map.to_sim(dataset_positions)
        recovered = joint_map.to_dataset(sim_positions)
        np.testing.assert_allclose(recovered, dataset_positions, atol=1e-9)

    def test_clamp_to_sim_limits(self) -> None:
        joint_map = contract.default_joint_map()
        # Gripper map output stays within URDF limits even for out-of-range
        # policy outputs (policy can emit anything in dataset units).
        clamped = joint_map.to_sim(np.full((1, 6), 10.0))
        limits = contract.SIM_JOINT_LIMITS_RAD
        for index, joint in enumerate(contract.SO101_JOINT_NAMES):
            self.assertGreaterEqual(clamped[0, index], limits[joint][0] - 1e-9)
            self.assertLessEqual(clamped[0, index], limits[joint][1] + 1e-9)

    def test_serialization(self) -> None:
        joint_map = contract.default_joint_map()
        restored = contract.JointMap.from_dict(
            json.loads(json.dumps(joint_map.to_dict())))
        np.testing.assert_allclose(restored.scale, joint_map.scale)
        np.testing.assert_allclose(restored.offset, joint_map.offset)


class EpisodeNpzTests(unittest.TestCase):
    def test_save_and_inspect(self) -> None:
        import tempfile

        steps = 5
        images = {
            camera: np.zeros((steps, contract.IMAGE_HEIGHT // 40,
                              contract.IMAGE_WIDTH // 40, 3), dtype=np.uint8)
            for camera in contract.CAMERA_KEYS
        }
        state = np.zeros((steps, 6), dtype=np.float32)
        action = np.ones((steps, 6), dtype=np.float32)
        joint_map = contract.default_joint_map()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollouts_raw" / "episode_000001.npz"
            contract.save_episode_npz(
                path,
                images=images,
                state=state,
                action=action,
                sim_state=state + 1.0,
                sim_action=action + 1.0,
                task="pick up the cube",
                success=True,
                reason="success",
                source="sim",
                joint_map=joint_map,
            )
            with np.load(path) as data:
                self.assertEqual(data[contract.NPZ_TASK], "pick up the cube")
                self.assertTrue(bool(data[contract.NPZ_SUCCESS]))
                self.assertEqual(data[contract.NPZ_REASON], "success")
                self.assertEqual(data[contract.NPZ_SOURCE], "sim")
                self.assertEqual(float(data[contract.NPZ_FPS]),
                                 contract.CONTROL_FREQUENCY_HZ)
                self.assertEqual(
                    list(data[contract.NPZ_JOINT_NAMES]),
                    list(contract.SO101_JOINT_NAMES))
                for camera in contract.CAMERA_KEYS:
                    self.assertIn(contract.camera_feature_key(camera), data)
                restored_map = contract.JointMap.from_dict(
                    json.loads(str(data[contract.NPZ_JOINT_MAP])))
                np.testing.assert_allclose(restored_map.scale, joint_map.scale)

    def test_missing_camera_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                contract.save_episode_npz(
                    Path(tmp) / "episode_000000.npz",
                    images={"wrist": np.zeros((1, 2, 3, 3), dtype=np.uint8)},
                    state=np.zeros((1, 6)),
                    action=np.zeros((1, 6)),
                    task="t",
                    success=False,
                    reason="timeout",
                    source="sim",
                    joint_map=contract.default_joint_map(),
                )


class ManifestTests(unittest.TestCase):
    def test_jsonl_roundtrip(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / contract.ROLL_FILENAME
            self.assertEqual(contract.read_jsonl(path), [])
            contract.append_jsonl(path, {"episode_index": 0, "success": True})
            contract.append_jsonl(path, {"episode_index": 1, "success": False})
            entries = contract.read_jsonl(path)
            self.assertEqual([e["episode_index"] for e in entries], [0, 1])


class VisualContractCrossCheck(unittest.TestCase):
    """The duplicated SO101 constants must match so101_rl exactly."""

    def test_matches_so101_rl(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        so101_rl_root = repo_root / "isaaclab" / "source" / "so101_rl"
        self.assertTrue(so101_rl_root.exists(),
                        "isaaclab/source/so101_rl not found")
        sys.path.insert(0, str(so101_rl_root))
        try:
            import so101_rl.visual_contract as vc  # noqa: E402
        finally:
            sys.path.remove(str(so101_rl_root))
        self.assertEqual(tuple(vc.SO101_JOINT_NAMES),
                         contract.SO101_JOINT_NAMES)
        self.assertEqual(set(vc.SO101_ACTOR_OBSERVATION_GROUPS[1:]),
                         set(contract.CAMERA_KEYS))


class ImageKeyMapTests(unittest.TestCase):
    def test_base_preset_maps_openpi_keys(self) -> None:
        self.assertEqual(
            contract.PI05_BASE_IMAGE_KEY_MAP["observation.images.wrist"],
            "observation.images.left_wrist_0_rgb")
        self.assertEqual(
            contract.PI05_BASE_IMAGE_KEY_MAP["observation.images.overhead_1"],
            "observation.images.base_0_rgb")
        self.assertEqual(
            contract.PI05_BASE_IMAGE_KEY_MAP["observation.images.overhead_2"],
            "observation.images.right_wrist_0_rgb")

    def test_preset_resolution(self) -> None:
        self.assertEqual(contract.resolve_image_key_map(["pi05_base"]),
                         contract.PI05_BASE_IMAGE_KEY_MAP)

    def test_pairs_merge_with_preset(self) -> None:
        resolved = contract.resolve_image_key_map([
            "observation.images.wrist=observation.images.left_wrist_0_rgb",
            "pi05_base",
        ])
        self.assertEqual(len(resolved), 3)
        self.assertEqual(resolved["observation.images.overhead_1"],
                         "observation.images.base_0_rgb")

    def test_none_resolves_empty(self) -> None:
        self.assertEqual(contract.resolve_image_key_map(None), {})

    def test_bad_entry_exits(self) -> None:
        with self.assertRaises(SystemExit):
            contract.resolve_image_key_map(["not-a-preset"])


if __name__ == "__main__":
    unittest.main()
