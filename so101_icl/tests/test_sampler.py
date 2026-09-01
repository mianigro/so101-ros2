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

"""Sampler invariants (ICL §4.4) against the on-disk SO-101 datasets.

Uses ``local/so101_test`` + ``algorithmtheworld/so101-pick-and-place``
(8 episodes, one task) via the smoke registry — no downloads. Skipped when
those datasets are absent.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from so101_icl.configuration_pi05_icl import icl_config_from_base  # noqa: E402
from so101_icl.data import (  # noqa: E402
    DatasetSpec,
    ICLDataset,
    build_task_registry,
    normalize_task,
)

DATASETS_PRESENT = (
    Path.home() / ".cache/huggingface/lerobot/local/so101_test/meta/info.json"
).exists() and (
    Path.home()
    / ".cache/huggingface/lerobot/algorithmtheworld/so101-pick-and-place/meta/info.json"
).exists()


def _config():
    return icl_config_from_base(device="cpu")


def _registry_path():
    tmp = tempfile.mkdtemp(prefix="icl_registry_")
    return Path(tmp) / "registry.json"


@unittest.skipUnless(DATASETS_PRESENT, "local SO-101 datasets not on disk")
class TestSampler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()
        cls.registry_path = _registry_path()
        build_task_registry(
            [DatasetSpec("local/so101_test"), DatasetSpec("algorithmtheworld/so101-pick-and-place")],
            holdout={"eval": 0, "test": 0},
            out_path=cls.registry_path,
        )

    def test_groups_merge_across_datasets(self):
        import json

        reg = json.loads(self.registry_path.read_text())
        groups = reg["groups"]
        self.assertEqual(len(groups), 1)
        eps = next(iter(groups.values()))
        self.assertEqual(len(eps), 8)
        self.assertEqual({e[0] for e in eps}, {0, 1})  # both datasets contribute

    def test_item_shapes_and_ranges(self):
        ds = ICLDataset(self.registry_path, self.cfg, split="train", seed=0)
        self.assertEqual(len(ds), 8)
        item = ds[3]
        de = self.cfg.demo_encoder
        self.assertEqual(item["icl.demo_frames"].shape, (de.k_max, de.frames_per_demo, 3, 224, 224))
        self.assertEqual(item["icl.demo_traj"].shape,
                         (de.k_max, de.traj_steps, self.cfg.max_state_dim + self.cfg.max_action_dim))
        self.assertTrue(item["icl.demo_frames"].min() >= -1.0)
        self.assertTrue(item["icl.demo_frames"].max() <= 1.0)
        k = int(item["icl.demo_mask"].sum())
        self.assertGreaterEqual(k, 1)
        self.assertLessEqual(k, de.k_max)
        self.assertTrue(set(item["icl.demo_traj_ok"].tolist()) <= {0.0, 1.0})
        # absent slots are zero-padded
        if k < de.k_max:
            self.assertEqual(item["icl.demo_frames"][k:].abs().max().item(), 0.0)
            self.assertFalse(item["icl.demo_mask"][k:].any())
        # query fields
        self.assertEqual(item["action"].shape, (self.cfg.chunk_size, 6))
        self.assertEqual(item["observation.state"].shape, (6,))
        for key in ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb",
                    "observation.images.right_wrist_0_rgb"):
            self.assertIn(key, item)

    def test_determinism_per_index_and_epoch(self):
        ds = ICLDataset(self.registry_path, self.cfg, split="train", seed=0)
        a = ds[2]
        b = ds[2]
        self.assertTrue(torch.equal(a["icl.demo_frames"], b["icl.demo_frames"]))
        self.assertTrue(torch.equal(a["action"], b["action"]))
        ds.set_epoch(1)
        c = ds[2]
        self.assertFalse(torch.equal(a["icl.demo_traj"], c["icl.demo_traj"]) and
                         torch.equal(a["icl.demo_frames"], c["icl.demo_frames"]))

    def test_holdout_isolation(self):
        """With the single group held out entirely, training must refuse."""
        import json

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            build_task_registry(
                [DatasetSpec("local/so101_test")],
                holdout={"eval": 1, "test": 0},
                out_path=path,
            )
            reg = json.loads(path.read_text())
            self.assertEqual(len(reg["splits"]["train"]), 0)
            self.assertEqual(len(reg["splits"]["eval"]), 1)
            with self.assertRaises(ValueError):
                ICLDataset(path, self.cfg, split="train")

    def test_supports_share_query_task_string(self):
        """Rev 6 sampling contract: support demos come from episodes with the
        query's task string (not just the category) — on this registry the
        task strings are consistent, so every draw must be same-task."""
        ds = ICLDataset(self.registry_path, self.cfg, split="train", seed=0)
        for i in range(len(ds)):
            q_ds, group, q_ep = ds._index[i]
            q_task = ds._ep_task.get((q_ds, q_ep))
            rng = np.random.default_rng([ds.seed, i, ds._epoch])
            members = [m for m in ds._group_members[group] if m != (q_ds, q_ep)]
            for k in range(1, min(ds.k_max, len(members)) + 1):
                picks = ds._select_supports(members, q_task, k, rng)
                self.assertEqual(len(picks), k)
                self.assertNotIn((q_ds, q_ep), picks)
                for m in picks:
                    self.assertEqual(ds._ep_task.get(m), q_task)

    def test_normalize_task(self):
        self.assertEqual(normalize_task("  Pick   Up "), "pick up")


class TestKeyframeIndices(unittest.TestCase):
    """Pure indexing invariants — no datasets needed."""

    def test_long_episode_strictly_increasing(self):
        idx = ICLDataset._sample_keyframe_indices(805, 6)
        self.assertEqual(len(idx), 6)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 804)
        self.assertTrue((np.diff(idx) > 0).all())

    def test_short_episode_returns_exact_count(self):
        # Fewer frames than requested: timesteps repeat instead of shrinking
        # (the traj slot must stay [traj_steps, d] for the broadcast in
        # __getitem__).
        idx = ICLDataset._sample_keyframe_indices(5, 6)
        self.assertEqual(len(idx), 6)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 4)
        self.assertTrue(((idx >= 0) & (idx <= 4)).all())

    def test_single_frame_episode(self):
        idx = ICLDataset._sample_keyframe_indices(1, 16)
        self.assertEqual(len(idx), 16)
        self.assertTrue((idx == 0).all())


if __name__ == "__main__":
    unittest.main()
