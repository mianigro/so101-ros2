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

"""Training-free retrieval-fusion unit tests (pure tensor logic, no model)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from so101_icl.demo_index import (  # noqa: E402
    DemoRetrievalIndex,
    demo_action_at,
    embed_frames,
    fuse_with_demo,
    nearest_keyframe,
)


class TestEmbed(unittest.TestCase):
    def test_shape_and_normalization(self):
        frames = torch.rand(5, 3, 224, 224) * 2 - 1
        emb = embed_frames(frames)
        self.assertEqual(emb.shape, (5, 3 * 8 * 8))
        norms = emb.norm(dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(5), atol=1e-5))

    def test_similar_frames_embed_close(self):
        base = torch.rand(1, 3, 64, 64)
        near = base + 0.001 * torch.randn_like(base)
        far = torch.rand(1, 3, 64, 64)
        e_base, e_near, e_far = embed_frames(base)[0], embed_frames(near)[0], embed_frames(far)[0]
        self.assertLess(
            (e_base - e_near).norm().item(), (e_base - e_far).norm().item()
        )


class TestFusion(unittest.TestCase):
    def test_demo_action_interpolation(self):
        actions = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]], np.float32)
        self.assertTrue(np.allclose(demo_action_at(actions, 0.0), [0.0, 0.0]))
        self.assertTrue(np.allclose(demo_action_at(actions, 1.0), [2.0, 4.0]))
        self.assertTrue(np.allclose(demo_action_at(actions, 0.5), [1.0, 2.0]))
        # halfway between step 1 and 2 (progress 0.75)
        self.assertTrue(np.allclose(demo_action_at(actions, 0.75), [1.5, 3.0]))

    def test_near_match_follows_demo_far_match_follows_base(self):
        base = np.zeros(6, np.float32)
        demo = np.ones(6, np.float32)
        near = fuse_with_demo(base, demo, distance=0.0, lam=5.0)
        self.assertTrue(np.allclose(near, demo, atol=1e-5))
        far = fuse_with_demo(base, demo, distance=5.0, lam=5.0)
        self.assertTrue(np.allclose(far, base, atol=1e-3))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            fuse_with_demo(np.zeros(6), np.zeros(5), 0.1)


class TestIndex(unittest.TestCase):
    def _pack(self):
        # structured (not constant) frames: L2-normalized embeddings are
        # brightness-invariant, so constant-value test frames would collapse
        # to the same vector — seeded noise patterns stand in for scenes.
        # Each demo repeats ONE pattern across its keyframes.
        def _pattern(seed):
            return np.random.default_rng(seed).uniform(
                -1, 1, (3, 32, 32)
            ).astype(np.float32)

        pat_a, pat_b = _pattern(10), _pattern(20)
        frames = torch.stack([
            torch.from_numpy(np.tile(pat_a, (4, 1, 1, 1))),
            torch.from_numpy(np.tile(pat_b, (4, 1, 1, 1))),
        ])
        actions = np.stack([
            np.tile(np.array([[0.0, 0.0, 0.0]], np.float32), (4, 1)),
            np.tile(np.array([[1.0, 1.0, 1.0]], np.float32), (4, 1)),
        ])
        return frames, actions

    def test_query_retrieves_matching_demo(self):
        frames, actions = self._pack()
        index = DemoRetrievalIndex(frames, actions)
        self.assertEqual(index.size, 8)
        query_a = torch.from_numpy(
            np.random.default_rng(10).uniform(-1, 1, (3, 32, 32)).astype(np.float32)
        )
        action, dist_a = index.query(query_a)
        self.assertTrue(np.allclose(action, [0.0, 0.0, 0.0]))
        query_b = torch.from_numpy(
            np.random.default_rng(20).uniform(-1, 1, (3, 32, 32)).astype(np.float32)
        )
        action_b, dist_b = index.query(query_b)
        self.assertTrue(np.allclose(action_b, [1.0, 1.0, 1.0]))
        # both queries match their own demo's bank exactly (same pattern)
        self.assertAlmostEqual(dist_a, 0.0, places=5)
        self.assertAlmostEqual(dist_b, 0.0, places=5)

    def test_masked_demos_excluded(self):
        frames, actions = self._pack()
        index = DemoRetrievalIndex(frames, actions, np.array([True, False]))
        self.assertEqual(index.size, 4)
        query_b = torch.from_numpy(
            np.random.default_rng(20).uniform(-1, 1, (3, 32, 32)).astype(np.float32)
        )
        action, _ = index.query(query_b)  # bright pattern, demo 1 masked out
        self.assertTrue(np.allclose(action, [0.0, 0.0, 0.0]))  # only demo 0 left

    def test_all_masked_raises(self):
        frames, actions = self._pack()
        with self.assertRaises(ValueError):
            DemoRetrievalIndex(frames, actions, np.array([False, False]))

    def test_progress_positions(self):
        # keyframes spanning a trajectory: querying the FIRST/LAST keyframe
        # pattern of demo 0 retrieves its initial/final action
        rng = np.random.default_rng(30)
        demo0 = rng.uniform(-1, 1, (4, 3, 32, 32)).astype(np.float32)
        frames = torch.stack([
            torch.from_numpy(demo0),
            torch.from_numpy(
                np.random.default_rng(40).uniform(-1, 1, (4, 3, 32, 32)).astype(np.float32)
            ),
        ])
        actions = np.stack([
            np.array([[0.0], [1.0], [2.0], [3.0]], np.float32),
            np.zeros((4, 1), np.float32),
        ])
        index = DemoRetrievalIndex(frames, actions)
        action, _ = index.query(torch.from_numpy(demo0[0]))
        self.assertTrue(np.allclose(action, [0.0]))  # first keyframe
        action, _ = index.query(torch.from_numpy(demo0[3]))
        self.assertTrue(np.allclose(action, [3.0]))  # last keyframe


if __name__ == "__main__":
    unittest.main()
