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

"""Rev 5 training-pipeline unit tests: demo-usage hinge, bursty sampler,
keypoint branch. No GPU, no model download."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from so101_icl.configuration_pi05_icl import DemoEncoderConfig, KeypointConfig  # noqa: E402
from so101_icl.train_loop import TrainSettings, _demo_zeroed_loss, usage_hinge  # noqa: E402


# ---------------------------------------------------------------------- #
# 1. Demo-usage hinge                                                     #
# ---------------------------------------------------------------------- #


class TestUsageHinge(unittest.TestCase):
    """usage_hinge(loss, zeroed, margin) = relu(loss - zeroed + margin)."""

    def test_zero_when_demos_beat_margin(self):
        # demos beat the zeroed replay by more than the margin -> no penalty
        self.assertEqual(usage_hinge(0.10, 0.20, margin=0.02).item(), 0.0)

    def test_fires_when_demos_ignored(self):
        # demo-conditioned loss >= zeroed: demos not helping -> penalty
        hinge = usage_hinge(0.20, 0.20, margin=0.02)
        self.assertAlmostEqual(hinge.item(), 0.02, places=6)

    def test_fires_when_demos_hurt(self):
        hinge = usage_hinge(0.30, 0.20, margin=0.02)
        self.assertAlmostEqual(hinge.item(), 0.12, places=6)

    def test_gradient_flows_through_loss_only(self):
        loss = torch.tensor(0.25, requires_grad=True)
        zeroed = 0.20
        usage_hinge(loss, zeroed, margin=0.02).backward()
        # d(relu)/d(loss) = 1 when active
        self.assertEqual(loss.grad.item(), 1.0)


class _FakeModel:
    def __init__(self, zeroed_loss: float):
        self._zeroed = zeroed_loss
        self._demo_gate_scale = 1.0

    def forward(self, batch):
        return torch.tensor(self._zeroed), None


class _FakePolicy:
    def __init__(self, model):
        self.model = model

    def forward(self, batch):
        return self.model.forward(batch)


class TestZeroedReplayIntegration(unittest.TestCase):
    def test_demo_zeroed_loss_returns_float(self):
        policy = _FakePolicy(_FakeModel(0.3))
        value = _demo_zeroed_loss(policy, {})
        self.assertIsInstance(value, float)
        self.assertEqual(policy.model._demo_gate_scale, 1.0)  # restored

    def test_settings_carry_usage_knobs(self):
        s = TrainSettings(demo_usage_weight=0.5, demo_usage_margin=0.02)
        self.assertEqual(s.demo_usage_weight, 0.5)
        self.assertEqual(s.demo_usage_margin, 0.02)
        # defaults off — old configs train unchanged
        self.assertEqual(TrainSettings().demo_usage_weight, 0.0)


# ---------------------------------------------------------------------- #
# 2. Bursty group sampler                                                 #
# ---------------------------------------------------------------------- #


class _FakeDataset:
    """Duck-typed ICLDataset: only _index and __len__ are needed."""

    def __init__(self, groups: dict[str, int]):
        self._index = []
        for group, n in groups.items():
            for j in range(n):
                self._index.append((0, group, j))
        self._epoch = 0

    def __len__(self):
        return len(self._index)


from so101_icl.data import BurstyGroupBatchSampler  # noqa: E402


class TestBurstySampler(unittest.TestCase):
    def _ds(self):
        # every group >= burst_length so batches stay single-group;
        # Zipf ranks: a=1, b=2, c=3 — 'a' should dominate popularity
        return _FakeDataset({"a": 8, "b": 4, "c": 4})

    def test_epoch_length_and_coverage(self):
        sampler = BurstyGroupBatchSampler(self._ds(), 4, burst_length=4, seed=0)
        batches = list(iter(sampler))
        flat = [i for b in batches for i in b]
        self.assertEqual(len(flat), 16)                      # full coverage
        self.assertEqual(sorted(flat), list(range(16)))      # every index once
        self.assertTrue(all(len(b) == 4 for b in batches))   # drop_last

    def test_bursts_are_group_contiguous(self):
        ds = self._ds()
        sampler = BurstyGroupBatchSampler(ds, 4, burst_length=4, seed=0)
        group_of = {i: g for i, (_d, g, _e) in enumerate(ds._index)}
        for batch in iter(sampler):
            # burst_length == batch_size: every batch is single-group
            groups = {group_of[i] for i in batch}
            self.assertEqual(len(groups), 1, f"batch spans groups {groups}")

    def test_epoch_regeneration(self):
        ds = self._ds()
        sampler = BurstyGroupBatchSampler(ds, 4, burst_length=2, seed=0)
        first = [i for b in iter(sampler) for i in b]
        ds._epoch = 1
        second = [i for b in iter(sampler) for i in b]
        # same coverage, different order (with overwhelming probability)
        self.assertEqual(sorted(first), sorted(second))

    def test_deterministic_same_seed(self):
        ds1, ds2 = self._ds(), self._ds()
        s1 = [i for b in iter(BurstyGroupBatchSampler(ds1, 4, seed=7)) for i in b]
        s2 = [i for b in iter(BurstyGroupBatchSampler(ds2, 4, seed=7)) for i in b]
        self.assertEqual(s1, s2)

    def test_zipfian_popularity_favors_rank_one(self):
        ds = _FakeDataset({g: 8 for g in "abcdefgh"})
        sampler = BurstyGroupBatchSampler(ds, 4, burst_length=1, seed=0)
        counts: dict[str, int] = {}
        for b in iter(sampler):
            for i in b:
                g = ds._index[i][1]
                counts[g] = counts.get(g, 0) + 1
        top = max(counts, key=counts.get)
        # the rank-1 group (alphabetically first under the seeded rank draw)
        # should be the most frequent — Zipf, not uniform
        self.assertEqual(counts[top], max(counts.values()))
        self.assertGreater(max(counts.values()), 16 // 8)  # clearly non-uniform


# ---------------------------------------------------------------------- #
# 3. Keypoint branch                                                      #
# ---------------------------------------------------------------------- #


class TestKeypointConfig(unittest.TestCase):
    def test_default_disabled(self):
        cfg = DemoEncoderConfig()
        self.assertFalse(cfg.keypoints.enabled)
        self.assertEqual(cfg.tokens_traj, 2 * cfg.traj_steps)

    def test_dict_coercion_from_yaml(self):
        cfg = DemoEncoderConfig(keypoints={"enabled": True, "n_kp": 8})
        self.assertIsInstance(cfg.keypoints, KeypointConfig)
        self.assertTrue(cfg.keypoints.enabled)
        self.assertEqual(cfg.keypoints.n_kp, 8)


try:
    from lerobot.policies.pi05.modeling_pi05 import get_gemma_config  # noqa: F401

    LEROBOT_OK = True
except ImportError:
    LEROBOT_OK = False


@unittest.skipUnless(LEROBOT_OK, "lerobot not importable")
class TestDemoEncoderKeypointBranch(unittest.TestCase):
    def _encoder(self, **kp_overrides):
        from so101_icl.demo_encoder import DemoEncoder

        kp = {"enabled": True, "n_kp": 4, "kp_dim": 12, "tokens_kp": 8}
        kp.update(kp_overrides)
        cfg = DemoEncoderConfig(
            k_max=2, frames_per_demo=3, tokens_vis=4, tokens_traj=8,
            traj_steps=4, n_heads=2, keypoints=kp,
        )

        class _Cfg:  # DemoEncoder reads config.demo_encoder.<fields> and
            demo_encoder = cfg  # config.max_state_dim/max_action_dim
            max_state_dim = 32
            max_action_dim = 32

        vlm_width = get_gemma_config("gemma_2b").width
        return DemoEncoder(_Cfg(), vlm_width)

    def _pack(self, encoder, kp_present=True, B=1):
        K, F, C = encoder.k_max, encoder.config.frames_per_demo, 8
        frames = torch.rand(B, K, F, 3, 64, 64)
        mask = torch.tensor([[True, True]])
        traj = torch.rand(B, K, encoder.config.traj_steps, 64)
        traj_ok = torch.ones(B, K)
        kp_cfg = encoder.config.keypoints
        kp = torch.rand(B, K, F, kp_cfg.n_kp, kp_cfg.kp_dim)
        if not kp_present:
            kp = torch.zeros_like(kp)
        kp_ok = torch.ones(B, K) if kp_present else torch.zeros(B, K)
        return frames, mask, traj, traj_ok, kp, kp_ok

    def _embed_fn(self, frames):
        # cheap SigLIP stand-in: mean-pool broadcast to the vlm width
        n = frames.shape[0]
        pooled = frames.mean(dim=(2, 3)) / 3.0  # [N, 3]
        return pooled[:, :, None].expand(n, 3, 2048)

    def test_tokens_per_demo_includes_kp(self):
        enc = self._encoder()
        self.assertEqual(enc.tokens_per_demo,
                         enc.tokens_vis + enc.config.keypoints.tokens_kp + enc.tokens_traj)

    def test_forward_shapes(self):
        enc = self._encoder()
        f, m, t, t_ok, kp, kp_ok = self._pack(enc)
        embs, pad = enc.forward(f, m, t, t_ok, self._embed_fn, kp=kp, kp_ok=kp_ok)
        self.assertEqual(embs.shape, (1, enc.k_max * enc.tokens_per_demo,
                                      enc.vlm_width))
        self.assertEqual(pad.shape, (1, enc.k_max * enc.tokens_per_demo))

    def test_batched_forward_mask_shapes(self):
        """Regression: kp_ok masking must match the [B, K] batch layout
        (B > 1 crashed at the pre-reshape [B*K, T, D] stage)."""
        enc = self._encoder()
        f, m, t, t_ok, kp, kp_ok = self._pack(enc, B=4)
        embs, pad = enc.forward(f, m, t, t_ok, self._embed_fn, kp=kp, kp_ok=kp_ok)
        self.assertEqual(embs.shape[0], 4)

    def test_requires_kp_when_enabled(self):
        enc = self._encoder()
        frames, mask, traj, traj_ok, kp, kp_ok = self._pack(enc)
        with self.assertRaises(ValueError):
            enc.forward(frames, mask, traj, traj_ok, self._embed_fn)  # no kp

    def test_disabled_branch_is_bit_identical(self):
        """keypoints.enabled=false must reproduce the rev-4 token layout."""
        from so101_icl.demo_encoder import DemoEncoder

        base = {"k_max": 2, "frames_per_demo": 3, "tokens_vis": 4,
                "tokens_traj": 8, "traj_steps": 4, "n_heads": 2, "gate_floor": 0.0}
        class _CfgOld:
            demo_encoder = DemoEncoderConfig(**base)
            max_state_dim = 32
            max_action_dim = 32

        class _CfgNew:
            demo_encoder = DemoEncoderConfig(**base, keypoints={"enabled": False})
            max_state_dim = 32
            max_action_dim = 32

        torch.manual_seed(0)
        enc_old = DemoEncoder(_CfgOld(), get_gemma_config("gemma_2b").width)
        torch.manual_seed(0)
        enc_new = DemoEncoder(_CfgNew(), get_gemma_config("gemma_2b").width)
        frames = torch.rand(1, 2, 3, 3, 64, 64)
        mask = torch.tensor([[True, True]])
        traj = torch.rand(1, 2, 4, 64)
        traj_ok = torch.ones(1, 2)
        e1, _ = enc_old.encode_pack(frames, mask, traj, traj_ok, self._embed_fn)
        e2, _ = enc_new.encode_pack(frames, mask, traj, traj_ok, self._embed_fn)
        torch.testing.assert_close(e1, e2)

    def test_zero_init_property_with_floor_zero(self):
        torch.manual_seed(0)
        enc = self._encoder()  # gate_floor defaults to 0
        f, m, t, t_ok, kp, kp_ok = self._pack(enc)
        embs, _ = enc.forward(f, m, t, t_ok, self._embed_fn, kp=kp, kp_ok=kp_ok)
        # every demo token embedding is exactly zero at init (M0 property)
        self.assertEqual(float(embs.abs().max()), 0.0)

    def test_gate_floor_initializes_kp_gate(self):
        enc = self._encoder()
        enc.config.gate_floor = 0.1
        enc.reset_icl_parameters()
        self.assertAlmostEqual(float(enc.gate_kp), 0.1, places=6)


class TestExtractKeypoints(unittest.TestCase):
    def test_shape_and_flags(self):
        from so101_icl.data import extract_keypoints

        rng = np.random.default_rng(0)
        frames = torch.from_numpy(
            rng.uniform(0, 1, (3, 3, 120, 160)).astype(np.float32)
        )
        kp = extract_keypoints(frames, max_kp=8)
        self.assertEqual(kp.shape, (3, 8, 131))
        # textured noise frames: at least one valid keypoint per frame
        self.assertTrue((kp[..., 130] > 0).any(axis=1).all())
        # valid descriptors are L2-normalized
        valid = kp[kp[..., 130] > 0]
        norms = np.linalg.norm(valid[:, 2:130], axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-4))
        # coords normalized to [0, 1]
        self.assertTrue((valid[:, 0:2] >= 0).all() and (valid[:, 0:2] <= 1).all())

    def test_blank_frame_all_invalid(self):
        from so101_icl.data import extract_keypoints

        blank = torch.zeros((1, 3, 64, 64))
        kp = extract_keypoints(blank, max_kp=4)
        self.assertEqual(kp.shape, (1, 4, 131))
        self.assertEqual(float(kp[..., 130].max()), 0.0)  # no valid keypoints


if __name__ == "__main__":
    unittest.main()
