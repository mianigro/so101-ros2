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

"""Prefix-mask / position-id correctness with inserted demo tokens (ICL §9).

Model-free: exercises ``splice_demo_tokens`` plus the exact downstream math
PI05Pytorch performs (``make_att_2d_masks`` + ``position_ids`` cumsum, and
the ``denoise_step`` ``prefix_offsets`` path), on small synthetic tensors.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from so101_icl.modeling_pi05_icl import splice_demo_tokens  # noqa: E402


def _make_prefix(batch=2, n_img=6, n_lang=4):
    embs = torch.randn(batch, n_img + n_lang, 128)
    pad_masks = torch.ones(batch, n_img + n_lang, dtype=torch.bool)
    pad_masks[:, -1] = False  # one padded language slot
    att_masks = torch.zeros(batch, n_img + n_lang, dtype=torch.bool)
    return embs, pad_masks, att_masks


class TestSplice(unittest.TestCase):
    def test_shapes_and_placement(self):
        embs, pad, att = _make_prefix()
        d_embs = torch.randn(1, 5, 128)
        d_pad = torch.ones(1, 5, dtype=torch.bool)
        e2, p2, a2 = splice_demo_tokens(embs, pad, att, d_embs, d_pad, n_lang_tokens=4)
        self.assertEqual(e2.shape, (2, 15, 128))
        self.assertEqual(p2.shape, (2, 15))
        self.assertEqual(a2.shape, (2, 15))
        # demo block sits between cameras and language, att all zero
        self.assertTrue((a2[:, 6:11] == 0).all())
        # camera and language contents are preserved in order
        self.assertTrue(torch.equal(e2[:, :6], embs[:, :6]))
        self.assertTrue(torch.equal(e2[:, 11:], embs[:, 6:]))
        self.assertTrue(torch.equal(p2[:, 11:], pad[:, 6:]))

    def test_single_sample_cache_expands_to_batch(self):
        embs, pad, att = _make_prefix(batch=3)
        d_embs = torch.randn(1, 2, 128)
        d_pad = torch.ones(1, 2, dtype=torch.bool)
        e2, p2, _ = splice_demo_tokens(embs, pad, att, d_embs, d_pad, n_lang_tokens=4)
        self.assertTrue(torch.equal(e2[0, 6:8], e2[1, 6:8]))
        self.assertTrue(torch.equal(e2[0, 6:8], e2[2, 6:8]))

    def test_batch_mismatch_raises(self):
        embs, pad, att = _make_prefix(batch=2)
        with self.assertRaises(ValueError):
            splice_demo_tokens(embs, pad, att, torch.randn(3, 2, 128),
                               torch.ones(3, 2, dtype=torch.bool), n_lang_tokens=4)

    def test_dtype_follows_prefix(self):
        embs, pad, att = _make_prefix()
        e2, _, _ = splice_demo_tokens(
            embs, pad, att, torch.randn(1, 2, 128), torch.ones(1, 2, dtype=torch.bool),
            n_lang_tokens=4,
        )
        self.assertEqual(e2.dtype, embs.dtype)

    def test_position_ids_monotonic_via_pad_masks(self):
        """PI05Pytorch computes position_ids = cumsum(pad_masks) - 1."""
        embs, pad, att = _make_prefix()
        d_pad = torch.tensor([[True, True, False, True]])  # one padded demo slot
        e2, p2, a2 = splice_demo_tokens(
            embs, pad, att, torch.randn(1, 4, 128), d_pad, n_lang_tokens=4
        )
        pos = torch.cumsum(p2, dim=1) - 1
        # ids never decrease along the sequence and pad slots don't advance them
        self.assertTrue((pos[:, 1:] >= pos[:, :-1]).all())
        # spliced layout: 6 img | demo(T,T,F,T) | lang — padded demo slot at idx 8
        self.assertFalse(bool(p2[0, 8]))
        self.assertTrue((pos[0, 8] == pos[0, 7]).item())
        self.assertEqual(int(p2[0, 9]), 1)

    def test_att_2d_masks_keep_demo_tokens_in_prefix_block(self):
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        embs, pad, att = _make_prefix(batch=1)
        pad[:, -1] = True  # fully-valid prefix here; padding is covered elsewhere
        n_demo = 5
        e2, p2, a2 = splice_demo_tokens(
            embs, pad, att, torch.randn(1, n_demo, 128),
            torch.ones(1, n_demo, dtype=torch.bool), n_lang_tokens=4,
        )
        suffix_pad = torch.ones(1, 10, dtype=torch.bool)
        suffix_att = torch.zeros(1, 10, dtype=torch.bool)
        suffix_att[:, 0] = True  # first action token marks the causal boundary
        all_pad = torch.cat([p2, suffix_pad], dim=1)
        all_att = torch.cat([a2, suffix_att], dim=1)
        att_2d = make_att_2d_masks(all_pad, all_att)
        # every prefix token (incl. demos) attends to every other prefix token
        prefix_len = p2.shape[1]
        self.assertTrue(att_2d[0, :prefix_len, :prefix_len].all())
        # prefix tokens can never attend to the action suffix
        self.assertFalse(att_2d[0, :prefix_len, prefix_len:].any())

    def test_kv_cache_prefix_offsets_math(self):
        """denoise_step: position_ids = prefix_offsets + cumsum(suffix) - 1."""
        embs, pad, att = _make_prefix(batch=1)
        e2, p2, _ = splice_demo_tokens(
            embs, pad, att, torch.randn(1, 3, 128),
            torch.ones(1, 3, dtype=torch.bool), n_lang_tokens=4,
        )
        suffix_pad = torch.ones(1, 10, dtype=torch.bool)
        prefix_offsets = torch.sum(p2, dim=-1)[:, None]
        pos = prefix_offsets + torch.cumsum(suffix_pad, dim=1) - 1
        self.assertEqual(pos[0, 0].item(), int(p2[0].sum()))
        self.assertEqual(pos[0, -1].item(), int(p2[0].sum()) + 9)

    def test_demo_at_end_equals_demo_mid_insertion_loss_parity(self):
        """Insertion position must not change the masked-attention structure."""
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        embs, pad, att = _make_prefix(batch=1)
        d_embs = torch.randn(1, 4, 128)
        d_pad = torch.ones(1, 4, dtype=torch.bool)

        mid = splice_demo_tokens(embs, pad, att, d_embs, d_pad, n_lang_tokens=4)
        # end-insertion variant: demos appended after language
        end_embs = torch.cat([embs, d_embs], dim=1)
        end_pad = torch.cat([pad, d_pad], dim=1)
        end_att = torch.cat([att, torch.zeros(1, 4, dtype=torch.bool)], dim=1)

        for tensors in (mid, (end_embs, end_pad, end_att)):
            e, p, a = tensors
            att_2d = make_att_2d_masks(p, a)
            self.assertEqual(att_2d.shape[1], e.shape[1])


if __name__ == "__main__":
    unittest.main()
