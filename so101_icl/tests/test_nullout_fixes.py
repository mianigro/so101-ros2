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

"""Gate floor + language dropout (null-out mitigations, Rev 3 §2.1/§4.5)."""

import random
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_icl.configuration_pi05_icl import DemoEncoderConfig, ICLConfig  # noqa: E402
from so101_icl.demo_encoder import DemoEncoder  # noqa: E402
from so101_icl.train_loop import (  # noqa: E402
    LANGUAGE_DROPOUT_PROMPT,
    _apply_language_dropout,
)


def _tiny_encoder(gate_floor: float) -> DemoEncoder:
    cfg = ICLConfig()
    cfg.demo_encoder = DemoEncoderConfig(
        k_max=2, tokens_vis=4, tokens_traj=4, traj_steps=2, gate_floor=gate_floor
    )
    return DemoEncoder(cfg, vlm_width=32)


def _dummy_pack(encoder: DemoEncoder, gate_scale=1.0):
    B, K, F = 1, encoder.k_max, encoder.config.frames_per_demo
    frames = torch.zeros(B, K, F, 3, 224, 224)
    mask = torch.tensor([[True, False]])
    traj = torch.zeros(B, K, encoder.config.traj_steps, 64)
    traj_ok = torch.ones(B, K)
    embed = lambda x: torch.zeros(x.shape[0], 2, 32)  # noqa: E731
    return encoder(frames, mask, traj, traj_ok, embed, gate_scale=gate_scale)


class TestGateFloor(unittest.TestCase):
    def test_gates_init_at_uniform_shares(self):
        # rev 8: raw scalars are logits init'd at 0; the effective gates
        # start uniform, i.e. each equals gate_floor (default budget).
        enc = _tiny_encoder(0.1)
        self.assertEqual(enc.gate_vis.item(), 0.0)
        eff = enc.effective_gates()
        for g in eff:
            self.assertAlmostEqual(g.item(), 0.1, places=6)

    def test_budget_keeps_mass_on_every_branch(self):
        # A branch can only lose loudness to the other, never to zero: even
        # a wildly negative vis logit still leaves some effective mass.
        enc = _tiny_encoder(0.1)
        with torch.no_grad():
            enc.gate_vis.fill_(-50.0)  # optimizer tries to shut the branch
        embs, _ = _dummy_pack(enc)
        self.assertGreater(embs.abs().max().item(), 0.0)  # still nonzero

    def test_gate_scale_zero_still_zeroes_demo_tokens(self):
        # loss_demo_zeroed must keep working despite the floor
        enc = _tiny_encoder(0.1)
        embs, _ = _dummy_pack(enc, gate_scale=0.0)
        self.assertEqual(embs.abs().max().item(), 0.0)

    def test_default_floor_zero_keeps_zero_init(self):
        enc = _tiny_encoder(0.0)
        self.assertEqual(enc.gate_vis.item(), 0.0)
        embs, _ = _dummy_pack(enc)
        self.assertEqual(embs.abs().max().item(), 0.0)


class TestLanguageDropout(unittest.TestCase):
    def test_dropped_batch_gets_vague_prompt(self):
        rng = random.Random(0)
        batch = {"task": ["pick up the cube", "place in the cup"], "action": None}
        applied = _apply_language_dropout(batch, rng, p=1.0)
        self.assertTrue(applied)
        self.assertEqual(batch["task"], [LANGUAGE_DROPOUT_PROMPT] * 2)


if __name__ == "__main__":
    unittest.main()
