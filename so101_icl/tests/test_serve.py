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

"""Demo transport round-trip and engine gating (no GPU, stub policy)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))  # policy_server is a flat-layout package dir

import numpy as np  # noqa: E402
import torch  # noqa: E402

from so101_icl import demo_transport  # noqa: E402
from so101_icl.demo_transport import DemoTransportClient, DemoTransportServer  # noqa: E402

PORT = 18661


class _StubModel:
    def __init__(self):
        self.demo_pack_is_set = False
        self.pack = None

    def set_demo_pack(self, frames, mask, traj, traj_ok):
        self.pack = (frames.clone(), mask.clone(), traj.clone(), traj_ok.clone())
        self.demo_pack_is_set = True

    def clear_demo_pack(self):
        self.demo_pack_is_set = False
        self.pack = None


class _StubPolicy:
    def __init__(self):
        self.model = _StubModel()


class TestDemoTransport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo_transport._current_policy["policy"] = _StubPolicy()
        cls.model = demo_transport._current_policy["policy"].model
        cls.server = DemoTransportServer(port=PORT).start()

    @classmethod
    def tearDownClass(cls):
        demo_transport._current_policy["policy"] = None

    def test_status_before_set(self):
        status = DemoTransportClient(port=PORT).status()
        self.assertEqual(status["status"], "ok")
        self.assertTrue(status["policy_loaded"])
        self.assertFalse(status["pack_set"])

    def test_set_clear_roundtrip(self):
        rng = np.random.default_rng(0)
        frames = rng.uniform(-1, 1, (2, 6, 3, 224, 224)).astype(np.float32)
        traj = rng.uniform(-1, 1, (2, 16, 64)).astype(np.float32)
        traj_ok = np.array([1.0, 1.0], dtype=np.float32)
        client = DemoTransportClient(port=PORT)
        reply = client.set_demo_pack(frames, traj=traj, traj_ok=traj_ok, k_max=4)
        self.assertEqual(reply["status"], "ok")
        self.assertEqual(reply["k"], 2)
        self.assertGreaterEqual(reply["encode_s"], 0.0)

        got_frames, got_mask, got_traj, got_ok = self.model.pack
        self.assertEqual(tuple(got_frames.shape), (4, 6, 3, 224, 224))  # padded to k_max
        self.assertEqual(int(got_mask.sum()), 2)
        self.assertEqual(tuple(got_traj.shape), (4, 16, 64))
        self.assertEqual(int(got_ok.sum()), 2)
        torch.testing.assert_close(got_frames[0], torch.from_numpy(frames[0]))
        torch.testing.assert_close(got_traj[1], torch.from_numpy(traj[1]))
        self.assertTrue(DemoTransportClient(port=PORT).status()["pack_set"])

        self.assertEqual(client.clear()["status"], "ok")
        self.assertFalse(self.model.demo_pack_is_set)

    def test_engine_gate_accepts_pi05_icl(self):
        """The one repo edit: SUPPORTED_POLICIES must include pi05_icl."""
        from policy_server.inference_engine import SUPPORTED_POLICIES

        self.assertIn("pi05_icl", SUPPORTED_POLICIES)


if __name__ == "__main__":
    unittest.main()
