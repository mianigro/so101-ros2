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

"""Latency harness tests for the M3/M4 gates (no GPU, no sockets)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from so101_icl.latency import (  # noqa: E402
    LatencyRecorder,
    TimedRolloutClient,
    latency_gate,
)


class TestRecorder(unittest.TestCase):
    def test_empty_stats(self):
        self.assertEqual(LatencyRecorder("x").stats()["n"], 0)

    def test_percentiles(self):
        rec = LatencyRecorder()
        for value in range(1, 101):  # 1..100 ms
            rec.record(value / 1000.0)
        stats = rec.stats()
        self.assertEqual(stats["n"], 100)
        # linear interpolation, numpy-percentile semantics: median of 1..100
        self.assertAlmostEqual(stats["p50_s"], 0.0505, places=4)
        self.assertAlmostEqual(stats["p95_s"], 0.09505, places=4)
        self.assertAlmostEqual(stats["max_s"], 0.100, places=3)


class FakeClient:
    def __init__(self):
        self.chunk_size = 16
        self.action_dim = 6
        self.infer_calls = 0

    def hello(self, *a, **k):
        return {"type": "welcome"}

    def infer(self, *a, **k):
        self.infer_calls += 1
        return "chunk"

    def close(self):
        pass


class TestTimedClient(unittest.TestCase):
    def test_times_every_infer(self):
        client = FakeClient()
        timed = TimedRolloutClient(client)
        self.assertEqual(timed.hello()["type"], "welcome")
        self.assertEqual(timed.chunk_size, 16)
        for _ in range(5):
            self.assertEqual(timed.infer("obs", "state", "task"), "chunk")
        self.assertEqual(client.infer_calls, 5)
        self.assertEqual(timed.recorder.stats()["n"], 5)
        self.assertGreater(timed.recorder.stats()["p50_s"], 0.0)


class TestGate(unittest.TestCase):
    def test_pass(self):
        base = LatencyRecorder("bare_prompt")
        full = LatencyRecorder("full_icl")
        for _ in range(20):
            base.record(0.100)
            full.record(0.110)  # +10%, within +20%
        self.assertIn("PASS", latency_gate({"bare_prompt": base, "full_icl": full}))

    def test_fail(self):
        base = LatencyRecorder("bare_prompt")
        full = LatencyRecorder("full_icl")
        for _ in range(20):
            base.record(0.100)
            full.record(0.150)  # +50%
        self.assertIn("FAIL", latency_gate({"bare_prompt": base, "full_icl": full}))

    def test_not_evaluable(self):
        self.assertIn("not evaluable", latency_gate({}))


if __name__ == "__main__":
    unittest.main()
